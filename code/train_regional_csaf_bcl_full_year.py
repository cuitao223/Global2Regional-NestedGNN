"""Full-year regional training with CSAF + BCL."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Iterator, Tuple, List, Optional

import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

try:
    import torch_npu  # noqa: F401
    from torch_npu.npu import amp
    from torch_npu.npu.amp import GradScaler
except ImportError:  # CPU/CUDA fallback
    from torch.cuda import amp  # type: ignore
    from torch.cuda.amp import GradScaler  # type: ignore

from graphcast import data_pipeline, npz_utils, normalization, model_utils
from graphcast.graphcast import GraphCast
from graphcast.training_utils import (
    build_loss_weights,
    normalize_datasets_for_training,
    stack_targets_to_tensor,
    weighted_mse_loss,
)
from nested_utils import (
    RegionConfig,
    blend_boundary_on_last_timestep,
    compute_region_slices,
    create_boundary_mask,
    load_boundary_manifest,
    load_boundary_patch,
    load_regional_checkpoint,
    save_regional_checkpoint,
)

torch.set_default_dtype(torch.float32)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/myout")
DEFAULT_BOUNDARY_ROOT = Path("/root/autodl-tmp/precomputed_boundaries_parts")
DEFAULT_GLOBAL_PARAM = ROOT.parent / "data" / "params" / (
    "params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - "
    "precipitation output only.npz"
)
DEFAULT_STATS_DIR = Path("/root/data/stats")
DEFAULT_SAVE_PATH = ROOT.parent / "data" / "dataset" / "params" / "regional_graphcast_full_year_v2.pt"
DEFAULT_LOG_DIR = ROOT / "logs_full_year_v2"


def _setup_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29640")


def _default_cache_dir() -> Path:
    env_path = os.environ.get("GRAPHCAST_CACHE_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return ROOT / ".graphcast_cache"


def _build_cache_path(param_path: Path, cache_dir: Path) -> Path:
    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(param_path).encode("utf-8")).hexdigest()
    safe_stem = param_path.stem.replace(" ", "_")
    return cache_dir / f"{safe_stem}-{digest}.model.pt"


def _attention_path_for(base: Path) -> Path:
    """Derive an attention checkpoint path from the main checkpoint path."""
    return base.with_name(base.stem + "_attn" + base.suffix)


def _save_attention_checkpoint(base: Path, attention: nn.Module) -> None:
    """Save BoundaryCrossAttention parameters next to the regional checkpoint."""
    attn_path = _attention_path_for(base)
    attn_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": attention.state_dict()}, attn_path)


def _ensure_device(device_id: int | None) -> torch.device:
    if torch.npu.is_available():
        torch.npu.set_device(device_id or 0)
        return torch.device(f"npu:{device_id or 0}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id or 0)
        return torch.device(f"cuda:{device_id or 0}")
    return torch.device("cpu")


def _autocast_context(device: torch.device, use_amp: bool = True):
    if use_amp and device.type in {"npu", "cuda"}:
        return amp.autocast()
    return contextlib.nullcontext()


def _subset_stats(stats: xr.Dataset, lat_slice: slice, lon_slice: slice) -> xr.Dataset:
    subset = stats
    if "lat" in subset.dims:
        subset = subset.isel(lat=lat_slice, drop=False)
    if "lon" in subset.dims:
        subset = subset.isel(lon=lon_slice, drop=False)
    return subset


class Logger:
    """强制 CSV 记录（简单可靠，便于手动绘图）。"""

    def __init__(self, log_dir: Path, run_name: str = "train_full_year_v2", project: str = "graphcast"):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = "csv"
        self.csv_path = log_dir / "metrics.csv"
        self.csv_fh = self.csv_path.open("w", encoding="utf-8")
        self.csv_fh.write("step,tag,value\n")
        self.csv_fh.flush()
        self.writer = None
        self.wandb = None

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.csv_fh.write(f"{step},{name},{value}\n")
        self.csv_fh.flush()

    def log_dict(self, name: str, dict_obj: Dict[str, float], step: int) -> None:
        for k, v in dict_obj.items():
            self.log_scalar(f"{name}/{k}", v, step)

    def log_image(self, name: str, image, step: int) -> None:
        return

    def save_checkpoint(self, path: Path, state: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def close(self) -> None:
        if self.csv_fh:
            self.csv_fh.close()


def _compute_gpu_mem(device: torch.device) -> Dict[str, float]:
    if device.type == "cuda" and torch.cuda.is_available():
        return {
            "max_allocated": float(torch.cuda.max_memory_allocated(device)),
            "max_reserved": float(torch.cuda.max_memory_reserved(device)),
        }
    if device.type == "npu" and torch.npu.is_available():
        try:
            import torch_npu

            return {
                "max_allocated": float(torch_npu.npu.max_memory_allocated(device)),
                "max_reserved": float(torch_npu.npu.max_memory_reserved(device)),
            }
        except Exception:
            return {"max_allocated": 0.0, "max_reserved": 0.0}
    return {"max_allocated": 0.0, "max_reserved": 0.0}


def _prefetch_training_windows(
    windows: List[Tuple[int, xr.Dataset]],
    boundary_dir: Path,
    boundary_entries,
    task_config,
    target_lead_time: str,
    max_prefetch: int = 3,
):
    """Prefetch global inputs/targets/forcings + boundary patches on CPU in a background thread.

    This overlaps xarray/IO work with NPU compute so that the device spends
    less time等待数据准备。"""
    import threading
    from queue import Queue

    sentinel = object()
    queue: "Queue[object]" = Queue(maxsize=max_prefetch)

    def _worker() -> None:
        try:
            for idx, (_, window) in enumerate(windows):
                global_inputs, global_targets, global_forcings = data_pipeline.prepare_example(
                    window, task_config, target_lead_time
                )
                boundary_patch = load_boundary_patch(boundary_dir, boundary_entries[idx]).astype("float32")
                queue.put((idx, global_inputs, global_targets, global_forcings, boundary_patch))
        finally:
            queue.put(sentinel)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = queue.get()
        if item is sentinel:
            break
        yield item


def _iter_windows(
    dataset: xr.Dataset,
    input_steps: int,
    target_step: int,
) -> Iterator[Tuple[int, xr.Dataset]]:
    total = dataset.sizes["time"]
    for end_idx in range(input_steps - 1, total - target_step):
        start_idx = end_idx - input_steps + 1
        window = dataset.isel(time=slice(start_idx, end_idx + target_step + 1))
        yield end_idx, window


def _ensure_time_progress(ds: xr.Dataset) -> xr.Dataset:
    """Add day/year progress sin/cos if missing (robust to timedelta time)."""
    required = ("day_progress_cos", "day_progress_sin", "year_progress_cos", "year_progress_sin")
    if "time" not in ds.coords:
        return ds
    missing = [name for name in required if name not in ds]
    if not missing:
        return ds
    time_values = ds["time"].values
    try:
        time_dt = pd.to_datetime(time_values)
    except Exception:
        base = np.datetime64("1970-01-01")
        try:
            time_dt = pd.to_datetime(base + time_values.astype("timedelta64[ns]"))
        except Exception:
            time_dt = pd.date_range("1970-01-01", periods=len(time_values), freq="6H")
    seconds_in_day = 24 * 60 * 60
    seconds_since_midnight = time_dt.hour * 3600 + time_dt.minute * 60 + time_dt.second
    day_of_year = time_dt.dayofyear - 1
    return ds.assign(
        day_progress_cos=("time", np.cos(2 * np.pi * seconds_since_midnight / seconds_in_day)),
        day_progress_sin=("time", np.sin(2 * np.pi * seconds_since_midnight / seconds_in_day)),
        year_progress_cos=("time", np.cos(2 * np.pi * day_of_year / 365.25)),
        year_progress_sin=("time", np.sin(2 * np.pi * day_of_year / 365.25)),
    )


def _parse_month_from_name(path: Path) -> int:
    stem = path.stem  # e.g., source-era5_date-2024-10-01_steps-12_part01
    try:
        parts = stem.split("_")
        for p in parts:
            if p.startswith("date-"):
                date_str = p.replace("date-", "")
                return int(date_str.split("-")[1])
    except Exception:
        pass
    return 1


def _split_train_val(paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    train, val = [], []
    for p in paths:
        m = _parse_month_from_name(p)
        if m in (10, 11):
            val.append(p)
        else:
            train.append(p)
    return train, val


def _tensor_to_dataset(tensor: torch.Tensor, template: xr.Dataset) -> xr.Dataset:
    """Convert BHWC torch tensor to xr.Dataset following template shapes."""
    arr = tensor.detach().cpu().numpy()
    var = xr.Variable(("batch", "lat", "lon", "channels"), arr)
    return model_utils.stacked_to_dataset(var, template)


def _compute_var_metrics(pred: xr.Dataset, target: xr.Dataset, mask: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Compute RMSE/Bias/ACC per variable; mask for strip_RMSE if provided."""
    out: Dict[str, Dict[str, float]] = {}
    for name in target.data_vars:
        p = pred[name]
        t = target[name]
        if mask is not None:
            m = xr.DataArray(mask, dims=("lat", "lon"))
            # broadcast if level dim exists
            if "level" in p.dims:
                m = m.expand_dims({"level": p.sizes["level"]}, axis=0)
                m = m.transpose("level", "lat", "lon")
            p = p * m
            t = t * m
        diff = (p - t).astype("float64")
        rmse = float(np.sqrt(np.nanmean(diff.values ** 2)))
        bias = float(np.nanmean(diff.values))
        p_mean = float(np.nanmean(p.values))
        t_mean = float(np.nanmean(t.values))
        p_std = float(np.nanstd(p.values))
        t_std = float(np.nanstd(t.values))
        acc = 0.0
        if p_std > 0 and t_std > 0:
            acc = float(np.nanmean(((p.values - p_mean) * (t.values - t_mean))) / (p_std * t_std))
        out[name] = {"rmse": rmse, "bias": bias, "acc": acc}
    return out


def _filter_task_config(task_config, ds: xr.Dataset):
    """Drop missing input/forcing variables to mirror train_regional robustness."""
    available = set(ds.data_vars)
    filtered_inputs = tuple(v for v in task_config.input_variables if v in available)
    filtered_forcings = tuple(v for v in task_config.forcing_variables if v in available)
    missing_inputs = set(task_config.input_variables) - set(filtered_inputs)
    missing_forcings = set(task_config.forcing_variables) - set(filtered_forcings)
    if missing_inputs:
        print(f"[warn] dropping missing input variables: {sorted(missing_inputs)}")
    if missing_forcings:
        print(f"[warn] dropping missing forcing variables: {sorted(missing_forcings)}")
    from graphcast.graphcast import TaskConfig as GC_TaskConfig

    return GC_TaskConfig(
        input_variables=filtered_inputs,
        target_variables=task_config.target_variables,
        forcing_variables=filtered_forcings,
        pressure_levels=task_config.pressure_levels,
        input_duration=task_config.input_duration,
    )


def _assert_finite(tensor: torch.Tensor, name: str):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"NaN/Inf detected in {name}")


def _load_model_with_cache(
    model_config,
    task_config,
    param_path: Path,
    sample_inputs: xr.Dataset,
    sample_targets: xr.Dataset,
    sample_forcings: xr.Dataset,
    device: torch.device,
    cache_dir: Path,
) -> Tuple[GraphCast, str]:
    cache_path = _build_cache_path(param_path, cache_dir)
    status = "from_npz"
    model = GraphCast(model_config, task_config)
    if cache_path.exists():
        try:
            model = torch.load(cache_path, map_location="cpu")
            status = "from_cache"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to load cache ({exc}), regenerating.")
            cache_path.unlink(missing_ok=True)
            model = GraphCast(model_config, task_config)
    if status == "from_npz":
        model.load_haiku_parameters(
            str(param_path), sample_inputs, sample_targets, sample_forcings)
        torch.save(model, cache_path)
    model = model.to(device)
    return model, status


class BoundaryCrossAttention(nn.Module):
    """Lightweight multi-head attention to fuse boundary features."""

    def __init__(self, dim: int, heads: int = 2, alpha: float = 1.0):
        super().__init__()
        if dim % heads != 0:
            heads = 1
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = (self.dim_head) ** -0.5
        self.alpha = alpha
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.last_attn: torch.Tensor | None = None

    def forward(self, h_reg: torch.Tensor, h_bnd: torch.Tensor) -> torch.Tensor:
        B, N, D = h_reg.shape
        H = self.heads

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(B, N, H, self.dim_head).transpose(1, 2)  # [B, H, N, Dh]

        q = split_heads(self.to_q(h_reg))
        k = split_heads(self.to_k(h_bnd))
        v = split_heads(self.to_v(h_bnd))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn_scores, dim=-1)
        self.last_attn = attn.detach()
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, D)
        out = self.to_out(ctx)
        return h_reg + self.alpha * out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential regional GraphCast training over multiple 12-step datasets.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--boundary-root", type=Path, default=DEFAULT_BOUNDARY_ROOT)
    parser.add_argument("--global-param", type=Path, default=DEFAULT_GLOBAL_PARAM)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-lead-time", type=str, default="6h")
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--buffer", type=float, default=5.0)
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=(3.0, 25.0, 100.0, 125.0),
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    parser.add_argument(
        "--main-weight",
        type=float,
        default=2.0,
        help="Weight for 2m_temperature in loss.",
    )
    parser.add_argument(
        "--mslp-weight",
        type=float,
        default=1.0,
        help="Weight for mean_sea_level_pressure in loss.",
    )
    parser.add_argument(
        "--deep-temp-weight",
        type=float,
        default=1.5,
        help="Extra weight for 3D temperature in loss.",
    )
    parser.add_argument(
        "--w-vert-weight",
        type=float,
        default=2.0,
        help="Extra weight for vertical_velocity in loss.",
    )
    parser.add_argument(
        "--precip-weight",
        type=float,
        default=0.5,
        help="Weight for total_precipitation_6hr in loss.",
    )
    parser.add_argument("--lat-chunks", type=int, default=2,
                        help="Split regional domain into this many latitude slices.")
    parser.add_argument("--lon-chunks", type=int, default=1,
                        help="Split regional domain into this many longitude slices.")
    parser.add_argument("--lambda-b", type=float, default=0.3,
                        help="Weight for boundary-consistent loss (BCL).")
    parser.add_argument("--use-precomputed-boundary", action="store_true",
                        help="Flag for interface compatibility; v2 assumes precomputed boundaries for each dataset.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (amp/autocast).")
    parser.add_argument("--reg-alpha", type=float, default=0.0,
                        help="L2 regularization strength (0 to disable).")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--save-every", type=int, default=0,
                        help="Optional: save checkpoint every N steps (0 to disable).")
    return parser.parse_args()


def _list_datasets(data_root: Path) -> List[Path]:
    paths = list(data_root.glob("source-era5_date-*steps-12*.nc"))
    paths += list(data_root.glob("source-era5_date-*steps-12*.nc4"))
    return sorted(set(paths))


def main() -> None:
    args = parse_args()
    if args.lat_chunks < 1:
        raise ValueError("--lat-chunks must be positive.")
    if args.lon_chunks < 1:
        raise ValueError("--lon-chunks must be positive.")
    _setup_env()
    device = _ensure_device(args.device_id)
    logger = Logger(args.log_dir, run_name="train_full_year_v2")

    dataset_paths = _list_datasets(args.data_root)
    if not dataset_paths:
        raise FileNotFoundError("No 12-step datasets found in data-root.")
    train_paths, val_paths = _split_train_val(dataset_paths)
    if not train_paths:
        raise FileNotFoundError("No training datasets after split.")

    # use first dataset to init model
    sample_ds = xr.load_dataset(train_paths[0], decode_timedelta=True).fillna(0)
    sample_ds = _ensure_time_progress(sample_ds)
    model_config, task_config, _ = npz_utils.load_config_from_npz(str(args.global_param))
    task_config = _filter_task_config(task_config, sample_ds)
    time_values = sample_ds.coords["time"].values
    step = pd.to_timedelta(time_values[1] - time_values[0])
    input_duration = pd.Timedelta(task_config.input_duration)
    input_steps = int(input_duration / step) + 1
    target_step = int(pd.Timedelta(args.target_lead_time) / step)
    if target_step <= 0:
        raise ValueError("target_lead_time must be positive.")

    region_cfg = RegionConfig(
        lat_min=args.region[0],
        lat_max=args.region[1],
        lon_min=args.region[2],
        lon_max=args.region[3],
        buffer_deg=args.buffer,
    )

    lat_slice_ref, lon_slice_ref = compute_region_slices(sample_ds.lat.values, sample_ds.lon.values, region_cfg)
    patch_lat = sample_ds.lat.isel(lat=lat_slice_ref)
    patch_lon = sample_ds.lon.isel(lon=lon_slice_ref)
    boundary_mask_ref = create_boundary_mask(patch_lat, patch_lon, region_cfg)

    stats_mean_global, stats_std_global, diff_std_global = data_pipeline.load_stats(
        str(args.stats_dir), task_config)
    stats_mean_reg_ref = _subset_stats(stats_mean_global, lat_slice_ref, lon_slice_ref)
    stats_std_reg_ref = _subset_stats(stats_std_global, lat_slice_ref, lon_slice_ref)
    diff_std_reg_ref = _subset_stats(diff_std_global, lat_slice_ref, lon_slice_ref)

    sample_end_idx = input_steps - 1
    sample_window = sample_ds.isel(time=slice(sample_end_idx - input_steps + 1,
                                              sample_end_idx + target_step + 1))
    sample_inputs, sample_targets, sample_forcings = data_pipeline.prepare_example(
        sample_window, task_config, args.target_lead_time)
    sample_reg_inputs = sample_inputs.isel(lat=lat_slice_ref, lon=lon_slice_ref)
    sample_reg_targets = sample_targets.isel(lat=lat_slice_ref, lon=lon_slice_ref)
    sample_reg_forcings = sample_forcings.isel(lat=lat_slice_ref, lon=lon_slice_ref)

    global_model, cache_status = _load_model_with_cache(
        model_config, task_config, Path(args.global_param), sample_inputs, sample_targets,
        sample_forcings, device, Path(args.cache_dir))
    if hasattr(global_model, "eval"):
        global_model.eval()
    for parameter in global_model.parameters():
        parameter.requires_grad_(False)
    print(f"[info] Global GraphCast ready ({cache_status}).")

    if args.resume and Path(args.resume).exists():
        regional_model, model_config, task_config = load_regional_checkpoint(
            Path(args.resume), device, model_config, task_config,
            sample_reg_inputs, sample_reg_targets, sample_reg_forcings)
        print(f"[info] Loaded regional checkpoint from {args.resume}.")
    else:
        regional_model = GraphCast(model_config, task_config)
        regional_model.load_haiku_parameters(
            str(args.global_param), sample_inputs, sample_targets, sample_forcings)
        regional_model = regional_model.to(device)
    if hasattr(regional_model, "train"):
        regional_model.train()

    lat_len = sample_reg_inputs.sizes["lat"]
    lon_len = sample_reg_inputs.sizes["lon"]
    lat_chunks = min(lat_len, max(1, args.lat_chunks))
    lon_chunks = min(lon_len, max(1, args.lon_chunks))
    lat_indices = [idx for idx in np.array_split(np.arange(lat_len), lat_chunks) if idx.size > 0]
    lon_indices = [idx for idx in np.array_split(np.arange(lon_len), lon_chunks) if idx.size > 0]
    chunk_pairs = [
        (slice(int(lat_idx[0]), int(lat_idx[-1]) + 1),
         slice(int(lon_idx[0]), int(lon_idx[-1]) + 1))
        for lat_idx in lat_indices
        for lon_idx in lon_indices
    ]
    num_chunks = len(chunk_pairs)
    if num_chunks == 0:
        raise ValueError("Chunk configuration resulted in zero sub-domains.")
    print(f"[info] chunks -> lat: {lat_chunks}, lon: {lon_chunks}, total: {num_chunks}")

    loss_weights_chunks = []
    for lat_chunk_slice, lon_chunk_slice in chunk_pairs:
        chunk_inputs = sample_reg_inputs.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        chunk_targets = sample_reg_targets.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        chunk_forcings = sample_reg_forcings.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        chunk_boundary = sample_inputs.isel(lat=lat_slice_ref, lon=lon_slice_ref).isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        chunk_mask = boundary_mask_ref.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        blended_chunk_inputs = blend_boundary_on_last_timestep(
            chunk_inputs, chunk_boundary, chunk_mask)
        norm_chunk = normalize_datasets_for_training(
            blended_chunk_inputs, chunk_targets, chunk_forcings,
            _subset_stats(stats_mean_reg_ref, lat_chunk_slice, lon_chunk_slice),
            _subset_stats(stats_std_reg_ref, lat_chunk_slice, lon_chunk_slice),
            _subset_stats(diff_std_reg_ref, lat_chunk_slice, lon_chunk_slice),
        )
        loss_weights_chunks.append(
            build_loss_weights(
                norm_chunk[1],
                {
                    "2m_temperature": args.main_weight,
                    "10m_u_component_of_wind": 0.1,
                    "10m_v_component_of_wind": 0.1,
                    "mean_sea_level_pressure": args.mslp_weight,
                    "total_precipitation_6hr": args.precip_weight,
                    "temperature": args.deep_temp_weight,
                    "vertical_velocity": args.w_vert_weight,
                },
            )
        )

    sample_channels = stack_targets_to_tensor(norm_chunk[1]).shape[-1]
    attention = BoundaryCrossAttention(sample_channels, heads=2, alpha=1.0).to(device)
    # If resuming from an existing regional checkpoint, try to restore the
    # corresponding attention weights from the derived *_attn.pt file.
    if args.resume is not None:
        attn_resume_path = _attention_path_for(Path(args.resume))
        if attn_resume_path.exists():
            try:
                attn_payload = torch.load(attn_resume_path, map_location=device)
                attn_state = attn_payload.get("state_dict", attn_payload)
                attention.load_state_dict(attn_state)
                print(f"[info] Loaded attention checkpoint from {attn_resume_path}.")
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Failed to load attention checkpoint {attn_resume_path}: {exc}")

    trainable_params = list(regional_model.parameters()) + list(attention.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scaler = GradScaler()

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"[info] epoch {epoch} start")
        update_norm_accum = 0.0
        update_norm_count = 0
        for ds_idx, ds_path in enumerate(train_paths, start=1):
            print(f"[info] dataset {ds_idx}/{len(train_paths)} -> {ds_path.name}")
            ds = xr.load_dataset(ds_path, decode_timedelta=True).fillna(0)
            ds = _ensure_time_progress(ds)
            lat_slice, lon_slice = compute_region_slices(ds.lat.values, ds.lon.values, region_cfg)
            patch_lat = ds.lat.isel(lat=lat_slice)
            patch_lon = ds.lon.isel(lon=lon_slice)
            boundary_mask = create_boundary_mask(patch_lat, patch_lon, region_cfg)
            stats_mean_reg = _subset_stats(stats_mean_global, lat_slice, lon_slice)
            stats_std_reg = _subset_stats(stats_std_global, lat_slice, lon_slice)
            diff_std_reg = _subset_stats(diff_std_global, lat_slice, lon_slice)

            windows = list(_iter_windows(ds, input_steps, target_step))
            boundary_dir = args.boundary_root / ds_path.stem
            if not boundary_dir.exists():
                raise FileNotFoundError(f"boundary dir not found: {boundary_dir}")
            boundary_entries = load_boundary_manifest(boundary_dir)
            if len(boundary_entries) < len(windows):
                raise ValueError(f"boundary manifest entries {len(boundary_entries)} < required {len(windows)}")

            for idx, global_inputs, global_targets, global_forcings, boundary_patch in _prefetch_training_windows(
                windows, boundary_dir, boundary_entries, task_config, args.target_lead_time
            ):
                region_inputs = global_inputs.isel(lat=lat_slice, lon=lon_slice)
                region_targets = global_targets.isel(lat=lat_slice, lon=lon_slice)
                region_forcings = global_forcings.isel(lat=lat_slice, lon=lon_slice)

                optimizer.zero_grad(set_to_none=True)
                window_loss = 0.0
                main_loss_val = 0.0
                bcl_loss_val = 0.0
                boundary_rmse = 0.0
                fusion_rmse = 0.0
                var_metrics_sum: Dict[str, Dict[str, float]] = {}
                attn_entropy_val = 0.0
                attn_mean_val = 0.0
                attn_std_val = 0.0
                step_start = time.time()
                for chunk_idx, (lat_chunk_slice, lon_chunk_slice) in enumerate(chunk_pairs):
                    region_inputs_chunk = region_inputs.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                    region_targets_chunk = region_targets.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                    region_forcings_chunk = region_forcings.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                    boundary_chunk = boundary_patch.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                    mask_chunk = boundary_mask.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)

                    stats_mean_chunk = _subset_stats(stats_mean_reg, lat_chunk_slice, lon_chunk_slice)
                    stats_std_chunk = _subset_stats(stats_std_reg, lat_chunk_slice, lon_chunk_slice)
                    diff_std_chunk = _subset_stats(diff_std_reg, lat_chunk_slice, lon_chunk_slice)

                    blended_inputs_chunk = blend_boundary_on_last_timestep(
                        region_inputs_chunk, boundary_chunk, mask_chunk)

                    norm_reg_inputs, norm_reg_targets, norm_reg_forcings = normalize_datasets_for_training(
                        blended_inputs_chunk, region_targets_chunk, region_forcings_chunk,
                        stats_mean_chunk,
                        stats_std_chunk,
                        diff_std_chunk,
                    )
                    for name, da in norm_reg_inputs.data_vars.items():
                        if not np.isfinite(da.values).all():
                            raise SystemExit(f"NaN/Inf in norm_reg_inputs {name}")
                        if np.abs(da.values).max() > 1e4:
                            raise SystemExit(f"huge values in norm_reg_inputs {name} max {np.abs(da.values).max():.2e}")
                    for name, da in norm_reg_targets.data_vars.items():
                        if not np.isfinite(da.values).all():
                            raise SystemExit(f"NaN/Inf in norm_reg_targets {name}")
                    target_tensor = stack_targets_to_tensor(norm_reg_targets).to(device=device)

                    norm_boundary_chunk = {}
                    for name, data in boundary_chunk.data_vars.items():
                        if name in blended_inputs_chunk:
                            residual = data - blended_inputs_chunk[name].isel(time=-1)
                            if name == "specific_humidity":
                                r = residual.values
                                print("[DEBUG] residual specific_humidity:", "min", r.min(), "max", r.max(), "mean", r.mean())
                            norm_boundary_chunk[name] = normalization.normalize(residual, diff_std_chunk, None)
                        else:
                            norm_boundary_chunk[name] = normalization.normalize(data, stats_std_chunk, stats_mean_chunk)
                        arr = norm_boundary_chunk[name].values
                        if not np.isfinite(arr).all():
                            raise SystemExit(f"NaN/Inf in norm_boundary_chunk {name}")
                        if np.abs(arr).max() > 1e4:
                            print(f"[warn] large values in norm_boundary_chunk {name} max={np.abs(arr).max():.2e}")
                    norm_boundary_chunk = xr.Dataset(norm_boundary_chunk)
                    boundary_tensor = stack_targets_to_tensor(norm_boundary_chunk).to(device=device)

                    if hasattr(regional_model, "reset_grid_cache"):
                        regional_model.reset_grid_cache()
                    with _autocast_context(device, use_amp=not args.no_amp):
                        outputs = regional_model.forward_grid_node_outputs(
                            norm_reg_inputs, norm_reg_forcings)
                        outputs = outputs.to(device)
                        _assert_finite(outputs, "outputs")
                        pred_bhwc = outputs.permute(1, 0, 2).reshape(target_tensor.shape)
                        bnd_bhwc = boundary_tensor
                        reg_flat = pred_bhwc.reshape(pred_bhwc.shape[0], -1, pred_bhwc.shape[-1])
                        bnd_flat = bnd_bhwc.reshape(bnd_bhwc.shape[0], -1, bnd_bhwc.shape[-1])
                        fused_flat = attention(reg_flat, bnd_flat)
                        fused_bhwc = fused_flat.view_as(pred_bhwc)
                        fused_outputs = fused_bhwc.reshape(outputs.shape[1], outputs.shape[0], outputs.shape[2]).permute(1, 0, 2)
                        _assert_finite(fused_outputs, "fused_outputs")
                        loss_main = weighted_mse_loss(fused_outputs.float(), target_tensor, loss_weights_chunks[chunk_idx])

                        buffer_mask_tensor = torch.from_numpy(mask_chunk.values.astype(np.float32)).to(device)
                        if buffer_mask_tensor.dim() == 2:
                            buffer_mask_tensor = buffer_mask_tensor.unsqueeze(0).unsqueeze(-1)
                        if buffer_mask_tensor.shape != fused_bhwc.shape:
                            buffer_mask_tensor = buffer_mask_tensor.expand_as(fused_bhwc)
                        diff_b = (fused_bhwc - boundary_tensor) * buffer_mask_tensor
                        denom = buffer_mask_tensor.sum()
                        loss_bcl = (diff_b.pow(2).sum() / denom) if denom > 0 else torch.tensor(0.0, device=device)

                        # boundary vs GT & fused vs GT (buffer区) RMSE
                        if denom > 0:
                            diff_boundary_gt = (boundary_tensor - target_tensor) * buffer_mask_tensor
                            boundary_rmse += torch.sqrt(diff_boundary_gt.pow(2).sum() / denom).item()
                            diff_fused_gt = (fused_bhwc - target_tensor) * buffer_mask_tensor
                            fusion_rmse += torch.sqrt(diff_fused_gt.pow(2).sum() / denom).item()

                        # L2 正则
                        reg_l2 = torch.tensor(0.0, device=device)
                        if args.reg_alpha > 0:
                            for p in regional_model.parameters():
                                if p.requires_grad and p.ndim > 1:
                                    reg_l2 = reg_l2 + p.pow(2).sum()
                        reg_term = args.reg_alpha * reg_l2

                        loss = loss_main + args.lambda_b * loss_bcl + reg_term
                        window_loss += loss.detach().item()
                        main_loss_val += loss_main.detach().item()
                        bcl_loss_val += loss_bcl.detach().item()
                        loss = loss / num_chunks

                    # 逐变量指标（全域与buffer带）
                    pred_ds = _tensor_to_dataset(fused_bhwc, norm_reg_targets)
                    var_metrics = _compute_var_metrics(pred_ds, norm_reg_targets)
                    mask_np = mask_chunk.values.astype(np.float32)
                    strip_metrics = _compute_var_metrics(pred_ds, norm_reg_targets, mask=mask_np)
                    for name, vals in var_metrics.items():
                        acc = var_metrics_sum.get(name, {"rmse": 0.0, "bias": 0.0, "acc": 0.0, "count": 0})
                        acc["rmse"] += vals["rmse"]
                        acc["bias"] += vals["bias"]
                        acc["acc"] += vals["acc"]
                        acc["count"] += 1
                        var_metrics_sum[name] = acc
                        logger.log_dict(f"train_metrics/{name}", vals, global_step)
                        logger.log_scalar(f"train_metrics/{name}_strip_rmse", strip_metrics[name]["rmse"], global_step)

                    # 简单时序指标：仅记录当前步作为 6h
                    for name, vals in var_metrics.items():
                        logger.log_dict(f"roll/6h/{name}", vals, global_step)
                    scaler.scale(loss).backward()

                    if torch.npu.is_available() and device.type == "npu":
                        del outputs
                        torch.npu.empty_cache()
                    del (
                        loss,
                        target_tensor,
                        norm_reg_inputs,
                        norm_reg_targets,
                        norm_reg_forcings,
                        blended_inputs_chunk,
                        boundary_chunk,
                        mask_chunk,
                        region_inputs_chunk,
                        region_targets_chunk,
                        region_forcings_chunk,
                    )

                # gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                lr_now = optimizer.param_groups[0]["lr"]
                update_norm_accum += lr_now * float(grad_norm.item())
                update_norm_count += 1
                # attention统计（如果可用）
                if attention.last_attn is not None:
                    a = attention.last_attn
                    attn_mean_val = float(a.mean().item())
                    attn_std_val = float(a.std().item())
                    attn_entropy_val = float((-(a * (a + 1e-8).log()).sum(dim=-1)).mean().item())
                metrics_log = {
                    "loss/total": window_loss / num_chunks,
                    "loss/main": main_loss_val / num_chunks,
                    "loss/boundary": bcl_loss_val / num_chunks,
                    "loss/reg_l2": float(reg_term.item()) if args.reg_alpha > 0 else 0.0,
                    "optim/grad_norm": float(grad_norm.item()),
                    "optim/lr": lr_now,
                    "time/batch": time.time() - step_start,
                    "boundary/rmse_patch": boundary_rmse / num_chunks,
                    "boundary/rmse_fused": fusion_rmse / num_chunks,
                    "attn/entropy": attn_entropy_val,
                    "attn/mean": attn_mean_val,
                    "attn/std": attn_std_val,
                }
                logger.log_dict("train", metrics_log, global_step)
                logger.log_dict("gpu", _compute_gpu_mem(device), global_step)
                if var_metrics_sum:
                    for name, vals in var_metrics_sum.items():
                        cnt = max(1, vals["count"])
                        logger.log_dict(f"train_metrics/{name}", {
                            "rmse": vals["rmse"] / cnt,
                            "bias": vals["bias"] / cnt,
                            "acc": vals["acc"] / cnt,
                        }, global_step)

                if window_loss / num_chunks < best_loss:
                    best_loss = window_loss / num_chunks
                    ckpt_base = Path(args.save_path)
                    save_regional_checkpoint(ckpt_base, regional_model, model_config, task_config)
                    _save_attention_checkpoint(ckpt_base, attention)

                if args.save_every and global_step % args.save_every == 0:
                    ckpt_path = Path(args.save_path).with_name(f"step{global_step}.pt")
                    save_regional_checkpoint(ckpt_path, regional_model, model_config, task_config)
                    _save_attention_checkpoint(ckpt_path, attention)
                    logger.save_checkpoint(
                        ckpt_path.with_suffix(".optim.pt"),
                        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": global_step},
                    )
                print(f"[info] epoch {epoch} ds {ds_idx} win {idx+1}/{len(windows)} loss {window_loss/num_chunks:.6f}")
                del boundary_patch
                if torch.npu.is_available() and device.type == "npu":
                    torch.npu.empty_cache()

            # save after each dataset (graph + attention)
            ckpt_base = Path(args.save_path)
            save_regional_checkpoint(ckpt_base, regional_model, model_config, task_config)
            _save_attention_checkpoint(ckpt_base, attention)
            print(f"[info] checkpoint saved after dataset {ds_path.name} -> {args.save_path}")
        # validation
        if val_paths:
            val_total = val_main = val_bcl = 0.0
            val_count = 0
            val_bnd_rmse = 0.0
            val_fused_rmse = 0.0
            val_var_sum: Dict[str, Dict[str, float]] = {}
            with torch.no_grad():
                for v_idx, v_path in enumerate(val_paths, start=1):
                    print(f"[val] dataset {v_idx}/{len(val_paths)} -> {v_path.name}")
                    ds = xr.load_dataset(v_path, decode_timedelta=True).fillna(0)
                    ds = _ensure_time_progress(ds)
                    lat_slice, lon_slice = compute_region_slices(ds.lat.values, ds.lon.values, region_cfg)
                    patch_lat = ds.lat.isel(lat=lat_slice)
                    patch_lon = ds.lon.isel(lon=lon_slice)
                    boundary_mask = create_boundary_mask(patch_lat, patch_lon, region_cfg)
                    stats_mean_reg = _subset_stats(stats_mean_global, lat_slice, lon_slice)
                    stats_std_reg = _subset_stats(stats_std_global, lat_slice, lon_slice)
                    diff_std_reg = _subset_stats(diff_std_global, lat_slice, lon_slice)
                    windows = list(_iter_windows(ds, input_steps, target_step))
                    boundary_dir = args.boundary_root / v_path.stem
                    boundary_entries = load_boundary_manifest(boundary_dir)
                    for idx, (_, window) in enumerate(windows):
                        global_inputs, global_targets, global_forcings = data_pipeline.prepare_example(
                            window, task_config, args.target_lead_time)
                        region_inputs = global_inputs.isel(lat=lat_slice, lon=lon_slice)
                        region_targets = global_targets.isel(lat=lat_slice, lon=lon_slice)
                        region_forcings = global_forcings.isel(lat=lat_slice, lon=lon_slice)
                        boundary_patch = load_boundary_patch(boundary_dir, boundary_entries[idx]).astype("float32")

                        window_loss = main_l = bcl_l = 0.0
                        b_rmse = f_rmse = 0.0
                        for chunk_idx, (lat_chunk_slice, lon_chunk_slice) in enumerate(chunk_pairs):
                            region_inputs_chunk = region_inputs.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                            region_targets_chunk = region_targets.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                            region_forcings_chunk = region_forcings.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                            boundary_chunk = boundary_patch.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                            mask_chunk = boundary_mask.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)

                            stats_mean_chunk = _subset_stats(stats_mean_reg, lat_chunk_slice, lon_chunk_slice)
                            stats_std_chunk = _subset_stats(stats_std_reg, lat_chunk_slice, lon_chunk_slice)
                            diff_std_chunk = _subset_stats(diff_std_reg, lat_chunk_slice, lon_chunk_slice)

                            blended_inputs_chunk = blend_boundary_on_last_timestep(
                                region_inputs_chunk, boundary_chunk, mask_chunk)
                            norm_reg_inputs, norm_reg_targets, norm_reg_forcings = normalize_datasets_for_training(
                                blended_inputs_chunk, region_targets_chunk, region_forcings_chunk,
                                stats_mean_chunk, stats_std_chunk, diff_std_chunk)
                            target_tensor = stack_targets_to_tensor(norm_reg_targets).to(device=device)

                            norm_boundary_chunk = {}
                            for name, data in boundary_chunk.data_vars.items():
                                if name in blended_inputs_chunk:
                                    residual = data - blended_inputs_chunk[name].isel(time=-1)
                                    norm_boundary_chunk[name] = normalization.normalize(residual, diff_std_chunk, None)
                                else:
                                    norm_boundary_chunk[name] = normalization.normalize(data, stats_std_chunk, stats_mean_chunk)
                            norm_boundary_chunk = xr.Dataset(norm_boundary_chunk)
                            boundary_tensor = stack_targets_to_tensor(norm_boundary_chunk).to(device=device)

                            outputs = regional_model.forward_grid_node_outputs(norm_reg_inputs, norm_reg_forcings)
                            outputs = outputs.to(device)
                            pred_bhwc = outputs.permute(1, 0, 2).reshape(target_tensor.shape)
                            bnd_bhwc = boundary_tensor
                            reg_flat = pred_bhwc.reshape(pred_bhwc.shape[0], -1, pred_bhwc.shape[-1])
                            bnd_flat = bnd_bhwc.reshape(bnd_bhwc.shape[0], -1, bnd_bhwc.shape[-1])
                            fused_flat = attention(reg_flat, bnd_flat)
                            fused_bhwc = fused_flat.view_as(pred_bhwc)
                            fused_outputs = fused_bhwc.reshape(outputs.shape[1], outputs.shape[0], outputs.shape[2]).permute(1, 0, 2)
                            loss_main = weighted_mse_loss(fused_outputs.float(), target_tensor, loss_weights_chunks[chunk_idx])

                            buffer_mask_tensor = torch.from_numpy(mask_chunk.values.astype(np.float32)).to(device)
                            if buffer_mask_tensor.dim() == 2:
                                buffer_mask_tensor = buffer_mask_tensor.unsqueeze(0).unsqueeze(-1)
                            if buffer_mask_tensor.shape != fused_bhwc.shape:
                                buffer_mask_tensor = buffer_mask_tensor.expand_as(fused_bhwc)
                            diff_b = (fused_bhwc - boundary_tensor) * buffer_mask_tensor
                            denom = buffer_mask_tensor.sum()
                            loss_bcl = (diff_b.pow(2).sum() / denom) if denom > 0 else torch.tensor(0.0, device=device)

                            if denom > 0:
                                diff_boundary_gt = (boundary_tensor - target_tensor) * buffer_mask_tensor
                                b_rmse += torch.sqrt(diff_boundary_gt.pow(2).sum() / denom).item()
                                diff_fused_gt = (fused_bhwc - target_tensor) * buffer_mask_tensor
                                f_rmse += torch.sqrt(diff_fused_gt.pow(2).sum() / denom).item()

                            window_loss += loss_main.detach().item() + args.lambda_b * loss_bcl.detach().item()
                            main_l += loss_main.detach().item()
                            bcl_l += loss_bcl.detach().item()

                            pred_ds = _tensor_to_dataset(fused_bhwc, norm_reg_targets)
                            var_metrics = _compute_var_metrics(pred_ds, norm_reg_targets)
                            mask_np = mask_chunk.values.astype(np.float32)
                            strip_metrics = _compute_var_metrics(pred_ds, norm_reg_targets, mask=mask_np)
                            for name, vals in var_metrics.items():
                                acc = val_var_sum.get(name, {"rmse": 0.0, "bias": 0.0, "acc": 0.0, "count": 0})
                                acc["rmse"] += vals["rmse"]
                                acc["bias"] += vals["bias"]
                                acc["acc"] += vals["acc"]
                                acc["count"] += 1
                                val_var_sum[name] = acc
                                logger.log_dict(f"val_metrics/{name}", vals, global_step)
                                logger.log_scalar(f"val_metrics/{name}_strip_rmse", strip_metrics[name]["rmse"], global_step)

                        val_total += window_loss / num_chunks
                        val_main += main_l / num_chunks
                        val_bcl += bcl_l / num_chunks
                        val_bnd_rmse += b_rmse / num_chunks
                        val_fused_rmse += f_rmse / num_chunks
                        val_count += 1
            if val_count > 0:
                logger.log_dict("val", {
                    "loss/total": val_total / val_count,
                    "loss/main": val_main / val_count,
                    "loss/boundary": val_bcl / val_count,
                    "boundary/rmse_patch": val_bnd_rmse / val_count,
                    "boundary/rmse_fused": val_fused_rmse / val_count,
                }, global_step)
                for name, vals in val_var_sum.items():
                    cnt = max(1, vals["count"])
                    logger.log_dict(f"val_epoch/{name}", {
                        "rmse": vals["rmse"] / cnt,
                        "bias": vals["bias"] / cnt,
                        "acc": vals["acc"] / cnt,
                    }, global_step)

        avg_update_norm = update_norm_accum / max(update_norm_count, 1)
        logger.log_dict("epoch", {"update_norm": avg_update_norm}, epoch)

    ckpt_base = Path(args.save_path)
    save_regional_checkpoint(ckpt_base, regional_model, model_config, task_config)
    _save_attention_checkpoint(ckpt_base, attention)
    print(f"[done] training finished, final checkpoint at {args.save_path}")
    logger.close()


if __name__ == "__main__":
    main()
