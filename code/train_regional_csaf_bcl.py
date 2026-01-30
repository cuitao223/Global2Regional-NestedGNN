"""Train a regional GraphCast model with CSAF + BCL."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Iterator, Tuple

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

from graphcast import data_pipeline, npz_utils, normalization
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
DATA_DIR = ROOT.parent / "data"
DEFAULT_DATASET = DATA_DIR / "dataset" / "source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc"
DEFAULT_GLOBAL_PARAM = DATA_DIR / "params" / (
    "params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - "
    "precipitation output only.npz"
)
DEFAULT_STATS_DIR = Path("/root/data/stats")
DEFAULT_SAVE_DIR = DATA_DIR / "dataset" / "params"


def _setup_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29610")


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


def _ensure_device(device_id: int | None) -> torch.device:
    if torch.npu.is_available():
        torch.npu.set_device(device_id or 0)
        return torch.device(f"npu:{device_id or 0}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id or 0)
        return torch.device(f"cuda:{device_id or 0}")
    return torch.device("cpu")


def _denormalize_predictions(
    norm_predictions: xr.Dataset,
    inputs: xr.Dataset,
    stats_mean: xr.Dataset,
    stats_std: xr.Dataset,
    diff_std: xr.Dataset,
) -> xr.Dataset:
    restored = {}
    for name, norm_var in norm_predictions.data_vars.items():
        wrapped = xr.Dataset({name: norm_var})
        if name in inputs.data_vars:
            residual = normalization.unnormalize(wrapped, diff_std, None)[name]
            baseline = inputs[name].isel(time=-1)
            restored[name] = residual + baseline
        else:
            restored[name] = normalization.unnormalize(wrapped, stats_std, stats_mean)[name]
    return xr.Dataset(restored)


def _grid_outputs_to_predictions(
    model: GraphCast,
    grid_outputs: torch.Tensor,
    norm_targets: xr.Dataset,
    inputs: xr.Dataset,
    stats_mean: xr.Dataset,
    stats_std: xr.Dataset,
    diff_std: xr.Dataset,
) -> xr.Dataset:
    norm_predictions = model._grid_node_outputs_to_prediction(  # pylint: disable=protected-access
        grid_outputs, norm_targets
    )
    return _denormalize_predictions(norm_predictions, inputs, stats_mean, stats_std, diff_std)


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


def _autocast_context(device: torch.device):
    if device.type == "npu":
        return amp.autocast()
    if device.type == "cuda":
        return amp.autocast()
    return contextlib.nullcontext()


def _subset_stats(stats: xr.Dataset, lat_slice: slice, lon_slice: slice) -> xr.Dataset:
    subset = stats
    if "lat" in subset.dims:
        subset = subset.isel(lat=lat_slice, drop=False)
    if "lon" in subset.dims:
        subset = subset.isel(lon=lon_slice, drop=False)
    return subset


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Regional GraphCast with nested boundary.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--global-param", type=Path, default=DEFAULT_GLOBAL_PARAM)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_DIR / "regional_graphcast.pt")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-lead-time", type=str, default="6h")
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument(
        "--boundary-dir",
        type=Path,
        default=None,
        help="Directory containing precomputed boundaries (manifest.json).",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--buffer", type=float, default=5.0)
    parser.add_argument("--region", type=float, nargs=4, default=(3.0, 25.0, 100.0, 125.0),
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--main-weight", type=float, default=1.0,
                        help="Weight for 2m_temperature in loss.")
    parser.add_argument("--lat-chunks", type=int, default=2,
                        help="Split regional domain into this many latitude slices for training.")
    parser.add_argument("--lon-chunks", type=int, default=1,
                        help="Split regional domain into this many longitude slices for training.")
    parser.add_argument("--lambda-b", type=float, default=1.0,
                        help="Weight for boundary-consistent loss (BCL).")
    return parser.parse_args()


class BoundaryCrossAttention(nn.Module):
    """Lightweight multi-head attention to fuse boundary features."""

    def __init__(self, dim: int, heads: int = 2, alpha: float = 1.0):
        super().__init__()
        if dim % heads != 0:
            heads = 1  # fall back to single head if not divisible
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = (self.dim_head) ** -0.5
        self.alpha = alpha
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, h_reg: torch.Tensor, h_bnd: torch.Tensor) -> torch.Tensor:
        # h_reg, h_bnd: [B, N, D]
        B, N, D = h_reg.shape
        H = self.heads
        def split_heads(x):
            return x.view(B, N, H, self.dim_head).transpose(1, 2)  # [B, H, N, Dh]
        q = split_heads(self.to_q(h_reg))
        k = split_heads(self.to_k(h_bnd))
        v = split_heads(self.to_v(h_bnd))
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = torch.softmax(attn_scores, dim=-1)
        ctx = torch.matmul(attn, v)  # [B, H, N, Dh]
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, D)
        out = self.to_out(ctx)
        return h_reg + self.alpha * out


def main() -> None:
    args = parse_args()
    if args.lat_chunks < 1:
        raise ValueError("--lat-chunks 必须为正整数。")
    if args.lon_chunks < 1:
        raise ValueError("--lon-chunks 必须为正整数。")
    _setup_env()
    device = _ensure_device(args.device_id)

    use_precomputed = args.boundary_dir is not None
    boundary_dir: Path | None = None
    boundary_entries = []
    if use_precomputed:
        boundary_dir = Path(args.boundary_dir).expanduser()
        boundary_entries = load_boundary_manifest(boundary_dir)

    region_cfg = RegionConfig(
        lat_min=args.region[0],
        lat_max=args.region[1],
        lon_min=args.region[2],
        lon_max=args.region[3],
        buffer_deg=args.buffer,
    )

    dataset = data_pipeline.load_dataset(str(args.dataset))
    lat_slice, lon_slice = compute_region_slices(dataset.lat.values, dataset.lon.values, region_cfg)
    patch_lat = dataset.lat.isel(lat=lat_slice)
    patch_lon = dataset.lon.isel(lon=lon_slice)
    boundary_mask = create_boundary_mask(patch_lat, patch_lon, region_cfg)

    model_config, task_config, _ = npz_utils.load_config_from_npz(str(args.global_param))
    stats_mean_global, stats_std_global, diff_std_global = data_pipeline.load_stats(
        str(args.stats_dir), task_config)
    stats_mean_reg, stats_std_reg, diff_std_reg = data_pipeline.load_stats(
        str(args.stats_dir), task_config, lat_slice=lat_slice, lon_slice=lon_slice)

    time_values = dataset.coords["time"].values
    if len(time_values) < 4:
        raise ValueError("Dataset must contain at least four time steps.")
    step = pd.to_timedelta(time_values[1] - time_values[0])
    input_duration = pd.Timedelta(task_config.input_duration)
    input_steps = int(input_duration / step) + 1
    target_step = int(pd.Timedelta(args.target_lead_time) / step)
    if target_step <= 0:
        raise ValueError("target_lead_time must be positive.")

    windows = list(_iter_windows(dataset, input_steps, target_step))
    if not windows:
        raise ValueError("No valid training windows were generated from the dataset.")

    if use_precomputed and len(boundary_entries) < len(windows):
        raise ValueError(
            f"Boundary manifest contains {len(boundary_entries)} entries, "
            f"but {len(windows)} windows are required.")

    sample_end_idx = input_steps - 1
    sample_window = dataset.isel(time=slice(sample_end_idx - input_steps + 1,
                                            sample_end_idx + target_step + 1))
    sample_inputs, sample_targets, sample_forcings = data_pipeline.prepare_example(
        sample_window, task_config, args.target_lead_time)
    sample_reg_inputs = sample_inputs.isel(lat=lat_slice, lon=lon_slice)
    sample_reg_targets = sample_targets.isel(lat=lat_slice, lon=lon_slice)
    sample_reg_forcings = sample_forcings.isel(lat=lat_slice, lon=lon_slice)

    global_model: GraphCast | None = None
    if not use_precomputed:
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

    if use_precomputed:
        sample_boundary = load_boundary_patch(boundary_dir, boundary_entries[0]).astype("float32")
    else:
        norm_sample_inputs, norm_sample_targets, norm_sample_forcings = normalize_datasets_for_training(
            sample_inputs, sample_targets, sample_forcings,
            stats_mean_global, stats_std_global, diff_std_global)
        with torch.no_grad():
            with _autocast_context(device):
                sample_raw = global_model.forward_grid_node_outputs(
                    norm_sample_inputs, norm_sample_forcings)
            sample_pred = _grid_outputs_to_predictions(
                global_model, sample_raw, norm_sample_targets,
                sample_inputs, stats_mean_global, stats_std_global, diff_std_global)
        sample_boundary = sample_pred.isel(lat=lat_slice, lon=lon_slice).astype("float32")

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
        chunk_boundary = sample_boundary.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        chunk_targets = sample_reg_targets.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        chunk_forcings = sample_reg_forcings.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        chunk_mask = boundary_mask.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
        blended_chunk_inputs = blend_boundary_on_last_timestep(
            chunk_inputs, chunk_boundary, chunk_mask)
        norm_chunk = normalize_datasets_for_training(
            blended_chunk_inputs, chunk_targets, chunk_forcings,
            _subset_stats(stats_mean_reg, lat_chunk_slice, lon_chunk_slice),
            _subset_stats(stats_std_reg, lat_chunk_slice, lon_chunk_slice),
            _subset_stats(diff_std_reg, lat_chunk_slice, lon_chunk_slice),
        )
        loss_weights_chunks.append(build_loss_weights(norm_chunk[1], {
            "2m_temperature": args.main_weight,
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation_6hr": 0.1,
        }))

    sample_channels = stack_targets_to_tensor(norm_chunk[1]).shape[-1]
    attention = BoundaryCrossAttention(sample_channels, heads=2, alpha=1.0).to(device)

    trainable_params = list(regional_model.parameters()) + list(attention.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scaler = GradScaler()

    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        if hasattr(regional_model, "train"):
            regional_model.train()
        epoch_loss = 0.0
        sample_count = 0
        for idx, (_, window) in enumerate(windows):
            global_inputs, global_targets, global_forcings = data_pipeline.prepare_example(
                window, task_config, args.target_lead_time)
            region_inputs = global_inputs.isel(lat=lat_slice, lon=lon_slice)
            region_targets = global_targets.isel(lat=lat_slice, lon=lon_slice)
            region_forcings = global_forcings.isel(lat=lat_slice, lon=lon_slice)

            if use_precomputed:
                boundary_patch = load_boundary_patch(boundary_dir, boundary_entries[idx]).astype("float32")
            else:
                norm_global_inputs, norm_global_targets, norm_global_forcings = normalize_datasets_for_training(
                    global_inputs, global_targets, global_forcings,
                    stats_mean_global, stats_std_global, diff_std_global)

                with torch.no_grad():
                    with _autocast_context(device):
                        raw_global = global_model.forward_grid_node_outputs(
                            norm_global_inputs, norm_global_forcings)
                    pred_global = _grid_outputs_to_predictions(
                        global_model, raw_global, norm_global_targets,
                        global_inputs, stats_mean_global, stats_std_global, diff_std_global)
                boundary_patch = pred_global.isel(lat=lat_slice, lon=lon_slice).astype("float32")
                del raw_global, pred_global

            optimizer.zero_grad(set_to_none=True)

            window_loss = 0.0
            for chunk_idx, (lat_chunk_slice, lon_chunk_slice) in enumerate(chunk_pairs):
                region_inputs_chunk = region_inputs.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                region_targets_chunk = region_targets.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                region_forcings_chunk = region_forcings.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                boundary_chunk = boundary_patch.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)
                mask_chunk = boundary_mask.isel(lat=lat_chunk_slice, lon=lon_chunk_slice)

                blended_inputs_chunk = blend_boundary_on_last_timestep(
                    region_inputs_chunk, boundary_chunk, mask_chunk)

                norm_reg_inputs, norm_reg_targets, norm_reg_forcings = normalize_datasets_for_training(
                    blended_inputs_chunk, region_targets_chunk, region_forcings_chunk,
                    _subset_stats(stats_mean_reg, lat_chunk_slice, lon_chunk_slice),
                    _subset_stats(stats_std_reg, lat_chunk_slice, lon_chunk_slice),
                    _subset_stats(diff_std_reg, lat_chunk_slice, lon_chunk_slice),
                )
                target_tensor = stack_targets_to_tensor(norm_reg_targets).to(device=device)

                # Normalize boundary patch for buffer-consistent loss.
                def _normalize_boundary(
                    boundary: xr.Dataset,
                    inputs: xr.Dataset,
                    stats_mean: xr.Dataset,
                    stats_std: xr.Dataset,
                    diff_std: xr.Dataset,
                ) -> xr.Dataset:
                    norm_vars = {}
                    for name, data in boundary.data_vars.items():
                        if name in inputs:
                            residual = data - inputs[name].isel(time=-1)
                            norm_vars[name] = normalization.normalize(residual, diff_std, None)
                        else:
                            norm_vars[name] = normalization.normalize(data, stats_std, stats_mean)
                    return xr.Dataset(norm_vars)

                norm_boundary_chunk = _normalize_boundary(
                    boundary_chunk,
                    blended_inputs_chunk,
                    _subset_stats(stats_mean_reg, lat_chunk_slice, lon_chunk_slice),
                    _subset_stats(stats_std_reg, lat_chunk_slice, lon_chunk_slice),
                    _subset_stats(diff_std_reg, lat_chunk_slice, lon_chunk_slice),
                )
                boundary_tensor = stack_targets_to_tensor(norm_boundary_chunk).to(device=device)

                if hasattr(regional_model, "reset_grid_cache"):
                    regional_model.reset_grid_cache()
                with amp.autocast():
                    outputs = regional_model.forward_grid_node_outputs(
                        norm_reg_inputs, norm_reg_forcings)
                    outputs = outputs.to(device)
                    # Cross-scale attention fusion with boundary features.
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

                    loss = loss_main + args.lambda_b * loss_bcl
                    window_loss += loss.detach().item()
                    loss = loss / num_chunks
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

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += window_loss / num_chunks
            sample_count += 1
            del boundary_patch
            if torch.npu.is_available() and device.type == "npu":
                torch.npu.empty_cache()

        avg_loss = epoch_loss / max(sample_count, 1)
        print(f"[info] epoch {epoch} | loss {avg_loss:.6f} | samples {sample_count}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_regional_checkpoint(Path(args.save_path), regional_model, model_config, task_config)
            print(f"[info] saved checkpoint to {args.save_path}")

    elapsed = time.time() - start_time
    print(f"[done] training finished in {elapsed/60:.2f} minutes, best loss {best_loss:.6f}")


if __name__ == "__main__":
    main()
