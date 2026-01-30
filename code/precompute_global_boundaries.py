"""Precompute regional boundary patches from global GraphCast."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Tuple

import contextlib
import pandas as pd
import torch
import xarray as xr

try:
    import torch_npu  # noqa: F401
    from torch_npu.npu import amp
except ImportError:  # CPU / CUDA fallback
    from torch.cuda import amp  # type: ignore

try:
    import h5netcdf  # noqa: F401
    _H5NETCDF_AVAILABLE = True
except ImportError:
    _H5NETCDF_AVAILABLE = False

try:
    import netCDF4  # noqa: F401
    _NETCDF4_AVAILABLE = True
except ImportError:
    _NETCDF4_AVAILABLE = False

from graphcast import data_pipeline, npz_utils, normalization
from graphcast.graphcast import GraphCast
from graphcast.training_utils import normalize_datasets_for_training
from nested_utils import (
    RegionConfig,
    compute_region_slices,
)

torch.set_default_dtype(torch.float32)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
DEFAULT_DATASET = DATA_DIR / "dataset" / "source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc"
DEFAULT_GLOBAL_PARAM = DATA_DIR / "params" / (
    "params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - "
    "precipitation output only.npz"
)
DEFAULT_STATS_DIR = DATA_DIR / "stats"
DEFAULT_OUTPUT_DIR = DATA_DIR / "dataset" / "precomputed_boundaries"


def _setup_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29620")


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


def _ensure_device(device_type: str, device_id: int | None) -> torch.device:
    device_type = device_type.lower()
    if device_type == "cpu":
        return torch.device("cpu")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        torch.cuda.set_device(device_id or 0)
        return torch.device(f"cuda:{device_id or 0}")
    if device_type == "npu":
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch.npu.is_available() is False.")
        torch.npu.set_device(device_id or 0)
        return torch.device(f"npu:{device_id or 0}")
    if torch.npu.is_available():
        torch.npu.set_device(device_id or 0)
        return torch.device(f"npu:{device_id or 0}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id or 0)
        return torch.device(f"cuda:{device_id or 0}")
    return torch.device("cpu")


def _autocast_context(device: torch.device):
    if device.type == "npu":
        return amp.autocast()
    if device.type == "cuda":
        return amp.autocast()
    return contextlib.nullcontext()


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


@dataclass
class BoundaryRecord:
    index: int
    file: str
    forecast_reference_time: str
    target_time: str
    lead_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute nested GraphCast boundaries.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--global-param", type=Path, default=DEFAULT_GLOBAL_PARAM)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument("--device-type", type=str, default="auto",
                        choices=("auto", "cpu", "cuda", "npu"),
                        help="Device type for global inference (default auto).")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Device index when using CUDA/NPU (ignored for CPU).")
    parser.add_argument("--target-lead-time", type=str, default="6h")
    parser.add_argument("--buffer", type=float, default=5.0)
    parser.add_argument("--region", type=float, nargs=4, default=(3.0, 25.0, 100.0, 125.0),
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Optional cap on the number of windows to process.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip windows when the boundary file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_env()
    device = _ensure_device(args.device_type, args.device_id)
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

    model_config, task_config, _ = npz_utils.load_config_from_npz(str(args.global_param))
    stats_mean, stats_std, diff_std = data_pipeline.load_stats(
        str(args.stats_dir), task_config)

    time_values = dataset.coords["time"].values
    if len(time_values) < 4:
        raise ValueError("Dataset must contain at least four time steps.")
    step = pd.to_timedelta(time_values[1] - time_values[0])
    input_duration = pd.Timedelta(task_config.input_duration)
    input_steps = int(input_duration / step) + 1
    target_step = int(pd.Timedelta(args.target_lead_time) / step)
    if target_step <= 0:
        raise ValueError("target_lead_time must be positive.")

    sample_end_idx = input_steps - 1
    sample_window = dataset.isel(time=slice(sample_end_idx - input_steps + 1,
                                            sample_end_idx + target_step + 1))
    sample_inputs, sample_targets, sample_forcings = data_pipeline.prepare_example(
        sample_window, task_config, args.target_lead_time)
    global_model, cache_status = _load_model_with_cache(
        model_config, task_config, Path(args.global_param),
        sample_inputs, sample_targets, sample_forcings,
        device, Path(args.cache_dir))
    if hasattr(global_model, "eval"):
        global_model.eval()
    for parameter in global_model.parameters():
        parameter.requires_grad_(False)
    print(f"[info] Global GraphCast ready ({cache_status}).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = list(_iter_windows(dataset, input_steps, target_step))
    total_windows = len(windows)
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    print(f"[info] Preparing {len(windows)} windows (from {total_windows} available).")

    if _H5NETCDF_AVAILABLE:
        output_engine = "h5netcdf"
    elif _NETCDF4_AVAILABLE:
        output_engine = "netcdf4"
    else:
        raise RuntimeError(
            "Neither 'h5netcdf' nor 'netCDF4' backend is available. "
            "Please install one of them to save boundary files, e.g. "
            "'pip install h5netcdf' (recommended) or 'pip install netCDF4'."
        )

    manifest: list[BoundaryRecord] = []
    for idx, (end_idx, window) in enumerate(windows):
        forecast_reference = window.coords["datetime"].isel(batch=0, time=input_steps - 1).item()
        target_time = window.coords["datetime"].isel(batch=0, time=input_steps - 1 + target_step).item()
        lead_hours = float((pd.Timestamp(target_time) - pd.Timestamp(forecast_reference)).total_seconds() / 3600)

        file_name = f"boundary_{idx:05d}.nc"
        file_path = output_dir / file_name
        if args.skip_existing and file_path.exists():
            manifest.append(BoundaryRecord(
                index=idx,
                file=file_name,
                forecast_reference_time=str(forecast_reference),
                target_time=str(target_time),
                lead_hours=lead_hours,
            ))
            continue

        global_inputs, global_targets, global_forcings = data_pipeline.prepare_example(
            window, task_config, args.target_lead_time)
        norm_inputs, norm_targets, norm_forcings = normalize_datasets_for_training(
            global_inputs, global_targets, global_forcings,
            stats_mean, stats_std, diff_std)

        with torch.no_grad():
            with _autocast_context(device):
                grid_outputs = global_model.forward_grid_node_outputs(
                    norm_inputs, norm_forcings)
            pred_global = global_model._grid_node_outputs_to_prediction(  # pylint: disable=protected-access
                grid_outputs, norm_targets)

        boundary_patch = pred_global.isel(lat=lat_slice, lon=lon_slice).astype("float32")
        boundary_patch.coords["lat"] = patch_lat
        boundary_patch.coords["lon"] = patch_lon
        boundary_patch.attrs["forecast_reference_time"] = str(forecast_reference)
        boundary_patch.attrs["target_time"] = str(target_time)
        boundary_patch.attrs["lead_hours"] = lead_hours
        boundary_patch.to_netcdf(file_path, engine=output_engine)

        manifest.append(BoundaryRecord(
            index=idx,
            file=file_name,
            forecast_reference_time=str(forecast_reference),
            target_time=str(target_time),
            lead_hours=lead_hours,
        ))

        del grid_outputs, pred_global, boundary_patch
        if torch.npu.is_available() and device.type == "npu":
            torch.npu.empty_cache()

        print(f"[info] window {idx+1}/{len(windows)} saved to {file_name}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(rec) for rec in manifest], f, ensure_ascii=False, indent=2)
    print(f"[done] Generated {len(manifest)} boundary files under {output_dir}")
    print(f"[done] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
