"""Batch precompute regional boundary patches from global GraphCast.

Usage example:

  python precompute_global_boundaries_batch.py \
    --data-dir /root/autodl-tmp/myout \
    --pattern "source-era5_date-2024-*_steps-12_part*.nc" \
    --output-root /root/autodl-tmp/precomputed_boundaries_parts \
    --global-param "../data/params/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz" \
    --stats-dir ../data/stats \
    --target-lead-time 6h \
    --region 3 25 100 125 --buffer 5 \
    --device-type npu --device-id 0

Each input NetCDF will create a subfolder under `output-root`, containing
`boundary_*.nc` and `manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import xarray as xr

from graphcast import data_pipeline, npz_utils, normalization
from graphcast.training_utils import normalize_datasets_for_training
from nested_utils import RegionConfig, compute_region_slices
import precompute_global_boundaries as pb  # reuse helper functions

torch.set_default_dtype(torch.float32)


def _subset_stats(stats: xr.Dataset, lat_slice: slice, lon_slice: slice) -> xr.Dataset:
    subset = stats
    if "lat" in subset.dims:
        subset = subset.isel(lat=lat_slice, drop=False)
    if "lon" in subset.dims:
        subset = subset.isel(lon=lon_slice, drop=False)
    return subset


def _load_global_model(
    model_config,
    task_config,
    param_path: Path,
    sample_inputs: xr.Dataset,
    sample_targets: xr.Dataset,
    sample_forcings: xr.Dataset,
    device: torch.device,
    cache_dir: Path,
):
    return pb._load_model_with_cache(  # type: ignore[attr-defined]
        model_config, task_config, param_path, sample_inputs, sample_targets, sample_forcings, device, cache_dir
    )


def _select_engine() -> str:
    if getattr(pb, "_H5NETCDF_AVAILABLE", False):
        return "h5netcdf"
    if getattr(pb, "_NETCDF4_AVAILABLE", False):
        return "netcdf4"
    raise RuntimeError("Neither h5netcdf nor netCDF4 backend is available;请安装 h5netcdf 或 netCDF4。")


def process_dataset(
    dataset_path: Path,
    output_dir: Path,
    region_cfg: RegionConfig,
    model,
    task_config,
    stats_mean: xr.Dataset,
    stats_std: xr.Dataset,
    diff_std: xr.Dataset,
    target_lead_time: str,
    device: torch.device,
) -> None:
    ds = data_pipeline.load_dataset(str(dataset_path))
    lat_slice, lon_slice = compute_region_slices(ds.lat.values, ds.lon.values, region_cfg)
    patch_lat = ds.lat.isel(lat=lat_slice)
    patch_lon = ds.lon.isel(lon=lon_slice)

    time_values = ds.coords["time"].values
    if len(time_values) < 4:
        raise ValueError(f"{dataset_path} 必须至少包含 4 个时间步。")
    step = pd.to_timedelta(time_values[1] - time_values[0])
    input_duration = pd.Timedelta(task_config.input_duration)
    input_steps = int(input_duration / step) + 1
    target_step = int(pd.Timedelta(target_lead_time) / step)
    if target_step <= 0:
        raise ValueError("target_lead_time must be positive.")

    windows = list(pb._iter_windows(ds, input_steps, target_step))  # type: ignore[attr-defined]
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = _select_engine()
    manifest: list[pb.BoundaryRecord] = []  # type: ignore[attr-defined]

    for idx, (end_idx, window) in enumerate(windows):
        forecast_reference = window.coords["datetime"].isel(batch=0, time=input_steps - 1).item()
        target_time = window.coords["datetime"].isel(batch=0, time=input_steps - 1 + target_step).item()
        lead_hours = float((pd.Timestamp(target_time) - pd.Timestamp(forecast_reference)).total_seconds() / 3600)

        file_name = f"boundary_{idx:05d}.nc"
        file_path = output_dir / file_name

        global_inputs, global_targets, global_forcings = data_pipeline.prepare_example(
            window, task_config, target_lead_time)
        norm_inputs, norm_targets, norm_forcings = normalize_datasets_for_training(
            global_inputs, global_targets, global_forcings,
            stats_mean, stats_std, diff_std)

        with torch.no_grad():
            with pb._autocast_context(device):  # type: ignore[attr-defined]
                grid_outputs = model.forward_grid_node_outputs(norm_inputs, norm_forcings)
            pred_global = model._grid_node_outputs_to_prediction(  # pylint: disable=protected-access
                grid_outputs, norm_targets)

        boundary_patch = pred_global.isel(lat=lat_slice, lon=lon_slice)
        stats_mean_reg = _subset_stats(stats_mean, lat_slice, lon_slice)
        stats_std_reg = _subset_stats(stats_std, lat_slice, lon_slice)
        diff_std_reg = _subset_stats(diff_std, lat_slice, lon_slice)
        # 反归一化回物理量
        denorm_vars = {}
        for name, var in boundary_patch.data_vars.items():
            wrapped = xr.Dataset({name: var})
            if name in global_inputs.data_vars:
                residual = normalization.unnormalize(wrapped, diff_std_reg, None)[name]
                base = global_inputs[name].isel(time=-1)
                if "lat" in base.dims and "lon" in base.dims:
                    base = base.isel(lat=lat_slice, lon=lon_slice)
                denorm_vars[name] = residual + base
            else:
                denorm_vars[name] = normalization.unnormalize(wrapped, stats_std_reg, stats_mean_reg)[name]
        boundary_patch = xr.Dataset(denorm_vars).astype("float32")
        boundary_patch.coords["lat"] = patch_lat
        boundary_patch.coords["lon"] = patch_lon
        boundary_patch.attrs["forecast_reference_time"] = str(forecast_reference)
        boundary_patch.attrs["target_time"] = str(target_time)
        boundary_patch.attrs["lead_hours"] = lead_hours
        boundary_patch.to_netcdf(file_path, engine=engine)

        manifest.append(pb.BoundaryRecord(  # type: ignore[attr-defined]
            index=idx,
            file=file_name,
            forecast_reference_time=str(forecast_reference),
            target_time=str(target_time),
            lead_hours=lead_hours,
        ))

        del grid_outputs, pred_global, boundary_patch
        if torch.npu.is_available() and device.type == "npu":
            torch.npu.empty_cache()

        print(f"[info] {dataset_path.name}: window {idx+1}/{len(windows)} saved to {file_name}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(rec) for rec in manifest], f, ensure_ascii=False, indent=2)
    print(f"[done] {dataset_path.name}: wrote {len(manifest)} boundary files under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量为切片后的 2024 数据集生成边界文件。")
    parser.add_argument("--data-dir", type=Path, default=Path("myout"),
                        help="切片后的 2024 数据所在目录")
    parser.add_argument("--pattern", type=str, default="source-era5_date-2024-*_steps-12_part*.nc",
                        help="匹配数据文件的通配符")
    parser.add_argument("--output-root", type=Path, default=Path("precomputed_boundaries_parts"),
                        help="边界输出根目录")
    parser.add_argument("--global-param", type=Path, required=True,
                        help="GraphCast Haiku 参数 NPZ")
    parser.add_argument("--stats-dir", type=Path, required=True,
                        help="统计文件目录")
    parser.add_argument("--target-lead-time", type=str, default="6h")
    parser.add_argument("--buffer", type=float, default=5.0)
    parser.add_argument("--region", type=float, nargs=4, default=(3.0, 25.0, 100.0, 125.0),
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--device-type", type=str, default="auto",
                        choices=("auto", "cpu", "cuda", "npu"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=pb._default_cache_dir())  # type: ignore[attr-defined]
    parser.add_argument("--max-datasets", type=int, default=None,
                        help="可选，最多处理多少个文件。")
    args = parser.parse_args()

    pb._setup_env()  # type: ignore[attr-defined]
    device = pb._ensure_device(args.device_type, args.device_id)  # type: ignore[attr-defined]

    all_files = sorted(args.data_dir.glob(args.pattern))
    if not all_files:
        raise FileNotFoundError(f"{args.data_dir} 下未找到匹配 {args.pattern} 的文件。")
    if args.max_datasets is not None:
        all_files = all_files[:args.max_datasets]

    # 用第一个文件初始化模型与样本
    first_ds = data_pipeline.load_dataset(str(all_files[0]))
    model_config, task_config, _ = npz_utils.load_config_from_npz(str(args.global_param))
    stats_mean, stats_std, diff_std = data_pipeline.load_stats(str(args.stats_dir), task_config)

    time_values = first_ds.coords["time"].values
    if len(time_values) < 4:
        raise ValueError("样本文件时间步不足 4。")
    step = pd.to_timedelta(time_values[1] - time_values[0])
    input_duration = pd.Timedelta(task_config.input_duration)
    input_steps = int(input_duration / step) + 1
    target_step = int(pd.Timedelta(args.target_lead_time) / step)
    sample_end_idx = input_steps - 1
    sample_window = first_ds.isel(time=slice(sample_end_idx - input_steps + 1,
                                             sample_end_idx + target_step + 1))
    sample_inputs, sample_targets, sample_forcings = data_pipeline.prepare_example(
        sample_window, task_config, args.target_lead_time)

    global_model, cache_status = _load_global_model(
        model_config, task_config, Path(args.global_param),
        sample_inputs, sample_targets, sample_forcings,
        device, Path(args.cache_dir))
    if hasattr(global_model, "eval"):
        global_model.eval()
    for parameter in global_model.parameters():
        parameter.requires_grad_(False)
    print(f"[info] Global GraphCast ready ({cache_status}).")

    region_cfg = RegionConfig(
        lat_min=args.region[0],
        lat_max=args.region[1],
        lon_min=args.region[2],
        lon_max=args.region[3],
        buffer_deg=args.buffer,
    )

    for path in all_files:
        out_dir = args.output_root / path.stem
        process_dataset(
            dataset_path=path,
            output_dir=out_dir,
            region_cfg=region_cfg,
            model=global_model,
            task_config=task_config,
            stats_mean=stats_mean,
            stats_std=stats_std,
            diff_std=diff_std,
            target_lead_time=args.target_lead_time,
            device=device,
        )


if __name__ == "__main__":
    main()
