#!/usr/bin/env python3
"""Single-step global GraphCast inference with surface maps.

This script runs one GraphCast forecast step (e.g. +6 h) on a single ERA5
example and generates surface diagnostics similar to `eval_nested_csaf.py`:

  * 2m temperature (K)
  * Mean sea-level pressure (hPa)
  * 10 m wind vectors (m/s)

It compares the GraphCast forecast with the verifying ERA5 field and saves:

  * global_surface_step_{lead}h_pretty.png
  * global_wind_step_{lead}h.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from graphcast import data_pipeline, npz_utils
from graphcast.graphcast import GraphCast
from graphcast.training_utils import normalize_datasets_for_training
from eval_nested_baseline import (  # type: ignore
    _grid_outputs_to_predictions,
)
from eval_nested_csaf import (  # type: ignore
    plot_surface_step_pretty,
    plot_wind_step,
    squeeze_batch,
    to_hours,
)


def _ensure_device(device_id: int | None) -> torch.device:
    if torch.npu.is_available():  # type: ignore[attr-defined]
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_id or 0)  # type: ignore[attr-defined]
        return torch.device(f"npu:{device_id or 0}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id or 0)
        return torch.device(f"cuda:{device_id or 0}")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-step global GraphCast surface maps.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="ERA5 NetCDF file with multiple time steps (same layout as training data).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        help="GraphCast Haiku parameter npz (e.g. operational 0.25° config).",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        required=True,
        help="Directory containing stats-mean/stddev/diffs_stddev NetCDF files.",
    )
    parser.add_argument(
        "--target-lead-time",
        type=str,
        default="48h",
        help="Forecast lead time for the single step (e.g. 48h).",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device id for NPU/CUDA; CPU is used if no accelerator is available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./single_step_outputs"),
        help="Directory to write output figures.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args()


def log(msg: str, *, enable: bool) -> None:
    if enable:
        print(msg, flush=True)


def run_single_step(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading dataset {args.dataset}", enable=args.verbose)
    dataset = data_pipeline.load_dataset(str(args.dataset))

    # Lead times (in hours) for which to generate maps.
    lead_hours_list = [6, 12, 24, 48]

    # Load GraphCast config
    model_config, task_config, _ = npz_utils.load_config_from_npz(str(args.params))

    # Determine how many past steps are needed from input_duration
    time_values = dataset.coords["time"].values
    if len(time_values) < 4:
        raise ValueError("Dataset must contain at least four time steps.")
    step = pd.to_timedelta(time_values[1] - time_values[0])
    step_hours = step / pd.Timedelta("1h")
    input_duration = pd.Timedelta(task_config.input_duration)
    input_steps = int(input_duration / step) + 1
    max_lead = max(lead_hours_list)
    target_step = int(max_lead / step_hours)
    if target_step <= 0:
        raise ValueError("target_lead_time must be positive.")

    # Use the earliest possible forecast window: last input index = input_steps-1
    sample_end_idx = input_steps - 1
    sample_window = dataset.isel(
        time=slice(sample_end_idx - input_steps + 1, sample_end_idx + target_step + 1)
    )

    log(
        f"Prepared sample window: time indices [{sample_end_idx - input_steps + 1}, "
        f"{sample_end_idx + target_step}]",
        enable=args.verbose,
    )

    lead_time_strs = [f"{h}h" for h in lead_hours_list]
    inputs, targets, forcings = data_pipeline.prepare_example(
        sample_window,
        task_config,
        target_lead_time=lead_time_strs,
    )

    stats_mean, stats_std, diff_std = data_pipeline.load_stats(
        str(args.stats_dir),
        task_config,
    )

    device = _ensure_device(args.device_id)
    log(f"Using device {device}", enable=args.verbose)

    # Initialize GraphCast and load parameters
    model = GraphCast(model_config, task_config).to(device)
    init_targets = targets.isel(time=slice(0, 1))
    init_forcings = forcings.isel(time=slice(0, 1))
    log("Loading GraphCast Haiku parameters...", enable=args.verbose)
    model.load_haiku_parameters(
        str(args.params),
        sample_inputs=inputs,
        sample_targets_template=init_targets,
        sample_forcings=init_forcings,
    )
    if hasattr(model, "eval"):
        model.eval()

    # Normalize inputs/targets/forcings as in training
    norm_inputs, norm_targets, norm_forcings = normalize_datasets_for_training(
        inputs, targets, forcings, stats_mean, stats_std, diff_std
    )

    with torch.no_grad():
        grid_outputs = model.forward_grid_node_outputs(norm_inputs, norm_forcings)
        predictions = _grid_outputs_to_predictions(
            model,
            grid_outputs,
            norm_targets,
            inputs,
            stats_mean,
            stats_std,
            diff_std,
        )

    pred = squeeze_batch(predictions)

    # Use global bounds
    lat_min = float(pred.lat.min().values)
    lat_max = float(pred.lat.max().values)
    lon_min = float(pred.lon.min().values)
    lon_max = float(pred.lon.max().values)
    lat_bounds = (lat_min, lat_max)
    lon_bounds = (lon_min, lon_max)

    # ERA5 verifying field for all requested lead times
    truth = squeeze_batch(targets)

    # Surface diagnostics with same style as eval_nested_csaf
    for lead_h in lead_hours_list:
        surface_path = plot_surface_step_pretty(
            predictions=predictions,
            targets=truth,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            lead_hour=lead_h,
            out_dir=args.output_dir,
        )
        wind_path = plot_wind_step(
            predictions=predictions,
            targets=truth,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            lead_hour=lead_h,
            out_dir=args.output_dir,
        )
        log(f"Surface map (+{lead_h} h) saved to {surface_path}", enable=True)
        log(f"Wind map (+{lead_h} h) saved to {wind_path}", enable=True)


def main() -> None:
    args = parse_args()
    run_single_step(args)


if __name__ == "__main__":
    main()
