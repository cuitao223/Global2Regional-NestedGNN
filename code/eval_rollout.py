#!/usr/bin/env python3
"""Rolling GraphCast evaluation (auto-regressive rollout).

This script loads ERA5 samples, runs GraphCast in an auto-regressive rollout
from +6 h to +72 h with 6-hourly cadence, and evaluates skill using RMSE, Bias
and ACC. It also generates spatial diagnostics (including wind vectors),
derives a simple precipitation proxy (-ω × q), and tracks the minimum sea-level
pressure as a proxy for tropical cyclone intensity.

Outputs (under --output-dir):
  * metrics.csv : per-variable metrics at each lead time
  * heatmap_{rmse,bias,acc}.png : metric heatmaps (variables × lead hours)
  * surface_step_{lead}.png : surface diagnostics for selected lead steps
  * precip_proxy_maps.png : lead × precipitation proxy comparison
  * summary.json : consolidated metrics and artefact locations
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from graphcast import data_pipeline, normalization, npz_utils, rollout
from graphcast.graphcast import GraphCast

plt.switch_backend("agg")


SURFACE_VARS: Sequence[str] = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
)

ATMOS_VARS: Sequence[str] = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
)

BATCH_DIM = "batch"


@dataclass
class Artefacts:
    predictions: xr.Dataset
    targets: xr.Dataset
    inputs: xr.Dataset
    forcings: xr.Dataset
    stats_mean: xr.Dataset
    stats_std: xr.Dataset
    diff_std: xr.Dataset
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphCast 0–72 h rolling evaluation")
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=[
            Path("/root/autodl-tmp/myout/source-era5_date-2023-01-01_steps-12_part04.nc")
        ],
        help="One or more ERA5 NetCDF files ordered in time.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=Path(
            "/home/ma-user/data/params/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"
        ),
        help="GraphCast Haiku parameter archive.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("/home/ma-user/data/stats"),
        help="Directory containing stats-mean/stddev/diffs_stddev NetCDF files.",
    )
    parser.add_argument(
        "--lat-bounds",
        type=float,
        nargs=2,
        default=(3.0, 25.0),
        help="Latitude bounds for the South China Sea region (degrees).",
    )
    parser.add_argument(
        "--lon-bounds",
        type=float,
        nargs=2,
        default=(105.0, 125.0),
        help="Longitude bounds for the South China Sea region (degrees, 0–360).",
    )
    parser.add_argument(
        "--lead-hours",
        type=int,
        default=72,
        help="Maximum forecast horizon in hours (inclusive).",
    )
    parser.add_argument(
        "--lead-step",
        type=int,
        default=6,
        help="Lead time increment in hours.",
    )
    parser.add_argument(
        "--precip-level",
        type=int,
        default=850,
        help="Pressure level (hPa) to compute the precipitation proxy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. cpu / cuda:0 / npu:0.",
    )
    parser.add_argument(
        "--plot-leads",
        type=int,
        nargs="*",
        default=(6, 24, 48, 72),
        help="Subset of lead hours to generate detailed surface maps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./rolling_outputs"),
        help="Directory where outputs will be written.",
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


def load_and_concat_datasets(paths: Iterable[Path], verbose: bool = False) -> xr.Dataset:
    datasets = []
    for path in sorted(paths):
        log(f"Loading dataset {path}", enable=verbose)
        ds = data_pipeline.load_dataset(str(path))
        datasets.append(ds)
    if not datasets:
        raise ValueError("No datasets were provided.")
    combined = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )
    combined = combined.sortby("time")
    time_index = combined.get_index("time")
    if hasattr(time_index, "duplicated"):
        unique = ~time_index.duplicated()
        combined = combined.isel(time=np.nonzero(unique)[0])
    sizes_desc = ", ".join(f"{k}={v}" for k, v in combined.sizes.items())
    if "time" in combined.coords:
        start_time = str(combined.coords["time"].values[0])
        end_time = str(combined.coords["time"].values[-1])
        log(f"Combined dataset sizes: {sizes_desc} (time range {start_time} → {end_time})", enable=verbose)
    else:
        log(f"Combined dataset sizes: {sizes_desc}", enable=verbose)
    return combined


def compute_index_slice(values: np.ndarray, bounds: tuple[float, float]) -> slice:
    """Return index slice for given bounds; handle global 0–360 grids.

    If the longitude coordinate is 0–360 and the requested bounds cover
    the full globe (e.g. [-180, 180]), return the full slice so that we
    don't accidentally drop half the domain.
    """
    lower, upper = bounds
    vmin = float(values.min())
    vmax = float(values.max())
    # Heuristic: 0–360 grid requested with -180..180 bounds -> full domain.
    if vmin >= 0.0 and vmax > 180.0 and lower <= -180.0 and upper >= 180.0:
        return slice(0, len(values))
    start = int(np.searchsorted(values, lower, side="left"))
    stop = int(np.searchsorted(values, upper, side="right"))
    return slice(start, stop)


def maybe_interp_to_0p25(
    pred: xr.Dataset,
    obs: xr.Dataset,
    inputs: xr.Dataset,
    forcings: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """Interpolate fields to a ~0.25° lat/lon grid for evaluation.

    If the current resolution is already close to 0.25°, datasets are
    returned unchanged. Assumes monotonic lat/lon coordinates.
    """
    if "lat" not in pred.dims or "lon" not in pred.dims:
        return pred, obs, inputs, forcings
    lat = pred.lat
    lon = pred.lon
    if lat.size < 2 or lon.size < 2:
        return pred, obs, inputs, forcings
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    if dlat == 0.0 or dlon == 0.0:
        return pred, obs, inputs, forcings
    # Already ~0.25°: keep as is.
    if np.isclose(abs(dlat), 0.25, atol=1e-4) and np.isclose(abs(dlon), 0.25, atol=1e-4):
        return pred, obs, inputs, forcings

    lat_min = float(lat.min().values)
    lat_max = float(lat.max().values)
    lon_min = float(lon.min().values)
    lon_max = float(lon.max().values)

    n_lat = max(2, int(round((lat_max - lat_min) / 0.25)) + 1)
    n_lon = max(2, int(round((lon_max - lon_min) / 0.25)) + 1)
    target_lat = np.linspace(lat_min, lat_max, n_lat)
    target_lon = np.linspace(lon_min, lon_max, n_lon)

    kwargs = {"lat": target_lat, "lon": target_lon, "method": "linear"}
    pred_i = pred.interp(**kwargs)
    obs_i = obs.interp(**kwargs)
    inputs_i = inputs.interp(**kwargs)
    forcings_i = forcings.interp(**kwargs)
    return pred_i, obs_i, inputs_i, forcings_i


def prepare_region_artifacts(
    dataset: xr.Dataset,
    stats_dir: Path,
    params_path: Path,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    lead_hours: int,
    lead_step: int,
    device: str,
    verbose: bool = False,
) -> Artefacts:
    if device.lower().startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Requested NPU device but torch_npu is not installed. "
                "Please install torch_npu or choose a different --device."
            ) from exc

    device_obj = torch.device(device)
    model_config, task_config, _ = npz_utils.load_config_from_npz(str(params_path))

    lat_slice = compute_index_slice(dataset.lat.values, lat_bounds)
    lon_slice = compute_index_slice(dataset.lon.values, lon_bounds)

    lead_times = [pd.Timedelta(hours=hour) for hour in range(lead_step, lead_hours + 1, lead_step)]

    inputs, targets, forcings = data_pipeline.prepare_example(
        dataset,
        task_config,
        target_lead_time=lead_times,
    )
    input_time = inputs.dims.get("time", 0)
    target_steps = targets.dims.get("time", 0)
    global_lat_points = inputs.dims.get("lat", 0)
    global_lon_points = inputs.dims.get("lon", 0)
    log(
        f"Prepared global example: inputs time={input_time}, targets={target_steps} steps, "
        f"lat={global_lat_points}, lon={global_lon_points}",
        enable=verbose,
    )
    stats_mean, stats_std, diff_std = data_pipeline.load_stats(
        str(stats_dir),
        task_config,
    )
    log("Statistics loaded for full domain", enable=verbose)

    if device_obj.type == "npu":
        torch.npu.set_device(device_obj)

    log(f"Initializing GraphCast on device {device_obj}", enable=verbose)
    model = GraphCast(model_config, task_config).to(device_obj)
    log("Loading GraphCast parameters...", enable=verbose)
    init_targets = targets.isel(time=slice(0, 1))
    init_forcings = forcings.isel(time=slice(0, 1))
    model.load_haiku_parameters(
        str(params_path),
        sample_inputs=inputs,
        sample_targets_template=init_targets,
        sample_forcings=init_forcings,
    )
    predictor = normalization.InputsAndResiduals(
        model,
        diffs_stddev_by_level=diff_std,
        mean_by_level=stats_mean,
        stddev_by_level=stats_std,
    )
    log("Running autoregressive rollout...", enable=verbose)
    n_chunks = len(to_hours(targets.time.values))
    log(f"Total rollout steps: {n_chunks}", enable=verbose)
    predictions = rollout.chunked_prediction(
        predictor_fn=predictor,
        inputs=inputs,
        targets_template=targets,
        forcings=forcings,
        num_steps_per_chunk=1,
        verbose=verbose,
    )
    log("Rollout completed", enable=verbose)

    region_inputs = inputs.isel(lat=lat_slice, lon=lon_slice)
    region_targets = targets.isel(lat=lat_slice, lon=lon_slice)
    region_forcings = forcings.isel(lat=lat_slice, lon=lon_slice)
    region_predictions = predictions.isel(lat=lat_slice, lon=lon_slice)

    # Optionally interpolate regional subset to a 0.25° grid for evaluation
    # so that metrics are comparable with 0.25° nested/regional runs.
    region_predictions, region_targets, region_inputs, region_forcings = maybe_interp_to_0p25(
        region_predictions,
        region_targets,
        region_inputs,
        region_forcings,
    )
    region_stats_mean = stats_mean
    if "lat" in region_stats_mean.dims:
        region_stats_mean = region_stats_mean.isel(lat=lat_slice, drop=False)
    if "lon" in region_stats_mean.dims:
        region_stats_mean = region_stats_mean.isel(lon=lon_slice, drop=False)

    region_stats_std = stats_std
    if "lat" in region_stats_std.dims:
        region_stats_std = region_stats_std.isel(lat=lat_slice, drop=False)
    if "lon" in region_stats_std.dims:
        region_stats_std = region_stats_std.isel(lon=lon_slice, drop=False)

    region_diff_std = diff_std
    if "lat" in region_diff_std.dims:
        region_diff_std = region_diff_std.isel(lat=lat_slice, drop=False)
    if "lon" in region_diff_std.dims:
        region_diff_std = region_diff_std.isel(lon=lon_slice, drop=False)
    region_lat_points = region_inputs.dims.get("lat", 0)
    region_lon_points = region_inputs.dims.get("lon", 0)
    log(
        f"Regional subset ready: lat={region_lat_points}, lon={region_lon_points}",
        enable=verbose,
    )
    return Artefacts(
        predictions=region_predictions,
        targets=region_targets,
        inputs=region_inputs,
        forcings=region_forcings,
        stats_mean=region_stats_mean,
        stats_std=region_stats_std,
        diff_std=region_diff_std,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )


def squeeze_batch(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    if isinstance(data, xr.Dataset):
        if BATCH_DIM in data.dims:
            return data.isel({BATCH_DIM: 0}, drop=True)
        return data
    if BATCH_DIM in data.dims:
        return data.isel({BATCH_DIM: 0}, drop=True)
    return data


def lat_weights(latitudes: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(latitudes))
    return xr.DataArray(weights, coords={"lat": latitudes}, dims=("lat",))


def to_hours(timedelta_values: np.ndarray) -> np.ndarray:
    return np.asarray(timedelta_values, dtype="timedelta64[h]").astype(int)


def compute_rmse_bias_acc(
    pred: xr.DataArray,
    obs: xr.DataArray,
    weights: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    diff = pred - obs
    rmse = np.sqrt((diff ** 2).weighted(weights).mean(dim=("lat", "lon"), skipna=True))
    bias = diff.weighted(weights).mean(dim=("lat", "lon"), skipna=True)

    pred_anom = pred - pred.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    obs_anom = obs - obs.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    numerator = (pred_anom * obs_anom).weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    denom = np.sqrt(
        (pred_anom ** 2).weighted(weights).mean(dim=("lat", "lon"), skipna=True)
        * (obs_anom ** 2).weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = numerator / denom
    acc = acc.where(np.isfinite(acc), 0.0)
    return rmse, bias, acc


def reduce_levels(value: xr.DataArray) -> xr.DataArray:
    if "level" in value.dims:
        return value.mean(dim="level", skipna=True)
    return value


def build_metrics_dataframe(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    precip_proxy_level: int,
) -> pd.DataFrame:
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)
    weights = lat_weights(pred.lat)
    leads = to_hours(pred.time.values)

    records: list[dict[str, float | int | str]] = []
    variable_groups = list(SURFACE_VARS) + list(ATMOS_VARS)

    for name in variable_groups:
        if name not in pred or name not in obs:
            continue
        pred_var = pred[name]
        obs_var = obs[name]
        rmse, bias, acc = compute_rmse_bias_acc(pred_var, obs_var, weights)
        rmse = reduce_levels(rmse)
        bias = reduce_levels(bias)
        acc = reduce_levels(acc)
        for idx, lead in enumerate(leads):
            records.append(
                {
                    "variable": name,
                    "lead_hours": int(lead),
                    "rmse": float(rmse.isel(time=idx).values),
                    "bias": float(bias.isel(time=idx).values),
                    "acc": float(acc.isel(time=idx).values),
                }
            )

    if "vertical_velocity" in pred and "specific_humidity" in pred:
        precip_pred = (-pred["vertical_velocity"] * pred["specific_humidity"]).sel(
            level=precip_proxy_level, method="nearest"
        )
        precip_obs = (-obs["vertical_velocity"] * obs["specific_humidity"]).sel(
            level=precip_proxy_level, method="nearest"
        )
        rmse, bias, acc = compute_rmse_bias_acc(precip_pred, precip_obs, weights)
        for idx, lead in enumerate(leads):
            records.append(
                {
                    "variable": f"precip_proxy_{precip_proxy_level}",
                    "lead_hours": int(leads[idx]),
                    "rmse": float(rmse.isel(time=idx).values),
                    "bias": float(bias.isel(time=idx).values),
                    "acc": float(acc.isel(time=idx).values),
                }
            )

    return pd.DataFrame.from_records(records)


def plot_heatmap(metrics: pd.DataFrame, metric: str, out_dir: Path) -> Path:
    pivot = metrics.pivot(index="variable", columns="lead_hours", values=metric)
    fig, ax = plt.subplots(figsize=(0.6 * pivot.shape[1] + 4, 0.45 * pivot.shape[0] + 2))
    cax = ax.imshow(pivot, aspect="auto", origin="lower", cmap="coolwarm" if metric != "acc" else "viridis")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=45)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_xlabel("Lead time (h)")
    ax.set_title(metric.upper())
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    path = out_dir / f"heatmap_{metric}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_surface_step(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    lead_hour: int,
    out_dir: Path,
) -> Path:
    """Surface fields with the same layout/units as nested_rolling_evaluation."""
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)
    lead_idx = int(np.where(to_hours(pred.time.values) == lead_hour)[0][0])

    pred_t2m = pred["2m_temperature"].isel(time=lead_idx)
    obs_t2m = obs["2m_temperature"].isel(time=lead_idx)
    diff_t2m = pred_t2m - obs_t2m

    pred_mslp = pred["mean_sea_level_pressure"].isel(time=lead_idx)
    obs_mslp = obs["mean_sea_level_pressure"].isel(time=lead_idx)
    # work in hPa for MSLP-related visualisation
    pred_mslp_hpa = pred_mslp / 100.0
    obs_mslp_hpa = obs_mslp / 100.0
    diff_mslp = pred_mslp_hpa - obs_mslp_hpa

    # If longitude is in 0–360 convention, wrap to [-180, 180] for nicer global plots.
    def _wrap_lon(da: xr.DataArray) -> xr.DataArray:
        lon_da = da.lon
        lon_min = float(lon_da.min().values)
        lon_max = float(lon_da.max().values)
        if lon_min >= 0.0 and lon_max > 190.0:
            new_lon = (((lon_da + 180.0) % 360.0) - 180.0)
            da_wrapped = da.assign_coords(lon=new_lon)
            return da_wrapped.sortby("lon")
        return da

    pred_t2m = _wrap_lon(pred_t2m)
    obs_t2m = _wrap_lon(obs_t2m)
    diff_t2m = _wrap_lon(diff_t2m)
    pred_mslp_hpa = _wrap_lon(pred_mslp_hpa)
    obs_mslp_hpa = _wrap_lon(obs_mslp_hpa)
    diff_mslp = _wrap_lon(diff_mslp)

    lon = pred_t2m.lon
    lat = pred_t2m.lat
    proj = ccrs.PlateCarree()

    # Physically constrained colour ranges (copied from nested_rolling_evaluation)
    T_MIN_DEFAULT, T_MAX_DEFAULT = 250.0, 320.0  # K
    P_MIN_DEFAULT, P_MAX_DEFAULT = 98000.0, 103000.0  # Pa
    T_DIFF_DEFAULT = 5.0  # K
    P_DIFF_DEFAULT = 10.0  # hPa

    t_min_raw = float(np.nanmin([pred_t2m.min().values, obs_t2m.min().values]))
    t_max_raw = float(np.nanmax([pred_t2m.max().values, obs_t2m.max().values]))
    t_range = t_max_raw - t_min_raw

    p_min_raw = float(np.nanmin([pred_mslp.min().values, obs_mslp.min().values]))
    p_max_raw = float(np.nanmax([pred_mslp.max().values, obs_mslp.max().values]))
    p_range = p_max_raw - p_min_raw

    if (t_min_raw < 150.0) or (t_max_raw > 400.0) or (t_range > 80.0):
        t_vmin, t_vmax = T_MIN_DEFAULT, T_MAX_DEFAULT
    else:
        t_vmin, t_vmax = t_min_raw, t_max_raw

    if (p_min_raw < 80000.0) or (p_max_raw > 120000.0) or (p_range > 8000.0):
        p_vmin_pa, p_vmax_pa = P_MIN_DEFAULT, P_MAX_DEFAULT
    else:
        p_vmin_pa, p_vmax_pa = p_min_raw, p_max_raw
    p_vmin, p_vmax = p_vmin_pa / 100.0, p_vmax_pa / 100.0

    t_diff_abs_raw = float(np.nanmax(np.abs(diff_t2m.values)))
    if not np.isfinite(t_diff_abs_raw) or t_diff_abs_raw > 20.0:
        t_diff_max = T_DIFF_DEFAULT
    else:
        t_diff_max = max(t_diff_abs_raw, T_DIFF_DEFAULT / 2.0)
    t_diff_vmin, t_diff_vmax = -t_diff_max, t_diff_max

    # ΔMSLP fixed ±10 hPa
    p_diff_vmin, p_diff_vmax = -P_DIFF_DEFAULT, P_DIFF_DEFAULT

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        subplot_kw={"projection": proj},
    )
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    fields = [
        (pred_t2m, "t", "GraphCast 2m T (K)"),
        (obs_t2m, "t", "ERA5 2m T (K)"),
        (diff_t2m, "t_diff", "Δ2m T (K)"),
        (pred_mslp_hpa, "p", "GraphCast MSLP (hPa)"),
        (obs_mslp_hpa, "p", "ERA5 MSLP (hPa)"),
        (diff_mslp, "p_diff", "ΔMSLP (hPa)"),
    ]

    for idx, (ax, (field, kind, title)) in enumerate(zip(axes.ravel(), fields, strict=True)):
        row, col = divmod(idx, 3)
        if kind == "t":
            vmin, vmax = t_vmin, t_vmax
            cmap = "coolwarm"
        elif kind == "t_diff":
            vmin, vmax = t_diff_vmin, t_diff_vmax
            cmap = "RdBu_r"
        elif kind == "p":
            vmin, vmax = p_vmin, p_vmax
            cmap = "viridis"
        else:  # "p_diff"
            vmin, vmax = p_diff_vmin, p_diff_vmax
            cmap = "RdBu_r"

        mesh = ax.pcolormesh(
            lon,
            lat,
            field,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=proj,
        )
        ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=proj)
        ax.coastlines(resolution="10m", linewidth=0.8)

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False
        if row == 0:
            gl.bottom_labels = False
        if col > 0:
            gl.left_labels = False

        ax.set_title(f"{title} @ +{lead_hour} h", fontsize=11)

        if kind == "t":
            label = "2m T (K)"
        elif kind == "t_diff":
            label = "Δ2m T (K)"
        elif kind == "p":
            label = "MSLP (hPa)"
        else:
            label = "ΔMSLP (hPa)"
        cbar = fig.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            fraction=0.04,
        )
        cbar.set_label(label)

    fig.suptitle(f"GraphCast surface fields @ +{lead_hour} h", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))

    path = out_dir / f"regional_surface_step_{lead_hour:03d}h_pretty.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_precip_proxy_maps(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    level: int,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    out_dir: Path,
) -> Path:
    """Precipitation proxy maps with the same layout as nested_rolling_evaluation."""
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)
    proxy_pred = (-pred["vertical_velocity"] * pred["specific_humidity"]).sel(level=level, method="nearest")
    proxy_obs = (-obs["vertical_velocity"] * obs["specific_humidity"]).sel(level=level, method="nearest")
    diff = proxy_pred - proxy_obs
    lon = proxy_pred.lon
    lat = proxy_pred.lat

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3,
        proxy_pred.sizes["time"],
        figsize=(3.2 * proxy_pred.sizes["time"], 9),
        subplot_kw={"projection": proj},
    )
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    leads = to_hours(proxy_pred.time.values)
    vmax = float(np.nanmax(np.abs(diff.values)))
    vmax = vmax if vmax > 0 else 1.0

    meshes_row: list[list] = [[], [], []]

    for idx, lead in enumerate(leads):
        for row, (data, title, cmap, vlims) in enumerate(
            (
                (proxy_pred.isel(time=idx), f"GraphCast ω×q {level} hPa @ +{lead}h", "PuOr", (-vmax, vmax)),
                (proxy_obs.isel(time=idx), f"ERA5 ω×q {level} hPa @ +{lead}h", "PuOr", (-vmax, vmax)),
                (diff.isel(time=idx), f"Δω×q {level} hPa @ +{lead}h", "PiYG", (-vmax, vmax)),
            ),
            start=0,
        ):
            ax = axes[row, idx]
            mesh = ax.pcolormesh(
                lon,
                lat,
                data,
                shading="auto",
                cmap=cmap,
                vmin=vlims[0],
                vmax=vlims[1],
                transform=proj,
            )
            ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=proj)
            ax.coastlines(resolution="10m", linewidth=0.5)

            gl = ax.gridlines(draw_labels=row == 2, linestyle="--", linewidth=0.3)
            gl.top_labels = False
            gl.right_labels = False
            if row < 2:
                gl.bottom_labels = False
            if idx > 0:
                gl.left_labels = False

            ax.set_title(title, fontsize=9)
            meshes_row[row].append(mesh)

    labels = [
        f"GraphCast proxy ω×q @ {level} hPa",
        f"ERA5 proxy ω×q @ {level} hPa",
        f"Δ proxy ω×q @ {level} hPa",
    ]
    for row in range(3):
        if meshes_row[row]:
            cbar = fig.colorbar(
                meshes_row[row][0],
                ax=axes[row, :],
                orientation="horizontal",
                pad=0.04,
                fraction=0.04,
            )
            cbar.set_label(labels[row])

    fig.suptitle(f"GraphCast precip proxy (ω×q) @ {level} hPa", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))

    path = out_dir / "regional_precip_proxy_maps.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_precip_maps(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    out_dir: Path,
) -> Path:
    """Direct total precipitation maps (GraphCast / ERA5 / Δ, in mm/6h)."""
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)

    if "total_precipitation_6hr" not in pred or "total_precipitation_6hr" not in obs:
        raise ValueError("total_precipitation_6hr not found in predictions/targets; cannot plot precipitation maps.")

    precip_pred = pred["total_precipitation_6hr"] * 1000.0
    precip_obs = obs["total_precipitation_6hr"] * 1000.0
    diff = precip_pred - precip_obs

    lon = precip_pred.lon
    lat = precip_pred.lat
    proj = ccrs.PlateCarree()

    n_time = precip_pred.sizes["time"]
    fig, axes = plt.subplots(
        3,
        n_time,
        figsize=(3.2 * n_time, 9),
        subplot_kw={"projection": proj},
    )
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    leads = to_hours(precip_pred.time.values)

    precip_max = float(
        np.nanmax(
            [
                float(precip_pred.max().values),
                float(precip_obs.max().values),
            ]
        )
    )
    if not np.isfinite(precip_max) or precip_max <= 0.0:
        precip_max = 1.0
    precip_max = min(precip_max, 60.0)

    diff_max = float(np.nanmax(np.abs(diff.values)))
    if not np.isfinite(diff_max) or diff_max <= 0.0:
        diff_max = 1.0
    diff_max = min(diff_max, 30.0)

    meshes_row: list[list] = [[], [], []]

    for idx, lead in enumerate(leads):
        for row, (data, title, cmap, vlims) in enumerate(
            (
                (
                    precip_pred.isel(time=idx),
                    f"GraphCast TP (mm/6h) @ +{lead}h",
                    "Blues",
                    (0.0, precip_max),
                ),
                (
                    precip_obs.isel(time=idx),
                    f"ERA5 TP (mm/6h) @ +{lead}h",
                    "Blues",
                    (0.0, precip_max),
                ),
                (
                    diff.isel(time=idx),
                    f"ΔTP (mm/6h) @ +{lead}h",
                    "RdBu_r",
                    (-diff_max, diff_max),
                ),
            ),
            start=0,
        ):
            ax = axes[row, idx]
            mesh = ax.pcolormesh(
                lon,
                lat,
                data,
                shading="auto",
                cmap=cmap,
                vmin=vlims[0],
                vmax=vlims[1],
                transform=proj,
            )
            ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=proj)
            ax.coastlines(resolution="10m", linewidth=0.5)

            gl = ax.gridlines(draw_labels=row == 2, linestyle="--", linewidth=0.3)
            gl.top_labels = False
            gl.right_labels = False
            if row < 2:
                gl.bottom_labels = False
            if idx > 0:
                gl.left_labels = False

            ax.set_title(title, fontsize=9)
            meshes_row[row].append(mesh)

    labels = [
        "GraphCast total precipitation (mm/6h)",
        "ERA5 total precipitation (mm/6h)",
        "Δ total precipitation (mm/6h)",
    ]
    for row in range(3):
        if meshes_row[row]:
            cbar = fig.colorbar(
                meshes_row[row][0],
                ax=axes[row, :],
                orientation="horizontal",
                pad=0.04,
                fraction=0.04,
            )
            cbar.set_label(labels[row])

    fig.suptitle("GraphCast regional total precipitation (mm/6h)", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))

    path = out_dir / "regional_precip_maps.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_wind_step(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    lead_hour: int,
    out_dir: Path,
) -> Path:
    """Standalone 10 m wind vectors (GraphCast, ERA5, difference), matching nested layout."""
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)
    lead_idx = int(np.where(to_hours(pred.time.values) == lead_hour)[0][0])

    pred_u = pred["10m_u_component_of_wind"].isel(time=lead_idx)
    pred_v = pred["10m_v_component_of_wind"].isel(time=lead_idx)
    obs_u = obs["10m_u_component_of_wind"].isel(time=lead_idx)
    obs_v = obs["10m_v_component_of_wind"].isel(time=lead_idx)

    # Wrap longitudes to [-180, 180] if dataset is in 0–360 convention.
    def _wrap_lon_uv(da: xr.DataArray) -> xr.DataArray:
        lon_da = da.lon
        lon_min = float(lon_da.min().values)
        lon_max = float(lon_da.max().values)
        if lon_min >= 0.0 and lon_max > 190.0:
            new_lon = (((lon_da + 180.0) % 360.0) - 180.0)
            da_wrapped = da.assign_coords(lon=new_lon)
            return da_wrapped.sortby("lon")
        return da

    pred_u = _wrap_lon_uv(pred_u)
    pred_v = _wrap_lon_uv(pred_v)
    obs_u = _wrap_lon_uv(obs_u)
    obs_v = _wrap_lon_uv(obs_v)

    lon = pred_u.lon
    lat = pred_u.lat
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        subplot_kw={"projection": proj},
    )

    stride_lat = max(1, len(lat) // 10)
    stride_lon = max(1, len(lon) // 10)

    panels = [
        (pred_u, pred_v, "GraphCast 10m wind"),
        (obs_u, obs_v, "ERA5 10m wind"),
        (pred_u - obs_u, pred_v - obs_v, "Δ10m wind"),
    ]

    for ax, (u, v, title) in zip(axes, panels, strict=True):
        ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=proj)
        ax.coastlines(resolution="10m", linewidth=0.8)

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False

        ax.quiver(
            lon.values[::stride_lon],
            lat.values[::stride_lat],
            u.values[::stride_lat, ::stride_lon],
            v.values[::stride_lat, ::stride_lon],
            transform=proj,
            color="k",
            scale=300,
            width=0.002,
        )
        ax.set_title(f"{title} @ +{lead_hour} h", fontsize=11)

    fig.suptitle(f"GraphCast 10m wind @ +{lead_hour} h", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.95))

    path = out_dir / f"regional_wind_step_{lead_hour:03d}h.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def compute_mslp_minima(
    predictions: xr.Dataset,
    targets: xr.Dataset,
) -> list[dict[str, float | int]]:
    pred = squeeze_batch(predictions)["mean_sea_level_pressure"]
    obs = squeeze_batch(targets)["mean_sea_level_pressure"]
    leads = to_hours(pred.time.values)
    results: list[dict[str, float | int]] = []

    for idx, lead in enumerate(leads):
        pred_field = pred.isel(time=idx)
        obs_field = obs.isel(time=idx)

        pred_min = float(pred_field.min().values)
        obs_min = float(obs_field.min().values)

        pred_argmin = np.unravel_index(np.nanargmin(pred_field.values), pred_field.shape)
        obs_argmin = np.unravel_index(np.nanargmin(obs_field.values), obs_field.shape)

        results.append(
            {
                "lead_hours": int(lead),
                "regional_mslp_min": pred_min,
                "regional_min_lat": float(pred_field.lat.values[pred_argmin[0]]),
                "regional_min_lon": float(pred_field.lon.values[pred_argmin[1]]),
                "era5_mslp_min": obs_min,
                "era5_min_lat": float(obs_field.lat.values[obs_argmin[0]]),
                "era5_min_lon": float(obs_field.lon.values[obs_argmin[1]]),
                "mslp_error": float(pred_min - obs_min),
            }
        )
    return results


def build_summary(
    metrics: pd.DataFrame,
    mslp_minima: list[dict[str, float | int]],
    heatmaps: dict[str, str],
    surface_maps: dict[int, str],
    precip_map: str,
    metrics_csv: str,
    precip_total_map: str | None = None,
) -> dict:
    summary: dict[str, object] = {
        "metrics_csv": metrics_csv,
        "heatmaps": heatmaps,
        "surface_maps": surface_maps,
        "precip_proxy_map": precip_map,
        "mslp_minima": mslp_minima,
        "leads": [],
    }
    if precip_total_map is not None:
        summary["precip_map"] = precip_total_map

    grouped = metrics.groupby("lead_hours")
    for lead, frame in grouped:
        lead_metrics = {
            row["variable"]: {"rmse": row["rmse"], "bias": row["bias"], "acc": row["acc"]}
            for _, row in frame.iterrows()
        }
        summary["leads"].append({"lead_hours": int(lead), "metrics": lead_metrics})
    summary["leads"].sort(key=lambda item: item["lead_hours"])
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_and_concat_datasets(args.datasets, verbose=args.verbose)
    log("Preparing region artefacts and executing rollout", enable=args.verbose)
    artefacts = prepare_region_artifacts(
        dataset,
        stats_dir=args.stats_dir,
        params_path=args.params,
        lat_bounds=tuple(args.lat_bounds),
        lon_bounds=tuple(args.lon_bounds),
        lead_hours=args.lead_hours,
        lead_step=args.lead_step,
        device=args.device,
        verbose=args.verbose,
    )

    metrics = build_metrics_dataframe(
        artefacts.predictions,
        artefacts.targets,
        args.precip_level,
    )
    log(f"Metrics table generated with {len(metrics)} rows", enable=args.verbose)

    metrics_csv_path = args.output_dir / "metrics.csv"
    metrics.to_csv(metrics_csv_path, index=False)
    log(f"Metrics written to {metrics_csv_path}", enable=args.verbose)

    heatmap_paths = {}
    for metric_name in ("rmse", "bias", "acc"):
        heatmap_paths[metric_name] = str(plot_heatmap(metrics, metric_name, args.output_dir))
        log(f"Generated heatmap for {metric_name}: {heatmap_paths[metric_name]}", enable=args.verbose)

    surface_paths = {}
    requested_leads = set(args.plot_leads) if args.plot_leads else set()
    available_leads = set(to_hours(squeeze_batch(artefacts.predictions).time.values))
    for lead in sorted(available_leads & requested_leads):
        surface_paths[int(lead)] = str(
            plot_surface_step(
                artefacts.predictions,
                artefacts.targets,
                artefacts.lat_bounds,
                artefacts.lon_bounds,
                lead,
                args.output_dir,
            )
        )
        # Standalone wind-only maps to mirror nested_rolling_evaluation
        plot_wind_step(
            artefacts.predictions,
            artefacts.targets,
            artefacts.lat_bounds,
            artefacts.lon_bounds,
            lead,
            args.output_dir,
        )
    log(f"Generated {len(surface_paths)} surface maps (plus wind maps)", enable=args.verbose)

    precip_path = str(
        plot_precip_proxy_maps(
            artefacts.predictions,
            artefacts.targets,
            args.precip_level,
            artefacts.lat_bounds,
            artefacts.lon_bounds,
            args.output_dir,
        )
    )
    log(f"Precipitation proxy map saved to {precip_path}", enable=args.verbose)

    precip_total_path: str | None = None
    pred_core = squeeze_batch(artefacts.predictions)
    obs_core = squeeze_batch(artefacts.targets)
    if "total_precipitation_6hr" in pred_core and "total_precipitation_6hr" in obs_core:
        precip_total_path = str(
            plot_precip_maps(
                artefacts.predictions,
                artefacts.targets,
                artefacts.lat_bounds,
                artefacts.lon_bounds,
                args.output_dir,
            )
        )
        log(f"Direct precipitation maps written to {precip_total_path}", enable=args.verbose)
    else:
        log(
            "total_precipitation_6hr not found in predictions/targets; skipping direct precipitation maps.",
            enable=args.verbose,
        )

    mslp_minima = compute_mslp_minima(artefacts.predictions, artefacts.targets)
    log(f"Computed MSLP minima for {len(mslp_minima)} lead steps", enable=args.verbose)

    summary = build_summary(
        metrics=metrics,
        mslp_minima=mslp_minima,
        heatmaps=heatmap_paths,
        surface_maps=surface_paths,
        precip_map=precip_path,
        metrics_csv=str(metrics_csv_path),
        precip_total_map=precip_total_path,
    )
    log("Summary dictionary assembled", enable=args.verbose)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Summary JSON saved to {summary_path}", enable=args.verbose)

    log(f"Evaluation complete. Summary written to {summary_path}", enable=args.verbose or True)


    log(f"Generated {len(surface_paths)} surface maps", enable=args.verbose)
if __name__ == "__main__":
    main()
