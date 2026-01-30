#!/usr/bin/env python3
"""Rolling evaluation for nested regional GraphCast forecasts with CSAF fusion.

This is a CSA-aware variant of `eval_nested_baseline.py`. It expects
regional checkpoints trained with train_regional_full_year_v2.py or
train_regional_full_year_v2_csa_only.py, together with a matching
BoundaryCrossAttention checkpoint saved as *_attn.pt.

The metrics and outputs (metrics.csv, heatmaps, precip proxy maps, summary.json)
follow the same structure as `eval_nested_baseline.py` so results can be
compared directly.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_npu  # noqa: F401
import xarray as xr

from graphcast import data_pipeline, normalization, npz_utils, model_utils
from graphcast.graphcast import GraphCast
from graphcast.training_utils import (
    normalize_datasets_for_training,
    stack_targets_to_tensor,
)
from nested_utils import (
    RegionConfig,
    blend_boundary_on_last_timestep,
    compute_region_slices,
    create_boundary_mask,
    _update_regional_interior,
    load_boundary_manifest,
    load_boundary_patch,
    load_regional_checkpoint,
)

plt.switch_backend("agg")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
DEFAULT_DATASET = DATA_DIR / "dataset" / "source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc"
DEFAULT_GLOBAL_PARAM = DATA_DIR / "params" / (
    "params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - "
    "precipitation output only.npz"
)
DEFAULT_STATS_DIR = DATA_DIR / "stats"
DEFAULT_REGIONAL_CKPT = DATA_DIR / "dataset" / "params" / "regional_graphcast_full_year_v2.pt"

BATCH_DIM = "batch"

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


@dataclass
class EvalArtefacts:
    predictions: xr.Dataset
    targets: xr.Dataset
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]


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
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(param_path).encode("utf-8")).hexdigest()
    safe_stem = param_path.stem.replace(" ", "_")
    return cache_dir / f"{safe_stem}-{digest}.model.pt"


def _attention_path_for(base: Path) -> Path:
    return base.with_name(base.stem + "_attn" + base.suffix)


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
    """Invert normalize_datasets_for_training / residual normalisation."""
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


def _tensor_to_dataset(tensor: torch.Tensor, template: xr.Dataset) -> xr.Dataset:
    """Convert BHWC torch tensor to xr.Dataset following template shapes."""
    arr = tensor.detach().cpu().numpy()
    var = xr.Variable(("batch", "lat", "lon", "channels"), arr)
    return model_utils.stacked_to_dataset(var, template)


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


def coarsen_to_1deg(ds: xr.Dataset) -> xr.Dataset:
    """Downsample a regular lat/lon grid dataset to ~1° via mean coarsening.

    Assumes uniform spacing in lat/lon (e.g. 0.25° or 0.1°). If the inferred
    resolution is already >= 1° or the grid is too small, the input is
    returned unchanged.
    """
    if "lat" not in ds.dims or "lon" not in ds.dims:
        return ds
    lat = ds.lat
    lon = ds.lon
    if lat.size < 2 or lon.size < 2:
        return ds

    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    if dlat == 0.0 or dlon == 0.0:
        return ds

    lat_factor = int(round(1.0 / abs(dlat)))
    lon_factor = int(round(1.0 / abs(dlon)))

    # Already at ~1° or coarser – no need to coarsen further.
    if lat_factor <= 1 and lon_factor <= 1:
        return ds

    # boundary="trim" drops partial edge cells so pred/obs share the same domain.
    return ds.coarsen(
        lat=max(lat_factor, 1),
        lon=max(lon_factor, 1),
        boundary="trim",
    ).mean()


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
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)

    # Restrict to core region (exclude buffer) if bounds are provided.
    if lat_bounds is not None and lon_bounds is not None:
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        lat_mask = (pred.lat >= lat_min) & (pred.lat <= lat_max)
        lon_mask = (pred.lon >= lon_min) & (pred.lon <= lon_max)
        pred = pred.sel(lat=lat_mask, lon=lon_mask)
        obs = obs.sel(lat=lat_mask, lon=lon_mask)

    # Evaluate skill on a coarsened ~1° grid for fair comparison with
    # global models (0.25° → 1° via mean over 4×4 cells, etc.).
    pred = coarsen_to_1deg(pred)
    obs = coarsen_to_1deg(obs)

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
            return x.view(B, N, H, self.dim_head).transpose(1, 2)

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


def plot_heatmap(metrics: pd.DataFrame, metric: str, out_dir: Path) -> Path:
    pivot = metrics.pivot(index="variable", columns="lead_hours", values=metric)
    fig, ax = plt.subplots(figsize=(0.6 * max(1, pivot.shape[1]) + 4, 0.45 * max(1, pivot.shape[0]) + 2))
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
    """Re-use baseline surface plots (no CSA-specific change)."""
    # Implementation identical to nested_rolling_evaluation.plot_surface_step;
    # omitted here for brevity since main CSA behaviour is in rollout.
    # For consistent outputs, you can still call nested_rolling_evaluation
    # if you need these baseline-style plots.
    raise NotImplementedError("Use plot_surface_step_pretty for CSA evaluation.")


def plot_surface_step_pretty(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    lead_hour: int,
    out_dir: Path,
) -> Path:
    """Pretty surface fields for CSA eval (no wind arrows on temperature panels)."""
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)

    lead_idx = int(np.where(to_hours(pred.time.values) == lead_hour)[0][0])

    pred_t2m = pred["2m_temperature"].isel(time=lead_idx)
    obs_t2m = obs["2m_temperature"].isel(time=lead_idx)
    diff_t2m = pred_t2m - obs_t2m

    pred_mslp = pred["mean_sea_level_pressure"].isel(time=lead_idx)
    obs_mslp = obs["mean_sea_level_pressure"].isel(time=lead_idx)
    pred_mslp_hpa = pred_mslp / 100.0
    obs_mslp_hpa = obs_mslp / 100.0
    diff_mslp = pred_mslp_hpa - obs_mslp_hpa

    lon = pred_t2m.lon
    lat = pred_t2m.lat
    proj = ccrs.PlateCarree()

    # Physically constrained colour ranges
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

    p_diff_vmin, p_diff_vmax = -P_DIFF_DEFAULT, P_DIFF_DEFAULT

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        subplot_kw={"projection": proj},
    )
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    fields = [
        (pred_t2m, "t", "Regional 2m T (K)"),
        (obs_t2m, "t", "ERA5 2m T (K)"),
        (diff_t2m, "t_diff", "Δ2m T (K)"),
        (pred_mslp_hpa, "p", "Regional MSLP (hPa)"),
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
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
        ax.coastlines(resolution="10m", linewidth=0.8, zorder=1)

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

    fig.suptitle(f"Nested regional surface fields (CSA) @ +{lead_hour} h", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))

    path = out_dir / f"regional_surface_step_{lead_hour:03d}h_pretty.png"
    fig.savefig(path, dpi=200)
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
    """Standalone 10 m wind vectors for CSA eval (pred, ERA5, difference)."""
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)
    lead_idx = int(np.where(to_hours(pred.time.values) == lead_hour)[0][0])

    pred_u = pred["10m_u_component_of_wind"].isel(time=lead_idx)
    pred_v = pred["10m_v_component_of_wind"].isel(time=lead_idx)
    obs_u = obs["10m_u_component_of_wind"].isel(time=lead_idx)
    obs_v = obs["10m_v_component_of_wind"].isel(time=lead_idx)

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
        (pred_u, pred_v, "Regional 10m wind"),
        (obs_u, obs_v, "ERA5 10m wind"),
        (pred_u - obs_u, pred_v - obs_v, "Δ10m wind"),
    ]

    for ax, (u, v, title) in zip(axes, panels, strict=True):
        ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=proj)
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
        ax.coastlines(resolution="10m", linewidth=0.8, zorder=1)

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False

        q = ax.quiver(
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

    fig.suptitle(f"Nested regional 10m wind (CSA) @ +{lead_hour} h", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.95))

    path = out_dir / f"regional_wind_step_{lead_hour:03d}h.png"
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
    """Delegate to precip plotting in baseline script to keep outputs aligned."""
    from eval_nested_baseline import plot_precip_proxy_maps as _pp  # type: ignore

    return _pp(predictions, targets, level, lat_bounds, lon_bounds, out_dir)


def plot_precip_maps(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    out_dir: Path,
) -> Path:
    """Direct total precipitation maps (regional / ERA5 / Δ, in mm/6h)."""
    pred = squeeze_batch(predictions)
    obs = squeeze_batch(targets)

    if "total_precipitation_6hr" not in pred or "total_precipitation_6hr" not in obs:
        raise ValueError("total_precipitation_6hr not found in predictions/targets; cannot plot precipitation maps.")

    # Convert to mm / 6 h for readability.
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
    precip_max = min(precip_max, 60.0)  # cap to keep colour scale reasonable

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
                    f"Regional TP (mm/6h) @ +{lead}h",
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
            ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
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
        "Regional total precipitation (mm/6h)",
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

    fig.suptitle("Nested regional total precipitation (mm/6h)", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))

    path = out_dir / "regional_precip_maps.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def compute_mslp_minima(
    predictions: xr.Dataset,
    targets: xr.Dataset,
) -> list[dict[str, float | int]]:
    """Same definition as in eval_nested_baseline (still in Pa)."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested GraphCast rolling evaluation with CSA.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--global-param", type=Path, default=DEFAULT_GLOBAL_PARAM)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--regional-checkpoint", type=Path, default=DEFAULT_REGIONAL_CKPT)
    parser.add_argument(
        "--attention-checkpoint",
        type=Path,
        default=None,
        help="BoundaryCrossAttention checkpoint; default derives from regional checkpoint.",
    )
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_dir())
    parser.add_argument(
        "--boundary-dir",
        type=Path,
        default=None,
        help="Directory with precomputed boundaries (manifest.json).",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "nested_eval_outputs")
    parser.add_argument("--lead-hours", type=int, default=72)
    parser.add_argument("--target-lead-time", type=str, default="6h")
    parser.add_argument("--precip-level", type=int, default=850)
    parser.add_argument("--plot-leads", type=int, nargs="*", default=(6, 24, 48, 72))
    parser.add_argument(
        "--buffer",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--region",
        type=float,
        nargs=4,
        default=(3.0, 25.0, 100.0, 125.0),
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(msg: str, *, enable: bool) -> None:
    if enable:
        print(msg, flush=True)


def run_nested_rollout_csa(args: argparse.Namespace) -> EvalArtefacts:
    dataset = data_pipeline.load_dataset(str(args.dataset))
    truth_dataset = dataset.copy(deep=True)
    log(
        f"Dataset loaded: time={dataset.sizes['time']}, lat={dataset.sizes['lat']}, lon={dataset.sizes['lon']}",
        enable=args.verbose,
    )

    region_cfg = RegionConfig(
        lat_min=args.region[0],
        lat_max=args.region[1],
        lon_min=args.region[2],
        lon_max=args.region[3],
        buffer_deg=args.buffer,
    )
    lat_slice, lon_slice = compute_region_slices(dataset.lat.values, dataset.lon.values, region_cfg)
    patch_lat = dataset.lat.isel(lat=lat_slice)
    patch_lon = dataset.lon.isel(lon=lon_slice)
    boundary_mask = create_boundary_mask(patch_lat, patch_lon, region_cfg)

    model_config, task_config, _ = npz_utils.load_config_from_npz(str(args.global_param))
    stats_mean, stats_std, diff_std = data_pipeline.load_stats(str(args.stats_dir), task_config)

    stats_mean_reg = stats_mean
    if "lat" in stats_mean_reg.dims:
        stats_mean_reg = stats_mean_reg.isel(lat=lat_slice, drop=False)
    if "lon" in stats_mean_reg.dims:
        stats_mean_reg = stats_mean_reg.isel(lon=lon_slice, drop=False)

    stats_std_reg = stats_std
    if "lat" in stats_std_reg.dims:
        stats_std_reg = stats_std_reg.isel(lat=lat_slice, drop=False)
    if "lon" in stats_std_reg.dims:
        stats_std_reg = stats_std_reg.isel(lon=lon_slice, drop=False)

    diff_std_reg = diff_std
    if "lat" in diff_std_reg.dims:
        diff_std_reg = diff_std_reg.isel(lat=lat_slice, drop=False)
    if "lon" in diff_std_reg.dims:
        diff_std_reg = diff_std_reg.isel(lon=lon_slice, drop=False)

    time_values = dataset.coords["time"].values
    if len(time_values) < 4:
        raise ValueError("Dataset must contain at least four time steps.")
    step = pd.to_timedelta(time_values[1] - time_values[0])
    step_hours = step / pd.Timedelta("1h")
    input_duration = pd.Timedelta(task_config.input_duration)
    input_steps = int(input_duration / step) + 1
    target_step = int(pd.Timedelta(args.target_lead_time) / step)
    if target_step <= 0:
        raise ValueError("target_lead_time must be positive.")

    total_steps = math.ceil(args.lead_hours / step_hours)
    max_steps = dataset.sizes["time"] - input_steps
    if total_steps > max_steps:
        log(f"[warn] Dataset limits forecast steps to {max_steps}; truncating requested {total_steps}.", enable=True)
        total_steps = max_steps
    log(f"Rolling evaluation will cover {total_steps} steps (~{total_steps * step_hours:.0f} h).", enable=args.verbose)

    sample_end_idx = input_steps - 1
    sample_window = dataset.isel(time=slice(sample_end_idx - input_steps + 1, sample_end_idx + target_step + 1))
    sample_inputs, sample_targets, sample_forcings = data_pipeline.prepare_example(
        sample_window, task_config, args.target_lead_time
    )
    sample_reg_inputs = sample_inputs.isel(lat=lat_slice, lon=lon_slice)
    sample_reg_targets = sample_targets.isel(lat=lat_slice, lon=lon_slice)
    sample_reg_forcings = sample_forcings.isel(lat=lat_slice, lon=lon_slice)

    device = _ensure_device(args.device_id)
    log(f"Using device {device}.", enable=args.verbose)

    use_precomputed = args.boundary_dir is not None
    boundary_entries = []
    boundary_dir: Path | None = None
    if use_precomputed:
        boundary_dir = Path(args.boundary_dir).expanduser()
        boundary_entries = load_boundary_manifest(boundary_dir)
        if len(boundary_entries) < total_steps:
            raise ValueError(
                f"Boundary manifest has {len(boundary_entries)} entries < required {total_steps} steps."
            )
        log(f"Loaded {len(boundary_entries)} boundary entries from {boundary_dir}.", enable=args.verbose)
    else:
        from eval_nested_baseline import _load_model_with_cache as _load_global  # type: ignore

        global_model, cache_status = _load_global(
            model_config,
            task_config,
            Path(args.global_param),
            sample_inputs,
            sample_targets,
            sample_forcings,
            device,
            Path(args.cache_dir),
        )
        log(f"Global GraphCast ready ({cache_status}).", enable=args.verbose)

    if not Path(args.regional_checkpoint).exists():
        raise FileNotFoundError(f"Regional checkpoint not found: {args.regional_checkpoint}")
    regional_model, model_config, task_config = load_regional_checkpoint(
        Path(args.regional_checkpoint),
        device,
        model_config,
        task_config,
        sample_reg_inputs,
        sample_reg_targets,
        sample_reg_forcings,
    )
    if hasattr(regional_model, "eval"):
        regional_model.eval()
    log(f"Regional model loaded from {args.regional_checkpoint}.", enable=args.verbose)

    # Instantiate CSA module and load its weights.
    sample_channels = stack_targets_to_tensor(sample_reg_targets).shape[-1]
    attention = BoundaryCrossAttention(sample_channels, heads=2, alpha=1.0).to(device)
    attn_path = (
        Path(args.attention_checkpoint)
        if args.attention_checkpoint is not None
        else _attention_path_for(Path(args.regional_checkpoint))
    )
    if not attn_path.exists():
        raise FileNotFoundError(f"Attention checkpoint not found: {attn_path}")
    attn_payload = torch.load(attn_path, map_location=device)
    attn_state = attn_payload.get("state_dict", attn_payload)
    attention.load_state_dict(attn_state)
    attention.eval()
    log(f"Loaded attention weights from {attn_path}", enable=args.verbose)

    rolling_dataset = dataset.copy(deep=True)
    predictions: list[xr.Dataset] = []
    truths: list[xr.Dataset] = []
    valid_times: list[np.datetime64] = []
    lead_offsets: list[np.timedelta64] = []

    for step_idx in range(total_steps):
        target_index = step_idx + input_steps
        lead_hours = int((step_idx + 1) * step_hours)
        log(f"[step {step_idx+1}/{total_steps}] preparing window for +{lead_hours} h.", enable=args.verbose)

        window = rolling_dataset.isel(time=slice(step_idx, step_idx + input_steps + target_step))
        global_inputs, global_targets, global_forcings = data_pipeline.prepare_example(
            window, task_config, args.target_lead_time
        )
        region_inputs = global_inputs.isel(lat=lat_slice, lon=lon_slice)
        region_targets = global_targets.isel(lat=lat_slice, lon=lon_slice)
        region_forcings = global_forcings.isel(lat=lat_slice, lon=lon_slice)

        if use_precomputed:
            boundary_patch = load_boundary_patch(boundary_dir, boundary_entries[step_idx]).astype("float32")
            boundary_patch = _denormalize_predictions(
                boundary_patch,
                region_inputs,
                stats_mean_reg,
                stats_std_reg,
                diff_std_reg,
            )
        else:
            from eval_nested_baseline import _grid_outputs_to_predictions as _grid_to_pred  # type: ignore

            norm_inputs, norm_targets, norm_forcings = normalize_datasets_for_training(
                global_inputs, global_targets, global_forcings, stats_mean, stats_std, diff_std
            )
            with torch.no_grad():
                grid_outputs = global_model.forward_grid_node_outputs(norm_inputs, norm_forcings)
                pred_global = _grid_to_pred(
                    global_model, grid_outputs, norm_targets, global_inputs, stats_mean, stats_std, diff_std
                )
            boundary_patch = pred_global.isel(lat=lat_slice, lon=lon_slice).astype("float32")

        # Apply boundary patch into the rolling dataset and blend into inputs.
        from eval_nested_baseline import _apply_boundary_to_dataset  # type: ignore

        _apply_boundary_to_dataset(rolling_dataset, boundary_patch, target_index, lat_slice, lon_slice)
        blended_inputs = blend_boundary_on_last_timestep(region_inputs, boundary_patch, boundary_mask)

        if args.verbose and step_idx == 0:
            name = "mean_sea_level_pressure"
            if name in blended_inputs:
                baseline = float(blended_inputs[name].isel(time=-1).mean().values)
                log(f"[debug] baseline({name}) last input mean={baseline:.2f}", enable=True)

        # CSA forward pass in normalized space, then denormalize to physical units.
        norm_reg_inputs, norm_reg_targets, norm_reg_forcings = normalize_datasets_for_training(
            blended_inputs,
            region_targets,
            region_forcings,
            stats_mean_reg,
            stats_std_reg,
            diff_std_reg,
        )
        target_tensor = stack_targets_to_tensor(norm_reg_targets).to(device=device)

        # Normalize boundary patch in the same way as training.
        norm_boundary_vars = {}
        for name, data in boundary_patch.data_vars.items():
            if name in blended_inputs:
                residual = data - blended_inputs[name].isel(time=-1)
                norm_boundary_vars[name] = normalization.normalize(
                    xr.Dataset({name: residual}),
                    diff_std_reg,
                    None,
                )[name]
            else:
                norm_boundary_vars[name] = normalization.normalize(
                    xr.Dataset({name: data}),
                    stats_std_reg,
                    stats_mean_reg,
                )[name]
        norm_boundary = xr.Dataset(norm_boundary_vars)
        boundary_tensor = stack_targets_to_tensor(norm_boundary).to(device=device)

        with torch.no_grad():
            outputs = regional_model.forward_grid_node_outputs(norm_reg_inputs, norm_reg_forcings)
            outputs = outputs.to(device)
            pred_bhwc = outputs.permute(1, 0, 2).reshape(target_tensor.shape)
            bnd_bhwc = boundary_tensor
            reg_flat = pred_bhwc.reshape(pred_bhwc.shape[0], -1, pred_bhwc.shape[-1])
            bnd_flat = bnd_bhwc.reshape(bnd_bhwc.shape[0], -1, bnd_bhwc.shape[-1])
            fused_flat = attention(reg_flat, bnd_flat)
            fused_bhwc = fused_flat.view_as(pred_bhwc)

        fused_norm = _tensor_to_dataset(fused_bhwc, norm_reg_targets)
        pred_regional = _denormalize_predictions(
            fused_norm,
            blended_inputs,
            stats_mean_reg,
            stats_std_reg,
            diff_std_reg,
        )

        _update_regional_interior(rolling_dataset, pred_regional, target_index, (~boundary_mask).astype(bool))

        pred_frame = pred_regional[list(task_config.target_variables)].isel(time=-1)
        truth_frame = truth_dataset[list(task_config.target_variables)].isel(
            time=target_index, lat=lat_slice, lon=lon_slice
        )

        time_coord = truth_dataset.coords["time"][target_index].values
        pred_frame = pred_frame.expand_dims(time=[np.timedelta64(lead_hours, "h")])
        pred_frame = pred_frame.assign_coords(valid_time=("time", [time_coord]))
        truth_frame = truth_frame.expand_dims(time=[np.timedelta64(lead_hours, "h")])
        truth_frame = truth_frame.assign_coords(valid_time=("time", [time_coord]))

        predictions.append(pred_frame)
        truths.append(truth_frame)
        valid_times.append(time_coord)
        lead_offsets.append(np.timedelta64(lead_hours, "h"))

        log(f"[step {step_idx+1}/{total_steps}] +{lead_hours} h forecast ready.", enable=args.verbose)

    predictions_ds = xr.concat(predictions, dim="time")
    truths_ds = xr.concat(truths, dim="time")
    predictions_ds = predictions_ds.assign_coords(valid_time=("time", valid_times))
    truths_ds = truths_ds.assign_coords(valid_time=("time", valid_times))

    return EvalArtefacts(
        predictions=predictions_ds,
        targets=truths_ds,
        lat_bounds=(args.region[0], args.region[1]),
        lon_bounds=(args.region[2], args.region[3]),
    )


def main() -> None:
    args = parse_args()
    _setup_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Outputs will be written to {args.output_dir}", enable=True)

    artefacts = run_nested_rollout_csa(args)

    metrics = build_metrics_dataframe(
        artefacts.predictions,
        artefacts.targets,
        args.precip_level,
        artefacts.lat_bounds,
        artefacts.lon_bounds,
    )
    metrics_csv = args.output_dir / "metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    log(f"Metrics saved to {metrics_csv}", enable=True)

    heatmap_paths = {}
    for metric_name in ("rmse", "bias", "acc"):
        heatmap_paths[metric_name] = str(plot_heatmap(metrics, metric_name, args.output_dir))
        log(f"Heatmap for {metric_name} written to {heatmap_paths[metric_name]}", enable=args.verbose)

    surface_paths: dict[int, str] = {}
    available_leads = set(to_hours(squeeze_batch(artefacts.predictions).time.values))
    requested = set(args.plot_leads) if args.plot_leads else set()
    for lead in sorted(available_leads & requested):
        surface_paths[int(lead)] = str(
            plot_surface_step_pretty(
                artefacts.predictions,
                artefacts.targets,
                artefacts.lat_bounds,
                artefacts.lon_bounds,
                lead,
                args.output_dir,
            )
        )
        # Standalone wind-only figure for each requested lead
        plot_wind_step(
            artefacts.predictions,
            artefacts.targets,
            artefacts.lat_bounds,
            artefacts.lon_bounds,
            lead,
            args.output_dir,
        )

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

    summary = build_summary(
        metrics=metrics,
        mslp_minima=mslp_minima,
        heatmaps=heatmap_paths,
        surface_maps=surface_paths,
        precip_map=precip_path,
        metrics_csv=str(metrics_csv),
        precip_total_map=precip_total_path,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Summary saved to {summary_path}", enable=True)
    log("Nested rolling evaluation with CSA completed.", enable=True)


if __name__ == "__main__":
    main()
