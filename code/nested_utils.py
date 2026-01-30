"""嵌套 GraphCast 场景下的区域辅助工具."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Optional

import numpy as np
import torch
import xarray as xr

from graphcast.graphcast import ModelConfig, TaskConfig


@dataclass(frozen=True)
class RegionConfig:
    """定义区域嵌套所需的核心范围与缓冲参数."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    buffer_deg: float = 0.0

    @property
    def expanded_lat_range(self) -> Tuple[float, float]:
        return (
            max(-90.0, self.lat_min - self.buffer_deg),
            min(90.0, self.lat_max + self.buffer_deg),
        )

    @property
    def expanded_lon_range(self) -> Tuple[float, float]:
        lon_min = (self.lon_min - self.buffer_deg) % 360.0
        lon_max = (self.lon_max + self.buffer_deg) % 360.0
        return lon_min, lon_max


def _build_index_slice(values: np.ndarray, min_val: float, max_val: float) -> slice:
    start = int(np.searchsorted(values, min_val, side="left"))
    stop = int(np.searchsorted(values, max_val, side="right"))
    return slice(start, max(stop, start + 1))


def compute_region_slices(
    latitudes: Iterable[float],
    longitudes: Iterable[float],
    region: RegionConfig,
) -> Tuple[slice, slice]:
    """根据区域设定返回索引切片."""
    lat_values = np.asarray(latitudes)
    lon_values = np.asarray(longitudes)
    lat_min, lat_max = region.expanded_lat_range
    lon_min, lon_max = region.expanded_lon_range
    if lon_min <= lon_max:
        lon_slice = _build_index_slice(lon_values, lon_min, lon_max)
    else:
        raise ValueError(
            "当前实现不支持跨越 0°/360° 的经度缓冲，请调整区域或缓冲设置。")
    lat_slice = _build_index_slice(lat_values, lat_min, lat_max)
    return lat_slice, lon_slice


def create_boundary_mask(
    lat_coord: xr.DataArray,
    lon_coord: xr.DataArray,
    region: RegionConfig,
) -> xr.DataArray:
    """创建 True 表示缓冲区（边界）的位置掩码."""
    lat_vals = lat_coord.values
    lon_vals = lon_coord.values
    core_lat = (lat_vals >= region.lat_min) & (lat_vals <= region.lat_max)
    core_lon = (lon_vals >= region.lon_min) & (lon_vals <= region.lon_max)
    core = np.outer(core_lat, core_lon)
    boundary = ~core
    return xr.DataArray(
        boundary,
        coords={"lat": lat_coord, "lon": lon_coord},
        dims=("lat", "lon"),
    )


def blend_boundary_on_last_timestep(
    inputs: xr.Dataset,
    boundary: xr.Dataset,
    boundary_mask: xr.DataArray,
) -> xr.Dataset:
    """用 boundary 数据替换 inputs 最后一个时间层上的缓冲区，保证维度对齐。"""
    result = inputs.copy(deep=True)
    if "time" not in result.dims:
        return result
    last_time = result.coords["time"][-1]
    for name, data in boundary.data_vars.items():
        if name not in result:
            continue

        if "time" not in result[name].dims:
            continue

        boundary_slice = data.astype("float32")
        if "batch" in boundary_slice.dims:
            boundary_slice = boundary_slice.isel(batch=0, drop=True)
        if "time" in boundary_slice.dims:
            boundary_slice = boundary_slice.isel(time=0, drop=True)

        target = result[name].sel(time=last_time)
        if target.ndim == 0 or not {"lat", "lon"}.issubset(set(target.dims)):
            continue
        boundary_slice = boundary_slice.broadcast_like(target)

        mask = boundary_mask.astype("bool")
        for dim in target.dims:
            if dim not in mask.dims:
                mask = mask.expand_dims({dim: target.coords[dim]})
        mask = mask.transpose(*target.dims)
        boundary_slice = boundary_slice.transpose(*target.dims)

        updated = xr.where(mask, boundary_slice, target)
        updated = updated.transpose(*target.dims, missing_dims="ignore")
        updated = updated.assign_coords(target.coords)
        result[name].loc[{"time": last_time}] = updated
    return result

def _update_regional_interior(
    dataset: xr.Dataset,
    prediction: xr.Dataset,
    target_index: int,
    core_mask: xr.DataArray,
) -> None:
    """Overwrite interior grid points inside the regional window with predictions."""
    if "time" not in dataset.dims:
        return
    target_time = dataset.coords["time"][target_index]
    mask = core_mask.astype("bool")
    region_lats = mask.coords["lat"]
    region_lons = mask.coords["lon"]

    for name, pred in prediction.data_vars.items():
        if name not in dataset.data_vars:
            continue
        pred_slice = pred

        if "batch" in pred_slice.dims:
            pred_slice = pred_slice.isel(batch=0)
        if "time" in pred_slice.dims:
            pred_slice = pred_slice.isel(time=-1)

        target_region = dataset[name].sel(time=target_time, lat=region_lats, lon=region_lons)
        pred_region = pred_slice.transpose(*target_region.dims, missing_dims="ignore").copy()
        mask_region = mask
        for dim in target_region.dims:
            if dim not in mask_region.dims:
                mask_region = mask_region.expand_dims({dim: target_region.coords[dim]})
        mask_region = mask_region.transpose(*target_region.dims)
        updated_region = xr.where(mask_region, pred_region, target_region)
        dataset[name].loc[{"time": target_time, "lat": region_lats, "lon": region_lons}] = updated_region


def serialize_model_config(model_config: ModelConfig) -> Dict[str, object]:
    return {
        "resolution": model_config.resolution,
        "mesh_size": model_config.mesh_size,
        "latent_size": model_config.latent_size,
        "gnn_msg_steps": model_config.gnn_msg_steps,
        "hidden_layers": model_config.hidden_layers,
        "radius_query_fraction_edge_length": model_config.radius_query_fraction_edge_length,
        "mesh2grid_edge_normalization_factor": model_config.mesh2grid_edge_normalization_factor,
    }


def serialize_task_config(task_config: TaskConfig) -> Dict[str, object]:
    return {
        "input_variables": list(task_config.input_variables),
        "target_variables": list(task_config.target_variables),
        "forcing_variables": list(task_config.forcing_variables),
        "pressure_levels": list(task_config.pressure_levels),
        "input_duration": task_config.input_duration,
    }


def deserialize_model_config(data: Dict[str, object]) -> ModelConfig:
    return ModelConfig(
        resolution=float(data["resolution"]),
        mesh_size=int(data["mesh_size"]),
        latent_size=int(data["latent_size"]),
        gnn_msg_steps=int(data["gnn_msg_steps"]),
        hidden_layers=int(data["hidden_layers"]),
        radius_query_fraction_edge_length=float(data["radius_query_fraction_edge_length"]),
        mesh2grid_edge_normalization_factor=float(data["mesh2grid_edge_normalization_factor"]),
    )


def deserialize_task_config(data: Dict[str, object]) -> TaskConfig:
    return TaskConfig(
        input_variables=tuple(data["input_variables"]),
        target_variables=tuple(data["target_variables"]),
        forcing_variables=tuple(data["forcing_variables"]),
        pressure_levels=tuple(int(x) for x in data["pressure_levels"]),
        input_duration=str(data["input_duration"]),
    )


@dataclass
class BoundaryEntry:
    index: int
    file: str
    forecast_reference_time: Optional[str]
    target_time: Optional[str]
    lead_hours: float


def load_boundary_manifest(boundary_dir: Path) -> List[BoundaryEntry]:
    manifest_path = boundary_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Boundary manifest not found at {manifest_path}. "
            "Please run precompute_global_boundaries.py first.")

    import json

    with manifest_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    entries: List[BoundaryEntry] = []
    for item in raw:
        entries.append(
            BoundaryEntry(
                index=int(item["index"]),
                file=str(item["file"]),
                forecast_reference_time=item.get("forecast_reference_time"),
                target_time=item.get("target_time"),
                lead_hours=float(item.get("lead_hours", 0.0)),
            )
        )
    entries.sort(key=lambda e: e.index)
    return entries


def _graphcast_state_dict(model: "GraphCast") -> Dict[str, object]:
    """Collect GraphCast parameters even if the class lacks state_dict()."""
    if hasattr(model, "state_dict"):
        return model.state_dict()  # type: ignore[return-value]
    return {
        "grid2mesh_gnn": model._grid2mesh_gnn.state_dict(),  # pylint: disable=protected-access
        "mesh_gnn": model._mesh_gnn.state_dict(),  # pylint: disable=protected-access
        "mesh2grid_gnn": model._mesh2grid_gnn.state_dict(),  # pylint: disable=protected-access
    }


def _load_graphcast_state_dict(model: "GraphCast", state_dict: Dict[str, object]) -> None:
    """Load GraphCast parameters regardless of explicit load_state_dict()."""
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(state_dict)  # type: ignore[arg-type]
        return
    for key, module in (
        ("grid2mesh_gnn", model._grid2mesh_gnn),  # pylint: disable=protected-access
        ("mesh_gnn", model._mesh_gnn),  # pylint: disable=protected-access
        ("mesh2grid_gnn", model._mesh2grid_gnn),  # pylint: disable=protected-access
    ):
        if key not in state_dict:
            raise KeyError(f"Missing state dict entry for {key}")
        module.load_state_dict(state_dict[key])
    if hasattr(model, "reset_grid_cache"):
        model.reset_grid_cache()


def load_boundary_patch(boundary_dir: Path, entry: BoundaryEntry) -> xr.Dataset:
    boundary_path = boundary_dir / entry.file
    if not boundary_path.exists():
        raise FileNotFoundError(
            f"Boundary file {boundary_path} referenced in manifest not found.")

    engine = None
    if boundary_path.suffix.lower() in (".h5", ".hdf5"):
        engine = "h5netcdf"

    try:
        dataset = xr.load_dataset(boundary_path, engine=engine, decode_cf=False)
    except ValueError:
        dataset = xr.load_dataset(boundary_path, decode_cf=False)

    for var_name in dataset.variables:
        attrs = dataset[var_name].attrs
        if "dtype" in attrs:
            attrs.pop("dtype")
    if "dtype" in dataset.attrs:
        dataset.attrs.pop("dtype")

    try:
        dataset = xr.decode_cf(dataset)
    except Exception:
        pass
    return dataset


def load_regional_checkpoint(
    path: Path,
    device: torch.device,
    fallback_model_config: ModelConfig,
    fallback_task_config: TaskConfig,
    sample_inputs: xr.Dataset | None = None,
    sample_targets: xr.Dataset | None = None,
    sample_forcings: xr.Dataset | None = None,
) -> Tuple["GraphCast", ModelConfig, TaskConfig]:
    from graphcast.graphcast import GraphCast  # Local import to avoid cycle

    payload = torch.load(path, map_location="cpu")
    model_conf = payload.get("model_config")
    task_conf = payload.get("task_config")
    model_config = deserialize_model_config(model_conf) if model_conf else fallback_model_config
    task_config = deserialize_task_config(task_conf) if task_conf else fallback_task_config
    model = GraphCast(model_config, task_config)
    if (
        sample_inputs is not None
        and sample_targets is not None
        and sample_forcings is not None
    ):
        with torch.no_grad():
            model(sample_inputs, targets_template=sample_targets, forcings=sample_forcings)
    _load_graphcast_state_dict(model, payload["state_dict"])
    model = model.to(device)
    return model, model_config, task_config


def save_regional_checkpoint(
    path: Path,
    model: "GraphCast",
    model_config: ModelConfig,
    task_config: TaskConfig,
) -> None:
    from graphcast.graphcast import GraphCast  # Local import to avoid cycle

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": _graphcast_state_dict(model),
        "model_config": serialize_model_config(model_config),
        "task_config": serialize_task_config(task_config),
    }
    torch.save(payload, path)
