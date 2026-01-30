"""Data loading utilities for PyTorch GraphCast."""

from __future__ import annotations

import os
from typing import Iterable, Set, Tuple

import xarray as xr

from graphcast import data_utils
from graphcast.graphcast import TaskConfig


def load_dataset(path: str) -> xr.Dataset:
  """Loads an ERA5 NetCDF file and enriches it with derived features."""
  dataset = xr.open_dataset(path)
  # Ensure derived temporal features are present.
  data_utils.add_derived_vars(dataset)
  return dataset


def prepare_example(
    dataset: xr.Dataset,
    task_config: TaskConfig,
    target_lead_time: str = "6h",
    lat_slice: slice | None = None,
    lon_slice: slice | None = None,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
  """Extracts inputs, targets and forcing datasets matching TaskConfig."""
  if lat_slice is not None or lon_slice is not None:
    dataset = dataset.isel(
        lat=lat_slice if lat_slice is not None else slice(None),
        lon=lon_slice if lon_slice is not None else slice(None),
    )
  inputs_all, targets_all = data_utils.extract_input_target_times(
      dataset,
      input_duration=task_config.input_duration,
      target_lead_times=target_lead_time)

  inputs = inputs_all[list(task_config.input_variables)]
  targets = targets_all[list(task_config.target_variables)]
  forcings = targets_all[list(task_config.forcing_variables)]

  return inputs, targets, forcings


def load_stats(
    stats_dir: str,
    task_config: TaskConfig,
    extra_variables: Iterable[str] = (),
    lat_slice: slice | None = None,
    lon_slice: slice | None = None,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
  """Loads mean/std statistics and aligns them with the current task."""

  variables: Set[str] = set(task_config.input_variables)
  variables.update(task_config.target_variables)
  variables.update(task_config.forcing_variables)
  variables.update(extra_variables)

  def _read(name: str) -> xr.Dataset:
    """支持不同命名风格的统计文件."""
    candidates = [
        name,
        name.replace("-", "_", 1),
        name.replace("-", "_"),
    ]
    for candidate in candidates:
      path = os.path.join(stats_dir, candidate)
      if os.path.exists(path):
        return xr.open_dataset(path)
    raise FileNotFoundError(
        f"无法在 {stats_dir} 找到以下任一统计文件：{', '.join(candidates)}")

  def _subset(ds: xr.Dataset) -> xr.Dataset:
    selected = {var: ds[var] for var in variables if var in ds.data_vars}
    processed = {}
    for name, data in selected.items():
      arr = data
      if "level" in arr.dims:
        arr = arr.sel(level=list(task_config.pressure_levels))
      if lat_slice is not None and "lat" in arr.dims:
        arr = arr.isel(lat=lat_slice)
      if lon_slice is not None and "lon" in arr.dims:
        arr = arr.isel(lon=lon_slice)
      processed[name] = arr.astype("float32")
    return xr.Dataset(processed)

  mean = _subset(_read("stats-mean_by_level.nc"))
  std = _subset(_read("stats-stddev_by_level.nc"))
  diffs_std = _subset(_read("stats-diffs_stddev_by_level.nc"))
  return mean, std, diffs_std
