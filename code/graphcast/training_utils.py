"""Helper utilities for PyTorch-based GraphCast training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import xarray as xr

from graphcast import losses, model_utils, normalization


@dataclass
class LossWeights:
  lat_weights: torch.Tensor
  channel_weights: torch.Tensor
  lon_size: int


def stack_targets_to_tensor(
    dataset: xr.Dataset,
) -> torch.Tensor:
  """Converts a target Dataset to a BHWC torch tensor."""
  stacked = model_utils.dataset_to_stacked(dataset)
  array = np.asarray(stacked.data, dtype=np.float32)
  return torch.from_numpy(array)


def build_loss_weights(
    template_dataset: xr.Dataset,
    per_variable_weights: Dict[str, float],
) -> LossWeights:
  """Builds latitude and channel weighting tensors."""
  preserved_dims = ("batch", "lat", "lon")

  lat_reference_var = template_dataset[list(sorted(template_dataset.data_vars))[0]]
  lat_size = lat_reference_var.sizes.get("lat", 0)
  if lat_size <= 1:
    lat_weights_np = np.ones(max(lat_size, 1), dtype=np.float32)
  else:
    lat_weights_np = losses.normalized_latitude_weights(lat_reference_var).values.astype(
        np.float32)
  lat_weights = torch.from_numpy(lat_weights_np)

  total_channels = 0
  metadata = []
  for name in sorted(template_dataset.data_vars.keys()):
    template_var = template_dataset[name]
    dims = [dim for dim in template_var.dims if dim not in preserved_dims]
    shape = [template_var.sizes[dim] for dim in dims]
    size = int(np.prod(shape)) if shape else 1
    metadata.append((name, dims, shape, total_channels, total_channels + size))
    total_channels += size

  channel_weights = torch.ones(total_channels, dtype=torch.float32)
  for name, dims, shape, start, end in metadata:
    base_weight = float(per_variable_weights.get(name, 1.0))
    weights = torch.full((end - start,), base_weight, dtype=torch.float32)
    if "level" in dims:
      level_weights = losses.normalized_level_weights(
          template_dataset[name]).values.astype(np.float32)
      tensor = torch.ones(shape, dtype=torch.float32)
      idx = dims.index("level")
      view_shape = [1] * len(shape)
      view_shape[idx] = len(level_weights)
      tensor = tensor * torch.from_numpy(level_weights).view(view_shape)
      weights = weights * tensor.reshape(-1)
    channel_weights[start:end] = weights

  return LossWeights(
      lat_weights=lat_weights,
      channel_weights=channel_weights,
      lon_size=int(template_dataset.sizes["lon"]),
  )


def weighted_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: LossWeights,
) -> torch.Tensor:
  """Computes a latitude/level weighted MSE in torch."""
  batch, lat, lon, channels = targets.shape
  pred = predictions.permute(1, 0, 2).reshape(batch, lat, lon, channels)

  lat_weights = loss_weights.lat_weights.to(predictions.device).view(1, lat, 1, 1)
  channel_weights = loss_weights.channel_weights.to(predictions.device).view(
      1, 1, 1, channels)
  weights = lat_weights * channel_weights

  diff = pred - targets.to(predictions.device)
  numerator = (diff.pow(2) * weights).sum()
  denominator = weights.sum() * loss_weights.lon_size
  return numerator / denominator


def normalize_datasets_for_training(
    inputs: xr.Dataset,
    targets: xr.Dataset,
    forcings: xr.Dataset,
    stats_mean: xr.Dataset,
    stats_std: xr.Dataset,
    diff_std: xr.Dataset,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
  """Prepares normalized inputs, targets and forcings."""

  norm_inputs = normalization.normalize(inputs, stats_std, stats_mean)
  norm_forcings = normalization.normalize(forcings, stats_std, stats_mean)

  def _normalize_target(target: xr.DataArray) -> xr.DataArray:
    if target.name in inputs:
      last_input = inputs[target.name].isel(time=-1)
      residual = target - last_input
      return normalization.normalize(residual, diff_std, None)
    return normalization.normalize(target, stats_std, stats_mean)

  norm_targets = xr.Dataset({
      name: _normalize_target(targets[name])
      for name in targets.data_vars.keys()
  })
  return norm_inputs, norm_targets, norm_forcings
