"""Utilities for loading GraphCast configuration files exported from Haiku."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from graphcast.graphcast import ModelConfig, TaskConfig


def _collect_list(data: np.lib.npyio.NpzFile, prefix: str, cast=str):
  values = []
  index = 0
  while True:
    key = f"{prefix}:{index}"
    if key not in data:
      break
    value = data[key]
    if isinstance(value, np.ndarray):
      if value.shape == ():
        value = value.item()
    values.append(cast(value))
    index += 1
  return tuple(values)


def load_config_from_npz(npz_path: str) -> Tuple[ModelConfig, TaskConfig, str]:
  """Parses GraphCast model and task configs stored in an NPZ archive."""

  with np.load(npz_path) as data:
    model_config = ModelConfig(
        resolution=float(data["model_config:resolution"]),
        mesh_size=int(data["model_config:mesh_size"]),
        latent_size=int(data["model_config:latent_size"]),
        gnn_msg_steps=int(data["model_config:gnn_msg_steps"]),
        hidden_layers=int(data["model_config:hidden_layers"]),
        radius_query_fraction_edge_length=float(
            data["model_config:radius_query_fraction_edge_length"]),
        mesh2grid_edge_normalization_factor=float(
            data["model_config:mesh2grid_edge_normalization_factor"]))

    input_variables = _collect_list(data, "task_config:input_variables", str)
    target_variables = _collect_list(data, "task_config:target_variables", str)
    forcing_variables = _collect_list(data, "task_config:forcing_variables", str)
    pressure_levels = _collect_list(
        data, "task_config:pressure_levels", lambda x: int(x))
    input_duration = str(data["task_config:input_duration"])

    task_config = TaskConfig(
        input_variables=input_variables,
        target_variables=target_variables,
        forcing_variables=forcing_variables,
        pressure_levels=pressure_levels,
        input_duration=input_duration,
    )

    description = str(data["description"]) if "description" in data else ""

  return model_config, task_config, description
