"""自回归包装器：将单步预测器扩展为多步预测。"""

from __future__ import annotations

import numpy as np
import xarray

from typing import Optional

from absl import logging

from graphcast import predictor_base


class Predictor(predictor_base.Predictor):
  """将单步Predictor扩展为自回归多步预测。"""

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      noise_level: Optional[float] = None,
      gradient_checkpointing: bool = False,
      ):
    self._predictor = predictor
    self._noise_level = noise_level
    self._gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
      logging.warning('PyTorch版本暂未支持gradient_checkpointing，参数将被忽略。')

  def _get_and_validate_constant_inputs(self, inputs, targets, forcings):
    constant_inputs = inputs.drop_vars(targets.keys(), errors='ignore')
    constant_inputs = constant_inputs.drop_vars(forcings.keys(), errors='ignore')
    for name, var in constant_inputs.items():
      if 'time' in var.dims:
        raise ValueError(
            f'时间维输入 {name} 既不是目标也不是强迫项，无法进行自回归反馈。')
    return constant_inputs

  def _validate_targets_and_forcings(self, targets, forcings):
    for name, var in targets.items():
      if 'time' not in var.dims:
        raise ValueError(f'目标变量 {name} 必须包含time维度。')
    for name, var in forcings.items():
      if 'time' not in var.dims:
        raise ValueError(f'强迫变量 {name} 必须包含time维度。')
    overlap = forcings.keys() & targets.keys()
    if overlap:
      raise ValueError(f'变量 {overlap} 同时出现在targets和forcings中。')

  def _update_inputs(self, inputs: xarray.Dataset, next_frame: xarray.Dataset) -> xarray.Dataset:
    num_inputs = inputs.dims['time']
    predicted_or_forced_inputs = next_frame[list(inputs.keys())]
    updated = xarray.concat([inputs, predicted_or_forced_inputs], dim='time')
    updated = updated.tail(time=num_inputs)
    return updated.assign_coords(time=inputs.coords['time'])

  def _apply_noise(self, dataset: xarray.Dataset) -> xarray.Dataset:
    if not self._noise_level:
      return dataset
    rng = np.random.default_rng()

    def add_noise(da: xarray.DataArray) -> xarray.DataArray:
      if np.issubdtype(da.dtype, np.floating):
        noise = rng.standard_normal(size=da.shape).astype(da.dtype)
        return da + self._noise_level * noise
      return da

    return dataset.map(add_noise)

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs) -> xarray.Dataset:
    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets_template, forcings)
    self._validate_targets_and_forcings(targets_template, forcings)

    time_dependent_inputs = inputs.drop_vars(constant_inputs.keys())
    time_dependent_inputs = self._apply_noise(time_dependent_inputs)

    predictions_per_step = []
    target_times = targets_template.coords['time']
    target_template = targets_template.isel(time=[0])

    current_inputs = time_dependent_inputs
    for step, time_value in enumerate(target_times):
      current_forcing = forcings.isel(time=[step])
      all_inputs = xarray.merge([constant_inputs, current_inputs])
      prediction = self._predictor(
          all_inputs,
          target_template.assign_coords(time=[time_value]),
          forcings=current_forcing,
          **kwargs)
      predictions_per_step.append(prediction)
      next_frame = xarray.merge([prediction, current_forcing])
      current_inputs = self._update_inputs(current_inputs, next_frame)

    predictions = xarray.concat(predictions_per_step, dim='time')
    predictions = predictions.assign_coords(time=target_times)
    return predictions

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs) -> predictor_base.LossAndDiagnostics:
    if targets.sizes['time'] == 1:
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets, forcings)
    self._validate_targets_and_forcings(targets, forcings)

    time_dependent_inputs = inputs.drop_vars(constant_inputs.keys())
    time_dependent_inputs = self._apply_noise(time_dependent_inputs)

    target_times = targets.coords['time']
    current_inputs = time_dependent_inputs

    loss_series = []
    diagnostics_series = []

    for step, time_value in enumerate(target_times):
      target_step = targets.isel(time=[step])
      forcing_step = forcings.isel(time=[step])
      all_inputs = xarray.merge([constant_inputs, current_inputs])
      try:
        (loss, diagnostics), predictions = self._predictor.loss_and_predictions(
            all_inputs, target_step, forcings=forcing_step, **kwargs)
      except NotImplementedError:
        loss, diagnostics = self._predictor.loss(
            all_inputs, target_step, forcings=forcing_step, **kwargs)
        predictions = self._predictor(
            all_inputs, target_step, forcings=forcing_step, **kwargs)

      loss_series.append(loss.expand_dims(time=[time_value]))
      diagnostics_series.append({
          name: value.expand_dims(time=[time_value])
          for name, value in diagnostics.items()
      })

      next_frame = xarray.merge([predictions, forcing_step])
      current_inputs = self._update_inputs(current_inputs, next_frame)

    stacked_loss = xarray.concat(loss_series, dim='time').mean('time', skipna=False)

    diagnostics_mean = {}
    if diagnostics_series:
      keys = diagnostics_series[0].keys()
      for name in keys:
        diag_concat = xarray.concat([d[name] for d in diagnostics_series], dim='time')
        diagnostics_mean[name] = diag_concat.mean('time', skipna=False)

    return stacked_loss, diagnostics_mean
