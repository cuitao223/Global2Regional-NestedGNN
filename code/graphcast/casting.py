"""预测器包装器：兼容PyTorch版本的BF16封装。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import xarray

from graphcast import predictor_base


def _cast_dataset(ds: xarray.Dataset, dtype: np.dtype) -> xarray.Dataset:
  def _maybe_cast(da: xarray.DataArray) -> xarray.DataArray:
    if np.issubdtype(da.dtype, np.floating):
      return da.astype(dtype)
    return da
  return ds.map(_maybe_cast)


class Bfloat16Cast(predictor_base.Predictor):
  """将浮点数据转换为bfloat16运行，再恢复到原始dtype。"""

  def __init__(self, predictor: predictor_base.Predictor, enabled: bool = True):
    self._predictor = predictor
    self._enabled = enabled

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs) -> xarray.Dataset:
    if not self._enabled:
      return self._predictor(inputs, targets_template, forcings, **kwargs)

    inputs_bf16 = _cast_dataset(inputs, np.dtype(np.bfloat16))
    targets_template_bf16 = _cast_dataset(targets_template, np.dtype(np.bfloat16))
    forcings_bf16 = _cast_dataset(forcings, np.dtype(np.bfloat16))

    predictions = self._predictor(inputs_bf16, targets_template_bf16,
                                  forcings_bf16, **kwargs)
    return _cast_dataset(predictions, np.dtype(np.float32))

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs) -> predictor_base.LossAndDiagnostics:
    if not self._enabled:
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    inputs_bf16 = _cast_dataset(inputs, np.dtype(np.bfloat16))
    targets_bf16 = _cast_dataset(targets, np.dtype(np.bfloat16))
    forcings_bf16 = _cast_dataset(forcings, np.dtype(np.bfloat16))
    loss, diagnostics = self._predictor.loss(inputs_bf16, targets_bf16,
                                             forcings_bf16, **kwargs)
    loss = loss.astype(np.float32)
    diagnostics = {k: v.astype(np.float32) if hasattr(v, 'astype') else v
                   for k, v in diagnostics.items()}
    return loss, diagnostics

  def loss_and_predictions(self,
                           inputs: xarray.Dataset,
                           targets: xarray.Dataset,
                           forcings: xarray.Dataset,
                           **kwargs
                           ) -> tuple[predictor_base.LossAndDiagnostics,
                                      xarray.Dataset]:
    if not self._enabled:
      return self._predictor.loss_and_predictions(inputs, targets, forcings,
                                                  **kwargs)

    inputs_bf16 = _cast_dataset(inputs, np.dtype(np.bfloat16))
    targets_bf16 = _cast_dataset(targets, np.dtype(np.bfloat16))
    forcings_bf16 = _cast_dataset(forcings, np.dtype(np.bfloat16))
    (loss, diagnostics), predictions = self._predictor.loss_and_predictions(
        inputs_bf16, targets_bf16, forcings_bf16, **kwargs)
    loss = loss.astype(np.float32)
    diagnostics = {k: v.astype(np.float32) if hasattr(v, 'astype') else v
                   for k, v in diagnostics.items()}
    predictions = _cast_dataset(predictions, np.dtype(np.float32))
    return (loss, diagnostics), predictions


LossAndDiagnostics = predictor_base.LossAndDiagnostics
MappingLike = Mapping[str, xarray.DataArray]
