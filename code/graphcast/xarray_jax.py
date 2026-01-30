"""兼容层：提供与旧版JAX封装类似的最小接口。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Mapping, Tuple

import numpy as np
import xarray


class JaxArrayWrapper:
  """简单的数组包装器，占位以兼容旧接口。"""

  def __init__(self, array: Any):
    self.array = array

  def __array__(self, dtype=None):
    return np.asarray(self.array, dtype=dtype)

  @property
  def shape(self):
    return np.asarray(self.array).shape

  @property
  def dtype(self):
    return np.asarray(self.array).dtype

  def __repr__(self) -> str:
    return f'JaxArrayWrapper({repr(self.array)})'


def Variable(dims, data, **kwargs) -> xarray.Variable:  # pylint:disable=invalid-name
  return xarray.Variable(dims, data, **kwargs)


def DataArray(*args, **kwargs) -> xarray.DataArray:  # pylint:disable=invalid-name
  return xarray.DataArray(*args, **kwargs)


def Dataset(*args, **kwargs) -> xarray.Dataset:  # pylint:disable=invalid-name
  return xarray.Dataset(*args, **kwargs)


def wrap(value: Any) -> Any:
  return value


def unwrap(value: Any, require_jax: bool = False) -> Any:
  del require_jax
  return value


def unwrap_data(data_array: xarray.DataArray, require_jax: bool = False):
  return unwrap(data_array.data, require_jax)


def unwrap_vars(dataset: xarray.Dataset, require_jax: bool = False):
  return {k: unwrap(v.data, require_jax) for k, v in dataset.data_vars.items()}


def dims_change_on_unflatten(fn: Callable[[Tuple[str, ...]], Tuple[str, ...]]):
  """上下文管理器占位符，与旧接口兼容。"""
  @contextmanager
  def _context():
    yield
  return _context()


def apply_ufunc(func: Callable[..., Any], *args, **kwargs):
  return xarray.apply_ufunc(func, *args, **kwargs)


def pmap(func: Callable[..., Any], dim: str):
  del dim
  return func


jax_data = unwrap_data
jax_vars = unwrap_vars
