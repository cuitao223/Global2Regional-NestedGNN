"""Convert raw ERA5 exports into the GraphCast-style NetCDF schema.

This script aligns variable names, dimensions, and coordinates to match the
canonical dataset layout used by this project (e.g.
`source-era5_date-YYYY-MM-DD_res-0.25_levels-13_steps-*.nc`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
from typing import Dict, Iterable, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr

try:
    import h5netcdf  # noqa: F401
    _H5_AVAILABLE = True
except ImportError:
    _H5_AVAILABLE = False
try:
    import netCDF4  # noqa: F401
    _NETCDF4_AVAILABLE = True
except ImportError:
    _NETCDF4_AVAILABLE = False


def _open_nc(path: Path) -> xr.Dataset:
    """优先 h5netcdf，其次默认引擎，统一报错。"""
    open_kwargs = {}
    if _H5_AVAILABLE:
        open_kwargs["engine"] = "h5netcdf"
    try:
        return xr.open_dataset(path, **open_kwargs)
    except Exception as exc:  # noqa: BLE001
        try:
            return xr.open_dataset(path)
        except Exception as exc2:  # noqa: BLE001
            raise OSError(f"无法打开 {path}: first={exc}, fallback={exc2}") from exc2

# 映射：原始变量名 -> 目标变量名
PART1_VAR_MAP: Dict[str, str] = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
}

ACCUM_VAR_MAP: Dict[str, str] = {
    "tp": "total_precipitation_6hr",
    "tisr": "toa_incident_solar_radiation",
}

INSTANT_VAR_MAP: Dict[str, str] = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "z": "geopotential_at_surface",
    "lsm": "land_sea_mask",
}

TARGET_VARS: Tuple[str, ...] = (
    "geopotential_at_surface",
    "land_sea_mask",
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation",
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)


def debug_dataset(ds: xr.Dataset, name: str) -> None:
    """打印调试信息，便于确认原始格式。"""
    print(f"\n==== {name} ====")
    print("Dims:")
    for k, v in ds.sizes.items():
        print(f"  - {k}: {v}")
    print("Vars:")
    for var in ds.data_vars:
        shape = tuple(ds[var].shape)
        print(f"  - {var}: shape={shape}, dtype={ds[var].dtype}")
    if "valid_time" in ds.coords:
        times = ds["valid_time"].values
        print(f"time head: {times[:3]} tail: {times[-3:]}")
    print("================")


def ensure_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """标准化经纬度坐标到 lat (升序 -90..90), lon (0..360)。"""
    rename_map = {}
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    ds = ds.rename(rename_map)

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        lon = ds["lon"]
        if lon.min() < 0:
            lon = (lon + 360) % 360
        ds = ds.assign_coords(lon=lon)
        ds = ds.sortby("lon")
    return ds


def to_time_coords(valid_time: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """生成 time(小时 int32) 与 datetime(batch,time) 坐标。"""
    if valid_time.size == 0:
        raise ValueError("valid_time 为空，无法生成时间坐标")
    datetimes_np = np.asarray(pd.to_datetime(valid_time.values), dtype="datetime64[ns]")
    base = datetimes_np[0]
    # time 用小时偏移，保持与 2022 基准一致（int32 + units=hours）
    hours = ((datetimes_np - base) / np.timedelta64(1, "h")).astype("int32")
    time = xr.DataArray(hours, dims=("time",), attrs={"units": "hours"})
    datetime_coord = xr.DataArray(
        np.expand_dims(datetimes_np, 0),
        dims=("batch", "time"),
        coords={"batch": [0], "time": time},
    )
    return time, datetime_coord


def subset_month(ds: xr.Dataset, month: int) -> xr.Dataset:
    """按月份切分年度文件，保持原有坐标。"""
    if "valid_time" not in ds.coords:
        raise ValueError("数据缺少 valid_time 坐标，无法按月切分")
    idx = pd.to_datetime(ds["valid_time"].values).month == month
    return ds.isel(valid_time=idx)


def reorder_level_var(da: xr.DataArray) -> xr.DataArray:
    """将含 level 的变量调维为 (time, level, lat, lon)，并转 float32。"""
    target_order = ("valid_time", "level", "lat", "lon")
    da = da.transpose(*target_order)
    return da.astype(np.float32)


def reorder_surface_var(da: xr.DataArray) -> xr.DataArray:
    """将不含 level 的时变变量调维为 (time, lat, lon)，并转 float32。"""
    target_order = ("valid_time", "lat", "lon")
    da = da.transpose(*target_order)
    return da.astype(np.float32)


def extract_static(da: xr.DataArray) -> xr.DataArray:
    """将时变面变量转为静态 (lat, lon)，默认取第一个时间步。"""
    if "valid_time" in da.dims:
        da = da.isel(valid_time=0, drop=True)
    da = da.transpose("lat", "lon")
    return da.astype(np.float32)


def build_month_dataset(
    part1: xr.Dataset,
    accum: xr.Dataset,
    instant: xr.Dataset,
    month: int,
    start_idx: int = 0,
    chunk_size: int | None = None,
) -> xr.Dataset:
    """将三类数据合并为目标结构的月度 dataset，可按时间分片。"""
    part1 = ensure_lat_lon(part1)
    accum = ensure_lat_lon(accum)
    instant = ensure_lat_lon(instant)

    part1 = part1.rename({"pressure_level": "level"}) if "pressure_level" in part1.coords else part1
    part1_month = part1
    accum_month = subset_month(accum, month)
    instant_month = subset_month(instant, month)

    # 校验时间长度
    if part1_month.dims.get("valid_time", 0) == 0:
        raise ValueError(f"月文件缺少 valid_time：month={month}")
    if accum_month.dims.get("valid_time", 0) < part1_month.dims["valid_time"]:
        raise ValueError(f"累积量时间不足：month={month}")
    if instant_month.dims.get("valid_time", 0) < part1_month.dims["valid_time"]:
        raise ValueError(f"瞬时量时间不足：month={month}")

    # 对齐时间：使用 part1 的时间戳进行精确 reindex
    vt_full = part1_month["valid_time"]
    end_idx = None if chunk_size is None else start_idx + chunk_size
    vt_full = vt_full.isel(valid_time=slice(start_idx, end_idx))
    part1_month = part1_month.isel(valid_time=slice(start_idx, end_idx))
    accum_month = accum_month.isel(valid_time=slice(start_idx, end_idx))
    instant_month = instant_month.isel(valid_time=slice(start_idx, end_idx))

    vt = vt_full
    accum_month = accum_month.reindex(valid_time=vt)
    instant_month = instant_month.reindex(valid_time=vt)

    time_coord, datetime_coord = to_time_coords(vt)
    level_coord = part1_month["level"] if "level" in part1_month.coords else None
    lat_coord = part1_month["lat"]
    lon_coord = part1_month["lon"]

    data_vars = {}

    # 三维（含 level）变量
    for raw, target in PART1_VAR_MAP.items():
        if raw not in part1_month:
            raise KeyError(f"缺少必需变量 {raw} (目标 {target}) in 2023_{month}.nc")
        da = reorder_level_var(part1_month[raw])
        da = da.rename({"valid_time": "time"})
        data_vars[target] = da

    # 累积量
    for raw, target in ACCUM_VAR_MAP.items():
        if raw not in accum_month:
            raise KeyError(f"缺少累积量变量 {raw} (目标 {target})")
        da = reorder_surface_var(accum_month[raw])
        da = da.rename({"valid_time": "time"})
        data_vars[target] = da

    # 面层瞬时量
    for raw, target in INSTANT_VAR_MAP.items():
        if raw not in instant_month:
            raise KeyError(f"缺少瞬时量变量 {raw} (目标 {target})")
        da = instant_month[raw]
        if raw in {"u10", "v10", "t2m", "msl", "z"}:
            da = reorder_surface_var(da)
            da = da.rename({"valid_time": "time"})
        else:  # lsm 等静态
            da = extract_static(da)
        data_vars[target] = da

    ds = xr.Dataset(data_vars)

    # 添加 batch 维度
    for name, da in ds.data_vars.items():
        if "time" in da.dims:
            ds[name] = da.expand_dims(batch=[0]).transpose("batch", "time", ...)

    # 设置坐标
    coords = {
        "time": time_coord,
        "datetime": datetime_coord,
        "lat": lat_coord,
        "lon": lon_coord,
        "batch": [0],
    }
    if level_coord is not None:
        coords["level"] = level_coord
    ds = ds.assign_coords(coords)

    # 静态变量确保维度顺序
    # 静态变量裁剪掉时间/批次后重排
    if "geopotential_at_surface" in ds and "time" in ds["geopotential_at_surface"].dims:
        ds["geopotential_at_surface"] = ds["geopotential_at_surface"].isel(batch=0, time=0, drop=True)
    if "land_sea_mask" in ds and "time" in ds["land_sea_mask"].dims:
        ds["land_sea_mask"] = ds["land_sea_mask"].isel(batch=0, time=0, drop=True)
    if "geopotential_at_surface" in ds:
        ds["geopotential_at_surface"] = ds["geopotential_at_surface"].transpose("lat", "lon")
    if "land_sea_mask" in ds:
        ds["land_sea_mask"] = ds["land_sea_mask"].transpose("lat", "lon")

    # 缺失检查
    missing = [v for v in TARGET_VARS if v not in ds]
    if missing:
        raise ValueError(f"输出缺少变量: {missing}")

    return ds


def process_month_worker(params: tuple[int, str, str, str, str, int | None, bool]) -> None:
    """多进程 worker：处理指定月份并按需分片输出。"""
    month, root_str, out_dir_str, accum_p, instant_p, chunk_arg, overwrite = params
    root_dir = Path(root_str)
    out_dir_local = Path(out_dir_str)
    part1_path = root_dir / f"2023_{month}.nc"
    if not part1_path.exists():
        raise FileNotFoundError(f"缺少月文件 {part1_path}")
    part1_ds = _open_nc(part1_path)
    accum_ds = _open_nc(Path(accum_p))
    instant_ds = _open_nc(Path(instant_p))

    total_steps = part1_ds.dims.get("valid_time", 0)
    if total_steps == 0:
        raise ValueError(f"{part1_path} 不含 valid_time")
    chunk = chunk_arg or total_steps
    num_parts = (total_steps + chunk - 1) // chunk

    print(f"\n>>> 处理月份 {month:02d}，total_steps={total_steps}, chunk={chunk}, parts={num_parts}")
    for idx in range(num_parts):
        start = idx * chunk
        time_len = min(chunk, total_steps - start)
        out_name = (
            f"source-era5_date-2023-{month:02d}-01_steps-{time_len:02d}_part{idx+1:02d}.nc"
        )
        out_path = out_dir_local / out_name

        def _is_valid(path: Path) -> bool:
            try:
                ds_chk = xr.open_dataset(path, decode_cf=False)
                if ds_chk.sizes.get("time", -1) != time_len:
                    return False
                missing = [v for v in TARGET_VARS if v not in ds_chk]
                return not missing
            except Exception:
                return False

        if out_path.exists() and not overwrite:
            if _is_valid(out_path):
                print(f"[skip] {out_path} 已存在且校验通过，跳过。")
                continue
            else:
                try:
                    out_path.unlink()
                    print(f"[warn] {out_path} 存在但校验失败，删除后重写。")
                except OSError:
                    print(f"[warn] 无法删除损坏文件 {out_path}，尝试覆盖重写。")

        ds_out = build_month_dataset(
            part1_ds, accum_ds, instant_ds, month, start_idx=start, chunk_size=chunk)

        for name in sorted(ds_out.data_vars):
            print(f"{name:28s}: dims={ds_out[name].dims}, shape={ds_out[name].shape}")

        encoding = {}
        engine = "netcdf4" if _NETCDF4_AVAILABLE else "scipy"
        ds_out.to_netcdf(
            out_path,
            engine=engine,
            format="NETCDF3_64BIT_OFFSET",
            encoding=encoding,
        )
        print(f"[ok] 已保存 {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge 2023 ERA5 monthly files to 2022-format.")
    parser.add_argument("--max-time-steps", type=int, default=None,
                        help="分片大小（时间步），例如 12 表示每个输出文件 12 步；缺省为整月。")
    parser.add_argument("--processes", type=int, default=6,
                        help="并行进程数，默认 6。")
    parser.add_argument("--overwrite", action="store_true",
                        help="若目标文件已存在则覆盖，默认跳过以支持断点续写。")
    args = parser.parse_args()

    root = Path(".")
    out_dir = root / "myout"
    out_dir.mkdir(parents=True, exist_ok=True)
    accum_path = root / "data_stream-oper_stepType-accum.nc"
    instant_path = root / "data_stream-oper_stepType-instant.nc"

    if not accum_path.exists() or not instant_path.exists():
        raise FileNotFoundError("缺少累积量或瞬时量年度文件，请确认当前目录包含两个年度 nc 文件。")

    # 调试：打印代表性文件信息
    rep_part1 = root / "2023_1.nc"
    if not rep_part1.exists():
        raise FileNotFoundError("缺少代表文件 2023_1.nc")

    debug_dataset(_open_nc(rep_part1), "2023_1.nc")
    debug_dataset(_open_nc(accum_path), "accum.nc")
    debug_dataset(_open_nc(instant_path), "instant.nc")

    params_list = [
        (m, str(root), str(out_dir), str(accum_path), str(instant_path),
         args.max_time_steps, args.overwrite)
        for m in range(1, 13)
    ]
    with mp.Pool(processes=args.processes) as pool:
        pool.map(process_month_worker, params_list)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        sys.exit(1)
