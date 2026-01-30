"""Optional helper to fetch datasets from OBS (Huawei Cloud).

This script is environment-specific (requires `moxing`). It is not needed for
core training/evaluation if you already have local NetCDF files.
"""

import moxing as mox

# 1. 设置 OBS 源路径与 Notebook 本地路径
obs_path = "obs://graphcast-ct/data/source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc"
local_path = "/home/ma-user/data/dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc"

# 2. 从 OBS 拷贝文件到本地 Notebook
mox.file.copy(obs_path, local_path)

# 3. 导入数据
import pandas as pd
df = pd.read_csv(local_path)
print(df.head())
