# Global2Regional-NestedGNN

**Language:** [`English`](README.md) | 中文

本仓库包含：

- **CSAF**：跨尺度注意力融合（`BoundaryCrossAttention`）
- **BCL**：缓冲带边界一致性损失（约束缓冲带与外部边界场一致）

## 项目概述

核心思想（概述）：

- **区域嵌套建模**：在“核心区 + 缓冲带”的区域窗口上进行预测。
- **边界融合**：每个滚动步，将输入序列最后一帧的**缓冲带**替换为全局模型提供的边界 patch。
- **CSAF**：轻量 cross-attention 将区域预测特征与边界特征融合。
- **BCL**：仅在缓冲带上施加 masked 一致性约束，使融合输出贴合边界场。

## 代码结构（论文发布子集）

- `code/graphcast/`：GraphCast 核心实现（Python）。
- `code/nested_utils.py`：区域裁剪、缓冲带 mask、边界融合等工具函数。

数据预处理：
- `code/preprocess_convert_era5_format.py`：将 ERA5 原始导出转换/对齐到本文使用的数据格式。
- `code/utils_fetch_dataset_obs.py`：可选 OBS 下载脚本（依赖具体平台环境）。

边界预计算（可选，用于加速训练/评估）：
- `code/precompute_global_boundaries.py`
- `code/precompute_global_boundaries_batch.py`

训练：
- `code/train_regional_csaf_bcl_full_year.py`：**主训练脚本（CSAF + BCL）**。
- `code/train_regional_csaf_bcl.py`：单数据集版本（CSAF + BCL）。

评估：
- `code/eval_nested_csaf.py`：**嵌套滚动评估（CSAF）**，输出指标与场图。
- `code/eval_nested_baseline.py`：baseline 评估（无 CSAF）。
- `code/eva_rollout.py`：全局 GraphCast 滚动评估（可选）。
- `code/eval_single_step_maps.py`：单步推理 + 场图输出（可选）。

## 依赖环境

推荐 Python 3.10+。

核心依赖：`numpy`、`pandas`、`xarray`、`netCDF4/h5netcdf`、`torch`、`matplotlib`、`cartopy`、`scipy`、`trimesh`。

参考 `Upgithub/requirements.txt`（最小依赖集合；建议发布时锁定版本号）。

## 快速开始

建议在 `/code` 下运行：

```bash
cd Upgithub/code
```

1)（可选）ERA5 数据预处理/对齐：

```bash
python preprocess_convert_era5_format.py --help
```

2)（可选）预计算边界（推荐用于加速训练/评估）：

```bash
python precompute_global_boundaries.py --help
python precompute_global_boundaries_batch.py --help
```

3) 训练区域模型（CSAF + BCL）：

```bash
python train_regional_csaf_bcl_full_year.py --help
```

4) 进行嵌套滚动评估（CSAF）：

```bash
python eval_nested_csaf.py --help
```

## 输出内容

评估通常会生成：

- `metrics.csv`（不同预报时效下各变量的 RMSE/Bias/ACC）
- 热力图（RMSE/Bias/ACC）
- 多个时效的 surface 场图与风场图
- `summary.json`（集中记录输出路径）

## 参考文献（GraphCast / ERA5）

如在学术工作中使用 GraphCast 与 ERA5，请引用：

- **GraphCast**：Lam, R. *et al.* “Learning skillful medium-range global weather forecasting.” *Science* (2023).
- **ERA5**：Hersbach, H. *et al.* “The ERA5 global reanalysis.” *QJRMS* (2020).

BibTeX（同英文版）：

```bibtex
@article{lam2023graphcast,
  title={Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and others},
  journal={Science},
  year={2023}
}

@article{hersbach2020era5,
  title={The ERA5 global reanalysis},
  author={Hersbach, Hans and Bell, Bill and Berrisford, Paul and Biavati, Gionata and Hor{\'a}nyi, Andr{\'a}s and Mu{\~n}oz-Sabater, Joaqu{\'\i}n and others},
  journal={Quarterly Journal of the Royal Meteorological Society},
  year={2020}
}
```

## 引用（本仓库/本文）

```text
TODO：补充本文/代码引用信息（作者/题目/期刊或会议/年份）。
```

