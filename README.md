# **Global2Regional-NestedGNN**

**Language:** English | [`中文`](README.zh-CN.md)

This repository provides a **reproducible code release** for **regional (nested) model** with:

- **CSAF**: Cross-Scale Attention Fusion (`BoundaryCrossAttention`)
- **BCL**: Boundary-Consistent Loss for buffer-zone constraints

All runnable scripts for the release are under `Upgithub/code/`.

## Overview

High-level ideas:

- **Nested regional modeling**: forecasts are produced on a regional window consisting of a **core region** plus a surrounding **buffer zone**.
- **Boundary blending**: at each rollout step, the **buffer zone** in the latest input frame is replaced with a boundary patch from a global model.
- **CSAF**: a lightweight cross-attention module fuses regional forecast features with boundary features.
- **BCL**: a masked consistency term constrains the fused output to match boundary values in the buffer zone.

## Repository Layout (release subset)

- `code/graphcast/`: GraphCast core implementation (Python).
- `code/nested_utils.py`: region slicing, buffer mask, boundary blending utilities.

Preprocessing:
- `code/preprocess_convert_era5_format.py`: convert raw ERA5 exports to the project NetCDF schema.
- `code/utils_fetch_dataset_obs.py`: optional OBS download helper (environment-specific).

Boundary precomputation (optional, speeds up training/eval):
- `code/precompute_global_boundaries.py`
- `code/precompute_global_boundaries_batch.py`

Training:
- `code/train_regional_csaf_bcl_full_year.py`: **main training script (CSAF + BCL)**.
- `code/train_regional_csaf_bcl.py`: single-dataset variant (CSAF + BCL).

Evaluation:
- `code/eval_nested_csaf.py`: **nested rolling evaluation (CSAF)**, metrics + maps.
- `code/eval_nested_baseline.py`: baseline evaluation (no CSAF).
- `code/eval_rollout.py`: global rollout evaluation (optional).
- `code/eval_single_step_maps.py`: single-step inference + maps (optional).

## Requirements

Recommended: Python 3.10+.

Core packages include: `numpy`, `pandas`, `xarray`, `netCDF4`/`h5netcdf`, `torch`, `matplotlib`, `cartopy`, `scipy`, `trimesh`.

See `Upgithub/requirements.txt` (minimal). For camera-ready release, please pin exact versions based on your runtime environment.

## Quick Start

Run from `/code/` so `graphcast` is importable:

```bash
cd /code
```

1) (Optional) Preprocess / convert ERA5 exports:

```bash
python preprocess_convert_era5_format.py --help
```

2) (Optional) Precompute boundary patches (recommended for speed):

```bash
python precompute_global_boundaries.py --help
python precompute_global_boundaries_batch.py --help
```

3) Train the regional model (CSAF + BCL):

```bash
python train_regional_csaf_bcl_full_year.py --help
```

4) Evaluate with nested rolling rollout (CSAF):

```bash
python eval_nested_csaf.py --help
```

## Outputs

Typical evaluation outputs include:

- `metrics.csv` (RMSE/Bias/ACC by variable and lead time)
- heatmaps (`heatmap_rmse.png`, `heatmap_bias.png`, `heatmap_acc.png`)
- surface and wind maps at selected lead times
- `summary.json` consolidating artefact paths

## References (GraphCast / ERA5)

Please cite the following works if you use GraphCast and ERA5 in academic work:

- **GraphCast**: Lam, R. *et al.* “Learning skillful medium-range global weather forecasting.” *Science* (2023).
- **ERA5**: Hersbach, H. *et al.* “The ERA5 global reanalysis.” *QJRMS* (2020).

BibTeX:

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

## Citation (this repository)

```text
TODO: Add citation for this code/paper (authors, title, venue, year).
```

