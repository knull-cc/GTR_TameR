# [ICLR 2026] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval

![Nissan skyline](assets/skyline.png)

> "The GT-R is not a supercar for a select few; it is a supercar for everyone, built to be enjoyed anywhere, anytime, by anyone." --**Nissan skyline**

[![arXiv](https://img.shields.io/badge/arXiv-2602.10847-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2602.10847)

This repository contains the official PyTorch implementation of the **(GTR)** — a lightweight, plug-and-play module designed to empower any multivariate time series forecasting (MTSF) model with the ability to capture global periodic patterns far beyond the fixed look-back window.

---

## 📣 News
* `2026/2/4` 🚀 Code has be released.
* `2026/1/26` 💥💥 GTR is honored to be accepted by ICLR 2026!

---

## 🌟 Key Innovation: The Global Temporal Retriever (GTR)

Existing MTSF models are fundamentally limited by their reliance on a fixed-length historical window, making them unable to capture crucial global periodic patterns (e.g., weekly, monthly, seasonal trends) that span cycles much longer than the input.

**GTR solves this by**:
1.  **Maintaining a Learnable Global Representation**: A parameter matrix `Q ∈ R^(L×N)` encodes the entire global cycle pattern for all `N` variables.
2.  **Dynamic Retrieval & Alignment**: For any input sequence, GTR identifies its position within the global cycle and retrieves the corresponding segment.
3.  **Joint Local-Global Modeling**: The retrieved global segment is stacked with the local input and processed by a 2D convolution to model dependencies across both scales.
4.  **Seamless Integration**: The enriched representation is fused back via a residual connection, making GTR compatible with *any* existing forecasting backbone (MLP, Transformer, Mamba, etc.) without architectural changes.

---

## 📈 Performance Highlights

*   **State-of-the-Art Results**: GTR+MLP achieves SOTA performance on 6 real-world datasets for both short-term and long-term forecasting.
*   **Significant Gains**: On the challenging Solar-Energy dataset, GTR outperforms the second-best model by **8.2% in MSE** and **6.5% in MAE**.
*   **Plug-and-Play Enhancement**: GTR consistently improves diverse SOTA models (iTransformer, PatchTST, DLinear) by up to **91.9% MSE reduction** (DLinear on PEMS04).
*   **Extreme Efficiency**: The GTR module itself adds only **40.1K parameters** and **4.50M MACs**. The full GTR+MLP model uses just **0.98M parameters**, which is only 19% of iTransformer's size.

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch 1.10+
*   Other dependencies (see `requirements.txt`)

### Installation

```bash
conda create -n GTR python=3.8
conda activate GTR
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Run
You can use the following script to obtain the prediction results (Recommended). For example, to reproduce all the experiment in the paper, you can run the following script:

```bash
bash run_main.sh
```

To reproduce all the ablation experiment in the paper, run the following script:

```bash
bash run_ablation.sh
```

### TameR-style recent-point robustness experiment

The test runner can evaluate a clean input and TameR's recent single anomalous
point from the same trained checkpoint. The perturbation changes only the last
observation of every test window:

```text
x[:, -1, :] += N(0, 1) * std(x, axis=time, ddof=0) * perturb_ratio
```

Run the representative ETTh1 experiment (input length 96; prediction lengths
96, 192, 336, and 720; perturbation ratio 3; seed 2024) with:

```bash
bash scripts/Perturb/etth1_recent_single_anomaly.sh 0
```

If the data is outside `./dataset`, set `GTR_DATA_ROOT` to the directory that
contains `ETTh1.csv`. Per-horizon JSON files are written below `./results`, and
the four-horizon CSV/JSON summary is written to `./results/perturbation`.

### ETTh1 perturbation sensitivity sweep

To measure how GTR's sensitivity changes with the perturbed point's position,
run:

```bash
bash run_sensitivity_sweep.sh 0
```

The script uses the original `GTR` model and only ETTh1. It trains (or reuses)
one checkpoint for each prediction length, then scans positions
`-96, -72, -50, -36, -24, -16, -10, -8, -6, -5, -4, -3, -2, -1` with the
same perturbation seed and noise ratio. Set `GTR_FORCE_RETRAIN=1` to ignore
existing compatible checkpoints. This fixed grid is intentional: it makes
runs directly reproducible while sampling the final 10 observations more
densely. By default, data is read from `./dataset/ETT-small/ETTh1.csv`; set
`GTR_DATA_DIR` to the directory containing `ETT-small/`, or set
`GTR_DATA_ROOT` directly to the directory containing `ETTh1.csv`.
CSV, JSON, SVG, PNG, and PDF outputs are
written to `./results/perturbation_sweep/` (PNG/PDF require matplotlib, which
is included in `requirements.txt`).

### Exchange boundary-fix probe

To compare the original frozen GTR predictions against a deterministic test-time
view that replaces only the latest observation with the preceding observation,
run:

```bash
bash run_exchange_boundary_fix.sh 0
```

The script reuses the existing Exchange checkpoints for prediction lengths 96,
192, 336, and 720; it does not retrain GTR. Each test batch is evaluated in both
forms so the original and fixed metrics use exactly the same samples. The
four-horizon CSV/JSON summary is written to `./results/boundary_fix/`.

### Experimental GTR + NTE plugin

`GTRNTE` wraps the complete GTR model with a parameter-free Noise-aware Trend
Extrapolation (NTE) layer. NTE removes an FFT low-frequency trend before GTR,
lets GTR forecast the residual, and then restores an inverse-SNR-damped
kinematic trend. Before the FFT, a prefix-fitted quadratic component is removed
and then added back to avoid the FFT's periodic boundary assumption corrupting a
non-periodic trend. The fit uses only the history prefix, and a prefix-only
robust innovation test replaces an anomalous most-recent point before either
the FFT or GTR sees it. Velocity is the current boundary derivative of the
first/middle/last quadratic fit, rather than the historical average velocity.
It does not change GTR's number of trainable parameters.

Run the same recent-point robustness protocol with NTE on one dataset:

```bash
bash run_nte_perturb.sh 0 ETTh1
```

Omit the dataset name to run all eight Table 1 datasets, or pass any subset of
`ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange Traffic Solar`. The NTE defaults can
be overridden in a direct `run.py` command with `--nte_cutoff_ratio`,
`--nte_alpha`, `--nte_gamma_max`, and `--nte_guard_sigma`. Results use the
distinct model name `GTRNTE`, so they do not overwrite the GTR baseline
summaries.
For batch runs, override the defaults with the environment variables
`NTE_CUTOFF_RATIO`, `NTE_ALPHA`, `NTE_GAMMA_MAX`, and `NTE_GUARD_SIGMA`; these
values are included in checkpoint, result, and summary identities so parameter
sweeps can coexist.

## 📜 Citation
If you find GTR useful, please consider citing our paper:
```bibtex
@inproceedings{
cao2026gtr,
title={Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval},
author={Fanpu Cao and Lu Dai and Jindong Han and Hui Xiong},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=QUJBPSfyui}
}
```
