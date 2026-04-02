# Tiny Autoencoder with Fuzzy Thresholding for Air-Quality Anomaly Detection

A leakage-controlled, reproducible anomaly-detection pipeline for air-quality time series.

## Data Sources (Used in This Project)

- Beijing Multi-Site Air Quality Data (primary):
  - [UCI dataset page](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- Air Quality Data Set (secondary/reference):
  - [UCI dataset page](https://archive.ics.uci.edu/dataset/360/air+quality)

Local storage paths (ignored from git):

- `data/beijing+multi+site+air+quality+data/`
- `data/air+quality/`

## Clean Project Structure

- `src/data_pipeline/`
  - `data_audit.py`
  - `preprocess.py`
  - `build_windows.py`
- `src/modeling/`
  - `model_autoencoder.py`
  - `train.py`
  - `baselines.py`
- `src/postprocess/`
  - `compute_errors.py`
  - `fuzzy_threshold.py`
  - `evaluate.py`
- `scr/`
  - `CLAIM_TREE.md`
  - `EVIDENCE_MAP.md`
  - `READING.md`

## Environment Setup

```powershell
conda create -n py312 python=3.12 -y
conda activate py312
pip install -r requirements.txt
```

## Reproducible Run Order

Run from repository root:

```powershell
python -m src.data_pipeline.preprocess --strict
python -m src.data_pipeline.build_windows --strict
python -m src.modeling.train --strict --epochs 20 --batch-size 256 --learning-rate 0.001 --seed 42
python -m src.postprocess.compute_errors --strict
python -m src.modeling.baselines --strict
python -m src.postprocess.evaluate --strict
```

## Key Outputs

- Training:
  - `data/training/tiny_ae_best.pt`
  - `data/training/tiny_ae_history.csv`
  - `data/training/tiny_ae_summary.json`
- Error extraction:
  - `errors/error_percentiles.csv`
  - `errors/error_summary.json`
- Evaluation:
  - `data/evaluation/evaluation_predictions.csv`
  - `data/evaluation/evaluation_metrics.csv`
  - `data/evaluation/evaluation_metrics.json`
  - `data/evaluation/evaluation_protocol.json`

Important quality condition:

- Reported metrics must be reproducible from `data/evaluation/evaluation_predictions.csv`.

## Git Cleanliness for Push

This repository ignores data, model artifacts, generated arrays, and figures via `.gitignore`.

Ignored working logs (as requested):

- `project_scope.md`
- `dataset_verification.md`
- `progress.md`

Only source code and intentional documentation should be committed.

## Current Claim Boundary

The pipeline reproducibility and anomaly-signal extraction are validated.
Any superiority claim for fuzzy thresholding is currently protocol-dependent and must remain conservative until stronger label protocols are added.
# air-quality-anomaly-benchmark
