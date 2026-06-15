# Anatomy-Informed TMJ Prediction

Code for the project:

**Establishment of Machine Learning Prediction Model for Temporomandibular Joint
Symptoms Based on CBCT and Cephalometric Measurement**

This repository contains a clean anatomy-informed temporomandibular joint (TMJ)
symptom prediction workflow.

The repository keeps only the training pipeline code:

1. Load and preprocess the tabular measurement workbook.
2. Map patient-level measurements to a fixed TMJ anatomy graph.
3. Train a PyTorch Geometric graph model with stratified K-fold validation.
4. Optionally sweep random seeds and summarize the best run by mean ROC-AUC.

Raw clinical data and generated experiment outputs are intentionally excluded.

## Layout

```text
.
├── DL/
│   └── data.py              # workbook loading, cleaning, feature engineering
├── GNN/
│   ├── model_pyg.py         # PyG GCN/GAT/GIN/SAGE model definitions
│   └── train_gnn.py         # K-fold GIN training entrypoint
├── environment.yml          # conda environment
├── requirements.txt         # pip dependency reference
├── run_seed_144.sh          # example GIN run command
└── sweep_seed.sh            # seed sweep helper
```

## Data

Place the approved source workbook at:

```text
data.xlsx
```

in the repository root. The file is ignored by Git.

The training code reads the sheet named `总` if present, otherwise the first
sheet. It expects the binary target column `关节紊乱`.

## Environment

```bash
conda env create -f environment.yml
conda activate anatomy-informed-tmj-prediction
```

If PyTorch Geometric needs a CUDA-specific install, install PyTorch and PyG using
versions that match the target machine, then use `requirements.txt` for the
remaining packages.

## Run

Single GIN run:

```bash
python GNN/train_gnn.py --model gin --seed 144 --folds 10
```

Example wrapper:

```bash
./run_seed_144.sh
```

Seed sweep:

```bash
./sweep_seed.sh 1 150 gin
```

The sweep creates `runs_seed/` locally. Generated logs and result files are
ignored by Git.

## Notes

- `GNN/train_gnn.py` defaults to `--model gin`.
- `--model gcn|gat|sage` remains available for ablation.
- No raw data, model checkpoints, notebooks, figures, or run logs are tracked.
