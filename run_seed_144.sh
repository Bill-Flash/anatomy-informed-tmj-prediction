#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python GNN/train_gnn.py \
  --model gin \
  --seed 144 \
  --folds 10 \
  --epochs 80 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 1e-3 \
  --patience 12 \
  --d-hidden 64 \
  --n-layers 3 \
  --dropout 0.2
