#!/usr/bin/env bash
#
# 用不同 seed 扫描训练，自动挑选“平均 ROC-AUC”最高的结果（主指标）。
#
# 用法：
#   chmod +x sweep_seed.sh
#   ./sweep_seed.sh                 # 默认扫 seed=1..50
#   ./sweep_seed.sh 1 100           # 扫 seed=1..100
#   ./sweep_seed.sh 1 150 gin       # 扫 seed=1..150，使用 PyG-GIN
#   CUDA_VISIBLE_DEVICES=0 ./sweep_seed.sh 1 50
#
# 输出：
#   runs_seed/{model}_seed_{seed}.log   # 每次运行完整日志
#   runs_seed/summary_{model}.csv       # 汇总表：seed,mean_auc,mean_acc,mean_f1,log
#
set -euo pipefail

SEED_START="${1:-1}"
SEED_END="${2:-50}"
MODEL="${3:-gin}"            # gcn|gat|gin|sage（见 GNN/train_gnn.py --help）
export MODEL
GAT_HEADS="${GAT_HEADS:-4}"  # 仅当 MODEL=gat 时使用

OUTDIR="runs_seed"
mkdir -p "${OUTDIR}"

CSV="${OUTDIR}/summary_${MODEL}.csv"
echo "seed,mean_auc,mean_acc,mean_f1,log" > "${CSV}"

# 选择 Python 解释器：
# - 可通过环境变量 PYTHON_BIN 显式指定（推荐：指向你装了 torch 的环境）
# - 否则自动探测 python3 / python 谁能 import torch
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "错误：找不到可用的 Python（需要能 import torch）。" >&2
    echo "请先激活你的 conda 环境，或指定解释器，例如：" >&2
    echo "  PYTHON_BIN=/path/to/python ./sweep_seed.sh ${SEED_START} ${SEED_END}" >&2
    exit 2
  fi
fi

best_seed=""
best_auc="-inf"
best_log=""

for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  log="${OUTDIR}/${MODEL}_seed_${seed}.log"

  echo "========== seed=${seed} =========="
  if [ "${MODEL}" = "gat" ]; then
    "${PYTHON_BIN}" GNN/train_gnn.py --seed "${seed}" --model "${MODEL}" --gat-heads "${GAT_HEADS}" > "${log}" 2>&1
  else
    "${PYTHON_BIN}" GNN/train_gnn.py --seed "${seed}" --model "${MODEL}" > "${log}" 2>&1
  fi

  mean_auc="$(grep -m1 '平均 ROC-AUC:' "${log}" | awk '{print $3}' || true)"
  mean_acc="$(grep -m1 '平均 ACC:' "${log}" | awk '{print $3}' || true)"
  mean_f1="$(grep -m1 '平均 F1:' "${log}" | awk '{print $3}' || true)"

  echo "${seed},${mean_auc},${mean_acc},${mean_f1},${log}" >> "${CSV}"
  echo "seed=${seed} mean_auc=${mean_auc} mean_acc=${mean_acc} mean_f1=${mean_f1} log=${log}"

  better="$(
    "${PYTHON_BIN}" - "${mean_auc:-nan}" "${best_auc}" <<'PY'
import sys, math
def tofloat(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("-inf")
cur = tofloat(sys.argv[1])
best = tofloat(sys.argv[2])
print(1 if (not math.isnan(cur) and cur > best) else 0)
PY
  )"

  if [ "${better}" = "1" ]; then
    best_auc="${mean_auc}"
    best_seed="${seed}"
    best_log="${log}"
  fi
done

echo
echo "BEST（按 mean ROC-AUC）：seed=${best_seed} mean_auc=${best_auc}"
echo "best log: ${best_log}"
echo "summary csv: ${CSV}"
echo
echo "Top 10 by mean ROC-AUC:"
"${PYTHON_BIN}" - <<'PY'
import csv

import os
model = os.environ.get("MODEL", "gcn")
path = f"runs_seed/summary_{model}.csv"
rows = []
with open(path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            row["mean_auc_f"] = float(row.get("mean_auc") or "-inf")
        except Exception:
            row["mean_auc_f"] = float("-inf")
        rows.append(row)

rows.sort(key=lambda x: x["mean_auc_f"], reverse=True)
for i, row in enumerate(rows[:10], 1):
    print(
        f"{i:02d}. seed={row['seed']} "
        f"mean_auc={row['mean_auc']} mean_acc={row.get('mean_acc','')} mean_f1={row.get('mean_f1','')} "
        f"log={row['log']}"
    )
PY

