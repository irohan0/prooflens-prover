#!/bin/bash --login
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH --job-name=pl_li_recall
#SBATCH --output=slurm-%j.out
#
#   sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/measure_li_recall.sbatch
#
# Measures what LI's two-stage candidate generation costs on REAL queries, and how much of it a
# larger first-stage budget recovers. See scripts/measure_li_recall.py for why the 0.992 printed at
# index-build time does not answer this.
#
# 64G: exact MaxSim holds a [n_query_tokens x chunk_tokens] score matrix — at 384 query tokens and
# 5000 premises (~394k tokens) that is ~600 MB per chunk, plus the 5.5 GB fp16 index.

set -euo pipefail
module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
DATA_ROOT="${PROOFLENS_DATA_ROOT:-$HOME/data/benchmarks/REAL-Prover/Realprover/data}"
INDEX="${INDEX:-data/index/li_ft_novel_bm25}"
BENCHMARK="${BENCHMARK:-fate_m}"
N_QUERIES="${N_QUERIES:-40}"
DEVICE="${PROOFLENS_DEVICE:-cuda}"

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
mkdir -p results/tables

echo "host    : $(hostname)"
echo "git sha : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "index   : $INDEX"
echo

"$VENV/bin/python" scripts/measure_li_recall.py \
  --index "$INDEX" \
  --benchmark "$BENCHMARK" \
  --data-root "$DATA_ROOT" \
  --n-queries "$N_QUERIES" \
  --device "$DEVICE" \
  --json-out "results/tables/li_recall_${BENCHMARK}.json"
