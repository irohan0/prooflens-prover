#!/bin/bash --login
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH --job-name=pl_li_recall
#SBATCH --output=slurm-%j.out
#
# Hypothesis H1: did the LI arm lose to its candidate generator rather than to late interaction?
#
#   sbatch -p multicore slurm/measure_li_recall.sbatch
#
# ## CPU, not GPU, and that is not a compromise
#
# The GPU would only encode 141 short statements — about 20 seconds of work. Everything after that is
# numpy MaxSim over a 5.5 GB fp16 index, which is CPU and memory-bandwidth bound. Requesting a GPU
# would idle an A100 for ~80 minutes and queue behind other GPU work for no benefit.
#
# Override with `PROOFLENS_DEVICE=cuda sbatch -p gpuA -A <account> -G 1 ...` if the CPU queue is
# worse than the GPU queue on the day.
#
# ## Cost, measured rather than guessed
#
# One exact full-corpus MaxSim pass at 384 query tokens over 21.7M premise token vectors was timed at
# **~26 s/query** with chunk=10000 (0.94 s per chunk x 28 chunks). For all 141 FATE-M problems:
#
#   exact ground truth      141 x 26 s          ~62 min
#   four two-stage budgets  ~27% of one pass    ~17 min
#   query encoding          141 statements      ~1 min
#                                               --------
#                                               ~80 min, inside the 4 h limit
#
# Peak memory ~13 GB: the fp16 index (5.5 GB), the chunk score matrix (384 x 790k fp32 = 1.2 GB) and
# the 50,000-candidate rerank (~6 GB). 64G leaves room.
#
# ## Robustness
#
# Per-query rows are appended to results/tables/li_recall_<benchmark>.jsonl and flushed as they
# complete, so a walltime kill still leaves a usable partial answer. Two jobs on this project were
# previously lost to all-or-nothing designs.

set -euo pipefail
module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
DATA_ROOT="${PROOFLENS_DATA_ROOT:-$HOME/data/benchmarks/REAL-Prover/Realprover/data}"
INDEX="${INDEX:-data/index/li_ft_novel_bm25}"
BENCHMARK="${BENCHMARK:-fate_m}"
N_QUERIES="${N_QUERIES:-0}"          # 0 = every problem, no sampling error
CHUNK="${CHUNK:-10000}"
DEVICE="${PROOFLENS_DEVICE:-cpu}"

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
mkdir -p results/tables

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no venv at $VENV" >&2; exit 1; }
[ -d "$INDEX" ] || { echo "ERROR: no LI index at $INDEX" >&2; exit 1; }

echo "=== environment ==="
echo "host      : $(hostname)"
echo "cpus      : ${SLURM_CPUS_ON_NODE:-?}"
echo "git sha   : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "index     : $INDEX"
echo "device    : $DEVICE (encoding only; scoring is numpy on CPU)"
echo

OUT="results/tables/li_recall_${BENCHMARK}.json"
# A stale per-query file would be appended to and silently double-count, so clear it first.
rm -f "results/tables/li_recall_${BENCHMARK}.jsonl"

"$PY" scripts/measure_li_recall.py \
  --index "$INDEX" \
  --benchmark "$BENCHMARK" \
  --data-root "$DATA_ROOT" \
  --n-queries "$N_QUERIES" \
  --chunk "$CHUNK" \
  --device "$DEVICE" \
  --json-out "$OUT"

echo
echo "H1 RESULT -> $OUT"
echo "The line marked '<-- what every LI result so far used' is the one that decides whether the"
echo "published LI-vs-SV comparison is an architecture result or a candidate-generation artefact."
