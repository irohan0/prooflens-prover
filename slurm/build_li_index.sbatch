#!/bin/bash --login
# NO `#SBATCH -p` HERE ON PURPOSE. GPU partition names are site- and allocation-specific, and a
# hardcoded guess (`gpuV`) was rejected outright with "Invalid partition name". Pass it at submit
# time instead, so this file cannot be wrong:
#
#     sinfo -o '%P %G' | grep -i gpu        # what you actually have access to
#     sbatch -p <partition> slurm/build_li_index.sbatch
#
# GPU partitions on CSF3 additionally require an account with a GPU allocation
# ("Invalid account or account/partition combination" means yours does not have one yet):
#     sacctmgr -n show assoc user=$USER format=Account%25,Partition%15,QOS%25
#     sbatch -p <partition> -A <account> slurm/build_li_index.sbatch
#
# Neither `-p` NOR `-G` is declared here, so the same file serves both paths:
#
#   GPU:  sbatch -p <gpu-partition> -A <account> -G 1 slurm/build_li_index.sbatch
#   CPU:  PROOFLENS_DEVICE=cpu sbatch -p multicore -n 16 -t 0-08:00:00 slurm/build_li_index.sbatch
#
# (A hardcoded `#SBATCH -G 1` would keep requesting a GPU on a CPU partition and fail there.)
#
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 0-02:00:00
#SBATCH --job-name=pl_li_index
# stderr is MERGED into stdout (one file, no --error). Splitting them meant a job whose stdout
# ended cleanly mid-run looked like it had simply stopped, while the traceback sat unread in a .err
# file nobody thought to open. One file, in causal order, is worth more than tidy separation.
#SBATCH --output=slurm-%j.out
#
# Build the LATE-INTERACTION premise index — the one GPU step standing between the current state
# and being able to run the LI arm on any benchmark.
#
#   sbatch slurm/build_li_index.sbatch
#
# Encoding ~276k short statements through a ModernBERT-sized ColBERT is minutes of GPU. The job is
# sized at 2h because the premise corpus is regenerated first if absent (~10 min of Lean, CPU only).
#
# WHY THE CORPUS IS REGENERATED RATHER THAN COPIED: it is gitignored (260 MB) and deterministic —
# the same extractor against the same pinned Mathlib v4.16.0 produces the same premises in the same
# order. `--assert-corpus-id` then proves it matches what BM25 indexed rather than assuming it.
#
# MEMORY: 48G is for the encoder plus the assembled index (~4.5 GB fp16 tokens, held in RAM before
# writing). It does NOT need a Lean REPL, so the ~5 GB/worker rule does not apply here.

set -euo pipefail
module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
LEAN_PROJECT="${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/mathlib_v4160}"
CKPT="${PROOFLENS_LI_CKPT:-$HOME/scratch/prooflens/checkpoints/li_ft_novel_bm25}"
CORPUS="$REPO/data/premises/mathlib_v4160.jsonl"
OUT="$REPO/data/index/li_ft_novel_bm25"
# CPU is a genuine fallback, not a degraded mode: the index it produces is identical, it just takes
# hours instead of minutes. Set PROOFLENS_DEVICE=cpu and submit to a CPU partition when GPU access
# is unavailable or queued — encoding is a one-off, and a slow index beats no index.
DEVICE="${PROOFLENS_DEVICE:-cuda}"

export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # the checkpoint is local; never phone home
cd "$REPO"

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no venv at $VENV" >&2; exit 1; }
[ -d "$CKPT" ] || { echo "ERROR: no LI checkpoint at $CKPT" >&2; exit 1; }

echo "=== environment ==="
echo "host       : $(hostname)"
echo "partition  : ${SLURM_JOB_PARTITION:-?}"
echo "gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "checkpoint : $CKPT"
echo "git sha    : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# pylate is in the `retrieval` extra, not the Tier-0 lock. Fail loudly and early rather than after
# ten minutes of Lean extraction. The venv is uv-created and therefore has NO pip — install with
# `uv pip`, which is also what created it.
PROOFLENS_DEVICE="$DEVICE" "$PY" - <<'PYCHK' || {
import os
import sys

import pylate
import torch

want = os.environ.get("PROOFLENS_DEVICE", "cuda")
print("pylate", pylate.__version__, "| torch", torch.__version__,
      "| requested device", want, "| cuda available", torch.cuda.is_available())
if want.startswith("cuda") and not torch.cuda.is_available():
    print("ERROR: PROOFLENS_DEVICE=cuda but no CUDA device is visible. Either submit to a GPU "
          "partition, or set PROOFLENS_DEVICE=cpu and use a CPU partition.", file=sys.stderr)
    sys.exit(1)
PYCHK
  echo "ERROR: pylate/torch missing, or cuda requested without a GPU." >&2
  echo "Install (uv venvs have no pip):" >&2
  echo "  PATH=\$HOME/.local/bin:\$PATH VIRTUAL_ENV=$VENV uv pip install 'pylate>=1.1' 'torch>=2.4' sentence-transformers" >&2
  exit 1
}

if [ ! -s "$CORPUS" ]; then
  echo "=== premise corpus absent — extracting (~10 min, CPU) ==="
  bash scripts/extract_premises.sh "$LEAN_PROJECT" "$CORPUS"
else
  echo "=== premise corpus present: $(wc -l < "$CORPUS") premises ==="
fi

echo
echo "=== encoding premises with the late-interaction checkpoint ==="
# --assert-corpus-id is intentionally omitted on the FIRST build (there is no BM25 index on the
# cluster to compare against yet). The printed corpus_id must match the local BM25 build report:
#   276070:31db61c63a9b7ee1
"$PY" scripts/build_dense_index.py \
  --kind li \
  --checkpoint "$CKPT" \
  --corpus "$CORPUS" \
  --out "$OUT" \
  --batch-size 64 \
  --device "$DEVICE"

echo
echo "LI INDEX BUILT — $OUT"
echo "Check the printed corpus_id is 276070:31db61c63a9b7ee1 (matches the BM25 candidate set),"
echo "and note the reported two-stage recall@10 — it must be quoted beside every LI result."
