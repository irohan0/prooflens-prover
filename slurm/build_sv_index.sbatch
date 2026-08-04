#!/bin/bash --login
# Build the SINGLE-VECTOR premise index — the matched control for the late-interaction arm.
#
#     sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/build_sv_index.sbatch
#     PROOFLENS_DEVICE=cpu sbatch -p multicore -n 16 slurm/build_sv_index.sbatch   # slow fallback
#
# Neither `-p` nor `-G` is declared, for the reasons documented in build_li_index.sbatch.
#
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 0-02:00:00
#SBATCH --job-name=pl_sv_index
#SBATCH --output=slurm-%j.out
#
# ## Which checkpoint, and why this one
#
# `sv_ft_novel_lr3e6`, paired with the LI arm's `li_ft_novel_bm25`. This pairing is not a choice
# made here — the predecessor study established it. Its `configs/dense_sv_ft_novel_lr3e6.yaml`
# describes itself as "the matched single-vector control ... the airtight baseline the FT-LI gap is
# compared against". Both checkpoints were trained on the same `novel_premises/train` split under
# the same protocol; only the architecture differs. Substituting any other SV checkpoint would make
# the comparison a training-configuration difference wearing an architecture label.
#
# ## The corpus assertion is the whole point
#
# `--assert-corpus-id` is MANDATORY here, unlike the first LI build where there was nothing to
# compare against yet. If SV and LI rank different candidate sets, every difference between the
# arms is uninterpretable. The id below is what both the BM25 and LI indices were built from.

set -euo pipefail
module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
LEAN_PROJECT="${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/mathlib_v4160}"
CKPT="${PROOFLENS_SV_CKPT:-$HOME/scratch/prooflens/checkpoints/sv_ft_novel_lr3e6}"
CORPUS="$REPO/data/premises/mathlib_v4160.jsonl"
OUT="$REPO/data/index/sv_ft_novel_lr3e6"
DEVICE="${PROOFLENS_DEVICE:-cuda}"
# The candidate set every arm must share. Printed by the BM25 build and matched by the LI index.
CORPUS_ID="${PROOFLENS_CORPUS_ID:-276070:31db61c63a9b7ee1}"

export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no venv at $VENV" >&2; exit 1; }
[ -d "$CKPT" ] || {
  echo "ERROR: no SV checkpoint at $CKPT" >&2
  echo "Available checkpoints:" >&2
  ls -1 "$(dirname "$CKPT")" 2>/dev/null >&2 || echo "  (checkpoint directory not found)" >&2
  exit 1
}

echo "=== environment ==="
echo "host       : $(hostname)"
echo "partition  : ${SLURM_JOB_PARTITION:-?}"
echo "gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "checkpoint : $CKPT"
echo "git sha    : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "corpus id  : asserting $CORPUS_ID"

PROOFLENS_DEVICE="$DEVICE" "$PY" - <<'PYCHK' || {
import os
import sys

import sentence_transformers
import torch

want = os.environ.get("PROOFLENS_DEVICE", "cuda")
print("sentence-transformers", sentence_transformers.__version__, "| torch", torch.__version__,
      "| requested device", want, "| cuda available", torch.cuda.is_available())
if want.startswith("cuda") and not torch.cuda.is_available():
    print("ERROR: PROOFLENS_DEVICE=cuda but no CUDA device is visible.", file=sys.stderr)
    sys.exit(1)
PYCHK
  echo "ERROR: sentence-transformers/torch missing, or cuda requested without a GPU." >&2
  echo "  PATH=\$HOME/.local/bin:\$PATH VIRTUAL_ENV=$VENV uv pip install sentence-transformers 'torch>=2.4'" >&2
  exit 1
}

if [ ! -s "$CORPUS" ]; then
  echo "=== premise corpus absent — extracting (~10 min, CPU) ==="
  bash scripts/extract_premises.sh "$LEAN_PROJECT" "$CORPUS"
else
  echo "=== premise corpus present: $(wc -l < "$CORPUS") premises ==="
fi

echo
echo "=== encoding premises with the single-vector checkpoint ==="
"$PY" scripts/build_dense_index.py \
  --kind sv \
  --checkpoint "$CKPT" \
  --corpus "$CORPUS" \
  --out "$OUT" \
  --batch-size 64 \
  --device "$DEVICE" \
  --assert-corpus-id "$CORPUS_ID"

echo
echo "SV INDEX BUILT — $OUT"
echo "corpus_id matched $CORPUS_ID, so SV and LI rank the identical candidate set."
