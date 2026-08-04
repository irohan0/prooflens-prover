#!/bin/bash --login
#SBATCH -p multicore            # CPU only: the Lean gate needs no GPU
#SBATCH -n 4
#SBATCH --mem=16G               # caps RSS properly (see note below). Budget ~5GB per concurrent
                                # Lean worker: this gate uses one, with headroom.
#SBATCH -t 0-01:00:00
#SBATCH --job-name=pl_lean_smoke
# stderr MERGED into stdout — see the note in build_li_index.sbatch. A split stream hid a traceback
# for a full debugging cycle.
#SBATCH --output=slurm-%j.out
#
# Tier-0 cluster gate: prove the SAME smoke test that passes locally also passes on CSF3, before
# any real compute is spent. The predecessor project's prover strand died on exactly this class of
# environment drift, so this runs early and unchanged rather than being discovered late.
#
#   sbatch slurm/lean_smoke.sbatch
#
# No `dos2unix` needed: .gitattributes forces LF on *.sbatch/*.sh, so a clone or a git-produced
# archive is already correct.
#
# MEMORY: cap real memory with `#SBATCH --mem` (a cgroup RSS limit), NOT with the harness's
# --memory-limit-mb, which caps VIRTUAL address space and makes Lean fail to start below ~32GB.
# See src/prooflens_prover/lean/leaninteract_backend.py for the measurement.

set -euo pipefail

# `module purge` only. We deliberately do NOT load a site python module: the venv was created by
# uv with its own standalone CPython 3.11, so the interpreter is byte-identical to the laptop's
# regardless of what the cluster's module tree offers. Loading a site python here would silently
# shadow it and reintroduce exactly the environment drift this gate exists to detect.
# `|| true` because some module implementations return non-zero from a no-op purge, which would
# abort the whole job under `set -e`.
module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
# Default matches scripts/setup_lean_project.sh. v4.16.0, NOT v4.31.0: the benchmarks only fully
# elaborate on v4.16.0 (141/141 FATE-M, 244/244 miniF2F, against 138 and 212 on v4.31.0), and a
# statement that fails to elaborate is silently scored as unproved. See LEARNINGS.
LEAN_PROJECT="${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/mathlib_v4160}"

export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # nothing here should touch the network

cd "$REPO"
mkdir -p results/logs          # gitignored, so absent on a fresh clone/extract
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: no venv at $VENV — run scripts/bootstrap_csf3.sh on a login node first" >&2
  exit 1
fi

echo "=== environment ==="
echo "host      : $(hostname)"
echo "repo      : $REPO"
echo "lean proj : $LEAN_PROJECT"
echo "python    : $("$PY" --version 2>&1)"
echo "elan      : $(elan --version 2>&1 || echo MISSING)"
# Run `lean --version` INSIDE the project so elan resolves the pinned `lean-toolchain`. Outside a
# project it resolves "latest" and silently downloads a whole toolchain (~1.3 GB into $HOME, which
# is quota-limited on CSF3) just to print a version string.
echo "lean      : $(cd "$LEAN_PROJECT" 2>/dev/null && lean --version 2>&1 || echo MISSING)"
echo "git sha   : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

echo "=== hermetic tests (no Lean, no GPU) ==="
"$PY" -m pytest tests/ -q -m "not lean"

echo
echo "=== Lean smoke gate ==="
"$PY" scripts/lean_smoke.py \
  --project-dir "$LEAN_PROJECT" \
  --json-out "results/logs/lean_smoke_csf3_${SLURM_JOB_ID:-local}.json"

echo
echo "=== live Lean regression tests ==="
PROOFLENS_LEAN_PROJECT="$LEAN_PROJECT" "$PY" -m pytest tests/ -q -m lean

echo
echo "ALL CSF3 TIER-0 CHECKS PASSED"
echo "Compare results/logs/lean_smoke_csf3_*.json against the local run to catch env drift."
echo "Run 'seff ${SLURM_JOB_ID:-<jobid>}' for the real resource accounting."
