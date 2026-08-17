#!/bin/bash --login
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 0-08:00:00
#SBATCH --job-name=pl_prove
# stderr MERGED into stdout. A split stream hid an encoding traceback for a full debugging cycle.
#SBATCH --output=slurm-%j.out
#
# Run one retrieval arm over one benchmark. Partition and account are NOT declared here — pass them
# at submit time, because they differ per account:
#
#   sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/prove_benchmark.sbatch
#
# Override anything via the environment:
#
#   BENCHMARK=fate_m ARM=li sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/prove_benchmark.sbatch
#
# This exists so a benchmark run is not a 400-character `--wrap` string. Every such string that has
# been typed by hand on this project has had at least one error in it (a wrong --data-root, a
# missing -n, a partition that does not exist), and each cost a submit/wait/diagnose cycle.

set -euo pipefail

module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
LEAN_PROJECT="${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/mathlib_v4160}"
DATA_ROOT="${PROOFLENS_DATA_ROOT:-$HOME/data/benchmarks/REAL-Prover/Realprover/data}"

BENCHMARK="${BENCHMARK:-fate_m}"
ARM="${ARM:-li}"
INDEX="${INDEX:-data/index/li_ft_novel_bm25}"
SAMPLES="${SAMPLES:-32}"
MIN_CLOSERS="${MIN_CLOSERS:-19}"
DEVICE="${DEVICE:-cuda}"
# li only: premises kept by the pooled first stage before exact MaxSim rerank. Empty = the index's
# stored 1000, which measured recall@10 = 0.443 against exact MaxSim on real queries. 50000 measured
# 0.979. See results/tables/li_recall_fate_m.json.
N_CANDIDATES="${N_CANDIDATES:-}"
EXTRA="${EXTRA:-}"          # e.g. EXTRA="--limit 5" for a smoke run, or "--resume results/logs/<id>"

export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$REPO"
mkdir -p results/logs
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no venv at $VENV" >&2; exit 1; }

echo "=== environment ==="
echo "host      : $(hostname)"
echo "git sha   : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "cpus      : ${SLURM_CPUS_ON_NODE:-?}"
echo "benchmark : $BENCHMARK | arm: $ARM | index: $INDEX"
echo "n_cand    : ${N_CANDIDATES:-(index default)}"
echo

# ---------------------------------------------------------------------------------------------
# Stage Mathlib to node-local disk. `import Mathlib` reads ~5.4 GB of .olean files, and from NFS
# that has been measured at 439 s (4 cores), 691 s (1 core), and >900 s on a contended node — where
# it exceeded the header timeout and killed job 18165549 outright. Every worker pays it once.
#
# Staging was described here as an OPTIMISATION that "must never be able to break a run", and the
# fallback message claimed NFS was "slower warm-up, still correct". Both were wrong, and the LLM arm
# paid for it: on ProofNet, the same arm at the same seed proved 26 under NFS and 28 staged, because
# NFS provoked 2 extra REPL restarts and their transient `Unknown proof state` failures. The penalty
# is also uneven across arms, so a mixed run set can understate one of them.
#
# A failure to stage therefore ABORTS. Set ALLOW_NFS_FALLBACK=1 to accept a run on NFS, and record
# that it is not comparable to a staged one.
# ---------------------------------------------------------------------------------------------
if [ "${STAGE_LEAN:-1}" = "1" ]; then
    if STAGED=$(bash scripts/stage_lean_project.sh "$LEAN_PROJECT"); then
        echo "using node-local Lean project: $STAGED"
        LEAN_PROJECT="$STAGED"
    elif [ "${ALLOW_NFS_FALLBACK:-0}" = "1" ]; then
        echo "WARNING: staging failed; running against NFS because ALLOW_NFS_FALLBACK=1."
        echo "         Not comparable to a staged run — expect extra REPL restarts."
    else
        echo "ERROR: could not stage Lean node-local, and NFS changes the result." >&2
        echo "       Measured: 26 vs 28 proved on ProofNet for the same arm and seed." >&2
        echo "       Re-queue, or set ALLOW_NFS_FALLBACK=1 to accept a non-comparable run." >&2
        exit 1
    fi
fi
echo

# shellcheck disable=SC2086  # EXTRA is deliberately word-split into separate arguments
"$PY" scripts/prove_benchmark.py \
    --benchmark "$BENCHMARK" \
    --arm "$ARM" \
    --index "$INDEX" \
    --data-root "$DATA_ROOT" \
    --lean-project "$LEAN_PROJECT" \
    --samples-per-step "$SAMPLES" \
    --min-closers "$MIN_CLOSERS" \
    --device "$DEVICE" \
    ${N_CANDIDATES:+--n-candidates "$N_CANDIDATES"} \
    $EXTRA

echo
echo "Run 'seff ${SLURM_JOB_ID:-<jobid>}' for resource accounting."
