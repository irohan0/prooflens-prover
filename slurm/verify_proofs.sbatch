#!/bin/bash --login
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
#SBATCH --job-name=pl_verify
#SBATCH --output=slurm-%j.out
#
# Independently re-check the proofs claimed by one or more runs.
#
#   RUNS="run_id_a run_id_b" sbatch -p multicore slurm/verify_proofs.sbatch
#   sbatch -p multicore slurm/verify_proofs.sbatch          # the 3 most recent runs
#
# Verifying several runs in ONE job is the point: each `verify_proofs.py` invocation builds its own
# Lean backend, and `import Mathlib` costs ~535 s from NFS. Three separate jobs would pay it three
# times. Here the project is staged to node-local disk once and every verify reuses it.
#
# Exit status is non-zero if ANY run has a failing proof, so a bad result cannot be skimmed past.

set -uo pipefail          # NOT -e: one failing run must not skip the others

module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
LEAN_PROJECT="${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/mathlib_v4160}"
DATA_ROOT="${PROOFLENS_DATA_ROOT:-$HOME/data/benchmarks/REAL-Prover/Realprover/data}"

export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$REPO" || exit 1
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no venv at $VENV" >&2; exit 1; }

# Default to the three most recent runs, newest last.
if [ -z "${RUNS:-}" ]; then
    RUNS=$(ls -1t results/logs | head -3 | tac | tr '\n' ' ')
fi
echo "verifying: $RUNS"
echo

if [ "${STAGE_LEAN:-1}" = "1" ]; then
    if STAGED=$(bash scripts/stage_lean_project.sh "$LEAN_PROJECT"); then
        echo "using node-local Lean project: $STAGED"
        LEAN_PROJECT="$STAGED"
    else
        echo "staging unavailable — using $LEAN_PROJECT over NFS"
    fi
fi

rc=0
for R in $RUNS; do
    echo
    echo "############################################################"
    echo "### $R"
    echo "############################################################"
    if [ ! -d "results/logs/$R" ]; then
        echo "MISSING: results/logs/$R"
        rc=1
        continue
    fi
    "$PY" scripts/verify_proofs.py \
        --run "results/logs/$R" \
        --data-root "$DATA_ROOT" \
        --lean-project "$LEAN_PROJECT" || rc=1
done

echo
if [ "$rc" -eq 0 ]; then
    echo "ALL RUNS VERIFIED CLEAN"
else
    echo "!! AT LEAST ONE RUN HAS A FAILING PROOF — its pass rate is invalid"
fi
exit "$rc"
