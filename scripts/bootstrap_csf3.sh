#!/usr/bin/env bash
# One-shot CSF3 bootstrap: Python env + Lean toolchain + Mathlib, then the Tier-0 gate.
#
# Run this ONCE on a LOGIN node (it needs network for downloads; heavy work is deferred to a
# batch job). It is idempotent — safe to re-run after a partial failure.
#
#   bash scripts/bootstrap_csf3.sh
#
# Deliberately does NOT depend on a `python/3.11` module existing: `uv` fetches its own
# standalone CPython, so the environment is identical on the laptop and the cluster regardless
# of what the site's module tree happens to offer. That is the whole point of pinning.
#
# Nothing here compiles Mathlib. `lake exe cache get` downloads prebuilt .olean files.

set -euo pipefail

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VENV="${PROOFLENS_PROVER_VENV:-$HOME/venvs/plprover}"
LEAN_PROJECT="${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/prooflens_mathlib}"
LEAN_VERSION="${LEAN_VERSION:-v4.31.0}"

echo "==================================================================="
echo " ProofLens-Prover — CSF3 bootstrap"
echo "==================================================================="
echo "repo         : $REPO"
echo "venv         : $VENV"
echo "lean project : $LEAN_PROJECT"
echo "lean version : $LEAN_VERSION"
echo

if [ ! -d "$REPO" ]; then
  echo "ERROR: repo not found at $REPO" >&2
  echo "Transfer it first, from your laptop:" >&2
  echo "  scp 'd:/msc thesis/prooflens-prover.bundle' \\" >&2
  echo "      $USER@csf3.itservices.manchester.ac.uk:~/scratch/" >&2
  echo "then on CSF3:" >&2
  echo "  cd ~/scratch && git clone prooflens-prover.bundle prooflens-prover" >&2
  exit 1
fi

# A real git repo is required, not an extracted archive: every run manifest records the commit SHA
# (see utils/manifest.py), and results produced from a tree with no git history cannot be traced
# back to the code that produced them. Cloning from a bundle preserves full history.
if ! git -C "$REPO" rev-parse --short HEAD >/dev/null 2>&1; then
  echo "WARNING: $REPO is not a git repository." >&2
  echo "  Run manifests will record git_commit='unknown', so results will not be traceable to a" >&2
  echo "  commit. Prefer cloning from the bundle:" >&2
  echo "    cd ~/scratch && git clone prooflens-prover.bundle prooflens-prover" >&2
  echo "Continuing anyway." >&2
fi

# ---------------------------------------------------------------- 1. uv ----
if ! command -v uv >/dev/null 2>&1; then
  echo "--- installing uv ---"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ------------------------------------------------------- 2. python env ----
echo
echo "--- creating venv (uv fetches a standalone CPython 3.11) ---"
uv venv "$VENV" --python 3.11
VIRTUAL_ENV="$VENV" uv pip install -r "$REPO/requirements.tier0.lock.txt"
echo "python: $("$VENV/bin/python" --version)"

# ------------------------------------------------------------- 3. elan ----
echo
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  echo "--- installing elan ---"
  curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"
elan --version

# ------------------------------------------- 4. Lean project + Mathlib ----
echo
echo "--- Lean project + prebuilt Mathlib (several GB download, no compilation) ---"
bash "$REPO/scripts/setup_lean_project.sh" "$LEAN_PROJECT" "$LEAN_VERSION"

# ------------------------------------------------------ 5. Tier-0 gate ----
echo
echo "--- hermetic tests ---"
cd "$REPO"
PYTHONPATH="$REPO/src" "$VENV/bin/python" -m pytest tests/ -q -m "not lean"

echo
echo "--- Lean smoke gate (login node) ---"
PYTHONPATH="$REPO/src" "$VENV/bin/python" scripts/lean_smoke.py \
  --project-dir "$LEAN_PROJECT" \
  --json-out "results/logs/lean_smoke_csf3_login.json"

cat <<EOF

===================================================================
 BOOTSTRAP COMPLETE
===================================================================
Add to your ~/.bashrc (or source before each session):

    export PATH="\$HOME/.local/bin:\$HOME/.elan/bin:\$PATH"
    export PROOFLENS_PROVER_REPO=$REPO
    export PROOFLENS_PROVER_VENV=$VENV
    export PROOFLENS_LEAN_PROJECT=$LEAN_PROJECT

Now confirm it also works on a COMPUTE node (the parity check that matters):

    cd $REPO && sbatch slurm/lean_smoke.sbatch

Then send back:
    results/logs/lean_smoke_csf3_login.json
    results/logs/lean_smoke_csf3_<jobid>.json
    slurm-<jobid>.out
EOF
