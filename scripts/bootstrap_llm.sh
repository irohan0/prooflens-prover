#!/usr/bin/env bash
# Finish the vLLM environment so `slurm/prove_benchmark_llm.sbatch` can run.
#
#   bash scripts/bootstrap_llm.sh
#
# Idempotent. Safe to re-run.
#
# ## Why this used to fail, and what changed
#
# `pip install vllm` brings torch, transformers, numpy and safetensors. It does not bring
# `lean-interact` (the Lean REPL wrapper) or `pylate` (the ColBERT query encoder for the li arm).
#
# The first version installed those with `--no-deps`, to stop them pulling their own torch pin and
# silently downgrading the torch vLLM was compiled against. That protected torch and broke everything
# else: `--no-deps` also skips genuine runtime requirements, so the environment was missing `scipy`,
# then `datasets`, then whatever came next. Each one was a two-minute cluster job and one traceback.
# Enumerating a dependency tree by reading tracebacks has no bound.
#
# The right tool is a **constraints file**. pip resolves each package's real dependency tree, but a
# constraint on `torch==<installed>` means the resolver either honours it or fails loudly — it cannot
# quietly move torch. So we get correct dependencies *and* an intact vLLM.
#
# Then `scripts/check_env.py` imports every third-party module this codebase imports at any nesting
# depth, found by walking the source with `ast`. Most of them are deliberately imported inside
# functions to keep the test suite hermetic, which is exactly why `import prooflens_prover` proves
# nothing about whether a run will start.

set -euo pipefail

VLLM_VENV="${PROOFLENS_VLLM_VENV:-$HOME/scratch/venvs/vllm}"
REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
#: The retrieval half. Not modified by this script — it already works and produced every verified
#: result — but validated by it, because half an environment is not a working setup.
RETRIEVAL_PYTHON="${PROOFLENS_RETRIEVAL_PYTHON:-$HOME/venvs/plprover/bin/python}"
PY="$VLLM_VENV/bin/python"

[ -x "$PY" ] || {
    echo "ERROR: no virtualenv at $VLLM_VENV." >&2
    echo "Create it first:" >&2
    echo '  BASE=$(~/venvs/plprover/bin/python -c "import sys; print(sys.base_prefix)")' >&2
    echo "  mkdir -p ~/scratch/venvs && \"\$BASE/bin/python3\" -m venv $VLLM_VENV" >&2
    echo "  $PY -m pip install --upgrade pip wheel vllm" >&2
    exit 1
}

# Packages whose versions must not move.
#
# Only two, and the list was longer once. Pinning `transformers`, `tokenizers`, `numpy` and
# `safetensors` as well produced `ResolutionImpossible`: vLLM installs `transformers==5.14.1` and
# sentence-transformers requires `transformers<5`, so nothing could satisfy both. That constraint was
# defensive rather than necessary — vLLM declares a *range* for transformers, and any version inside
# it is supported by definition.
#
# `torch` is the one that truly cannot move: vLLM ships compiled kernels linked against a specific
# build, and a downgrade surfaces as a CUDA error deep inside a run rather than at install time.
# `vllm` itself is pinned so the resolver cannot "solve" a conflict by replacing the thing we came for.
PROTECTED="torch vllm"

#: Reported after the install for information, not constrained. If one of these moved a long way, the
#: run may still work but the manifest should show what it actually used.
WATCHED="transformers tokenizers numpy safetensors"

CONSTRAINTS="$VLLM_VENV/prooflens-constraints.txt"
: > "$CONSTRAINTS"
echo "=== freezing protected versions -> $CONSTRAINTS ==="
for pkg in $PROTECTED; do
    ver=$("$PY" - "$pkg" <<'EOF' 2>/dev/null || true
import importlib.metadata as m, sys
try:
    print(m.version(sys.argv[1]))
except m.PackageNotFoundError:
    pass
EOF
)
    if [ -n "$ver" ]; then
        echo "$pkg==$ver" >> "$CONSTRAINTS"
        echo "  $pkg==$ver"
    else
        echo "  $pkg (not installed; unconstrained)"
    fi
done

# What the PROVER environment needs. Note what is absent: `pylate`, `sentence-transformers` and
# `bm25s`. Retrieval runs in a child process under the retrieval virtualenv
# (`--retrieval-python`), precisely because pylate requires `transformers<=5.3.0` and vLLM requires
# `transformers>=5.5.3`. Installing pylate here is what created the conflict, and it bought nothing:
# this process never encodes a query.
WANTED=(
    "lean-interact>=0.9,<1.0"
    "transformers>=5.5.3"          # vLLM's own floor; pip downgraded below it to satisfy pylate
    "orjson>=3.10"
    "pyyaml>=6.0"
    "tqdm>=4.66"
)

# Left over from the attempts to make one environment serve both. They are the reason transformers
# was pulled below vLLM's floor, and nothing in the prover process imports them.
echo
echo "=== removing retrieval-only packages from the prover environment ==="
for pkg in pylate sentence-transformers; do
    if "$PY" -m pip show "$pkg" >/dev/null 2>&1; then
        echo "  uninstalling $pkg (retrieval runs in a separate interpreter)"
        "$PY" -m pip uninstall -y -q "$pkg"
    else
        echo "  $pkg: absent, good"
    fi
done

echo
echo "=== what vLLM itself allows (the authority on what is safe) ==="
"$PY" - <<'EOF' || true
import importlib.metadata as m
for pkg in ("vllm", "sentence-transformers", "pylate"):
    try:
        reqs = m.requires(pkg) or []
    except m.PackageNotFoundError:
        print(f"  {pkg}: not installed")
        continue
    interesting = [r for r in reqs
                   if any(k in r.lower() for k in ("torch", "transformers", "tokenizers", "numpy"))]
    print(f"  {pkg}:")
    for r in interesting or ["    (no torch/transformers constraints)"]:
        print(f"    {r}")
EOF

# Resolve first, install second. A dry run takes seconds on a login node and reports a conflict
# without touching the environment — which is the difference between iterating on the constraint set
# in seconds and iterating in two-minute cluster jobs. Four round-trips were spent the other way.
echo
echo "=== resolving (dry run: nothing is installed yet) ==="
if ! "$PY" -m pip install --dry-run -c "$CONSTRAINTS" "${WANTED[@]}" 2>&1 | tail -25; then
    echo
    echo "ERROR: resolution failed. Nothing was installed, so the environment is unchanged." >&2
    echo "       Read the conflict above: it names the two requirements that cannot both hold." >&2
    echo "       Loosen \$PROTECTED in this script only for packages vLLM declares a range for." >&2
    exit 1
fi

echo
echo "=== installing ==="
# WITH dependencies. The constraints file is what keeps torch still.
"$PY" -m pip install --quiet -c "$CONSTRAINTS" "${WANTED[@]}"

# pip can "succeed" while leaving a declared requirement unsatisfied: it resolves the packages named
# on the command line and prints a warning about everything else. That is how this environment ended
# up with vllm 0.26.0 (needs transformers>=5.5.3) and transformers 5.3.0 (pylate's ceiling) at the
# same time — installed, warned about, and broken. `pip check` is the assertion that warning deserves.
echo
echo "=== pip check: are all declared requirements satisfied? ==="
if ! "$PY" -m pip check; then
    echo
    echo "ERROR: the environment does not satisfy its own declared requirements." >&2
    echo "       This is not a warning to skim past — the conflict above is why an import will fail" >&2
    echo "       mid-run. If two packages have disjoint ranges for a shared dependency, no pin can" >&2
    echo "       fix it: either move one package to a version whose range overlaps, or separate them" >&2
    echo "       into different processes (see scripts/encode_server.py)." >&2
    exit 1
fi

echo
echo "=== confirming the protected versions did not move ==="
moved=0
while IFS='=' read -r pkg _ ver; do
    [ -n "$pkg" ] || continue
    now=$("$PY" -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null || echo "GONE")
    if [ "$now" = "$ver" ]; then
        echo "  $pkg $now"
    else
        echo "  !! $pkg was $ver, is now $now"
        moved=1
    fi
done < <(sed 's/==/=/' "$CONSTRAINTS" | sed 's/=/==/')
if [ "$moved" = "1" ]; then
    echo "ERROR: a protected package moved despite the constraints file. Stopping — vLLM may be" >&2
    echo "       broken in a way that only appears mid-run." >&2
    exit 1
fi

echo
echo "=== versions of packages we let float ==="
for pkg in $WATCHED; do
    ver=$("$PY" -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null || echo "-")
    echo "  $pkg: $ver"
done

echo
echo "=== the PROVER environment (vllm; pylate expected ABSENT) ==="
PYTHONPATH="$REPO/src" "$PY" "$REPO/scripts/check_env.py" \
    --protected "$(echo "$PROTECTED $WATCHED" | tr ' ' ',')" --require vllm,vllm.LLM

echo
echo "=== the RETRIEVAL environment (pylate; vllm expected ABSENT) ==="
# Both halves from one command. A perfect prover environment plus a retrieval environment that
# cannot import pylate still cannot run an arm, and checking only the half you are standing in is
# how this took four round-trips.
if [ -x "$RETRIEVAL_PYTHON" ]; then
    PYTHONPATH="$REPO/src" "$RETRIEVAL_PYTHON" "$REPO/scripts/check_env.py" \
        --require pylate,pylate.models
else
    echo "  WARNING: no retrieval interpreter at $RETRIEVAL_PYTHON; li and sv need it." >&2
fi

echo
echo "=== the prompt, for one human look ==="
PYTHONPATH="$REPO/src" "$PY" - <<'EOF'
from prooflens_prover.prover.prompt import build_tactic_prompt
from prooflens_prover.retrieval.base import Premise

print(build_tactic_prompt("a b : G\n⊢ a * b = b * a",
                          [Premise("mul_comm", "∀ (a b : G), a * b = b * a", "mult. commutes")]))
EOF

echo
echo "READY. Smoke next, before anything long — STAGE_LEAN=0 because staging Mathlib has been"
echo "measured at up to 2624 s, which a 5-problem run never recovers:"
echo "  STAGE_LEAN=0 EXTRA=\"--limit 5\" BENCHMARK=fate_m ARM=li N_CANDIDATES=50000 \\"
echo "      sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/prove_benchmark_llm.sbatch"
