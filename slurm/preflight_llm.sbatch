#!/bin/bash --login
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH -t 0-00:30:00
#SBATCH --job-name=pl_pre
#SBATCH --output=slurm-%j.out
#
# Does the LLM arm start? Answered in ~5 minutes, not ~25.
#
#   sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/preflight_llm.sbatch
#
# Loads the model and samples once. No Lean, no retrieval, no benchmark data — so it isolates the
# only part of the path that has never yet run, without first paying `import Mathlib` over NFS
# (439-691 s, and 2624 s once when staged). Every failure this arm has hit is reachable from here:
# the CUDA-in-forked-subprocess crash, the missing logprobs that would have silently produced a
# meaningless result, and the decoding parameters.
#
# Run this after any environment change, then the 5-problem smoke, then the benchmark. Each stage
# costs roughly 5x the last, so a failure caught early is worth an order of magnitude.

set -euo pipefail

module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VLLM_VENV="${PROOFLENS_VLLM_VENV:-$HOME/scratch/venvs/vllm}"
MODEL="${MODEL:-$HOME/scratch/models/REAL-Prover-v1}"
SAMPLES="${SAMPLES:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# Sample under every candidate template in ONE job. The weights are already loaded, so each extra
# template costs seconds — which is the difference between settling the prompt format empirically and
# arguing about it across three cluster round-trips. Set to a single name to skip the comparison.
COMPARE_TEMPLATES="${COMPARE_TEMPLATES:-qwen_chatml,qwen,deepseek}"
# See the same knob in prove_benchmark_llm.sbatch: set to 1 if startup dies inside a JIT compiler.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EAGER_FLAG=""
[ "$ENFORCE_EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
true    # the `&&` above returns 1 when the test fails, and `set -e` would treat that as fatal

export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
PY="$VLLM_VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no vLLM venv at $VLLM_VENV — run scripts/bootstrap_llm.sh" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: no model weights at $MODEL" >&2; exit 1; }

echo "=== environment ==="
echo "host      : $(hostname)"
echo "git sha   : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "gpu       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "vllm      : $("$PY" -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
echo "transform : $("$PY" -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1)"
echo

"$PY" scripts/preflight_llm.py \
    --model "$MODEL" \
    --samples "$SAMPLES" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --compare-templates "$COMPARE_TEMPLATES" \
    $EAGER_FLAG

echo
echo "If that said PREFLIGHT OK, the 5-problem smoke is next:"
echo "  STAGE_LEAN=0 EXTRA=\"--limit 5\" BENCHMARK=fate_m ARM=li N_CANDIDATES=50000 \\"
echo "      sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/prove_benchmark_llm.sbatch"
