#!/bin/bash --login
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH --job-name=pl_llm
# stderr MERGED into stdout, as elsewhere: a split stream hid an encoding traceback for a full cycle.
#SBATCH --output=slurm-%j.out
#
# Tier 1: freeze REAL-Prover-v1 (7B) and vary only the retriever.
#
#   BENCHMARK=fate_m ARM=li N_CANDIDATES=50000 \
#       sbatch -p gpuA -A gpu-fse-ugpgt01 -G 1 slurm/prove_benchmark_llm.sbatch
#
# Smoke first, always — 5 problems, ~15 minutes, and it proves the whole path end to end:
#
#   EXTRA="--limit 5" BENCHMARK=fate_m ARM=li N_CANDIDATES=50000 sbatch ...
#
# ## Two virtualenvs, and two PROCESSES
#
#     vllm 0.26.0   requires transformers>=5.5.3
#     pylate 1.6.0  requires transformers<=5.3.0
#
# Disjoint ranges. No pin satisfies both, so the prover and the query encoder cannot share a process.
# The prover runs under `~/scratch/venvs/vllm`; retrieval runs as a **child process** under
# `~/venvs/plprover` — the environment that produced every verified model-free result, and that this
# does not modify. `--retrieval-python` connects them over newline-delimited JSON.
#
# That separation is worth more than a working install: the LLM arm queries bit-identically the same
# retriever as the model-free arm, so a difference between the two tracks cannot be a difference in
# retrieval. The `none` arm needs no retrieval process and does not start one.
#
# ## Memory: three consumers on one GPU, and the order they arrive in
#
# The 15 GB of bf16 weights are not what fills an A100. The KV cache is, and it grows with prompt
# length — which retrieval directly inflates by adding premises. `--max-model-len 4096` caps it.
#
# The retrieval child *does* use the GPU, contrary to what this header claimed for two revisions:
# `--device cuda` puts the ~149M-parameter query encoder there, which is what the model-free runs
# did and therefore what keeps this arm's retrieval bit-identical to theirs. Only the LI index
# itself is CPU-resident (5.5 GB of host memory, scored with numpy). The child loads eagerly, before
# vLLM profiles, so vLLM's budget already accounts for it — but the encoder's cuBLAS workspaces are
# allocated on its first real query, *after* vLLM has taken its share. Hence
# `--gpu-memory-utilization 0.85` rather than 0.90: ~12 GB of headroom on an 80 GB card, ~6 GB on a
# 40 GB one, and no run lost to an OOM three hours in.
#
# Only one process imports Mathlib, so the wall-clock accounting is unchanged.
#
# ## Wall clock: this benchmark does not fit one job, by design
#
# 141 problems x `--wall-clock 600` is 23.5 h worst case against an 8 h limit. That is expected and
# handled: `JsonlAppender` fsyncs every row, so resubmit with `--resume results/logs/<run_id>` and
# the run continues where it stopped. Shrinking the per-problem budget to fit the queue would
# quietly make this a different experiment from the model-free arms; resuming does not.

set -euo pipefail

module purge || true

REPO="${PROOFLENS_PROVER_REPO:-$HOME/scratch/prooflens-prover}"
VLLM_VENV="${PROOFLENS_VLLM_VENV:-$HOME/scratch/venvs/vllm}"
RETRIEVAL_PYTHON="${PROOFLENS_RETRIEVAL_PYTHON:-$HOME/venvs/plprover/bin/python}"
LEAN_PROJECT="${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/mathlib_v4160}"
DATA_ROOT="${PROOFLENS_DATA_ROOT:-$HOME/data/benchmarks/REAL-Prover/Realprover/data}"

BENCHMARK="${BENCHMARK:-fate_m}"
ARM="${ARM:-li}"
INDEX="${INDEX:-data/index/li_ft_novel_bm25}"
MODEL="${MODEL:-$HOME/scratch/models/REAL-Prover-v1}"
INFORMAL_NAMES="${INFORMAL_NAMES:-$HOME/scratch/data/mathlib_informal_v4160}"
# REAL-Prover's shipped conf/config.py, verbatim: NUM_SAMPLES=16, and PROVER_MODEL_PARAMS
# {"temperature": 1.5, "max_tokens": 256, "top_p": 0.9}. Every one of these was wrong here once —
# 32 samples (double their budget) at temperature 1.0 and top_p 1.0 — which would have made "differs
# from theirs only in the retriever" untrue in three places at once. temperature 1.5 looks like a
# typo and is not: best-first reranks the samples afterwards, so the sampler's job is coverage.
SAMPLES="${SAMPLES:-16}"
PROMPT_LIMIT="${PROMPT_LIMIT:-6}"
# The chat template the released checkpoint declares in its own tokenizer_config.json. REAL-Prover's
# code hard-codes `deepseek`; sending that to these weights produced multilingual token salad and
# lost a FATE-M run. TEMPLATE=deepseek reproduces their code exactly, as an ablation.
TEMPLATE="${TEMPLATE:-qwen_chatml}"
TEMPERATURE="${TEMPERATURE:-1.5}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# Set ENFORCE_EAGER=1 if startup dies inside a JIT compiler. A GPU compute node has the CUDA driver
# but no toolkit, so any kernel that compiles itself on first use fails there — `nvcc` not found
# during sampler warm-up cost one round-trip already. `ENGINE_ENV` disables the known offender
# (FlashInfer's sampler); this turns off CUDA-graph capture and torch.compile as well.
# Expanded through a plain variable rather than `$(test && echo)`, matching $EXTRA below: an empty
# string word-splits to no argument, with no dependence on how a given bash treats the exit status
# of a command substitution under `set -e`.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EAGER_FLAG=""
[ "$ENFORCE_EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"
true    # `[ ... ] && ...` returns 1 when the test fails, which `set -e` would treat as fatal
DEVICE="${DEVICE:-cuda}"
# li only. Empty = the index's stored 1000, which measured recall@10 = 0.443 on real queries.
N_CANDIDATES="${N_CANDIDATES:-}"
# The sampling DRAW, and the only thing a replicate may vary. `--seed` reaches the vLLM engine as
# `LLM(seed=...)` while `SamplingParams.seed` stays None, so a run is reproducible at a given SEED
# and a different SEED is an independent draw. The whole Tier 1 table was produced at SEED=0, which
# is one draw and therefore carries no estimate of its own variance -- see
# `scripts/replication_variance.py`. Was previously not settable here at all, so every run silently
# took argparse's default.
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"

export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# vLLM's own progress bars are noise in a SLURM log and hide the search's own logging.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
mkdir -p results/logs
PY="$VLLM_VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: no vLLM venv at $VLLM_VENV — run scripts/bootstrap_llm.sh" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "ERROR: no model weights at $MODEL" >&2; exit 1; }

# Checked HERE rather than after the Mathlib import, which costs 158-691 s. A broken retrieval
# environment discovered afterwards has already spent that, and four such cycles cost a day.
if [ "$ARM" != "none" ]; then
    [ -x "$RETRIEVAL_PYTHON" ] || {
        echo "ERROR: no retrieval interpreter at $RETRIEVAL_PYTHON." >&2
        echo "       It must be the venv with pylate — the one the model-free runs used." >&2
        exit 1
    }
    "$RETRIEVAL_PYTHON" -c "import pylate.models" 2>/dev/null || {
        echo "ERROR: $RETRIEVAL_PYTHON cannot import pylate.models." >&2
        echo "       Run: $RETRIEVAL_PYTHON scripts/check_env.py" >&2
        exit 1
    }
fi

echo "=== environment ==="
echo "host      : $(hostname)"
echo "git sha   : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "gpu       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "benchmark : $BENCHMARK | arm: $ARM | index: $INDEX"
echo "model     : $MODEL"
echo "n_cand    : ${N_CANDIDATES:-(index default)}"
echo "sampling  : $SAMPLES samples/step, temp $TEMPERATURE, top_p $TOP_P, max_tokens $MAX_TOKENS"
echo "seed      : $SEED  (the sampling draw; replicates differ ONLY here)"
echo "template  : $TEMPLATE  (watch the log for 'PROMPT FORMAT MISMATCH')"
echo "retrieval : $RETRIEVAL_PYTHON (separate process)"
echo "vllm      : $("$PY" -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1)"
echo

# Same node-local staging as the model-free arm: `import Mathlib` from NFS has been measured at
# 439-691 s and once exceeded the header timeout outright. Never allowed to break a run.
if [ "${STAGE_LEAN:-1}" = "1" ]; then
    if STAGED=$(bash scripts/stage_lean_project.sh "$LEAN_PROJECT"); then
        echo "using node-local Lean project: $STAGED"
        LEAN_PROJECT="$STAGED"
    else
        echo "staging unavailable — using $LEAN_PROJECT over NFS (slower warm-up, still correct)"
    fi
fi
echo

# shellcheck disable=SC2086  # EXTRA and EAGER_FLAG are deliberately word-split; empty adds nothing
"$PY" scripts/prove_benchmark.py \
    --policy vllm \
    --benchmark "$BENCHMARK" \
    --arm "$ARM" \
    --index "$INDEX" \
    --model "$MODEL" \
    --informal-names "$INFORMAL_NAMES" \
    --retrieval-python "$RETRIEVAL_PYTHON" \
    --data-root "$DATA_ROOT" \
    --lean-project "$LEAN_PROJECT" \
    --samples-per-step "$SAMPLES" \
    --prompt-limit "$PROMPT_LIMIT" \
    --template "$TEMPLATE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --device "$DEVICE" \
    --seed "$SEED" \
    ${N_CANDIDATES:+--n-candidates "$N_CANDIDATES"} \
    $EAGER_FLAG \
    $EXTRA

echo
echo "Confirm from the MANIFEST, not this header, that the run used what you intended:"
echo "  grep -E '\"policy_kind\"|\"n_candidates\"|\"informal_names\"|\"temperature\"|\"logprobs\"' \\"
echo "       results/logs/<run_id>/manifest.json"
echo
echo "Then check the arm is HEALTHY, which the pass rate alone will not tell you:"
echo "  python -c \"import json,sys; o=json.load(open(sys.argv[1]))['outcome']; \\"
echo "             print(o['policy_stats']); print(o['generator_stats'])\" \\"
echo "       results/logs/<run_id>/manifest.json"
echo "  TWO numbers, and neither alone is a gate:"
echo "    mean_candidates_per_expansion near $SAMPLES = the search has choices; near 1 = the model"
echo "      is repeating itself or nearly everything is rejected."
echo "    mean_candidate_logprob is the QUALITY half. Measured on this model: about -0.35 to -0.70"
echo "      in-distribution, about -2.78 with a wrong chat template."
echo "  Count alone is not health: the broken deepseek prompt produced MORE distinct candidates"
echo "  than the correct one (13 vs 8), because noise does not repeat itself, and a run generating"
echo "  token salad reported 11.33/16 and looked fine."
echo "  cheat_rate, n_undecodable and n_prompts_truncated should all be ~0."
echo
echo "Run 'seff ${SLURM_JOB_ID:-<jobid>}' for resource accounting."
