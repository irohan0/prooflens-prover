#!/usr/bin/env python
"""Answer "would this exact command have worked?" before a single GPU-hour is queued.

    python scripts/preflight_sweep.py --benchmark proofnet_test --arm li \
        --index data/index/li_ft_novel_bm25 --data-root ~/data/.../data \
        --model ~/scratch/models/REAL-Prover-v1 --samples-per-step 32 --slurm-time 8

No GPU, no Lean, no model weights. It runs on a login node in seconds, and every check it makes is
one that has already cost this project a submit/wait/diagnose cycle at least once:

* an index built over a different corpus, so the arms were not ranking the same premise set;
* a benchmark path that resolved to nothing, so the run "completed" with zero problems;
* a budget whose projected wall clock exceeded the SLURM limit, so the job was killed part-way and
  left a manifest describing an experiment that never finished;
* a policy configuration that raises on its first `propose` call — after `import Mathlib` and a
  15 GB model load have already been paid for.

`scripts/preflight_llm.py` is the complement: it starts the engine and checks the numbers are real.
This one checks everything *except* the engine, which is why it can run without a GPU allocation.

## The wall-clock projection

Every rate below is measured from the staged Tier 1 runs in `results/exported/`, not estimated.
Retrieval cost scales with the number of expansions (one query per expansion) while generation and
Lean cost scale with expansions *and* samples, so the two halves are projected separately. The
projection is deliberately an over-estimate: it assumes a doubled sample count doubles the Lean
work, when in practice some problems close sooner.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.lean.backend import ProofState  # noqa: E402
from prooflens_prover.prover.search import SearchConfig  # noqa: E402
from prooflens_prover.prover.vllm_policy import Generation, SamplingConfig, VLLMPolicy  # noqa: E402
from prooflens_prover.retrieval.base import Premise  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: The corpus every arm must rank, asserted at index build by `--assert-corpus-id` and again here.
#: Two indices over different corpora would still run, still fuse, and still produce a plausible
#: table — of an experiment where the arms were never comparable.
EXPECTED_CORPUS_ID = "276070:31db61c63a9b7ee1"

#: Measured hours for one pass at the published budget (64 expansions x 16 samples), split into the
#: half that scales with expansions alone and the half that scales with expansions x samples.
#: Source: the staged run set, `results/exported/logs/*_vllm_2026081[01]*`.
BASELINE: dict[tuple[str, str], tuple[float, float]] = {
    # (benchmark, arm): (retrieval hours, generation+Lean hours)
    ("proofnet_test", "none"): (0.00, 1.78),
    ("proofnet_test", "sv"): (0.06, 1.88),
    ("proofnet_test", "li"): (1.59, 1.86),
    ("fate_m", "none"): (0.00, 0.83),
    ("fate_m", "sv"): (0.04, 0.94),
    ("fate_m", "li"): (1.09, 0.99),
}

#: Expected problem counts, so a mis-resolved `--data-root` cannot look like a small benchmark.
BENCHMARK_SIZES = {"fate_m": 141, "proofnet_test": 186, "minif2f_test": 244}

#: Bytes of free space required under the results root. One full run's traces are tens of MB; the
#: margin is for a 32-job sweep landing in the same filesystem.
MIN_FREE_BYTES = 5 * 1024**3

#: Conservative characters-per-token for Lean-heavy prompt text, used only when the real tokenizer
#: cannot be loaded. Lean identifiers tokenize badly (`Nat.succ_le_succ` is several tokens), so a
#: low number is the safe direction for a *budget* check.
CHARS_PER_TOKEN_FLOOR = 2.0


class Check:
    """Collects failures rather than raising, so one run reports every problem at once.

    Raising on the first would send the user round the fix-and-rerun loop once per defect, which is
    the loop this script exists to remove.
    """

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  [ OK ] {label}" + (f" — {detail}" if detail else ""))

    def fail(self, label: str, detail: str) -> None:
        print(f"  [FAIL] {label} — {detail}")
        self.failures.append(f"{label}: {detail}")

    def warn(self, label: str, detail: str) -> None:
        print(f"  [warn] {label} — {detail}")
        self.notes.append(f"{label}: {detail}")


def index_corpus_id(index_dir: Path) -> str | None:
    meta = index_dir / "meta.json"
    if not meta.exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("corpus_id")


def check_index(c: Check, label: str, index_dir: Path | None) -> None:
    if index_dir is None:
        c.fail(label, "not given, but the arm requires it")
        return
    if not index_dir.exists():
        c.fail(label, f"{index_dir} does not exist")
        return
    cid = index_corpus_id(index_dir)
    if cid is None:
        c.fail(label, f"{index_dir}/meta.json missing or has no corpus_id")
    elif cid != EXPECTED_CORPUS_ID:
        c.fail(label, f"corpus_id {cid} != {EXPECTED_CORPUS_ID}. The arms would not be ranking the "
                      "same premise set, so no comparison between them means anything")
    else:
        c.ok(label, f"corpus_id {cid}")


def load_statements(data_root: Path, benchmark: str) -> list:
    from prooflens_prover.data.benchmarks import load_benchmark

    return load_benchmark(benchmark, data_root, check_count=False)


#: Wall-clock cost of premise-free mixing, measured in the budget pilot: 2.32 h at 64x32 against
#: 2.46 h with `--premise-free-fraction 0.25`. It is a second *prefill* per expansion, not extra
#: samples, so it is a small fixed surcharge rather than anything that scales with the fraction.
PREMISE_FREE_OVERHEAD = 1.06


def project_hours(benchmark: str, arm: str, expansions: int, samples: int,
                  n_problems: int | None, premise_free: float = 0.0) -> float | None:
    """Projected wall clock, scaling the measured baseline by the requested budget."""
    key = (benchmark, "li" if arm == "fusion" else arm)
    if key not in BASELINE:
        return None
    retrieval_h, gen_h = BASELINE[key]
    if arm == "fusion":
        # Fusion queries both retrievers per state; single-vector adds ~4% of a li query on GPU and
        # rather less than that matters next to li's ~930 ms.
        retrieval_h *= 1.1
    hours = (retrieval_h * (expansions / 64)
             + gen_h * (expansions / 64) * (samples / 16))
    if premise_free > 0:
        hours *= PREMISE_FREE_OVERHEAD
    if n_problems:
        hours *= n_problems / BENCHMARK_SIZES.get(benchmark, n_problems)
    return hours


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--arm", required=True, choices=["none", "bm25", "li", "sv", "fusion"])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--index-sv", type=Path, default=None)
    ap.add_argument("--index-li", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--max-expansions", type=int, default=64)
    ap.add_argument("--samples-per-step", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--premise-free-fraction", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-root", type=Path, default=Path("results/logs"))
    ap.add_argument("--slurm-time", type=float, default=8.0,
                    help="the job's -t limit in hours; the projection is checked against it")
    ap.add_argument("--headroom", type=float, default=0.8,
                    help="fraction of --slurm-time the projection must fit inside, leaving room "
                         "for Lean staging and the model load")
    args = ap.parse_args()
    ensure_utf8_output()
    c = Check()

    print(f"=== preflight: {args.benchmark} / {args.arm} "
          f"({args.max_expansions} nodes x {args.samples_per_step} samples) ===\n")

    # --- indices ------------------------------------------------------------------------------
    if args.arm == "fusion":
        check_index(c, "index (sv)", args.index_sv)
        check_index(c, "index (li)", args.index_li)
    elif args.arm != "none":
        check_index(c, "index", args.index)
    else:
        c.ok("index", "not needed for the no-retrieval control")

    # --- benchmark ----------------------------------------------------------------------------
    problems = []
    try:
        problems = load_statements(args.data_root, args.benchmark)
        expected = BENCHMARK_SIZES.get(args.benchmark)
        if expected and len(problems) != expected:
            c.fail("benchmark", f"loaded {len(problems)} problems, expected {expected}. A run on a "
                                "different problem set is not comparable to the published rate")
        else:
            c.ok("benchmark", f"{len(problems)} problems from {args.data_root}")
    except Exception as e:  # noqa: BLE001 — reporting the failure is the whole job here
        c.fail("benchmark", f"{type(e).__name__}: {e}")

    # --- model --------------------------------------------------------------------------------
    if args.model is None:
        c.warn("model", "not given; the engine path is unchecked")
    elif not (args.model / "config.json").exists():
        c.fail("model", f"{args.model} has no config.json — not a model directory")
    else:
        c.ok("model", str(args.model))

    # --- the policy actually constructs and proposes -------------------------------------------
    # This is the check that catches a bad --premise-free-fraction or a broken prompt path, and it
    # exercises the real `propose` rather than a paraphrase of it.
    prompt_chars = 0
    try:
        stub_premises = [
            Premise(formal_name=f"Some.Namespace.premise_{i}",
                    formal_statement="∀ {G : Type*} [Group G] (a b : G), a * b⁻¹ = 1 ↔ a = b")
            for i in range(10)
        ]

        class StubRetriever:
            name = args.arm

            def retrieve(self, query, k=10):  # noqa: ARG002
                return stub_premises[:k]

        class StubGenerator:
            def generate(self, prompt, n, sampling):  # noqa: ARG002
                nonlocal prompt_chars
                prompt_chars = max(prompt_chars, len(prompt))
                return [Generation(text="simp", cumulative_logprob=-1.0, n_tokens=2)]

        policy = VLLMPolicy(
            generator=StubGenerator(), retriever=StubRetriever(),
            sampling=SamplingConfig(), premise_free_fraction=args.premise_free_fraction,
        )
        longest = max((p.statement for p in problems), key=len, default="theorem t : True := by")
        candidates = policy.propose(ProofState(pid=1, goals=(longest,)), args.samples_per_step)
        if not candidates:
            c.fail("policy", "propose() returned no candidates from a healthy generator")
        else:
            c.ok("policy", f"{len(candidates)} candidate(s), "
                           f"premise_free_fraction={args.premise_free_fraction}")
    except Exception as e:  # noqa: BLE001
        c.fail("policy", f"{type(e).__name__}: {e}")

    # --- prompt budget ------------------------------------------------------------------------
    if prompt_chars:
        est_tokens = prompt_chars / CHARS_PER_TOKEN_FLOOR
        if est_tokens > args.max_model_len:
            c.fail("prompt budget",
                   f"worst-case prompt ~{est_tokens:.0f} tokens (>{args.max_model_len}). The "
                   "generator would truncate it, and a truncated premise block is a different arm")
        else:
            c.ok("prompt budget",
                 f"worst case {prompt_chars} chars, <={est_tokens:.0f} tokens of "
                 f"{args.max_model_len}")

    # --- search config ------------------------------------------------------------------------
    try:
        SearchConfig(max_expansions=args.max_expansions, samples_per_step=args.samples_per_step)
        c.ok("search config", f"{args.max_expansions} x {args.samples_per_step}")
    except Exception as e:  # noqa: BLE001
        c.fail("search config", f"{type(e).__name__}: {e}")

    # --- disk ---------------------------------------------------------------------------------
    probe = args.results_root if args.results_root.exists() else Path.cwd()
    free = shutil.disk_usage(probe).free
    if free < MIN_FREE_BYTES:
        c.fail("disk", f"{free / 1024**3:.1f} GB free under {probe}, want >= "
                       f"{MIN_FREE_BYTES / 1024**3:.0f} GB")
    else:
        c.ok("disk", f"{free / 1024**3:.1f} GB free")

    # --- the projection -----------------------------------------------------------------------
    n = args.limit or len(problems) or None
    hours = project_hours(args.benchmark, args.arm, args.max_expansions,
                          args.samples_per_step, n, args.premise_free_fraction)
    if hours is None:
        c.warn("wall clock", f"no measured baseline for {args.benchmark}/{args.arm}; unprojected")
    elif args.limit and args.limit < BENCHMARK_SIZES.get(args.benchmark, 0):
        # Pro-rata scaling assumes problems are uniformly hard, and they are not. Measured: the
        # budget pilot on ProofNet's first 60 was projected at 1.71 h and took 2.32 h — a 36%
        # under-estimate, which would eat the whole default headroom. Full-benchmark projections
        # are unaffected: the baselines are calibrated on full runs.
        c.warn("wall clock",
               f"projected {hours:.2f} h for the first {args.limit} problems, but a --limit subset "
               "is not a uniform sample — ProofNet's first 60 ran 36% over their projection")
    else:
        budget = args.slurm_time * args.headroom
        detail = (f"projected {hours:.2f} h for {n} problems, against {budget:.2f} h usable "
                  f"({args.slurm_time:.0f} h limit x {args.headroom:.0%})")
        if hours > budget:
            c.fail("wall clock", detail + ". Shard with --limit/--offset, or raise -t")
        else:
            c.ok("wall clock", detail)

    print()
    if c.failures:
        print(f"PREFLIGHT FAILED — {len(c.failures)} problem(s):")
        for f in c.failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PREFLIGHT CLEAN — this configuration is safe to submit.")
    if c.notes:
        print(f"({len(c.notes)} unchecked item(s): {'; '.join(c.notes)})")


if __name__ == "__main__":
    main()
