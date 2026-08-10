#!/usr/bin/env python
"""Load the model, sample once, and check the numbers are real. Nothing else.

    python scripts/preflight_llm.py --model ~/scratch/models/REAL-Prover-v1

No Lean, no retrieval, no benchmark data — so it answers "will the LLM arm start?" in the time it
takes to load 15 GB of weights (~4 min) instead of the ~25 min a smoke run costs after paying
`import Mathlib` over NFS. Run this first, every time the environment changes.

## What it actually proves

Each of these was a separate cluster round-trip, and every one of them is exercised here:

* **The engine starts at all.** vLLM v1 forks `EngineCore`, and a forked child cannot use CUDA if
  the parent ever called `torch.cuda.is_available()` — which `utils.seed` used to do, three hundred
  lines before the model loaded. `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
* **`cumulative_logprob` is not None.** vLLM v1 only populates it when logprobs are requested, and
  the search ranks on it. Without this check the failure is invisible: every candidate scores 0.0,
  best-first degenerates to alphabetical order, and the run reports a plausible, meaningless number.
* **The decoding parameters are REAL-Prover's**, not argparse's opinion of them.
* **Distinct candidates come back.** One tactic repeated sixteen times is a working engine and a
  useless search, and the pass rate alone will not tell you which you have.

Exits non-zero with a diagnosis on any failure, so it chains: `preflight && sbatch ...`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.prover.prompt import (  # noqa: E402
    DEFAULT_TEMPLATE,
    build_tactic_content,
    render_chat,
)
from prooflens_prover.prover.vllm_policy import (  # noqa: E402
    REPLACEMENT_CHAR,
    SamplingConfig,
    VLLMGenerator,
    clean_tactic,
)
from prooflens_prover.retrieval.base import Premise  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402
from prooflens_prover.utils.seed import set_global_seed  # noqa: E402

#: A real FATE-M-shaped goal with a real premise, so the prompt exercised here is the same shape the
#: run will build — including the full-width separators in the chat template, which is exactly the
#: kind of thing that renders fine locally and breaks on a different tokenizer.
STATE = "G : Type u_1\ninst✝ : CommGroup G\na b : G\n⊢ a * b = b * a"
PREMISES = [
    Premise("mul_comm", "∀ {G : Type u_1} [inst : CommMonoid G] (a b : G), a * b = b * a",
            "multiplication commutes"),
    Premise("CommGroup.to_commMonoid", "∀ {G : Type u_1} [self : CommGroup G], CommMonoid G", ""),
]


def main() -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=16,
                    help="REAL-Prover's NUM_SAMPLES")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--compare-templates", default="",
                    help="comma-separated templates to sample from in one job, e.g. "
                         "qwen_chatml,qwen,deepseek. The weights are already loaded, so each extra "
                         "template costs seconds — which is the difference between settling this "
                         "empirically and arguing about it across three cluster round-trips")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip CUDA-graph capture and torch.compile; the escape hatch when a "
                         "compute node lacks a toolchain some kernel wants to JIT")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Deliberately called, not skipped: this is the function whose `torch.cuda.is_available()`
    # broke the arm, so the preflight must run through it exactly as the real script does. A
    # preflight that avoids the dangerous path proves nothing about the run that takes it.
    set_global_seed(args.seed)

    sampling = SamplingConfig()
    print("=== configuration ===")
    print(f"model    : {args.model}")
    print(f"sampling : {sampling.to_dict()}")
    print()

    print("=== loading the engine (this is where the fork bug appeared) ===", flush=True)
    generator = VLLMGenerator(
        args.model,
        max_model_len=args.max_model_len,
        max_tokens=sampling.max_tokens,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    print(f"engine env: {generator.engine_env}")
    print()

    content = build_tactic_content(STATE, PREMISES)
    templates = [t for t in args.compare_templates.split(",") if t] or [args.template]
    results: dict[str, dict[str, Any]] = {}

    for template in templates:
        prompt = render_chat([{"role": "user", "content": content}], template)
        print("=" * 78)
        print(f"TEMPLATE: {template}")
        print("=" * 78)
        print(prompt)
        print("--- end of prompt ---")

        # Against the checkpoint's own `chat_template`. This is the check that would have caught the
        # deepseek format before it cost a run, so in the preflight it is a hard gate.
        mismatch = generator.check_prompt_format(prompt, content)
        print(f"matches the shipped chat_template: {'YES' if mismatch is None else 'NO'}")
        if mismatch is not None:
            print(f"  {mismatch}")

        generations = generator.generate(prompt, args.samples, sampling)
        distinct: dict[str, float] = {}
        n_garbled = 0
        print(f"\n{'mean logprob':>13}  {'tokens':>6}  tactic")
        for g in sorted(generations, key=lambda g: -g.mean_logprob):
            tactic, _ = clean_tactic(g.text)
            print(f"{g.mean_logprob:13.4f}  {g.n_tokens:6d}  {tactic!r}")
            if REPLACEMENT_CHAR in tactic:
                n_garbled += 1
            if tactic:
                distinct.setdefault(tactic, g.mean_logprob)

        mean_lp = (
            sum(g.mean_logprob for g in generations) / len(generations) if generations else 0.0
        )
        results[template] = {
            "matches_shipped": mismatch is None,
            "n_distinct": len(distinct),
            "n_garbled": n_garbled,
            "mean_logprob": round(mean_lp, 4),
            "generations": generations,
        }
        print()

    if len(templates) > 1:
        print("=" * 78)
        print("TEMPLATE COMPARISON")
        print("=" * 78)
        # Mean log-probability is the quantitative signal: a model given a format it was trained
        # on is markedly more confident about its own continuations than one improvising on an
        # unfamiliar one. Read it alongside the tactics themselves — a higher mean with visibly
        # worse Lean would mean something else is wrong.
        print(f"{'template':<14} {'shipped?':>9} {'distinct':>9} "
              f"{'garbled':>8} {'mean logprob':>13}")
        for name, r in results.items():
            print(f"{name:<14} {('yes' if r['matches_shipped'] else 'no'):>9} "
                  f"{r['n_distinct']:>9} {r['n_garbled']:>8} {r['mean_logprob']:>13.4f}")
        print("\nHigher mean logprob = the model recognises the format. Pick on this AND on")
        print("whether the tactics above are plausible Lean, not on either alone.")
        print()

    chosen = results[templates[0]]
    generations = chosen["generations"]
    print(f"stats     : {generator.stats()}")
    print()

    problems: list[str] = []
    if not generations:
        problems.append("the engine returned nothing at all")
    if generations and all(g.mean_logprob == 0.0 for g in generations):
        # Belt to the SystemExit's braces: a generator returning 0.0 rather than None would slip
        # past that check and produce the same silent, uniform, meaningless ranking.
        problems.append(
            "every mean logprob is exactly 0.0, so best-first would have nothing to rank on"
        )
    if chosen["n_distinct"] < 2:
        problems.append(
            f"only {chosen['n_distinct']} distinct non-empty tactic from {args.samples} samples — "
            "the engine works but the search would have nothing to choose between"
        )
    if not chosen["matches_shipped"]:
        # The failure this whole round exists to prevent. A format the model never saw yields
        # fluent-looking nonsense rather than an error, so nothing downstream can catch it.
        problems.append(
            f"template {templates[0]!r} does not match the chat_template shipped with these "
            "weights — see the mismatch above. This is the failure that cost a FATE-M run."
        )
    if chosen["n_garbled"] > args.samples // 4:
        problems.append(
            f"{chosen['n_garbled']} of {args.samples} generations contain U+FFFD. A quarter of the "
            "samples being undecodable is a symptom of an out-of-distribution prompt, not of the "
            "sampler"
        )

    if problems:
        print("PREFLIGHT FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"PREFLIGHT OK — {chosen['n_distinct']} distinct tactics, logprobs present, "
          f"prompt matches the shipped chat_template.")
    print("The LLM arm can now be smoke-tested against Lean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
