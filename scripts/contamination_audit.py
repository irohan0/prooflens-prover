#!/usr/bin/env python
"""Audit how often a benchmark's answer was already sitting in the retrieval corpus.

    python scripts/contamination_audit.py \
        --corpus data/premises/mathlib_v4160.jsonl \
        --run results/logs/fate_m_none_vllm_... \
        --run results/logs/fate_m_sv_vllm_... \
        --run results/logs/fate_m_li_vllm_...

## The question

The premise corpus is all 276,070 declarations of Mathlib v4.16.0. Several benchmark theorems are
restatements of lemmas Mathlib already contains — ProofNet is transcribed from textbook exercises
that Mathlib also formalises, and FATE-M is graduate algebra over the same library. When that
happens the retriever can hand the prover the theorem itself, and the proof closes in one step:

    exact Sylow.normalizer_normalizer P

That is a **valid Lean proof** and not a cheat in the `sorry` sense — the guard in
`lean/backend.py` is about `sorry`/`admit`/`sorryAx` and has nothing to say here. But it is the
degenerate case of premise retrieval, and an examiner will ask how much of the measured effect it
is. This script answers that from the run records, with no GPU and no Lean.

## What counts

A solved problem is a **one-step corpus answer** when its entire proof is a single `exact` or
`apply` whose head identifier resolves to a premise in the corpus. That is deliberately the
narrowest possible definition:

* One step, so anything requiring the prover to do work first is excluded.
* A bare head identifier, so `exact fun h => ...` and `exact ⟨_, _⟩` — which build a term rather
  than name one — do not count.
* Resolved against the corpus, so a local hypothesis (`exact h₁`) does not count.

It therefore **under**-counts: a two-step proof that is really `intro x; exact <lemma>` is not
flagged. An under-count is the right direction for a figure used to bound a concern.

## Why it does not threaten the architecture contrast

Both retrieval arms index the identical corpus (`--assert-corpus-id` enforces it at build time), so
whatever is in reach of one is in reach of the other. Corpus overlap inflates the *absolute* pass
rate and part of the retrieval-vs-none effect; it cannot manufacture a difference between SV and
LI. The control arm is reported alongside for exactly this reason: a one-step citation that the
no-retrieval arm also found came from the model's own memory, not from the retriever.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.compare import format_budget  # noqa: E402
from prooflens_prover.eval.premises import TACTIC_WORDS, load_corpus, resolve  # noqa: E402
from prooflens_prover.utils.io import read_jsonl  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: A whole proof that is one `exact`/`apply` naming something. The head must be an identifier, so a
#: term-building completion (`exact fun h => ...`, `exact ⟨0, by simp⟩`) does not match: those close
#: the goal by construction rather than by citing a premise that already proves it.
ONE_STEP = re.compile(r"^(?:exact|apply)\s+([A-Za-z_][A-Za-z0-9_.'!?₀-₉¹²³]*)\s*(.*)$", re.DOTALL)


def one_step_citation(
    tactics, exact: set[str], by_suffix: dict[str, set[str]]
) -> tuple[str, str] | None:
    """`(head identifier, full tactic)` if this proof is a single citation, else None."""
    steps = [s.strip() for s in (tactics or ()) if str(s).strip()]
    if len(steps) != 1:
        return None
    m = ONE_STEP.match(steps[0])
    if not m:
        return None
    head = m.group(1)
    # `exact fun hx => ...` builds a term; it does not cite a premise that already proves the goal.
    # It reaches here because `fun` happens to be the last component of a corpus name, so resolving
    # alone is not enough -- keyword heads have to go first. Without this the audit over-counted by
    # 4 problems and reported term construction as corpus overlap.
    if head in TACTIC_WORDS or not resolve(head, exact, by_suffix):
        return None
    return head, steps[0]


def read_run(run_dir: Path) -> tuple[str, str, dict[str, list[str]]]:
    """`(benchmark, arm label, {problem id: proof tactics})` over solved problems only."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config", {})
    n_candidates = cfg.get("n_candidates")
    arm = cfg.get("arm", "?")
    label = f"{arm}@{format_budget(n_candidates)}" if n_candidates else arm

    solved: dict[str, list[str]] = {}
    for row in read_jsonl(run_dir / "attempts.jsonl"):
        if row.get("proved"):
            solved[str(row["problem_id"])] = row.get("proof") or []
    return cfg.get("benchmark", "?"), label, solved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="repeat; runs of the same benchmark are compared against its control")
    ap.add_argument("--out", type=Path, help="write the full result as JSON")
    ap.add_argument("--show", type=int, default=5, help="example citations to print per arm")
    args = ap.parse_args()
    ensure_utf8_output()

    exact, by_suffix = load_corpus(args.corpus)
    print(f"corpus: {len(exact):,} premises\n")

    # benchmark -> arm -> {problem: (head, tactic)}
    found: dict[str, dict[str, dict[str, tuple[str, str]]]] = {}
    totals: dict[str, dict[str, int]] = {}
    solved_sets: dict[str, dict[str, set[str]]] = {}
    for run_dir in args.run:
        benchmark, arm, solved = read_run(run_dir)
        hits = {}
        for pid, tactics in solved.items():
            hit = one_step_citation(tactics, exact, by_suffix)
            if hit:
                hits[pid] = hit
        found.setdefault(benchmark, {})[arm] = hits
        totals.setdefault(benchmark, {})[arm] = len(solved)
        solved_sets.setdefault(benchmark, {})[arm] = set(solved)

    report: dict[str, dict] = {}
    for benchmark in sorted(found):
        arms = found[benchmark]
        # The control's whole solved set, not merely its one-step proofs. A problem the control
        # closed in three steps is not a problem retrieval won, however this arm closed it.
        control_solved = solved_sets[benchmark].get("none", set())
        print(f"=== {benchmark} ===")
        print(f"{'arm':<12} {'solved':>7} {'1-step corpus answer':>21} {'of solved':>10} "
              f"{'won by retrieval':>17} {'of those, 1-step':>17}")
        rows = {}
        for arm, hits in arms.items():
            n_solved = totals[benchmark][arm]
            frac = len(hits) / n_solved if n_solved else 0.0
            won = solved_sets[benchmark][arm] - control_solved
            won_one_step = sorted(set(hits) & won)
            print(f"{arm:<12} {n_solved:>7} {len(hits):>21} {frac:>9.1%} "
                  f"{len(won) if arm != 'none' else '-':>17} "
                  f"{len(won_one_step) if arm != 'none' else '-':>17}")
            rows[arm] = {
                "solved": n_solved,
                "one_step_corpus_answers": len(hits),
                "fraction_of_solved": round(frac, 4),
                "won_vs_control": len(won) if arm != "none" else None,
                "won_vs_control_by_one_step_citation": won_one_step if arm != "none" else None,
                "citations": {p: t for p, (_, t) in sorted(hits.items())},
            }
        report[benchmark] = rows

        for arm, hits in arms.items():
            if not hits:
                continue
            won = solved_sets[benchmark][arm] - control_solved
            print(f"\n  {arm}:")
            for pid, (_, tactic) in sorted(hits.items())[: args.show]:
                print(f"   {'*' if pid in won else ' '} {pid:>8}  {tactic[:78]}")
        print("\n  * = a problem the no-retrieval control did not solve at all\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
