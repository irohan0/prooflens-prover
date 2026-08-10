#!/usr/bin/env python
"""Assemble Table 1 (Track A') from every run in `results/logs`, with significance.

    python scripts/build_table1.py

Discovers runs rather than taking run ids on the command line: pasting run ids by hand is how a
`none` row gets compared against the wrong benchmark's `li` row, and the resulting table looks
entirely plausible. For each `(benchmark, arm)` the most recent **finalised** run wins — a run whose
manifest has no `outcome` did not complete and is ignored.

Prints a markdown table plus the per-benchmark paired analysis, and writes
`results/tables/table1.md` and `table1.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importable from a fresh clone with no install and no exported PYTHONPATH: a login-node
# `python scripts/<this>.py` must work, because that is how the analysis scripts get run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.compare import (
    Arm,
    compare,
    format_budget,
    format_report,
    oracle_union,
)
from prooflens_prover.utils.logging import ensure_utf8_output

#: Published reference numbers, for context in the table. These are NOT our results and are not
#: comparable to a model-free policy — see the caption emitted below.
PUBLISHED: dict[str, dict[str, str]] = {
    "fate_m": {
        "REAL-Prover-v1 (7B, LeanSearch-PS)": "56.7",
        "REAL-Prover-v1 (no retrieval)": "44.7",
    },
    "proofnet_test": {
        "REAL-Prover-v1 (7B, LeanSearch-PS)": "23.7",
        "REAL-Prover-v1 (no retrieval)": "22.6",
        "ReProver (<1B)": "13.8",
    },
    "minif2f_test": {"REAL-Prover-v1 (7B)": "54.1"},
}

BENCHMARK_ORDER = ["fate_m", "proofnet_test", "minif2f_test", "putnambench"]



def discover(results_root: Path, policy: str = "repertoire") -> dict[tuple[str, str], Path]:
    """Most recent finalised run per (benchmark, arm), restricted to one policy.

    The policy filter is not a convenience. A 7B language model and a 19-tactic repertoire are
    different systems whose pass rates are not comparable, and both produce runs for `arm=li` on
    `fate_m`. Keyed on (benchmark, arm) alone, the newer of the two would silently take the cell and
    the table would mix them — with a caption claiming the generator was held fixed.

    Runs from before `policy_kind` existed are all `repertoire`; that is what they were.
    """
    best: dict[tuple[str, str], tuple[str, Path]] = {}
    for d in sorted(results_root.iterdir()):
        mf = d / "manifest.json"
        if not d.is_dir() or not mf.exists() or not (d / "attempts.jsonl").exists():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        if not m.get("outcome"):
            continue                      # never finished; its counts would be partial
        cfg = m.get("config", {})
        if cfg.get("policy_kind", "repertoire") != policy:
            continue
        key = (cfg.get("benchmark", "?"), cfg.get("arm", "?"))
        started = m.get("started_utc", "")
        if key not in best or started > best[key][0]:
            best[key] = (started, d)
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=Path("results/logs"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="repertoire",
                    help="which generator's runs to tabulate: repertoire (Track A') or vllm "
                         "(Tier 1). Mixing them in one table would be meaningless.")
    args = ap.parse_args()

    runs = discover(args.results_root, args.policy)
    if not runs:
        raise SystemExit(
            f"no finalised --policy {args.policy!r} runs under {args.results_root}"
        )

    benchmarks = sorted(
        {b for b, _ in runs},
        key=lambda b: BENCHMARK_ORDER.index(b) if b in BENCHMARK_ORDER else 99,
    )
    print(f"discovered runs (policy={args.policy}):")
    for (b, a), d in sorted(runs.items()):
        print(f"  {b:<16} {a:<6} {d.name}")
    print()

    #: Each comparison is (baseline arm, treatment arm). `li vs none` and `sv vs none` measure
    #: whether retrieval helps at all; `li vs sv` is the architecture claim the dissertation
    #: is named after, and it is the one that needs the SV arm to exist.
    #:
    #: `sv vs none` earns its place: on miniF2F, SV solves 77 against the control's 78. A raw count
    #: one below the floor could be two problems gained and three lost, and only the paired
    #: comparison distinguishes that from a single displacement. Displacement is the confound this
    #: harness has already been bitten by (`none` 10/30 vs `bm25` 6/30, bm25's set a strict subset).
    PAIRS = [("none", "sv"), ("none", "li"), ("sv", "li")]

    unions: dict[str, dict] = {}

    def union_cell(bench: str) -> str:
        if (bench, "sv") not in runs or (bench, "li") not in runs:
            return "—"
        u = oracle_union(Arm.load(runs[(bench, "sv")]), Arm.load(runs[(bench, "li")]))
        unions[bench] = u
        return (f"{u['n_union']}/{u['n_problems']} ({100 * u['union_rate']:.1f}%) "
                f"[+{u['gain_over_best']}]")

    rows: list[str] = []
    comparisons: dict[str, dict] = {}
    li_budgets: dict[str, int | None] = {}
    for b in benchmarks:
        cells = []
        for arm in ("none", "sv", "li"):
            d = runs.get((b, arm))
            if d is None:
                cells.append("—")
                continue
            a = Arm.load(d)
            n = len(a.proved)
            cell = f"{a.n_proved}/{n} ({100 * a.n_proved / n:.1f}%)"
            if arm == "li":
                # An unlabelled li number is ambiguous between two experiments that differ by more
                # than any effect this table reports: at n_candidates=1,000 the pooled first stage
                # kept 0.443 of its own exact top-10, at 50,000 it kept 0.979. `discover` keeps the
                # newest run per (benchmark, arm), so without this the budget silently changes
                # underneath the column.
                li_budgets[b] = a.n_candidates
                cell += f" @{format_budget(a.n_candidates)}"
            cells.append(cell)

        deltas = []
        for base_arm, treat_arm in PAIRS:
            if (b, base_arm) not in runs or (b, treat_arm) not in runs:
                deltas.append("—")
                continue
            r = compare(
                Arm.load(runs[(b, base_arm)]), Arm.load(runs[(b, treat_arm)]),
                args.n_boot, args.n_perm, args.seed,
            )
            comparisons[f"{b}:{treat_arm}_vs_{base_arm}"] = r
            star = "**" if r["significant"] else ""
            deltas.append(
                f"{star}{r['delta_problems']:+d}{star} (p={r['p_mcnemar_exact']:.4f})"
            )
            print(format_report(r))
            print()
        rows.append(
            f"| {b} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{deltas[0]} | {deltas[1]} | {deltas[2]} | {union_cell(b)} |"
        )

    # A single "ProofLens-LI" column implies one system. If the benchmarks ran at different
    # first-stage budgets that is false, and any sentence beginning "LI achieves..." is then a
    # statement about a mixture. Loud, but not fatal: the per-cell annotation still says which.
    if len(set(li_budgets.values())) > 1:
        print("!! the LI column is not one system — first-stage budgets differ across benchmarks:")
        for b, n in sorted(li_budgets.items()):
            print(f"     {b:<16} n_candidates={format_budget(n)}")
        print("   Name the budget in any cross-benchmark claim, or re-run the odd ones out.\n")

    #: Both halves of the caption depend on the policy, and getting either wrong misreports the
    #: table. The model-free rows genuinely are not comparable to published 7B systems; the Tier 1
    #: rows are comparable in kind and wildly incomparable in *budget*, which is the thing a reader
    #: will otherwise get backwards when they see 32.6% next to a published 56.7%.
    if args.policy == "repertoire":
        title = "Table 1 — retrieval architecture, model-free policy (Track A')"
        detail_extra = ", and premise-needed rates"
        comparability = [
            "These come from systems built on 7B-class fine-tuned language models. The rows",
            "above use a **model-free** tactic policy with no language model at all, so the",
            "rates are not comparable. What Table 1 measures is the *effect of retrieval* with",
            "the generator held fixed; the comparable-to-published experiment is Tier 1",
            "(frozen REAL-Prover-v1).",
        ]
    else:
        title = "Table 1 — retrieval architecture, frozen REAL-Prover-v1 (7B) (Tier 1)"
        detail_extra = ""
        comparability = [
            "The rows above hold **REAL-Prover-v1 (7B) frozen** and vary only the retriever, so",
            "they are comparable in kind to these published numbers — but **not in budget**, and",
            "the gap is enormous. REAL-Prover's figures are Pass@64x64: 64 passes of 64 nodes x",
            "64 samples, about 4.2M generations per problem. Table 1 is a *single* pass at 64",
            "nodes x 16 samples — 1,024 generations, roughly **1/4,000** of their budget. Read",
            "any shortfall against 56.7 / 23.7 as a budget difference first and a system",
            "difference second.",
            "",
            "ReProver's ProofNet **13.8%** is the closest thing here to a like-for-like row: a",
            "single pass from a sub-1B model with single-vector retrieval.",
            "",
            "Premise attribution is **not reported** for these rows. 'Names a premise' is",
            "decidable from proof text only for the model-free repertoire, whose tactics are",
            "either a fixed closer or a premise template; every tactic a language model writes",
            "is 'not a closer', so the same test would mark all of them and report a rate of",
            "100% regardless of what retrieval contributed. See",
            "`eval/compare.PREMISE_ATTRIBUTABLE_POLICIES`.",
        ]

    lines = [
        f"# {title}",
        "",
        "| benchmark | none (floor) | ProofLens-SV | ProofLens-LI | Δ SV vs none | Δ LI vs none "
        "| Δ LI vs SV | SV ∪ LI (oracle) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
        *rows,
        "",
        "**Δ vs none** is problems gained over the no-retrieval control; **Δ LI vs SV** is the",
        "architecture comparison — same premise corpus, same search budget, same checkpoint",
        "training protocol, only multi-vector against single-vector. Bold marks a result where the",
        "bootstrap CI and the sign-flip permutation test agree.",
        "",
        "**SV ∪ LI** is the oracle union: problems solved by *either* retriever, with the gain",
        "over the better single arm in brackets. No single retriever can reach it, and it is not a",
        "fusion result — it is the ceiling a fusion arm could reach. It sits above both arms only",
        "because they disagree about *which* problems they solve, not about how many.",
        "",
        "`@1k` / `@50k` on the LI cells is that run's **first-stage candidate budget**: LI",
        "generates candidates with a mean-pooled vector, then reranks them with exact MaxSim, so",
        "this is how many premises the exact stage ever sees. Measured recall@10 of that first",
        "stage against exact full-corpus MaxSim is **0.443 at 1k** and **0.979 at 50k**. SV needs",
        "no such annotation: it ranks all 276,070 premises exactly. An LI number at 1k is a lower",
        "bound on the architecture, not a measurement of it.",
        "",
        "Per-comparison detail — the exact problems each arm won, the displacement check"
        f"{detail_extra} — is in `table1.json`.",
        "",
        "## Published reference numbers",
        "",
        *comparability,
        "",
    ]
    for b in benchmarks:
        if b in PUBLISHED:
            lines.append(f"* **{b}** — " + "; ".join(
                f"{k}: {v}" for k, v in PUBLISHED[b].items()
            ))
    text = "\n".join(lines) + "\n"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table1.md").write_text(text, encoding="utf-8")
    (args.out_dir / "table1.json").write_text(
        json.dumps(
            {"runs": {f"{b}/{a}": d.name for (b, a), d in runs.items()},
             "li_n_candidates": li_budgets,
             "oracle_union": unions,
             "comparisons": comparisons},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(text)
    print(f"written: {args.out_dir / 'table1.md'} and table1.json")


if __name__ == "__main__":
    main()
