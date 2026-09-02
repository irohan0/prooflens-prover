"""Every number a figure draws, loaded once from the same records the tables are built from.

Nothing here recomputes statistics a committed script already produces. `results/tables/*.json` is
the output of `build_table1.py`, `compare_arms.py` and friends; the pass@k quantities come from the
exported run records through `eval/draws.py`, so a figure and the sentence beside it cannot drift.

The one thing this adds is the *shape* the plots need: per-seed solved sets keyed by (arm, seed),
which no table stores because no table needs them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from budget_matched import passk  # noqa: E402
from passk_profile import BENCHMARKS, arm_of  # noqa: E402
from passk_union import discover  # noqa: E402
from prooflens_prover.eval.draws import load_draw  # noqa: E402
from prooflens_prover.utils.io import read_jsonl  # noqa: E402

SWEEP_MATCH = ["search.samples_per_step=32", "policy_config.premise_free_fraction=0.25"]
N_PROBLEMS = dict(BENCHMARKS)

#: REAL-Prover-v1, Pass@64x64: 64 passes of MAX_NODES=1024 x NUM_SAMPLES=64.
REALPROVER = {"proofnet_test": 23.7, "fate_m": 56.7}
REALPROVER_NO_RETRIEVAL = {"proofnet_test": 22.6, "fate_m": 44.7}
REPROVER = {"proofnet_test": 13.8}
REALPROVER_GENERATIONS = 4_194_304


# =================================================================================================
# Predecessor study (ProofLens premise selection), transcribed from `prooflens_results.md`
# =================================================================================================
# These are NOT recomputable from this repository: they were produced by the predecessor project's
# own frozen harness (`prooflens/src/prooflens/eval/`) on LeanDojo Benchmark 4, and are transcribed
# here so the figures that motivate this study can be drawn beside its own. Every value is quoted
# with its section in `prooflens_results.md`, which is the source of record.

#: §2, matched control, 5 seeds (42, 1, 2, 3, 4). R@10 mean +/- sample std, percent.
PL_CROSSOVER = {
    #  system              random          novel
    "FT-SV (control)": ((32.29, 0.53), (26.51, 0.68)),
    "FT-LI (no weighting)": ((27.36, 0.25), (27.27, 0.13)),
    "FT-LI (IDF)": ((28.10, 0.31), (29.05, 0.27)),
}

#: §2, per-seed paired test on `novel`: R@10 delta (LI-IDF - SV) and its permutation p.
PL_PER_SEED = [(42, 2.76, 0.0002), (1, 2.78, 0.0001), (2, 2.35, 0.0001),
               (3, 1.45, 0.0210), (4, 3.38, 0.0001)]

#: §4, novel examples split by whether the gold lemma's name appears in the proof state.
#: (label, n, LI R@10, SV R@10, delta, ci_low, ci_high)
PL_STRATIFIED = [
    ("lexical\n(gold name in state)", 1158, 30.6, 24.1, 6.5, 4.4, 8.5),
    ("structural\n(name not in state)", 3199, 27.7, 26.9, 0.8, -0.6, 2.2),
]

#: §5, cross-split leakage in the public dense checkpoint. (label, n, R@10, ci_low, ci_high)
PL_LEAKAGE = [("leaked\n(theorem in random-train)", 4284, 64.12, None, None),
              ("clean\n(never trained on)", 73, 37.04, 27.5, 46.6)]
PL_LEAKAGE_PUBLISHED = 27.6

#: §7, offline downstream evaluation with ReProver's generator held fixed. premise_name@8, percent.
PL_DOWNSTREAM = {
    "novel_premises": {"none": 23.98, "BM25": 30.80, "FT-LI": 35.94, "FT-SV": 35.80},
    "random": {"none": 19.89, "BM25": 26.36, "FT-LI": 33.26, "FT-SV": 33.72},
}
#: §7, match@8 (the strict lower bound) for the same conditions.
PL_DOWNSTREAM_MATCH = {
    "novel_premises": {"none": 2.50, "BM25": 4.73, "FT-LI": 6.11, "FT-SV": 5.81},
    "random": {"none": 1.81, "BM25": 3.91, "FT-LI": 5.51, "FT-SV": 5.59},
}


def table(name: str) -> dict:
    return json.loads((REPO / "results" / "tables" / f"{name}.json").read_text(encoding="utf-8"))


def sweep_draws(results_root: Path, bench: str, arms=("li", "sv", "fusion")) -> dict:
    """`{(arm, seed): solved set}` for one benchmark at the sweep configuration.

    Rejected proofs are already discounted: `load_draw` reads `verification.json` and drops them.
    A run without that file is reported by `unverified()` rather than silently trusted.
    """
    found = discover(results_root, bench, "vllm", [*SWEEP_MATCH, f"n_problems={N_PROBLEMS[bench]}"])
    out = {}
    for d in found:
        draw = load_draw(d)
        arm = arm_of(draw)
        if arm not in arms:
            continue
        out[(arm, draw.seed)] = set(draw.solved)
    return out


def unverified(results_root: Path) -> list[str]:
    names = []
    for bench in N_PROBLEMS:
        for d in discover(results_root, bench, "vllm",
                          [*SWEEP_MATCH, f"n_problems={N_PROBLEMS[bench]}"]):
            if not (d / "verification.json").exists():
                names.append(d.name)
    return sorted(names)


def status_counts(results_root: Path, bench: str, arm: str,
                  seeds: Iterable[int] | None = None) -> Counter:
    """Terminal statuses pooled over the requested seeds of one arm at the sweep configuration.

    `seeds` defaults to all eight of the sweep.
    """
    want = set(range(8) if seeds is None else seeds)
    c: Counter = Counter()
    for d in discover(results_root, bench, "vllm",
                      [*SWEEP_MATCH, f"n_problems={N_PROBLEMS[bench]}"]):
        draw = load_draw(d)
        if arm_of(draw) != arm or draw.seed not in want:
            continue
        c.update(r.get("status") for r in read_jsonl(d / "attempts.jsonl"))
    return c


def tier1_status_counts(results_root: Path, bench: str, arm: str) -> Counter:
    """The same, for the single-draw Tier 1 run at 64x16 named in `table1.json`.

    Read by run id rather than by filter: the export holds replicates of these runs, and unioning
    them would turn a published single-seed figure into a multi-seed one.
    """
    run_id = table("table1")["runs"][f"{bench}/{arm}"]
    c: Counter = Counter()
    c.update(r.get("status") for r in read_jsonl(results_root / run_id / "attempts.jsonl"))
    return c


def curve(draws: list[set[str]], ids: list[str], k: int) -> float:
    """Expected problems solved by k draws, summed over problems."""
    return sum(passk(draws, p, k) for p in ids)


def problem_ids(results_root: Path, bench: str) -> list[str]:
    found = discover(results_root, bench, "vllm",
                     [*SWEEP_MATCH, f"n_problems={N_PROBLEMS[bench]}"])
    ids: set[str] = set()
    for d in found:
        ids |= load_draw(d).attempted
    return sorted(ids)
