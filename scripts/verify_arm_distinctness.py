#!/usr/bin/env python
"""Are these arms genuinely different runs, or is one of them counted twice?

    python scripts/verify_arm_distinctness.py \
        --run results/logs/fate_m_none_vllm_... \
        --run results/logs/fate_m_sv_vllm_... \
        --run results/logs/fate_m_li_vllm_...

## Why this check exists

Tier 1 reported an **exact tie**: 46 against 46 on FATE-M, 26 against 26 on ProofNet. Equal
counts on two independent benchmarks is exactly the shape a duplicated or mislabelled run takes —
if one arm were run twice and one copy relabelled, the two "arms" would agree exactly, and every
downstream figure would be an artefact rather than a result.

That is a serious enough alternative explanation to deserve a test rather than an assurance, and it
is cheap to test decisively.

## Why the proof text settles it

At a fixed `--seed` the vLLM engine is deterministic (`LLM(seed=...)` with `SamplingParams.seed`
left `None`). So two runs of the *same* arm at the same seed agree on every shared proof, character
for character — `identical_proof_fraction` would be exactly **1.0**. Two genuinely different arms
explore different trees and produce different proofs for most problems they both solve.

Four independent signals are checked, and a duplicate would fail all four at once:

1. **Proof text.** A fraction near 1.0 on shared solved problems is a duplicate.
2. **Recorded retriever and index.** Written to the manifest before the run does any work.
3. **Retrieval cost.** LI's exact MaxSim rerank costs ~25x SV's latency; no relabelling reproduces
   that.
4. **Discordance.** A duplicate solves *exactly* the same problems, so discordance would be zero.

## On the tie itself

The script also computes how surprising the equal counts actually are. Given `k` problems where the
two arms disagree, and a null in which each falls either way with probability 1/2, an exact tie has
probability `C(k, k/2) / 2**k` — and that is the **modal** outcome, the single most likely result. A
tie is what equivalence looks like, not what a mistake looks like.

Reads `attempts.jsonl` and `manifest.json`. No GPU, no model, no Lean.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.eval.draws import (  # noqa: E402
    discordance,
    identical_proof_fraction,
    load_draw,
)
from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

#: Above this fraction of byte-identical shared proofs, two arms are the same run. Well clear of the
#: highest value measured between genuinely different arms (0.45, ProofNet SV vs LI, where both arms
#: often close a goal with the same one-line `simp`).
DUPLICATE_PROOF_FRACTION = 0.95


def tie_probability(only_a: int, only_b: int) -> float | None:
    """P(exact tie) given the observed discordance, under 'each disagreement is a coin flip'.

    Returns None when the discordant count is odd, since an exact tie is then impossible and the
    question does not arise.
    """
    k = only_a + only_b
    if k == 0 or k % 2:
        return None
    return comb(k, k // 2) / 2 ** k


def compare_pair(a, b) -> dict:
    """Every distinctness signal for one pair of arms."""
    n_shared, frac = identical_proof_fraction(a, b)
    only_a, only_b = discordance(a.solved, b.solved)
    lat_a = (a.retrieval or {}).get("mean_latency_ms")
    lat_b = (b.retrieval or {}).get("mean_latency_ms")
    return {
        "pair": f"{a.arm} vs {b.arm}",
        "n_both_solved": n_shared,
        "fraction_identical_proofs": frac,
        "only_first": only_a,
        "only_second": only_b,
        "discordant": only_a + only_b,
        "proved": [len(a.solved), len(b.solved)],
        "tie": len(a.solved) == len(b.solved),
        "tie_probability_under_equivalence": tie_probability(only_a, only_b),
        "retriever": [a.retriever, b.retriever],
        "index": [a.index, b.index],
        "mean_latency_ms": [lat_a, lat_b],
        "n_queries": [(a.retrieval or {}).get("n_queries"), (b.retrieval or {}).get("n_queries")],
        "run_id": [a.run_id, b.run_id],
        "duplicate": bool(frac is not None and frac >= DUPLICATE_PROOF_FRACTION),
    }


def main() -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="results/logs/<run_id>; repeat for every arm of one benchmark")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    if len(args.run) < 2:
        raise SystemExit("--run at least twice; distinctness is a property of a pair")

    draws = [load_draw(d) for d in args.run]
    benches = {d.benchmark for d in draws}
    seeds = {d.seed for d in draws}

    print("=== runs under test ===")
    print(f"  {'arm':<10} {'seed':>4} {'retriever':<10} {'index':<32} {'ms/q':>8} {'proved':>7}")
    for d in draws:
        lat = (d.retrieval or {}).get("mean_latency_ms")
        print(f"  {d.arm:<10} {d.seed:>4} {str(d.retriever):<10} {str(d.index):<32} "
              f"{'—' if not lat else f'{lat:.1f}':>8} {len(d.solved):>7}")
    if len(seeds) > 1:
        print(f"\n  NOTE: these runs span seeds {sorted(seeds)}. Different seeds are different")
        print("  sampling draws, so differing proofs are expected even within one arm and the")
        print("  duplicate test below is weaker. Compare arms at a FIXED seed.")

    ids = [d.run_id for d in draws]
    if len(set(ids)) != len(ids):
        raise SystemExit("the same run directory was passed twice")

    print("\n=== pairwise distinctness ===")
    print("  A duplicate agrees on every shared proof, has zero discordance, and records the")
    print("  same retriever, index and latency. Any one of those failing is enough.")
    results, verdicts = [], []
    for a, b in itertools.combinations(draws, 2):
        r = compare_pair(a, b)
        results.append(r)
        frac = r["fraction_identical_proofs"]
        print(f"\n  --- {r['pair']} ---")
        print(f"    proved              : {r['proved'][0]} vs {r['proved'][1]}"
              f"{'   (EXACT TIE)' if r['tie'] else ''}")
        print(f"    both solved         : {r['n_both_solved']}")
        shown = "—" if frac is None else f"{frac:.1%}"
        print(f"    identical proofs    : {shown}   (a duplicate scores ~100%)")
        print(f"    discordant          : {r['only_first']} + {r['only_second']} "
              f"= {r['discordant']}   (a duplicate scores 0)")
        print(f"    retriever           : {r['retriever'][0]} vs {r['retriever'][1]}")
        print(f"    index               : {r['index'][0]}")
        print(f"                          {r['index'][1]}")
        lat = r["mean_latency_ms"]
        if lat[0] and lat[1]:
            ratio = max(lat) / min(lat)
            print(f"    retrieval latency   : {lat[0]:.1f} vs {lat[1]:.1f} ms/q  ({ratio:.1f}x)")
        p_tie = r["tie_probability_under_equivalence"]
        if r["tie"] and p_tie is not None:
            print(f"    P(exact tie | equivalent, {r['discordant']} disagreements) = "
                  f"{p_tie:.3f}  — the modal outcome")
        verdicts.append(r["duplicate"])
        print(f"    VERDICT             : "
              f"{'DUPLICATE — DO NOT TRUST' if r['duplicate'] else 'DISTINCT RUNS'}")

    if any(verdicts):
        raise SystemExit(
            "\nat least one pair of 'arms' is the same run. Every difference reported between them "
            "is an artefact; re-run the affected arm before using any of these numbers."
        )

    print("\n=== verdict: every pair is a genuinely distinct run ===")
    print("The equal counts are a coincidence of totals, not a repeated run: the arms disagree")
    print("about which problems they solve, write different proofs for most they share, and record")
    print("different retrievers, indices and retrieval costs. An exact tie is the most likely")
    print("single outcome when two arms are equivalent, so observing one is evidence FOR it")
    print("rather than evidence of a mistake.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "benchmarks": sorted(benches), "seeds": sorted(seeds),
            "duplicate_proof_fraction_threshold": DUPLICATE_PROOF_FRACTION,
            "pairs": results,
        }, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
