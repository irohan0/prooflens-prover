#!/usr/bin/env python
"""Build the BM25 index over the extracted Mathlib premise corpus.

    python scripts/build_bm25_index.py \
        --corpus data/premises/mathlib_v4160.jsonl \
        --out    data/index/bm25_mathlib_v4160

Also reports **query latency on the real corpus**, because the `none` / `bm25` / `sv` / `li` arms
must be compared at equal search budget, and a retriever whose per-query cost is material changes
how many proof states a fixed wall-clock budget can visit. Measuring it at index-build time means
the cost table is populated before any GPU hours are spent, not reconstructed afterwards.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Importable from a fresh clone with no install and no exported PYTHONPATH: a login-node
# `python scripts/<this>.py` must work, because that is how the analysis scripts get run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.data.premises import corpus_id, load_premise_corpus
from prooflens_prover.retrieval.base import PROMPT_PREMISE_LIMIT
from prooflens_prover.retrieval.bm25 import BM25Index, BM25Params, TokenizerOptions
from prooflens_prover.utils.logging import ensure_utf8_output, get_logger

log = get_logger(__name__)

# Proof-state-shaped probes for the latency measurement. Real states, not single words: query cost
# is proportional to the total posting-list length of the query's terms, so a one-word probe would
# understate it by an order of magnitude.
PROBE_QUERIES = [
    "⊢ ∀ (n m : ℕ), n + m = m + n",
    "G : Type u_1\ninst✝ : Group G\na b : G\n⊢ a * b * b⁻¹ = a",
    "R : Type u\ninst✝ : CommRing R\nI : Ideal R\n⊢ I.IsPrime ↔ I.IsMaximal",
    "f : ℝ → ℝ\nhf : Continuous f\n⊢ ∀ (s : Set ℝ), IsOpen s → IsOpen (f ⁻¹' s)",
    "s t : Finset α\n⊢ (s ∪ t).card + (s ∩ t).card = s.card + t.card",
    "x : ℂ\n⊢ Complex.exp (x + 2 * ↑Real.pi * Complex.I) = Complex.exp x",
]


def main() -> None:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--props-only", action="store_true",
                    help="index only Prop-valued declarations (drops defs/inductives)")
    ap.add_argument("--module-prefix", action="append", default=None,
                    help="repeatable; e.g. --module-prefix Mathlib to exclude Lean core internals")
    ap.add_argument("--max-statement-chars", type=int, default=4000)
    ap.add_argument("--split-underscores", action="store_true",
                    help="also index underscore-separated parts of identifiers")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--k1", type=float, default=BM25Params.k1)
    ap.add_argument("--b", type=float, default=BM25Params.b)
    args = ap.parse_args()

    log.info("loading corpus from %s", args.corpus)
    t0 = time.perf_counter()
    records = load_premise_corpus(
        args.corpus,
        props_only=args.props_only,
        module_prefixes=args.module_prefix,
        max_statement_chars=args.max_statement_chars,
    )
    log.info("loaded %d premises in %.1fs", len(records), time.perf_counter() - t0)
    if not records:
        raise SystemExit("corpus is empty after filtering — check --module-prefix / --props-only")

    kinds: dict[str, int] = {}
    for r in records:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    log.info("kinds: %s", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    log.info("corpus_id: %s", corpus_id(records))

    log.info("building index (k1=%.2f b=%.2f, tokenizer: lowercase=%s split_underscores=%s)",
             args.k1, args.b, args.lowercase, args.split_underscores)
    t0 = time.perf_counter()
    index = BM25Index.build(
        records,
        params=BM25Params(k1=args.k1, b=args.b),
        tokenizer=TokenizerOptions(
            lowercase=args.lowercase, split_underscores=args.split_underscores
        ),
    )
    build_s = time.perf_counter() - t0
    log.info("built in %.1fs — %d docs, %d terms, %d postings",
             build_s, index.n_docs, index.n_terms, index.doc_ids.size)

    index.save(args.out)
    size_mb = sum(p.stat().st_size for p in args.out.rglob("*")) / 1e6
    log.info("saved to %s (%.1f MB)", args.out, size_mb)

    # -- latency, measured on the real index -------------------------------------------------
    for q in PROBE_QUERIES:          # warm the page cache so the numbers reflect steady state
        index.topk(q, k=PROMPT_PREMISE_LIMIT)
    timings = []
    for q in PROBE_QUERIES * 5:
        t = time.perf_counter()
        index.topk(q, k=PROMPT_PREMISE_LIMIT)
        timings.append(1000 * (time.perf_counter() - t))
    log.info("query latency over %d probes: median %.1f ms, mean %.1f ms, max %.1f ms",
             len(timings), statistics.median(timings), statistics.mean(timings), max(timings))

    print("\n=== sample retrievals (top 3) ===")
    for q in PROBE_QUERIES[:3]:
        print(f"\nSTATE: {q.splitlines()[-1]}")
        for i, s in index.topk(q, k=3):
            r = index.records[i]
            print(f"  {s:6.2f}  {r.name}")
            print(f"          {r.statement.splitlines()[0][:110]}")

    (args.out / "build_report.json").write_text(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "corpus_id": corpus_id(records),
                "n_docs": index.n_docs,
                "n_terms": index.n_terms,
                "n_postings": int(index.doc_ids.size),
                "index_size_mb": round(size_mb, 1),
                "build_seconds": round(build_s, 1),
                "kinds": kinds,
                "query_latency_ms": {
                    "median": round(statistics.median(timings), 2),
                    "mean": round(statistics.mean(timings), 2),
                    "max": round(max(timings), 2),
                },
                "filters": {
                    "props_only": args.props_only,
                    "module_prefixes": args.module_prefix,
                    "max_statement_chars": args.max_statement_chars,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nINDEX OK — {index.n_docs} premises, {size_mb:.1f} MB, "
          f"{statistics.median(timings):.1f} ms/query")


if __name__ == "__main__":
    main()
