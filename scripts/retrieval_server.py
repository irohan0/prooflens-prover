#!/usr/bin/env python
"""Serve premise retrieval over stdin/stdout, so the prover can live in a different environment.

    python scripts/retrieval_server.py --arm li --index data/index/li_ft_novel_bm25 \
        --n-candidates 50000

Not run by hand — `prooflens_prover.retrieval.subprocess_client` launches it. Run it directly only
to check whether an environment can serve an arm, which is useful to know without a GPU allocation.

## Why this exists

    vllm 0.26.0   requires transformers>=5.5.3
    pylate 1.6.0  requires transformers<=5.3.0

Disjoint. The prover needs vLLM, the li arm's query encoder needs pylate, and no version of
transformers satisfies both. So this runs under the **retrieval** virtualenv — the one that produced
every verified Track A' result — and the prover keeps vLLM's.

That separation buys more than a working install: the LLM arm queries bit-identically the same
retriever as the model-free arm, so a Track A'-to-Tier 1 difference cannot be a difference in
retrieval.

## Protocol

One JSON object per line in, one per line out. **stdout carries protocol only** — every log line
goes to stderr, which the parent leaves attached so it lands in the SLURM log. A stray `print` here
would be read as a response and desynchronise the stream for the rest of the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.retrieval.base import RetrievalStats  # noqa: E402
from prooflens_prover.utils.logging import ensure_utf8_output, get_logger  # noqa: E402

log = get_logger(__name__)


def build(arm: str, index_dir: Path | None, n_candidates: int | None,
          checkpoint: str | None, device: str | None, stats: RetrievalStats):
    """The same construction `prove_benchmark.build_retriever` performs, minus the CLI."""
    if arm == "none":
        from prooflens_prover.retrieval.base import NullRetriever

        return NullRetriever()
    if arm == "bm25":
        from prooflens_prover.retrieval.bm25 import load_bm25_retriever

        return load_bm25_retriever(index_dir, stats=stats)

    from prooflens_prover.retrieval.dense import load_retriever

    r = load_retriever(arm, index_dir, checkpoint=checkpoint, device=device, stats=stats)
    if n_candidates is not None:
        if arm != "li":
            raise SystemExit(f"--n-candidates applies to the li arm only, not {arm!r}")
        log.info("first-stage budget: %d -> %d", r.index.n_candidates, n_candidates)
        r.index.n_candidates = n_candidates
    return r


def main() -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["none", "bm25", "li", "sv"])
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--n-candidates", type=int, default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    stats = RetrievalStats()
    t0 = time.perf_counter()
    retriever = build(args.arm, args.index, args.n_candidates, args.checkpoint, args.device, stats)
    log.info("loaded arm=%s in %.1fs", args.arm, time.perf_counter() - t0)

    index = getattr(retriever, "index", None)
    encoder = getattr(index, "encoder", None)

    def reply(obj: dict) -> None:
        # The only writer to stdout in this process. Everything else logs to stderr.
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            reply({"error": f"malformed request: {exc}"})
            continue

        cmd = request.get("cmd")
        try:
            if cmd == "hello":
                reply({
                    "arm": args.arm,
                    "corpus_id": getattr(index, "corpus_id", None),
                    "n_docs": getattr(index, "n_docs", 0),
                    "n_candidates": getattr(index, "n_candidates", None),
                    "encoder": encoder.to_dict() if hasattr(encoder, "to_dict") else None,
                })
            elif cmd == "names":
                reply({"names": [r.name for r in getattr(index, "records", [])]})
            elif cmd == "stats":
                reply({"stats": stats.to_dict()})
            elif cmd == "shutdown":
                reply({"ok": True})
                log.info("shutting down; %s", stats.to_dict())
                return 0
            else:
                premises = retriever.retrieve(request["query"], k=int(request.get("k", 10)))
                reply({"premises": [
                    {"formal_name": p.formal_name, "formal_statement": p.formal_statement,
                     "informal_name": p.informal_name, "score": p.score}
                    for p in premises
                ]})
        except Exception as exc:                  # noqa: BLE001
            # Reported over the protocol AND logged. Dying here would strand the parent on a read
            # with no explanation; the parent raises on `{"error": ...}` and its traceback says why.
            log.exception("request failed: %s", request)
            reply({"error": f"{type(exc).__name__}: {exc}"})

    log.info("stdin closed; %s", stats.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
