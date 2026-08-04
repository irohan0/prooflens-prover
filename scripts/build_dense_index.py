#!/usr/bin/env python
"""Encode the Mathlib premise corpus with a ProofLens retriever checkpoint and write its index.

    # the arm we actually evaluate
    python scripts/build_dense_index.py --kind li \
        --checkpoint $HOME/scratch/prooflens/checkpoints/li_ft_novel_bm25 \
        --corpus data/premises/mathlib_v4160.jsonl \
        --out    data/index/li_ft_novel_bm25

    # same script builds the single-vector index if it is ever needed
    python scripts/build_dense_index.py --kind sv \
        --checkpoint $HOME/scratch/prooflens/checkpoints/sv_ft_novel_lr3e6 \
        --corpus data/premises/mathlib_v4160.jsonl --out data/index/sv_ft_novel_lr3e6

This is the **only** GPU step between here and running the late-interaction arm. Encoding ~276k
short statements through a ModernBERT-sized encoder is minutes, not hours; the run is dominated by
writing the index, not by the model.

## The corpus is not re-derived here

It is loaded from the same JSONL, with the same filters, that BM25 already indexed, and
`--assert-corpus-id` fails the build if the candidate set differs by even one premise. An arm that
ranks over a different corpus is a different experiment, not a different retriever.

## Storage

Tokens are stored **float16** (`--dtype`): 276k x ~64 tokens x 128 dims is ~4.5 GB at fp16 and ~9 GB
at fp32, and the index has to be resident alongside a Lean REPL. `_gather` casts the ~1000 gathered
candidates back to float32 for the matmul, so scoring precision is unaffected.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from prooflens_prover.data.premises import corpus_id, load_premise_corpus
from prooflens_prover.retrieval.dense import (
    DEFAULT_N_CANDIDATES,
    EncoderSpec,
    LateInteractionIndex,
    SingleVectorIndex,
    l2_normalise,
)
from prooflens_prover.retrieval.lean_text import premise_document
from prooflens_prover.utils.logging import get_logger

log = get_logger(__name__)

#: Sized from the predecessor's config, which encoded the same kind of text with the same model.
LI_QUERY_LENGTH = 256
LI_DOCUMENT_LENGTH = 300


def save_shard_atomically(path: Path, tokens: np.ndarray, lengths: np.ndarray) -> None:
    """Write a shard so a killed job can never leave a truncated file a later run would skip.

    **The temp name must end in `.npz`.** `np.savez` silently appends `.npz` to any filename that
    does not already have it, so an obvious `path.with_suffix('.npz.tmp')` produces
    `shard.npz.tmp.npz` on disk and the subsequent rename fails with `FileNotFoundError` — after the
    encoding work is already done. Hence `shard.tmp.npz`, which numpy leaves alone.

    `shard_NNNNN.tmp.npz` also cannot be mistaken for a finished `shard_NNNNN.npz` by the
    resume check, which tests that exact filename.
    """
    tmp = path.parent / f"{path.stem}.tmp.npz"
    np.savez(tmp, tokens=tokens, lengths=lengths)
    if not tmp.exists():  # numpy renamed it under us; fail loudly rather than silently skip a shard
        raise RuntimeError(f"expected np.savez to write {tmp}, but it does not exist")
    tmp.replace(path)


def encode_li(checkpoint: str, texts: list[str], batch_size: int, device: str | None,
              dtype: str, shard_dir: Path, shard_size: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode documents to ragged token embeddings; return `(tokens, offsets, pooled)`.

    Loading and `encode` arguments are copied from
    `prooflens/src/prooflens/retrievers/late_interaction.py` so this index is produced by the same
    procedure that produced the retrieval numbers we are building on.

    **Sharded and resumable.** Measured throughput on 16 CPU cores is ~7 premises/s, so the full
    corpus is ~11 hours; the first attempt was killed by an 8-hour walltime at 68% and lost all of
    it. Each shard is written as soon as it is encoded, and a rerun skips shards already on disk. A
    timeout now costs at most one shard, and successive submissions make progress.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(texts) + shard_size - 1) // shard_size
    model = None
    t0 = time.perf_counter()
    n_encoded_this_run = 0

    for s in range(n_shards):
        path = shard_dir / f"shard_{s:05d}.npz"
        if path.exists():
            continue
        lo, hi = s * shard_size, min((s + 1) * shard_size, len(texts))
        if model is None:                       # loaded lazily: a fully-cached rerun needs no model
            from pylate import models
            log.info("loading ColBERT checkpoint %s", checkpoint)
            model = models.ColBERT(
                model_name_or_path=checkpoint,
                query_length=LI_QUERY_LENGTH,
                document_length=LI_DOCUMENT_LENGTH,
                device=device,
            )

        blocks: list[np.ndarray] = []
        lengths = np.zeros(hi - lo, dtype=np.int64)
        for start in range(lo, hi, batch_size):
            chunk = texts[start:min(start + batch_size, hi)]
            embs = model.encode(
                chunk, is_query=False, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, output_value="token_embeddings",
                show_progress_bar=False,
            )
            if isinstance(embs, np.ndarray) and embs.ndim == 2:
                embs = [embs]
            for j, e in enumerate(embs):
                e = np.ascontiguousarray(e, dtype=np.float32)
                lengths[start - lo + j] = e.shape[0]
                blocks.append(e.astype(dtype, copy=False))
            n_encoded_this_run += len(chunk)

        save_shard_atomically(path, np.concatenate(blocks, axis=0), lengths)
        del blocks

        rate = n_encoded_this_run / max(time.perf_counter() - t0, 1e-9)
        remaining = len(texts) - hi
        log.info("shard %d/%d written (%d-%d) — %.1f/s this run, ~%.0f s for the remaining %d",
                 s + 1, n_shards, lo, hi, rate, remaining / max(rate, 1e-9), remaining)

    log.info("assembling %d shards", n_shards)
    block_list, len_list = [], []
    for s in range(n_shards):
        blob = np.load(shard_dir / f"shard_{s:05d}.npz", allow_pickle=False)
        block_list.append(blob["tokens"])
        len_list.append(blob["lengths"])
    lengths = np.concatenate(len_list)
    if len(lengths) != len(texts):
        raise SystemExit(
            f"shards cover {len(lengths)} premises but the corpus has {len(texts)} — "
            f"delete {shard_dir} and rebuild (a shard was written for a different corpus)"
        )
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = np.concatenate(block_list, axis=0)
    del block_list, len_list

    # Pooled vectors drive stage-1 candidate generation. Kept float32: at 276k x 128 that is only
    # ~140 MB, and it is touched on every single query.
    pooled = np.zeros((len(texts), tokens.shape[1]), dtype=np.float32)
    for i in range(len(texts)):
        pooled[i] = tokens[offsets[i]:offsets[i + 1]].astype(np.float32).mean(axis=0)
    return tokens, offsets, l2_normalise(pooled)


def encode_sv(checkpoint: str, texts: list[str], batch_size: int,
              device: str | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    from prooflens_prover.retrieval.dense import SV_MAX_SEQ_LENGTH

    model = SentenceTransformer(checkpoint, device=device)
    # Must match the query side in `dense.load_query_encoder`, and both must match the predecessor's
    # locked `max_length: 512`. Left unset, SentenceTransformer adopts the checkpoint's own config,
    # so premises and proof states could be truncated differently without any error.
    model.max_seq_length = SV_MAX_SEQ_LENGTH
    log.info("sv encoder max_seq_length=%d", model.max_seq_length)
    embs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    return np.ascontiguousarray(embs, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", required=True, choices=["li", "sv"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base-model", default="",
                    help="recorded in the index metadata; defaults per --kind")
    ap.add_argument("--seed-tag", default="",
                    help="e.g. s1 — which of the 5 training seeds this checkpoint is")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu; default lets the library choose")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    ap.add_argument("--shard-size", type=int, default=10_000,
                    help="premises per resumable shard (li only). Smaller = less lost to a "
                         "walltime kill, at the cost of more files")
    ap.add_argument("--limit", type=int, default=None, help="first N premises — smoke runs only")
    ap.add_argument("--assert-corpus-id", default=None,
                    help="fail unless the filtered corpus has this exact id (from a BM25 "
                         "build_report.json). This is what guarantees the arms are comparable.")
    # Filters must match the BM25 build or the corpus_id assertion will fire.
    ap.add_argument("--props-only", action="store_true")
    ap.add_argument("--module-prefix", action="append", default=None)
    ap.add_argument("--max-statement-chars", type=int, default=4000)
    args = ap.parse_args()

    records = load_premise_corpus(
        args.corpus,
        props_only=args.props_only,
        module_prefixes=args.module_prefix,
        max_statement_chars=args.max_statement_chars,
    )
    if args.limit:
        records = records[:args.limit]
    cid = corpus_id(records)
    log.info("corpus: %d premises, corpus_id=%s", len(records), cid)

    if args.assert_corpus_id and cid != args.assert_corpus_id:
        raise SystemExit(
            f"CORPUS MISMATCH\n  expected {args.assert_corpus_id}\n  got      {cid}\n"
            "The arms would be ranking different candidate sets, so any difference between them "
            "could not be attributed to the retriever. Check --props-only / --module-prefix / "
            "--max-statement-chars against the BM25 build_report.json."
        )

    texts = [premise_document(r.name, r.statement) for r in records]
    base = args.base_model or (
        "lightonai/GTE-ModernColBERT-v1" if args.kind == "li"
        else "Alibaba-NLP/gte-modernbert-base"
    )

    t0 = time.perf_counter()
    if args.kind == "li":
        # Shards live beside the index so a resumed run finds them without extra arguments.
        tokens, offsets, pooled = encode_li(
            args.checkpoint, texts, args.batch_size, args.device, args.dtype,
            shard_dir=args.out / "_shards", shard_size=args.shard_size,
        )
        spec = EncoderSpec(kind="li", checkpoint=args.checkpoint, base_model=base,
                           dim=int(tokens.shape[1]), seed_tag=args.seed_tag)
        index = LateInteractionIndex(records, tokens, offsets, pooled, spec, args.n_candidates)
        log.info("encoded %d premises -> %d token vectors (mean %.1f tokens/premise)",
                 len(records), tokens.shape[0], tokens.shape[0] / max(len(records), 1))
    else:
        embs = encode_sv(args.checkpoint, texts, args.batch_size, args.device)
        spec = EncoderSpec(kind="sv", checkpoint=args.checkpoint, base_model=base,
                           dim=int(embs.shape[1]), seed_tag=args.seed_tag)
        index = SingleVectorIndex(records, embs, spec)
    log.info("encoding took %.1f s", time.perf_counter() - t0)

    index.save(args.out)
    size_mb = sum(p.stat().st_size for p in args.out.rglob("*")) / 1e6
    log.info("saved %s (%.1f MB)", args.out, size_mb)

    # For LI, measure what the two-stage approximation costs. Reported now rather than assumed,
    # because it has to appear beside every LI result.
    if args.kind == "li" and len(records) > args.n_candidates:
        rng = np.random.default_rng(0)
        probes = [
            index.doc_tokens(int(i)).astype(np.float32)
            for i in rng.choice(len(records), size=min(25, len(records)), replace=False)
        ]
        recall = index.recall_at_k_vs_exact(probes, k=10)
        log.info("two-stage recall@10 vs exact MaxSim (25 probes): %.3f", recall)
        print(f"\nAPPROXIMATION recall@10 = {recall:.3f} "
              f"(n_candidates={args.n_candidates} of {len(records)})")

    print(f"\nINDEX OK — kind={args.kind} n={len(records)} size={size_mb:.1f} MB")
    print(f"  corpus_id : {cid}")
    print(f"  out       : {args.out}")


if __name__ == "__main__":
    main()
