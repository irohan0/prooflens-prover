"""Dense retrieval arms: single-vector (SV) and late-interaction (LI) over the shared corpus.

Both index the identical premise set BM25 uses (asserted via `corpus_id`), so the three arms differ
only in how a proof state is matched against a premise.

## The scoring maths is pure numpy, on purpose

Encoding needs torch and a GPU; **scoring does not**. Embeddings are stored as numpy arrays and
every ranking function here is plain array maths, so the part that decides which premises reach
the model is unit-testable on a laptop with no model, no GPU and no downloads. This mirrors the
split that worked in `prooflens/src/prooflens/retrievers/token_idf.py` ("the pure part -- no
torch, no tokenizer"), and it is why the MaxSim implementation can be checked against a reference.

## Why LI needs two stages and SV does not

SV is one 768-d vector per premise: 276k x 768 fp16 = ~420 MB, and a query is a single matrix-vector
product.

LI is ~64 token vectors per premise: 276k x 64 x 128 = ~17.7M token vectors. Exact MaxSim forms
`sims = query_tokens @ all_token_vectors.T`, which for a 32-token query is 32 x 17.7M floats --
**~2.3 GB per query**, at every proof state, tens of thousands of times per benchmark. Not viable.

So LI retrieves in two stages:

1. **Candidate generation** over mean-pooled premise vectors (one 128-d vector per premise, ~70 MB).
   Cheap, and *self-contained*: pooling the premise's own token embeddings keeps the candidate set a
   function of the LI encoder alone.
2. **Exact MaxSim rerank** of the top `n_candidates` (default 1000) — the same arithmetic the
   reference implements, just restricted to ~0.4% of the corpus.

**Deliberately NOT using BM25 for stage 1.** It would be simpler, and it would silently make LI's
recall a function of BM25's -- so an LI-vs-BM25 comparison would partly be BM25 against itself. The
pooled-vector stage keeps the arms independent.

**The approximation must be measured, not assumed.** `recall_at_k_vs_exact` computes how often the
two-stage top-k matches exact full-corpus MaxSim on a sample of states. That number belongs in the
write-up next to any LI result, because a two-stage retriever that loses 5% of its true top-6 is a
different system from the one the predecessor study evaluated.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prooflens_prover.data.premises import PremiseRecord, corpus_id, iter_premise_records
from prooflens_prover.retrieval.base import DEFAULT_TOP_K, Premise, RetrievalStats

__all__ = [
    "LateInteractionIndex",
    "LateInteractionRetriever",
    "SingleVectorIndex",
    "SingleVectorRetriever",
    "maxsim_score",
]

#: Premises exactly rescored by MaxSim after pooled-vector candidate generation.
DEFAULT_N_CANDIDATES = 1000

#: ColBERT query length, from `configs/late_interaction_ft_novel.yaml` — the config for the exact
#: checkpoint this project indexes (`li_ft_novel_bm25`).
#:
#: **384, not 256.** An earlier version of this file used 256, copied from the predecessor's *base*
#: `configs/late_interaction.yaml` rather than the FT-novel config that its Phase 11 locked. That
#: phase measured the cost directly: **26.4% of proof states exceed 256 tokens** (mean 222, median
#: 163, p95 593, max 2326), and the sweep found
#:
#:     | query_length | R@1  | R@10  | MRR   |
#:     |--------------|-----:|------:|------:|
#:     | 256          | 3.83 | 11.66 | .1045 |
#:     | 384 (locked) | 4.23 | 11.72 | .1099 |
#:     | 512          | 4.23 | 11.72 | .1100 |
#:
#: R@1 is the metric that matters most to a prover — the top-ranked premise is the one that gets
#: the highest-prior template — so truncating a quarter of all queries handicapped LI on precisely
#: its strongest signal. 512 buys nothing over 384.
#:
#: Changing this does **not** invalidate the premise index: query length does not affect premise
#: encoding (predecessor `scripts/audit.py`, `run_sweep`). Only re-running the search is required.
LI_QUERY_LENGTH = 384

#: ColBERT document length, unchanged from the FT-novel config and from the index build.
#: **This one must match the index**, since it governs premise-side truncation.
LI_DOCUMENT_LENGTH = 300

#: Single-vector sequence length, from `configs/dense_sv_ft_novel_lr3e6.yaml` (`max_length: 512`),
#: which the predecessor applies as `model.max_seq_length`. Set explicitly on both the premise and
#: the query side: left unset, `SentenceTransformer` silently adopts whatever the saved checkpoint
#: config carries, which makes the arm's truncation behaviour an uncontrolled variable in a
#: comparison whose entire purpose is that only the retriever differs.
SV_MAX_SEQ_LENGTH = 512


def l2_normalise(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit-normalise so dot products are cosines. Zero rows are left at zero rather than NaN."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def maxsim_score(
    query_emb: np.ndarray, doc_emb: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Reference MaxSim (ColBERT late interaction): `sum_i w_i * max_j (q_i . d_j)`.

    The obvious un-vectorised definition, ported unchanged from
    `prooflens/src/prooflens/retrievers/late_interaction.py`. The batched `reduceat` path in
    `LateInteractionIndex` is unit-tested against it — that agreement is the only thing that makes
    the fast path trustworthy.
    """
    sims = query_emb @ doc_emb.T
    maxs = sims.max(axis=1)
    if weights is None:
        return float(maxs.sum())
    return float((weights * maxs).sum())


@dataclass(frozen=True)
class EncoderSpec:
    """Which checkpoint produced an index. Recorded so a run manifest names the exact arm."""

    kind: str            # "sv" | "li"
    checkpoint: str      # path or HF id of the fine-tuned checkpoint
    base_model: str      # e.g. Alibaba-NLP/gte-modernbert-base
    dim: int
    seed_tag: str = ""   # e.g. "s1" — which of the 5 training seeds

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind, "checkpoint": self.checkpoint,
            "base_model": self.base_model, "dim": self.dim, "seed_tag": self.seed_tag,
        }


def _write_corpus(records: list[PremiseRecord], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({
                "name": r.name, "kind": r.kind, "statement": r.statement,
                "module": r.module, "is_prop": r.is_prop,
            }, ensure_ascii=False) + "\n")


# =============================================================================================
# Single vector
# =============================================================================================
class SingleVectorIndex:
    """One L2-normalised vector per premise. Ranking is a single matrix-vector product."""

    def __init__(
        self, records: list[PremiseRecord], embeddings: np.ndarray, encoder: EncoderSpec
    ) -> None:
        if embeddings.shape[0] != len(records):
            raise ValueError(
                f"embeddings/corpus mismatch: {embeddings.shape[0]} vectors, "
                f"{len(records)} premises"
            )
        self.records = records
        self.embeddings = embeddings          # [N, dim], unit-norm, float32
        self.encoder = encoder

    @property
    def n_docs(self) -> int:
        return len(self.records)

    @property
    def corpus_id(self) -> str:
        return corpus_id(self.records)

    def topk(self, query_emb: np.ndarray, k: int = DEFAULT_TOP_K) -> list[tuple[int, float]]:
        """Top-`k` `(index, cosine)` for a single unit-norm query vector."""
        if k <= 0 or self.n_docs == 0:
            return []
        q = l2_normalise(np.asarray(query_emb, dtype=np.float32).reshape(-1))
        scores = self.embeddings @ q
        k = min(k, self.n_docs)
        cand = np.argpartition(-scores, k - 1)[:k] if k < self.n_docs else np.arange(self.n_docs)
        # Deterministic ordering including ties: score desc, then corpus index.
        cand = cand[np.lexsort((cand, -scores[cand]))]
        return [(int(i), float(scores[i])) for i in cand]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "index.npz", embeddings=self.embeddings)
        _write_corpus(self.records, d / "corpus.jsonl")
        (d / "meta.json").write_text(json.dumps({
            "kind": "sv", "n_docs": self.n_docs, "dim": int(self.embeddings.shape[1]),
            "corpus_id": self.corpus_id, "encoder": self.encoder.to_dict(),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> SingleVectorIndex:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        blob = np.load(d / "index.npz", allow_pickle=False)
        records = list(iter_premise_records(d / "corpus.jsonl"))
        if len(records) != meta["n_docs"]:
            raise ValueError(f"index/corpus mismatch in {d}")
        return cls(records, blob["embeddings"], EncoderSpec(**meta["encoder"]))


# =============================================================================================
# Late interaction
# =============================================================================================
class LateInteractionIndex:
    """Ragged per-token embeddings in compressed-sparse-row layout, plus pooled vectors.

    `tokens[offsets[i] : offsets[i+1]]` are premise `i`'s token vectors. Same CSR-over-ragged-rows
    structure as the predecessor's index, which is the right shape for this and already understood
    in the codebase.
    """

    def __init__(
        self,
        records: list[PremiseRecord],
        tokens: np.ndarray,          # [T, dim] unit-norm
        offsets: np.ndarray,         # [N+1] int64
        pooled: np.ndarray,          # [N, dim] unit-norm mean of each premise's tokens
        encoder: EncoderSpec,
        n_candidates: int = DEFAULT_N_CANDIDATES,
    ) -> None:
        if offsets.shape[0] != len(records) + 1:
            raise ValueError(
                f"offsets must have len(records)+1 entries: got {offsets.shape[0]} "
                f"for {len(records)} premises"
            )
        if int(offsets[-1]) != tokens.shape[0]:
            raise ValueError(
                f"offsets[-1]={int(offsets[-1])} does not match token count {tokens.shape[0]}"
            )
        self.records = records
        self.tokens = tokens
        self.offsets = offsets
        self.pooled = pooled
        self.encoder = encoder
        self.n_candidates = n_candidates

    @property
    def n_docs(self) -> int:
        return len(self.records)

    @property
    def corpus_id(self) -> str:
        return corpus_id(self.records)

    def doc_tokens(self, i: int) -> np.ndarray:
        return self.tokens[int(self.offsets[i]):int(self.offsets[i + 1])]

    def _gather(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate the token blocks of premises `idx` and return `(D, seg)` where `seg` holds
        the `reduceat` boundaries. Fully vectorised — no Python loop over candidates.

        The gathered block is cast to float32. Tokens are stored **float16** because the full index
        is 276k premises x ~64 tokens x 128 dims — 9 GB as float32, which does not fit alongside a
        Lean REPL, and 4.5 GB as float16 which does. Casting only the ~1000 gathered candidates
        keeps the matmul in float32 (fast, and accurate enough that scores are stable) while the
        resident index stays half-size.
        """
        starts = self.offsets[idx]
        lengths = self.offsets[idx + 1] - starts
        seg = np.zeros(len(idx), dtype=np.int64)
        if len(idx) > 1:
            np.cumsum(lengths[:-1], out=seg[1:])
        doc_of_pos = np.repeat(np.arange(len(idx)), lengths)
        intra = np.arange(int(lengths.sum()), dtype=np.int64) - seg[doc_of_pos]
        block = self.tokens[starts[doc_of_pos] + intra]
        return block.astype(np.float32, copy=False), seg

    def maxsim_over(
        self, query_emb: np.ndarray, idx: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Exact MaxSim of `query_emb` [n_q, dim] against the premises `idx`. Returns [len(idx)]."""
        if len(idx) == 0:
            return np.zeros(0, dtype=np.float32)
        d, seg = self._gather(idx)
        sims = query_emb @ d.T                                   # [n_q, T_cand]
        maxs = np.maximum.reduceat(sims, seg, axis=1)            # [n_q, len(idx)]
        w = np.ones(query_emb.shape[0], dtype=np.float32) if weights is None else weights
        return w @ maxs

    def topk(
        self,
        query_emb: np.ndarray,
        k: int = DEFAULT_TOP_K,
        weights: np.ndarray | None = None,
        n_candidates: int | None = None,
    ) -> list[tuple[int, float]]:
        """Two-stage top-`k`: pooled-vector candidates, then exact MaxSim rerank.

        Set `n_candidates >= n_docs` to force the exact full-corpus path (used by the approximation
        measurement, and viable on small corpora in tests).
        """
        if k <= 0 or self.n_docs == 0:
            return []
        q = l2_normalise(np.asarray(query_emb, dtype=np.float32))
        if q.ndim == 1:
            q = q[None, :]

        n_cand = self.n_candidates if n_candidates is None else n_candidates
        n_cand = max(k, min(n_cand, self.n_docs))
        if n_cand >= self.n_docs:
            cand = np.arange(self.n_docs)
        else:
            # Stage 1: sum of per-query-token similarity to each premise's pooled vector. Summing
            # (not maxing) keeps this a smooth proxy for the MaxSim sum it is standing in for.
            pooled_scores = (q @ self.pooled.T).sum(axis=0)
            cand = np.argpartition(-pooled_scores, n_cand - 1)[:n_cand]
            cand = np.sort(cand)     # ascending order keeps `_gather` reading memory in sequence

        scores = self.maxsim_over(q, cand, weights)
        k = min(k, len(cand))
        top = np.argpartition(-scores, k - 1)[:k] if k < len(cand) else np.arange(len(cand))
        top = top[np.lexsort((cand[top], -scores[top]))]
        return [(int(cand[j]), float(scores[j])) for j in top]

    def recall_at_k_vs_exact(
        self, query_embs: list[np.ndarray], k: int = DEFAULT_TOP_K
    ) -> float:
        """Fraction of the exact full-corpus top-`k` that the two-stage path also returns.

        The honest measurement of what the approximation costs. Report it beside any LI result:
        without it, "LI scores X" is a claim about an unspecified retriever.
        """
        if not query_embs:
            return 1.0
        hits = 0
        total = 0
        for q in query_embs:
            approx = {i for i, _ in self.topk(q, k=k)}
            exact = {i for i, _ in self.topk(q, k=k, n_candidates=self.n_docs)}
            hits += len(approx & exact)
            total += len(exact)
        return hits / max(total, 1)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(
            d / "index.npz",
            tokens=self.tokens, offsets=self.offsets, pooled=self.pooled,
        )
        _write_corpus(self.records, d / "corpus.jsonl")
        (d / "meta.json").write_text(json.dumps({
            "kind": "li", "n_docs": self.n_docs, "n_tokens": int(self.tokens.shape[0]),
            "dim": int(self.tokens.shape[1]), "corpus_id": self.corpus_id,
            "n_candidates": self.n_candidates, "encoder": self.encoder.to_dict(),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> LateInteractionIndex:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        blob = np.load(d / "index.npz", allow_pickle=False)
        records = list(iter_premise_records(d / "corpus.jsonl"))
        if len(records) != meta["n_docs"]:
            raise ValueError(f"index/corpus mismatch in {d}")
        return cls(
            records, blob["tokens"], blob["offsets"], blob["pooled"],
            EncoderSpec(**meta["encoder"]), int(meta.get("n_candidates", DEFAULT_N_CANDIDATES)),
        )


# =============================================================================================
# Retriever arms
# =============================================================================================
class _DenseRetriever:
    """Shared plumbing: hold an encoder, time each query, return `Premise` objects."""

    name = "dense"

    def __init__(self, index, encode_fn, stats: RetrievalStats | None = None) -> None:
        self.index = index
        self._encode = encode_fn        # str -> np.ndarray; injected so tests need no model
        self.stats = stats if stats is not None else RetrievalStats()

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Premise]:
        t0 = time.perf_counter()
        hits = self.index.topk(self._encode(query), k=k)
        premises = [self.index.records[i].to_premise(score=s) for i, s in hits]
        self.stats.record(time.perf_counter() - t0, len(premises))
        return premises


class SingleVectorRetriever(_DenseRetriever):
    """The `sv` arm — ProofLens matched control (`sv_ft_novel_lr3e6`, gte-modernbert-base)."""

    name = "sv"


class LateInteractionRetriever(_DenseRetriever):
    """The `li` arm — ProofLens late interaction (`li_ft_novel_bm25`, GTE-ModernColBERT-v1)."""

    name = "li"


def load_query_encoder(kind: str, checkpoint: str, device: str | None = None):
    """Return `str -> np.ndarray`, encoding a proof state the way the index was built.

    Torch and pylate are imported here rather than at module scope so the whole scoring path — and
    every test of it — stays importable without them.

    **The checkpoint must be the one that built the index.** Query and premise vectors are only
    comparable if they come from the same weights; a mismatch produces a retriever that returns
    confident nonsense rather than an error.
    """
    if kind == "li":
        from pylate import models

        model = models.ColBERT(
            model_name_or_path=checkpoint,
            query_length=LI_QUERY_LENGTH,
            document_length=LI_DOCUMENT_LENGTH,
            device=device,
        )

        def encode_li_query(text: str) -> np.ndarray:
            emb = model.encode(
                [text], is_query=True, batch_size=1, convert_to_numpy=True,
                normalize_embeddings=True, output_value="token_embeddings",
                show_progress_bar=False,
            )
            if isinstance(emb, list):
                emb = emb[0]
            return np.ascontiguousarray(emb, dtype=np.float32)

        return encode_li_query

    if kind == "sv":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(checkpoint, device=device)
        # Explicit, not inherited — see SV_MAX_SEQ_LENGTH. Must match the premise side in
        # `scripts/build_dense_index.py:encode_sv`.
        model.max_seq_length = SV_MAX_SEQ_LENGTH

        def encode_sv_query(text: str) -> np.ndarray:
            return np.ascontiguousarray(
                model.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)[0],
                dtype=np.float32,
            )

        return encode_sv_query

    raise ValueError(f"unknown encoder kind {kind!r}")


def load_retriever(kind: str, index_dir: str | Path, checkpoint: str | None = None,
                   device: str | None = None, stats: RetrievalStats | None = None):
    """Load a dense arm from a saved index directory.

    `checkpoint` defaults to the one recorded in the index metadata, which is what keeps the query
    encoder and the premise vectors in step without the caller having to remember.
    """
    index_cls = LateInteractionIndex if kind == "li" else SingleVectorIndex
    index = index_cls.load(index_dir)
    ckpt = checkpoint or index.encoder.checkpoint
    encode = load_query_encoder(kind, ckpt, device)
    retriever_cls = LateInteractionRetriever if kind == "li" else SingleVectorRetriever
    return retriever_cls(index, encode, stats)
