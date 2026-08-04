"""BM25 over the full Mathlib premise corpus — the lexical baseline arm.

## Why this is not a strawman, and why that matters

It would be convenient for this project if BM25 were weak. It is not: sparse lexical retrieval
is a genuinely strong premise-selection baseline, and the predecessor study concluded that late
interaction's advantage over single-vector retrieval was **"largely lexical"**. If that holds,
BM25 is the arm most likely to be competitive with ProofLens-LI, and a BM25 implemented
carelessly would manufacture a win for the thesis's own hypothesis. It gets the same care as
the arm we hope wins.

## Why not `rank_bm25` (which the predecessor used) or `bm25s`

`prooflens` scored only the *accessible* premises for a state — a few thousand candidates — which
`rank_bm25`'s per-query Python loop handles fine. Here there is no accessibility filter: retrieval
runs over all ~300k Mathlib premises, at **every proof state**, for tens of thousands of states per
benchmark. That is a different performance regime.

An explicit inverted index in numpy is ~100 lines, adds no dependency, is deterministic, and
makes the scoring formula auditable in the file where the numbers come from. Per query it
touches only the posting lists of terms the query actually contains.

## Scoring

Okapi BM25 with Lucene's IDF variant:

    idf(t)          = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(q, d)     = sum over unique t in q of  idf(t) * tf(t,d)*(k1+1) / (tf(t,d) + norm(d))
    norm(d)         = k1 * (1 - b + b * len(d) / avg_len)

Lucene's IDF rather than textbook Okapi because the textbook form goes **negative** for terms
in more than half the corpus, and a Lean corpus is full of such terms (`∀`, `:`, `Type` appear
nearly everywhere). Negative IDF makes a document score *worse* for matching a query term, which
is indefensible here. `rank_bm25` patches this with an `epsilon` floor; using a formula that is
positive by construction is cleaner than clamping one that is not.

Query term frequency is ignored (unique terms only), the standard choice — a goal repeating `+` five
times should not weight `+` five times.
"""

from __future__ import annotations

import json
import time
from array import array
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from prooflens_prover.data.premises import PremiseRecord, corpus_id, iter_premise_records
from prooflens_prover.retrieval.base import DEFAULT_TOP_K, Premise, RetrievalStats
from prooflens_prover.retrieval.lean_text import lean_tokenize, premise_document

__all__ = ["BM25Index", "BM25Params", "BM25Retriever"]


@dataclass(frozen=True)
class BM25Params:
    """Standard Okapi parameters. These are the field's defaults, not tuned on our benchmarks —
    tuning them against FATE-M would make BM25 an unfairly *strong* baseline in exactly the place we
    measure, which is the mirror image of strawmanning it."""

    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class TokenizerOptions:
    lowercase: bool = False
    split_underscores: bool = False


class BM25Index:
    """An inverted index over a fixed premise corpus, in compressed-sparse-row form.

    Layout (V = vocabulary size, P = total postings, N = documents):
      `indptr[V+1]`  postings for term t are the slice `indptr[t] : indptr[t+1]`
      `doc_ids[P]`   document id of each posting
      `tfs[P]`       term frequency of that term in that document
      `idf[V]`       per-term inverse document frequency
      `norm[N]`      the precomputed length-normalisation denominator term

    Same CSR-over-ragged-rows idea as the predecessor's late-interaction index
    (`prooflens/src/prooflens/retrievers/late_interaction.py`), reused because it is the right
    structure for exactly this and is already understood in this codebase.
    """

    def __init__(
        self,
        records: list[PremiseRecord],
        vocab: dict[str, int],
        indptr: np.ndarray,
        doc_ids: np.ndarray,
        tfs: np.ndarray,
        idf: np.ndarray,
        norm: np.ndarray,
        params: BM25Params,
        tokenizer: TokenizerOptions,
    ) -> None:
        self.records = records
        self.vocab = vocab
        self.indptr = indptr
        self.doc_ids = doc_ids
        self.tfs = tfs
        self.idf = idf
        self.norm = norm
        self.params = params
        self.tokenizer = tokenizer

    @property
    def n_docs(self) -> int:
        return len(self.records)

    @property
    def n_terms(self) -> int:
        return len(self.vocab)

    @property
    def corpus_id(self) -> str:
        return corpus_id(self.records)

    # -- build -------------------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        records: list[PremiseRecord],
        *,
        params: BM25Params | None = None,
        tokenizer: TokenizerOptions | None = None,
        progress_every: int = 50_000,
    ) -> BM25Index:
        params = params or BM25Params()
        tokenizer = tokenizer or TokenizerOptions()
        n = len(records)
        if n == 0:
            raise ValueError("cannot build a BM25 index over an empty corpus")

        vocab: dict[str, int] = {}
        # `array` rather than a Python list: ~11M postings would cost hundreds of MB as boxed ints,
        # and 4 bytes each here. This is the difference between building the index and swapping.
        post_terms = array("i")
        post_docs = array("i")
        post_tfs = array("f")
        doc_len = np.zeros(n, dtype=np.float32)

        for i, rec in enumerate(records):
            tokens = lean_tokenize(
                premise_document(rec.name, rec.statement),
                lowercase=tokenizer.lowercase,
                split_underscores=tokenizer.split_underscores,
            )
            doc_len[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                tid = vocab.get(term)
                if tid is None:
                    tid = len(vocab)
                    vocab[term] = tid
                post_terms.append(tid)
                post_docs.append(i)
                post_tfs.append(float(tf))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [bm25] tokenised {i + 1}/{n} premises, |vocab|={len(vocab)}", flush=True)

        term_arr = np.frombuffer(post_terms, dtype=np.int32)
        doc_arr = np.frombuffer(post_docs, dtype=np.int32)
        tf_arr = np.frombuffer(post_tfs, dtype=np.float32)

        # Group postings by term. Stable sort keeps documents in corpus order inside each posting
        # list, which is what makes top-k tie-breaking reproducible run to run.
        order = np.argsort(term_arr, kind="stable")
        doc_ids = np.ascontiguousarray(doc_arr[order])
        tfs = np.ascontiguousarray(tf_arr[order])

        v = len(vocab)
        # One posting per (term, document) pair, so the postings-per-term count IS the document
        # frequency. No separate df pass needed.
        df = np.bincount(term_arr, minlength=v).astype(np.int64)
        indptr = np.zeros(v + 1, dtype=np.int64)
        np.cumsum(df, out=indptr[1:])

        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)

        avg_len = float(doc_len.mean()) if n else 0.0
        # An empty document would divide by zero in the length ratio; extraction should never
        # produce one, but the index must not depend on that.
        safe_avg = avg_len if avg_len > 0 else 1.0
        norm = (params.k1 * (1.0 - params.b + params.b * doc_len / safe_avg)).astype(np.float32)

        return cls(records, vocab, indptr, doc_ids, tfs, idf, norm, params, tokenizer)

    # -- query -------------------------------------------------------------------------------
    def score_all(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query`. Cost is proportional to the total length of
        the posting lists of the query's terms, not to the corpus size."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        terms = set(
            lean_tokenize(
                query,
                lowercase=self.tokenizer.lowercase,
                split_underscores=self.tokenizer.split_underscores,
            )
        )
        k1 = self.params.k1
        for term in terms:
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = int(self.indptr[tid]), int(self.indptr[tid + 1])
            if lo == hi:
                continue
            docs = self.doc_ids[lo:hi]
            tf = self.tfs[lo:hi]
            # Fancy-index accumulation is safe without `np.add.at` because a term appears at most
            # once per document in its posting list, so `docs` has no repeats.
            scores[docs] += self.idf[tid] * (tf * (k1 + 1.0)) / (tf + self.norm[docs])
        return scores

    def topk(self, query: str, k: int = DEFAULT_TOP_K) -> list[tuple[int, float]]:
        """Top-`k` `(document index, score)` pairs, best first.

        Documents scoring exactly 0 share no term with the query and are **dropped rather than used
        to pad the result to `k`**. Padding with arbitrary premises would put text in the model's
        prompt that retrieval has no reason to believe is relevant, which corrupts the comparison
        against the `none` arm: some of BM25's measured effect would be the effect of adding noise.
        """
        if k <= 0:
            return []
        scores = self.score_all(query)
        k = min(k, self.n_docs)
        if k < self.n_docs:
            cand = np.argpartition(-scores, k - 1)[:k]
        else:
            cand = np.arange(self.n_docs)
        # Primary key -score, secondary key document index: fully deterministic ordering, including
        # among ties, which `argsort` on scores alone does not give.
        cand = cand[np.lexsort((cand, -scores[cand]))]
        return [(int(i), float(scores[i])) for i in cand if scores[i] > 0.0]

    # -- persistence -------------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        """Write the index as a directory: arrays, the exact corpus in row order, and metadata.

        The corpus is stored *with* the index rather than referenced by path, because index row i
        and corpus row i must correspond. A drifting corpus file would silently return premises
        under the wrong names — the sort of bug that produces plausible, wrong results.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        terms = [""] * len(self.vocab)
        for term, tid in self.vocab.items():
            terms[tid] = term
        # Tokens can never contain whitespace (see the tokenizer regex), so newline-joining is a
        # lossless encoding and avoids a fixed-width unicode array.
        np.savez(
            d / "index.npz",
            indptr=self.indptr,
            doc_ids=self.doc_ids,
            tfs=self.tfs,
            idf=self.idf,
            norm=self.norm,
            vocab="\n".join(terms),
        )
        with open(d / "corpus.jsonl", "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(
                    json.dumps(
                        {
                            "name": rec.name,
                            "kind": rec.kind,
                            "statement": rec.statement,
                            "module": rec.module,
                            "is_prop": rec.is_prop,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "n_docs": self.n_docs,
                    "n_terms": self.n_terms,
                    "n_postings": int(self.doc_ids.size),
                    "corpus_id": self.corpus_id,
                    "params": asdict(self.params),
                    "tokenizer": asdict(self.tokenizer),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> BM25Index:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        blob = np.load(d / "index.npz", allow_pickle=False)
        terms = str(blob["vocab"].item()).split("\n")
        records = list(iter_premise_records(d / "corpus.jsonl"))
        if len(records) != meta["n_docs"]:
            raise ValueError(
                f"index/corpus mismatch in {d}: meta says {meta['n_docs']} docs, "
                f"corpus.jsonl has {len(records)}"
            )
        return cls(
            records=records,
            vocab={t: i for i, t in enumerate(terms)},
            indptr=blob["indptr"],
            doc_ids=blob["doc_ids"],
            tfs=blob["tfs"],
            idf=blob["idf"],
            norm=blob["norm"],
            params=BM25Params(**meta["params"]),
            tokenizer=TokenizerOptions(**meta["tokenizer"]),
        )


class BM25Retriever:
    """The `bm25` experimental arm. Satisfies the `Retriever` protocol."""

    name = "bm25"

    def __init__(self, index: BM25Index, stats: RetrievalStats | None = None) -> None:
        self.index = index
        self.stats = stats if stats is not None else RetrievalStats()

    @classmethod
    def from_directory(cls, directory: str | Path) -> BM25Retriever:
        return cls(BM25Index.load(directory))

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Premise]:
        t0 = time.perf_counter()
        hits = self.index.topk(query, k)
        premises = [self.index.records[i].to_premise(score=s) for i, s in hits]
        self.stats.record(time.perf_counter() - t0, len(premises))
        return premises
