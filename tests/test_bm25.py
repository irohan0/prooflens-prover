"""Hermetic tests for the lexical retrieval arm: tokenizer, corpus loader, BM25 index.

No Lean, no network, no checkpoints. The centrepiece is `test_scores_match_naive_reference`: the
index is a hand-rolled sparse structure with fancy-index accumulation, and the only way to know it
computes BM25 rather than something that merely correlates with BM25 is to check it against a
transparently correct implementation of the formula.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest

from prooflens_prover.data.premises import (
    PremiseRecord,
    corpus_id,
    load_premise_corpus,
)
from prooflens_prover.retrieval.bm25 import (
    BM25Index,
    BM25Params,
    BM25Retriever,
    TokenizerOptions,
)
from prooflens_prover.retrieval.lean_text import lean_tokenize, premise_document


def rec(name: str, statement: str, kind: str = "theorem", module: str = "Mathlib.Test",
        is_prop: bool = True) -> PremiseRecord:
    return PremiseRecord(name=name, kind=kind, statement=statement, module=module, is_prop=is_prop)


TINY_CORPUS = [
    rec("Nat.add_comm", "∀ (n m : ℕ), n + m = m + n"),
    rec("Nat.mul_comm", "∀ (n m : ℕ), n * m = m * n"),
    rec("Nat.add_assoc", "∀ (n m k : ℕ), n + m + k = n + (m + k)"),
    rec("Set.mem_union", "∀ {α : Type u} {s t : Set α} {a : α}, a ∈ s ∪ t ↔ a ∈ s ∨ a ∈ t"),
    rec("Finset.sum_const", "∀ {β : Type v} [AddCommMonoid β] (b : β), ∑ _x ∈ s, b = s.card • b"),
]


# ---------------------------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------------------------
class TestLeanTokenize:
    def test_splits_dotted_names_into_components(self):
        assert lean_tokenize("Nat.add_comm") == ["Nat", ".", "add_comm"]

    def test_keeps_underscored_identifier_whole_by_default(self):
        # `add_comm` must survive as one token: it is the single most discriminative term for that
        # premise, and splitting it by default would change every DF statistic the predecessor's
        # tokenizer produced.
        assert "add_comm" in lean_tokenize("Nat.add_comm : ∀ n m, n + m = m + n")

    def test_unicode_operators_become_individual_tokens(self):
        toks = lean_tokenize("a ∈ s ∪ t ↔ a ∈ s ∨ a ∈ t")
        assert "∈" in toks and "∪" in toks and "↔" in toks and "∨" in toks

    def test_greek_and_blackboard_letters_are_tokens(self):
        toks = lean_tokenize("∀ (n : ℕ) (x : ℝ), α")
        for sym in ("∀", "ℕ", "ℝ", "α"):
            assert sym in toks, f"{sym} missing from {toks}"

    def test_primes_stay_inside_identifiers(self):
        # `h'` is one hypothesis name, not `h` followed by a stray quote.
        assert "h'" in lean_tokenize("h' : a = b")

    def test_case_is_preserved_by_default(self):
        toks = lean_tokenize("IsUnit isUnit")
        assert "IsUnit" in toks and "isUnit" in toks

    def test_lowercase_flag_collapses_case(self):
        toks = lean_tokenize("IsUnit isUnit", lowercase=True)
        assert toks == ["isunit", "isunit"]

    def test_split_underscores_adds_parts_and_keeps_whole(self):
        toks = lean_tokenize("add_comm", split_underscores=True)
        assert toks[0] == "add_comm"          # whole token still first
        assert "add" in toks and "comm" in toks

    def test_split_underscores_drops_single_character_fragments(self):
        # `h_1` should not inject a bare `1` token into every document that names a hypothesis.
        toks = lean_tokenize("h_1", split_underscores=True)
        assert "1" not in toks

    def test_whitespace_never_appears_in_a_token(self):
        # The save/load path newline-joins the vocabulary, which is only lossless if this holds.
        for tok in lean_tokenize("∀ (n m : ℕ),\n  n + m\t= m + n"):
            assert not any(c.isspace() for c in tok)

    def test_premise_document_includes_name_then_statement(self):
        assert premise_document("Nat.add_comm", "n + m = m + n") == "Nat.add_comm n + m = m + n"


# ---------------------------------------------------------------------------------------------
# Corpus loading and filtering
# ---------------------------------------------------------------------------------------------
class TestPremiseCorpus:
    @staticmethod
    def write(tmp_path, records):
        p = tmp_path / "premises.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps({
                    "name": r.name, "kind": r.kind, "statement": r.statement,
                    "module": r.module, "is_prop": r.is_prop,
                }) + "\n")
        return p

    def test_roundtrips_all_fields(self, tmp_path):
        p = self.write(tmp_path, TINY_CORPUS)
        loaded = load_premise_corpus(p)
        assert [r.name for r in loaded] == [r.name for r in TINY_CORPUS]
        assert loaded[3].statement == TINY_CORPUS[3].statement

    def test_preserves_order(self, tmp_path):
        # Row order is the index's row order; a loader that reordered would silently invalidate a
        # saved index against a reloaded corpus.
        p = self.write(tmp_path, TINY_CORPUS)
        assert [r.name for r in load_premise_corpus(p)] == [r.name for r in TINY_CORPUS]

    def test_props_only_drops_definitions(self, tmp_path):
        records = [*TINY_CORPUS, rec("Foo.bar", "ℕ → ℕ", kind="def", is_prop=False)]
        p = self.write(tmp_path, records)
        assert len(load_premise_corpus(p)) == 6
        assert len(load_premise_corpus(p, props_only=True)) == 5

    def test_kind_filter(self, tmp_path):
        records = [*TINY_CORPUS, rec("Foo.bar", "ℕ → ℕ", kind="def", is_prop=False)]
        p = self.write(tmp_path, records)
        kept = load_premise_corpus(p, kinds={"def"})
        assert [r.name for r in kept] == ["Foo.bar"]

    def test_module_prefix_filter(self, tmp_path):
        records = [*TINY_CORPUS, rec("Lean.foo", "True", module="Lean.Elab")]
        p = self.write(tmp_path, records)
        kept = load_premise_corpus(p, module_prefixes=("Mathlib",))
        assert all(r.module.startswith("Mathlib") for r in kept)
        assert len(kept) == 5

    def test_long_statements_are_dropped(self, tmp_path):
        records = [*TINY_CORPUS, rec("Huge.thm", "x" * 5000)]
        p = self.write(tmp_path, records)
        assert len(load_premise_corpus(p, max_statement_chars=4000)) == 5
        assert len(load_premise_corpus(p, max_statement_chars=None)) == 6

    def test_blank_lines_are_skipped(self, tmp_path):
        p = self.write(tmp_path, TINY_CORPUS)
        p.write_text(p.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(load_premise_corpus(p)) == 5

    def test_corpus_id_is_stable_and_sensitive(self):
        assert corpus_id(TINY_CORPUS) == corpus_id(list(TINY_CORPUS))
        assert corpus_id(TINY_CORPUS) != corpus_id(TINY_CORPUS[:-1])
        # Order matters: a reordered corpus is a different index, even with identical members.
        assert corpus_id(TINY_CORPUS) != corpus_id(list(reversed(TINY_CORPUS)))

    def test_to_premise_uses_realprover_shape(self):
        p = TINY_CORPUS[0].to_premise(score=1.5)
        assert p.formal_name == "Nat.add_comm"
        assert p.informal_name == ""          # constant across every arm, by design
        assert p.score == 1.5
        assert set(p.to_realprover_dict()) == {
            "Formal name", "Informal name", "Formal statement"
        }


# ---------------------------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------------------------
def naive_bm25(doc_texts: list[str], query: str, k1: float, b: float) -> list[float]:
    """A deliberately slow, obviously-correct BM25. The oracle for the vectorised index."""
    docs = [lean_tokenize(t) for t in doc_texts]
    n = len(docs)
    lens = [len(d) for d in docs]
    avg = sum(lens) / n
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    out = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for t in set(lean_tokenize(query)):
            if t not in tf:
                continue
            idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1.0) / (tf[t] + k1 * (1.0 - b + b * lens[i] / avg))
        out.append(s)
    return out


@pytest.fixture(scope="module")
def index():
    return BM25Index.build(TINY_CORPUS, progress_every=0)


class TestBM25Index:

    def test_shapes_are_consistent(self, index):
        assert index.n_docs == len(TINY_CORPUS)
        assert index.indptr.shape == (index.n_terms + 1,)
        assert index.doc_ids.shape == index.tfs.shape
        assert int(index.indptr[-1]) == index.doc_ids.size
        assert index.idf.shape == (index.n_terms,)
        assert index.norm.shape == (index.n_docs,)

    @pytest.mark.parametrize(
        "query",
        [
            "n + m = m + n",
            "∀ (n m : ℕ), n * m = m * n",
            "a ∈ s ∪ t",
            "Nat.add_comm",
            "⊢ x + y = y + x",
            "nothing matches here zzzz",
        ],
    )
    def test_scores_match_naive_reference(self, index, query):
        doc_texts = [premise_document(r.name, r.statement) for r in TINY_CORPUS]
        expected = naive_bm25(doc_texts, query, index.params.k1, index.params.b)
        got = index.score_all(query)
        np.testing.assert_allclose(got, np.array(expected, dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_idf_is_never_negative(self):
        # The reason for Lucene's IDF over textbook Okapi. `∀` here appears in every document, which
        # under the textbook formula scores negative — matching a query term would *penalise* a
        # premise. Construct exactly that case and assert it cannot happen.
        corpus = [rec(f"t{i}", "∀ x, P x") for i in range(5)]
        idx = BM25Index.build(corpus, progress_every=0)
        assert float(idx.idf.min()) >= 0.0
        assert (idx.score_all("∀") >= 0.0).all()

    def test_document_query_ranks_its_own_premise_first(self, index):
        # Self-retrieval is over the *indexed document* (name + statement), not the statement alone.
        # Statement-only self-retrieval is NOT a BM25 guarantee and does fail here: querying
        # `Nat.add_comm`'s statement ranks `Nat.add_assoc` first, because `add_assoc` contains
        # `n + m` twice and on a 5-document corpus `+` still carries idf 0.875. That is a property
        # of the toy corpus, not of the implementation — see
        # `test_ubiquitous_tokens_are_suppressed_as_the_corpus_grows` for the mechanism that removes
        # it at Mathlib scale, where structural tokens appear in nearly every document.
        for r in TINY_CORPUS:
            hits = index.topk(premise_document(r.name, r.statement), k=1)
            assert hits, f"no hit for {r.name}"
            assert index.records[hits[0][0]].name == r.name

    def test_ubiquitous_tokens_are_suppressed_as_the_corpus_grows(self):
        # IDF is what stops Lean's structural tokens (`(`, `,`, `:`, `+`) from dominating scores.
        # It only bites when the corpus is large enough for them to be genuinely ubiquitous, which
        # is why toy-corpus rankings must not be read as evidence about the real index.
        def idf_of_common_token(n: int) -> float:
            corpus = [rec(f"t{i}", f"∀ (x : ℕ), P{i} x") for i in range(n)]
            idx = BM25Index.build(corpus, progress_every=0)
            return float(idx.idf[idx.vocab["("]])

        assert idf_of_common_token(5) > idf_of_common_token(100) > idf_of_common_token(5000)
        assert idf_of_common_token(5000) < 0.01

    def test_topk_is_sorted_descending(self, index):
        hits = index.topk("n + m = m + n", k=5)
        assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)

    def test_zero_score_documents_are_dropped_not_padded(self, index):
        # A query sharing nothing with the corpus must return nothing, rather than k arbitrary
        # premises — otherwise the bm25 arm silently injects noise the `none` arm does not have.
        assert index.topk("zzzz qqqq wwww", k=5) == []

    def test_topk_never_exceeds_k(self, index):
        assert len(index.topk("n + m", k=2)) <= 2

    def test_k_larger_than_corpus_is_safe(self, index):
        assert len(index.topk("∀", k=1000)) <= index.n_docs

    def test_k_zero_returns_empty(self, index):
        assert index.topk("n + m", k=0) == []

    def test_empty_query_returns_empty(self, index):
        assert index.topk("", k=5) == []

    def test_ranking_is_deterministic(self, index):
        a = index.topk("∀ (n m : ℕ), n + m = m + n", k=5)
        b = index.topk("∀ (n m : ℕ), n + m = m + n", k=5)
        assert a == b

    def test_ties_break_on_corpus_index(self):
        # Two identical documents must come back in corpus order every time, not in whatever order
        # argpartition happens to produce.
        corpus = [rec("b_second", "P x"), rec("a_first", "P x")]
        idx = BM25Index.build(corpus, progress_every=0)
        hits = idx.topk("P x", k=2)
        assert [i for i, _ in hits] == [0, 1]

    def test_empty_corpus_is_rejected(self):
        with pytest.raises(ValueError):
            BM25Index.build([], progress_every=0)

    def test_split_underscores_finds_premises_the_default_misses(self):
        # `add` alone cannot reach `Nat.add_comm` under the default tokenizer. This is the concrete
        # recall trade-off the flag exists to let us measure, not an assumption about which is best.
        corpus = [rec("Nat.add_comm", "n + m = m + n")]
        plain = BM25Index.build(corpus, progress_every=0)
        split = BM25Index.build(
            corpus, tokenizer=TokenizerOptions(split_underscores=True), progress_every=0
        )
        assert plain.topk("add", k=1) == []
        assert len(split.topk("add", k=1)) == 1


class TestPersistence:
    def test_roundtrip_preserves_scores_and_metadata(self, tmp_path):
        idx = BM25Index.build(TINY_CORPUS, progress_every=0)
        idx.save(tmp_path / "bm25")
        back = BM25Index.load(tmp_path / "bm25")

        assert back.n_docs == idx.n_docs
        assert back.n_terms == idx.n_terms
        assert back.corpus_id == idx.corpus_id
        assert back.params == idx.params
        assert back.tokenizer == idx.tokenizer
        assert [r.name for r in back.records] == [r.name for r in idx.records]
        q = "∀ (n m : ℕ), n + m = m + n"
        np.testing.assert_allclose(back.score_all(q), idx.score_all(q), rtol=1e-6)
        assert back.topk(q, k=5) == idx.topk(q, k=5)

    def test_roundtrip_preserves_unicode_vocabulary(self, tmp_path):
        idx = BM25Index.build(TINY_CORPUS, progress_every=0)
        idx.save(tmp_path / "bm25")
        back = BM25Index.load(tmp_path / "bm25")
        for sym in ("∀", "ℕ", "∪", "↔", "∑"):
            assert (sym in idx.vocab) == (sym in back.vocab)
            if sym in idx.vocab:
                assert idx.vocab[sym] == back.vocab[sym]

    def test_tokenizer_options_survive_the_roundtrip(self, tmp_path):
        idx = BM25Index.build(
            TINY_CORPUS, tokenizer=TokenizerOptions(split_underscores=True), progress_every=0
        )
        idx.save(tmp_path / "bm25")
        back = BM25Index.load(tmp_path / "bm25")
        # If these were not persisted, queries would be tokenised differently from the documents —
        # the index would still load and still return results, just quietly worse ones.
        assert back.tokenizer.split_underscores is True
        assert back.topk("add", k=1) == idx.topk("add", k=1)

    def test_meta_json_is_readable(self, tmp_path):
        idx = BM25Index.build(TINY_CORPUS, progress_every=0)
        idx.save(tmp_path / "bm25")
        meta = json.loads((tmp_path / "bm25" / "meta.json").read_text(encoding="utf-8"))
        assert meta["n_docs"] == 5
        assert meta["params"]["k1"] == pytest.approx(1.5)
        assert meta["corpus_id"] == idx.corpus_id

    def test_corpus_length_mismatch_is_detected(self, tmp_path):
        idx = BM25Index.build(TINY_CORPUS, progress_every=0)
        d = tmp_path / "bm25"
        idx.save(d)
        lines = (d / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        (d / "corpus.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="index/corpus mismatch"):
            BM25Index.load(d)


class TestBM25Retriever:
    def test_satisfies_the_retriever_protocol(self):
        from prooflens_prover.retrieval.base import Retriever

        r = BM25Retriever(BM25Index.build(TINY_CORPUS, progress_every=0))
        assert isinstance(r, Retriever)
        assert r.name == "bm25"

    def test_returns_premises_in_realprover_shape(self):
        r = BM25Retriever(BM25Index.build(TINY_CORPUS, progress_every=0))
        hits = r.retrieve(premise_document("Nat.add_comm", "∀ (n m : ℕ), n + m = m + n"), k=3)
        assert hits and hits[0].formal_name == "Nat.add_comm"
        assert all(h.informal_name == "" for h in hits)
        assert hits[0].score > 0.0

    def test_records_latency_statistics(self):
        # Closes prooflens open item #8: retrieval latency was never measured there, and this
        # project's argument requires knowing what multi-vector retrieval costs, not just what it
        # scores. Accounting is on by default so it cannot be forgotten.
        r = BM25Retriever(BM25Index.build(TINY_CORPUS, progress_every=0))
        r.retrieve("n + m", k=3)
        r.retrieve("a ∈ s", k=3)
        d = r.stats.to_dict()
        assert d["n_queries"] == 2
        assert d["total_latency_s"] >= 0.0
        assert d["mean_premises_returned"] > 0

    def test_params_are_the_documented_defaults(self):
        assert BM25Params() == BM25Params(k1=1.5, b=0.75)
