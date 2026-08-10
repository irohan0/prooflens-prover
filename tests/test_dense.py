"""Hermetic tests for the SV and LI retrieval arms — no torch, no GPU, no checkpoints.

The centrepiece is `test_maxsim_matches_reference_per_document`. `LateInteractionIndex` computes
MaxSim with a gather + `np.maximum.reduceat` over a ragged CSR layout, which is fast and completely
opaque; the only way to know it computes ColBERT MaxSim rather than something that merely correlates
with it is to check every document against the obvious one-document-at-a-time definition.

Embeddings here are random unit vectors. That is deliberate: these tests are about the *arithmetic*,
which must be right regardless of what the encoder produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from prooflens_prover.data.premises import PremiseRecord
from prooflens_prover.retrieval.base import Retriever
from prooflens_prover.retrieval.dense import (
    LI_DOCUMENT_LENGTH,
    LI_QUERY_LENGTH,
    SV_MAX_SEQ_LENGTH,
    EncoderSpec,
    LateInteractionIndex,
    LateInteractionRetriever,
    SingleVectorIndex,
    SingleVectorRetriever,
    l2_normalise,
    maxsim_score,
)

DIM = 16
RNG = np.random.default_rng(0)


def records(n: int) -> list[PremiseRecord]:
    return [
        PremiseRecord(name=f"Lemma.n{i}", kind="theorem", statement=f"statement {i}",
                      module="Mathlib.Test", is_prop=True)
        for i in range(n)
    ]


def unit(shape) -> np.ndarray:
    return l2_normalise(RNG.standard_normal(shape).astype(np.float32))


def build_li(n_docs: int = 25, min_tok: int = 1, max_tok: int = 7,
             n_candidates: int = 1000) -> LateInteractionIndex:
    lengths = RNG.integers(min_tok, max_tok + 1, size=n_docs)
    offsets = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = unit((int(offsets[-1]), DIM))
    pooled = l2_normalise(
        np.stack([tokens[offsets[i]:offsets[i + 1]].mean(axis=0) for i in range(n_docs)])
    )
    spec = EncoderSpec(kind="li", checkpoint="test", base_model="test", dim=DIM)
    return LateInteractionIndex(records(n_docs), tokens, offsets, pooled, spec, n_candidates)


def build_sv(n_docs: int = 25) -> SingleVectorIndex:
    spec = EncoderSpec(kind="sv", checkpoint="test", base_model="test", dim=DIM)
    return SingleVectorIndex(records(n_docs), unit((n_docs, DIM)), spec)


# ---------------------------------------------------------------------------------------------
class TestMaxSimReference:
    def test_reference_is_sum_of_per_query_token_maxima(self):
        q, d = unit((3, DIM)), unit((5, DIM))
        expected = sum(max(float(qi @ dj) for dj in d) for qi in q)
        assert maxsim_score(q, d) == pytest.approx(expected, rel=1e-5)

    def test_reference_applies_query_token_weights(self):
        q, d = unit((3, DIM)), unit((5, DIM))
        w = np.array([2.0, 0.5, 1.0], dtype=np.float32)
        expected = sum(wi * max(float(qi @ dj) for dj in d) for wi, qi in zip(w, q, strict=True))
        assert maxsim_score(q, d, w) == pytest.approx(expected, rel=1e-5)

    def test_identical_query_and_document_scores_one_per_token(self):
        d = unit((4, DIM))
        assert maxsim_score(d, d) == pytest.approx(4.0, rel=1e-5)


class TestLateInteractionIndex:
    def test_gather_reconstructs_each_document_block(self):
        idx = build_li()
        sel = np.array([0, 3, 7, 8])
        d, seg = idx._gather(sel)
        assert seg[0] == 0
        for j, i in enumerate(sel):
            lo = int(seg[j])
            hi = int(seg[j + 1]) if j + 1 < len(seg) else d.shape[0]
            np.testing.assert_allclose(d[lo:hi], idx.doc_tokens(int(i)), rtol=1e-6)

    def test_maxsim_matches_reference_per_document(self):
        """The test that makes the vectorised path trustworthy."""
        idx = build_li()
        q = unit((5, DIM))
        sel = np.arange(idx.n_docs)
        got = idx.maxsim_over(q, sel)
        expected = np.array([maxsim_score(q, idx.doc_tokens(i)) for i in sel])
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_maxsim_matches_reference_with_weights(self):
        idx = build_li()
        q = unit((4, DIM))
        w = np.array([1.0, 3.0, 0.25, 2.0], dtype=np.float32)
        sel = np.arange(idx.n_docs)
        got = idx.maxsim_over(q, sel, w)
        expected = np.array([maxsim_score(q, idx.doc_tokens(i), w) for i in sel])
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_single_token_documents_are_handled(self):
        # `reduceat` boundaries are easiest to get wrong on length-1 segments.
        idx = build_li(n_docs=10, min_tok=1, max_tok=1)
        q = unit((3, DIM))
        got = idx.maxsim_over(q, np.arange(idx.n_docs))
        expected = np.array([maxsim_score(q, idx.doc_tokens(i)) for i in range(idx.n_docs)])
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_maxsim_over_empty_selection(self):
        assert build_li().maxsim_over(unit((3, DIM)), np.array([], dtype=np.int64)).size == 0

    def test_a_document_retrieves_itself_under_exact_search(self):
        idx = build_li()
        for i in range(idx.n_docs):
            hits = idx.topk(idx.doc_tokens(i), k=1, n_candidates=idx.n_docs)
            assert hits[0][0] == i

    def test_two_stage_equals_exact_when_all_candidates_kept(self):
        idx = build_li()
        q = unit((4, DIM))
        assert idx.topk(q, k=5, n_candidates=idx.n_docs) == idx.topk(q, k=5, n_candidates=10_000)

    def test_two_stage_restricts_the_rescored_set(self):
        # With a tight candidate budget the result may differ from exact — that is the whole point
        # of measuring recall. What must hold is that it still returns k scored premises.
        idx = build_li(n_docs=40)
        hits = idx.topk(unit((4, DIM)), k=5, n_candidates=8)
        assert len(hits) == 5

    def test_topk_is_sorted_and_deterministic(self):
        idx = build_li()
        q = unit((4, DIM))
        a, b = idx.topk(q, k=6), idx.topk(q, k=6)
        assert a == b
        assert [s for _, s in a] == sorted((s for _, s in a), reverse=True)

    def test_k_zero_and_k_larger_than_corpus(self):
        idx = build_li(n_docs=5)
        assert idx.topk(unit((3, DIM)), k=0) == []
        assert len(idx.topk(unit((3, DIM)), k=99)) == 5

    def test_recall_is_one_when_no_approximation_is_made(self):
        idx = build_li(n_docs=20, n_candidates=20)
        assert idx.recall_at_k_vs_exact([unit((4, DIM)) for _ in range(5)], k=5) == 1.0

    def test_recall_is_reported_not_assumed(self):
        # A deliberately tiny candidate budget should be *measurable* as lossy, not silently wrong.
        idx = build_li(n_docs=60, n_candidates=3)
        r = idx.recall_at_k_vs_exact([unit((4, DIM)) for _ in range(8)], k=5)
        assert 0.0 <= r <= 1.0

    def test_offsets_length_is_validated(self):
        idx = build_li(n_docs=5)
        spec = EncoderSpec(kind="li", checkpoint="t", base_model="t", dim=DIM)
        with pytest.raises(ValueError, match="offsets must have"):
            LateInteractionIndex(idx.records, idx.tokens, idx.offsets[:-1], idx.pooled, spec)

    def test_token_count_mismatch_is_validated(self):
        idx = build_li(n_docs=5)
        spec = EncoderSpec(kind="li", checkpoint="t", base_model="t", dim=DIM)
        with pytest.raises(ValueError, match="does not match token count"):
            LateInteractionIndex(idx.records, idx.tokens[:-1], idx.offsets, idx.pooled, spec)


class TestSingleVectorIndex:
    def test_a_document_retrieves_itself(self):
        idx = build_sv()
        for i in range(idx.n_docs):
            assert idx.topk(idx.embeddings[i], k=1)[0][0] == i

    def test_scores_are_cosines(self):
        idx = build_sv()
        q = unit((DIM,))
        for i, s in idx.topk(q, k=5):
            assert s == pytest.approx(float(idx.embeddings[i] @ q), rel=1e-5)

    def test_topk_sorted_and_deterministic(self):
        idx = build_sv()
        q = unit((DIM,))
        a = idx.topk(q, k=7)
        assert a == idx.topk(q, k=7)
        assert [s for _, s in a] == sorted((s for _, s in a), reverse=True)

    def test_ties_break_on_corpus_index(self):
        spec = EncoderSpec(kind="sv", checkpoint="t", base_model="t", dim=DIM)
        v = unit((1, DIM))
        idx = SingleVectorIndex(records(3), np.repeat(v, 3, axis=0), spec)
        assert [i for i, _ in idx.topk(v[0], k=3)] == [0, 1, 2]

    def test_embedding_count_is_validated(self):
        spec = EncoderSpec(kind="sv", checkpoint="t", base_model="t", dim=DIM)
        with pytest.raises(ValueError, match="embeddings/corpus mismatch"):
            SingleVectorIndex(records(5), unit((4, DIM)), spec)


class TestPersistence:
    def test_li_roundtrip(self, tmp_path):
        idx = build_li()
        idx.save(tmp_path / "li")
        back = LateInteractionIndex.load(tmp_path / "li")
        assert back.n_docs == idx.n_docs
        assert back.corpus_id == idx.corpus_id
        assert back.encoder == idx.encoder
        q = unit((4, DIM))
        assert back.topk(q, k=5) == idx.topk(q, k=5)

    def test_sv_roundtrip(self, tmp_path):
        idx = build_sv()
        idx.save(tmp_path / "sv")
        back = SingleVectorIndex.load(tmp_path / "sv")
        assert back.corpus_id == idx.corpus_id
        q = unit((DIM,))
        assert back.topk(q, k=5) == idx.topk(q, k=5)

    def test_corpus_mismatch_is_detected(self, tmp_path):
        idx = build_sv()
        idx.save(tmp_path / "sv")
        p = tmp_path / "sv" / "corpus.jsonl"
        p.write_text("\n".join(p.read_text(encoding="utf-8").splitlines()[:-1]) + "\n",
                     encoding="utf-8")
        with pytest.raises(ValueError, match="index/corpus mismatch"):
            SingleVectorIndex.load(tmp_path / "sv")


class TestArmsShareOneCorpus:
    def test_sv_and_li_over_the_same_records_have_the_same_corpus_id(self):
        # The invariant the whole comparison rests on: arms must rank the SAME candidate set.
        recs = records(30)
        spec_sv = EncoderSpec(kind="sv", checkpoint="t", base_model="t", dim=DIM)
        sv = SingleVectorIndex(recs, unit((30, DIM)), spec_sv)
        li = build_li(n_docs=30)
        li.records = recs
        assert sv.corpus_id == li.corpus_id


class TestRetrieverArms:
    def test_both_satisfy_the_retriever_protocol(self):
        sv = SingleVectorRetriever(build_sv(), lambda _q: unit((DIM,)))
        li = LateInteractionRetriever(build_li(), lambda _q: unit((4, DIM)))
        assert isinstance(sv, Retriever) and isinstance(li, Retriever)
        assert sv.name == "sv" and li.name == "li"

    def test_return_premises_in_realprover_shape(self):
        li = LateInteractionRetriever(build_li(), lambda _q: unit((4, DIM)))
        hits = li.retrieve("⊢ a + b = b + a", k=3)
        assert len(hits) == 3
        assert all(h.informal_name == "" for h in hits)
        assert set(hits[0].to_realprover_dict()) == {
            "Formal name", "Informal name", "Formal statement"
        }

    def test_latency_is_recorded(self):
        li = LateInteractionRetriever(build_li(), lambda _q: unit((4, DIM)))
        li.retrieve("a", k=3)
        li.retrieve("b", k=3)
        assert li.stats.to_dict()["n_queries"] == 2

    def test_encoder_is_injected_so_no_model_is_needed(self):
        # Scoring must be testable with no torch, no checkpoint and no download. If this ever
        # requires a real encoder, the pure/impure split has been broken.
        seen = []
        sv = SingleVectorRetriever(build_sv(), lambda q: (seen.append(q), unit((DIM,)))[1])
        sv.retrieve("⊢ goal text", k=2)
        assert seen == ["⊢ goal text"]


class TestFloat16Storage:
    """The full LI index is stored float16 (9 GB -> 4.5 GB, so it fits beside a Lean REPL).

    `_gather` casts the gathered candidates back to float32 for the matmul. These tests pin that the
    halving is a memory decision, not a scoring one.
    """

    @staticmethod
    def to_fp16(idx: LateInteractionIndex) -> LateInteractionIndex:
        return LateInteractionIndex(
            idx.records, idx.tokens.astype(np.float16), idx.offsets, idx.pooled,
            idx.encoder, idx.n_candidates,
        )

    def test_gather_returns_float32_regardless_of_storage(self):
        idx = self.to_fp16(build_li())
        assert idx.tokens.dtype == np.float16
        block, _ = idx._gather(np.array([0, 1, 2]))
        assert block.dtype == np.float32, "matmul must not run in float16"

    def test_scores_stay_close_to_the_float32_reference(self):
        idx32 = build_li()
        idx16 = self.to_fp16(idx32)
        q = unit((5, DIM))
        sel = np.arange(idx32.n_docs)
        np.testing.assert_allclose(
            idx16.maxsim_over(q, sel), idx32.maxsim_over(q, sel), rtol=2e-3, atol=2e-3
        )

    def test_ranking_is_unchanged_by_the_dtype(self):
        # What actually matters: the same premises reach the model, in the same order.
        idx32 = build_li()
        idx16 = self.to_fp16(idx32)
        q = unit((5, DIM))
        assert [i for i, _ in idx16.topk(q, k=8)] == [i for i, _ in idx32.topk(q, k=8)]

    def test_roundtrip_preserves_float16_storage(self, tmp_path):
        idx = self.to_fp16(build_li())
        idx.save(tmp_path / "li16")
        back = LateInteractionIndex.load(tmp_path / "li16")
        assert back.tokens.dtype == np.float16
        q = unit((4, DIM))
        assert back.topk(q, k=5) == idx.topk(q, k=5)


class TestLockedSequenceLengths:
    """The sequence lengths are inherited experimental settings, not free parameters.

    An earlier version of `dense.py` took `query_length=256` from the predecessor's *base*
    `configs/late_interaction.yaml` instead of `configs/late_interaction_ft_novel.yaml`, which is
    the config for the checkpoint actually indexed here and which its Phase 11 locked at **384**
    after measuring that **26.4% of proof states exceed 256 tokens** and that 256 -> 384 recovers
    ~10% relative R@1. Three full benchmark runs were completed under the wrong value.

    Nothing downstream fails when these drift — retrieval simply gets quietly worse — so they are
    pinned here.
    """

    def test_li_query_length_is_the_phase11_knee(self):
        assert LI_QUERY_LENGTH == 384, (
            "configs/late_interaction_ft_novel.yaml locks query_length=384; 256 truncates a "
            "quarter of proof states and 512 was measured to add nothing"
        )

    def test_li_document_length_matches_the_index_build(self):
        # Unlike query length, this one governs premise-side truncation: changing it invalidates
        # every existing LI index.
        assert LI_DOCUMENT_LENGTH == 300

    def test_sv_max_seq_length_matches_its_locked_config(self):
        assert SV_MAX_SEQ_LENGTH == 512, (
            "configs/dense_sv_ft_novel_lr3e6.yaml sets max_length: 512"
        )

    def test_sv_length_is_set_explicitly_on_both_sides(self):
        # Left unset, SentenceTransformer adopts the checkpoint's own config, making truncation an
        # uncontrolled variable in a comparison whose whole point is that only the retriever varies.
        from pathlib import Path

        builder = Path(__file__).resolve().parent.parent / "scripts" / "build_dense_index.py"
        src = builder.read_text(encoding="utf-8")
        assert "model.max_seq_length = SV_MAX_SEQ_LENGTH" in src, (
            "scripts/build_dense_index.py:encode_sv must pin max_seq_length, or premises and "
            "proof states can be truncated differently with no error"
        )

    def test_query_and_document_lengths_are_independent(self):
        # Guards against a future "simplification" that collapses them into one constant: they come
        # from different config keys and only the document side is tied to the stored index.
        assert LI_QUERY_LENGTH != LI_DOCUMENT_LENGTH


class TestExactChunkedMatchesSingleShot:
    """`exact_topk_chunked` is the ground truth the two-stage approximation is measured against.

    If it differs from `topk(..., n_candidates=n_docs)` for any reason other than the candidate
    restriction, that difference is reported as approximation loss that does not exist — and it
    would be invisible, because both numbers look plausible. An earlier draft of
    `scripts/measure_li_recall.py` reimplemented the chunked loop in the caller and skipped
    `topk`'s query normalisation and tie-breaking, which is exactly this bug.
    """

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 24, 25, 26, 1000])
    def test_same_ranking_for_any_chunk_size(self, chunk):
        """Ranking must be chunk-invariant. Scores need only agree to float32 precision.

        Chunking changes the order BLAS accumulates the matmul, so scores differ in the last
        bits (observed: 1.4285389 vs 1.4285390, ~1e-7 relative). That is inherent to floating
        point, not a defect. What must not change is *which* premises are returned and in what
        order — recall is computed over index sets, so that is what the measurement depends on.
        """
        idx = build_li(n_docs=25)
        q = unit((4, DIM))
        got = idx.exact_topk_chunked(q, k=6, chunk=chunk)
        want = idx.topk(q, k=6, n_candidates=idx.n_docs)
        assert [i for i, _ in got] == [i for i, _ in want]
        np.testing.assert_allclose(
            [s for _, s in got], [s for _, s in want], rtol=1e-5, atol=1e-5
        )

    def test_chunk_larger_than_corpus(self):
        idx = build_li(n_docs=10)
        q = unit((3, DIM))
        # One chunk covering everything is the same computation, so this one IS bit-identical.
        assert idx.exact_topk_chunked(q, k=5, chunk=10_000) == idx.topk(
            q, k=5, n_candidates=idx.n_docs
        )

    def test_normalises_the_query_like_topk_does(self):
        """A non-unit query must rank identically to its normalised form.

        Skipping the normalisation that `topk` performs would silently score the two paths
        differently. Compared the same way as `test_same_ranking_for_any_chunk_size`, and for the
        same reason: dividing by 7.5 and then by the norm is a different float32 accumulation from
        dividing by the norm alone, so the scores differ in the last bits (~1e-7). This test
        asserted exact tuple equality and therefore failed whenever the unseeded random draw
        happened to put two scores within rounding distance of each other — a flaky test, which is
        worse than no test, because it makes a green suite stop meaning anything.
        """
        idx = build_li(n_docs=25)
        q = unit((4, DIM))
        got = idx.exact_topk_chunked(q * 7.5, k=6)
        want = idx.exact_topk_chunked(q, k=6)
        assert [i for i, _ in got] == [i for i, _ in want]
        np.testing.assert_allclose(
            [s for _, s in got], [s for _, s in want], rtol=1e-5, atol=1e-5
        )

    def test_accepts_a_one_dimensional_query(self):
        idx = build_li(n_docs=25)
        q = unit(DIM)
        assert len(idx.exact_topk_chunked(q, k=5)) == 5

    def test_k_beyond_corpus_size_is_clamped(self):
        idx = build_li(n_docs=8)
        assert len(idx.exact_topk_chunked(unit((3, DIM)), k=99)) == 8

    def test_k_zero(self):
        assert build_li(n_docs=8).exact_topk_chunked(unit((3, DIM)), k=0) == []

    def test_scores_are_descending(self):
        hits = build_li(n_docs=40).exact_topk_chunked(unit((4, DIM)), k=10, chunk=7)
        assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)

    def test_matches_the_reference_maxsim_definition(self):
        # Ties the chunked path back to the one-document-at-a-time definition of ColBERT MaxSim,
        # not merely to another implementation in this file.
        idx = build_li(n_docs=30)
        q = unit((4, DIM))
        best = sorted(
            ((maxsim_score(q, idx.doc_tokens(i)), i) for i in range(idx.n_docs)), reverse=True
        )[:5]
        got = idx.exact_topk_chunked(q, k=5, chunk=6)
        assert [i for _, i in best] == [i for i, _ in got]


class TestRecallMeasurement:
    def test_recall_is_one_when_the_budget_covers_the_corpus(self):
        idx = build_li(n_docs=30)
        qs = [unit((4, DIM)) for _ in range(5)]
        assert idx.recall_at_k_vs_exact(qs, k=5, n_candidates=idx.n_docs) == 1.0

    def test_a_tiny_budget_loses_recall(self):
        # The property the whole diagnostic rests on: restricting candidates must be *measurable*.
        idx = build_li(n_docs=200, n_candidates=3)
        qs = [unit((4, DIM)) for _ in range(10)]
        assert idx.recall_at_k_vs_exact(qs, k=10, n_candidates=3) < 1.0

    def test_larger_budgets_do_not_reduce_recall(self):
        # Monotonicity: a bigger first stage is a superset, so recall cannot fall. If this fails the
        # candidate selection is not a nested top-n and the sweep would be uninterpretable.
        idx = build_li(n_docs=300)
        qs = [unit((4, DIM)) for _ in range(8)]
        rs = [idx.recall_at_k_vs_exact(qs, k=10, n_candidates=b) for b in (5, 25, 100, 300)]
        assert rs == sorted(rs), f"recall not monotone in n_candidates: {rs}"

    def test_empty_query_list(self):
        assert build_li().recall_at_k_vs_exact([], k=5) == 1.0
