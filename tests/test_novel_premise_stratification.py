"""Novel-premise stratification — the end-to-end test of the predecessor study's robustness claim.

The metric is a near-neighbour of one this project deliberately *withheld* (premise attribution
for an LLM), so the tests pin exactly what it counts: a property of a proof's text, resolved
against a premise corpus, classified against the retriever's training positives — and never a
claim that retrieval supplied anything.

Hermetic: every fixture is written by this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from novel_premise_stratification import (  # noqa: E402
    fisher_exact_two_sided,
    gather,
    permutation_diff_means,
    seen_premise_names,
    stratum,
    stream_theorems,
    unseen_citations,
)

# Premise-name resolution is shared with `scripts/contamination_audit.py`, which asks the opposite
# question of the same citation (does this premise close the theorem outright, rather than was it
# ever trained on). It lives in the package so the two analyses cannot drift apart.
from prooflens_prover.eval.premises import (  # noqa: E402
    cited_premises,
    load_corpus,
    resolve,
)

REPO = Path(__file__).resolve().parent.parent


def write_split(tmp_path, theorems) -> Path:
    p = tmp_path / "train.json"
    p.write_text(json.dumps(theorems), encoding="utf-8")
    return p


def thm(full_name, tactics):
    """A LeanDojo theorem record. `tactics` is [(tactic, [full_name, ...]), ...]."""
    return {
        "url": "https://github.com/leanprover-community/mathlib4", "commit": "deadbeef",
        "file_path": "Mathlib/Foo.lean", "full_name": full_name, "start": [1, 1], "end": [2, 1],
        "traced_tactics": [
            {"tactic": t, "state_before": "s", "state_after": "t",
             "annotated_tactic": [t, [{"full_name": n, "def_path": "Mathlib/Bar.lean",
                                       "def_pos": [3, 1], "def_end_pos": [3, 9]} for n in prem]]}
            for t, prem in tactics
        ],
    }


def write_corpus(tmp_path, names) -> Path:
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(
        json.dumps({"name": n, "statement": f"theorem {n} : True", "module": "M",
                    "kind": "theorem", "is_prop": True}) for n in names
    ), encoding="utf-8")
    return p


def write_run(tmp_path, name, arm, solved, n_candidates=None):
    """`solved` maps problem id -> proof tactic list, or None for unsolved."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)      # the cache test builds the fixture twice
    cfg = {"benchmark": "fate_m", "arm": arm, "policy_kind": "vllm"}
    if n_candidates is not None:
        cfg["n_candidates"] = n_candidates
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "started_utc": "2026-08-10T00:00:00+00:00",
        "config": cfg, "outcome": {"n_proved": sum(1 for v in solved.values() if v)},
    }), encoding="utf-8")
    (d / "attempts.jsonl").write_text("\n".join(
        json.dumps({"problem_id": pid, "proved": bool(pf), "proof": pf,
                    "status": "proved" if pf else "exhausted"})
        for pid, pf in solved.items()
    ), encoding="utf-8")
    return d


class TestSeenSetExtraction:
    def test_it_streams_without_parsing_the_whole_array(self, tmp_path):
        p = write_split(tmp_path, [thm("a", []), thm("b", []), thm("c", [])])
        assert [t["full_name"] for t in stream_theorems(p)] == ["a", "b", "c"]

    def test_it_collects_premise_full_names_not_theorem_names(self, tmp_path):
        # The theorem's own `full_name` sits beside the provenances' `full_name` in the same file;
        # collecting both would put every training *theorem* into the "seen premise" set and make
        # almost nothing look novel.
        p = write_split(tmp_path, [thm("MyThm", [("rw [foo]", ["Nat.foo"])])])
        assert seen_premise_names(p) == {"Nat.foo"}

    def test_tactics_with_no_premises_contribute_nothing(self, tmp_path):
        p = write_split(tmp_path, [thm("T", [("rfl", []), ("exact bar", ["Set.bar"])])])
        assert seen_premise_names(p) == {"Set.bar"}

    def test_an_empty_split_is_an_empty_set(self, tmp_path):
        assert seen_premise_names(write_split(tmp_path, [])) == set()


class TestCorpusResolution:
    def test_a_qualified_citation_resolves_exactly(self, tmp_path):
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["Fintype.card_pi_const"]))
        assert resolve("Fintype.card_pi_const", exact, by_suffix) == {"Fintype.card_pi_const"}

    def test_an_abbreviated_citation_resolves_by_last_component(self, tmp_path):
        # `open Fintype` lets a tactic write only the tail; the corpus stores the full name.
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["Fintype.card_pi_const"]))
        assert resolve("card_pi_const", exact, by_suffix) == {"Fintype.card_pi_const"}

    def test_an_ambiguous_abbreviation_returns_every_candidate(self, tmp_path):
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["A.card", "B.card"]))
        assert resolve("card", exact, by_suffix) == {"A.card", "B.card"}

    def test_a_non_premise_token_resolves_to_nothing(self, tmp_path):
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["Nat.foo"]))
        assert resolve("hypothesis_h", exact, by_suffix) is None


class TestCitationExtraction:
    def test_it_finds_premises_across_every_tactic_of_a_proof(self, tmp_path):
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["Nat.foo", "Set.bar"]))
        cited = cited_premises(["rw [Nat.foo]", "exact Set.bar h"], exact, by_suffix)
        assert set(cited) == {"Nat.foo", "Set.bar"}

    def test_a_qualified_name_is_not_split_into_its_parts(self, tmp_path):
        # Without dots in the identifier pattern, `Fintype.card` would tokenise as `Fintype` and
        # `card` and resolve to whatever else happens to end in `.card`.
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["Fintype.card", "Other.card"]))
        cited = cited_premises(["simp [Fintype.card]"], exact, by_suffix)
        assert cited == {"Fintype.card": {"Fintype.card"}}

    def test_the_tactic_word_stoplist_is_opt_in(self, tmp_path):
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["rfl"]))
        assert set(cited_premises(["rfl"], exact, by_suffix)) == {"rfl"}
        assert cited_premises(["rfl"], exact, by_suffix, drop_tactic_words=True) == {}

    def test_an_empty_proof_cites_nothing(self, tmp_path):
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["Nat.foo"]))
        assert cited_premises(None, exact, by_suffix) == {}


class TestUnseenClassification:
    def test_a_premise_outside_the_training_set_is_unseen(self):
        assert unseen_citations({"tok": {"Nat.novel"}}, {"Nat.old"}) == {"tok"}

    def test_a_premise_inside_the_training_set_is_seen(self):
        assert unseen_citations({"tok": {"Nat.old"}}, {"Nat.old"}) == set()

    def test_ambiguity_resolves_toward_seen(self):
        """The documented conservatism: any seen resolution makes the token seen.

        This makes an unseen-premise finding harder to obtain, which is the direction an argument
        resting on unseen premises has to be biased.
        """
        assert unseen_citations({"card": {"A.card", "B.card"}}, {"B.card"}) == set()


class TestFractionStatistic:
    """The indicator '>=1 unseen premise' measured 89-92% for every arm on real data, because 78%
    of the corpus is unseen and proofs name several premises. These tests pin the statistic that
    replaced it."""

    def _fixtures(self, tmp_path):
        exact, by_suffix = load_corpus(write_corpus(tmp_path, ["S.a", "S.b", "N.c", "N.d"]))
        return exact, by_suffix, {"S.a", "S.b"}

    def test_the_fraction_is_per_problem_not_pooled_over_citations(self, tmp_path):
        exact, by_suffix, seen = self._fixtures(tmp_path)
        # p0 cites 1 unseen of 1; p1 cites 1 unseen of 3. Pooling citations gives 2/4 = 0.50;
        # averaging per problem gives (1.0 + 0.333)/2 = 0.667. A problem is the unit of analysis.
        proofs = {"p0": ["exact N.c"], "p1": ["simp [S.a, S.b, N.d]"]}
        s = stratum({"p0", "p1"}, proofs, exact, by_suffix, seen, False)
        assert s["mean_fraction_unseen"] == pytest.approx(2 / 3)

    def test_a_proof_citing_nothing_is_excluded_from_the_mean(self, tmp_path):
        exact, by_suffix, seen = self._fixtures(tmp_path)
        proofs = {"p0": ["exact N.c"], "p1": ["omega"]}
        s = stratum({"p0", "p1"}, proofs, exact, by_suffix, seen, False)
        assert s["n_problems"] == 2
        assert s["n_problems_citing_any_premise"] == 1
        assert s["mean_fraction_unseen"] == pytest.approx(1.0)

    def test_all_seen_gives_zero_and_all_unseen_gives_one(self, tmp_path):
        exact, by_suffix, seen = self._fixtures(tmp_path)
        allseen = stratum({"p"}, {"p": ["simp [S.a, S.b]"]}, exact, by_suffix, seen, False)
        allnovel = stratum({"p"}, {"p": ["simp [N.c, N.d]"]}, exact, by_suffix, seen, False)
        assert allseen["mean_fraction_unseen"] == 0.0
        assert allnovel["mean_fraction_unseen"] == 1.0


class TestPermutationDiffMeans:
    def test_a_clean_separation_is_significant(self):
        p = permutation_diff_means([0.0] * 12, [1.0] * 12, 2000, 0)
        assert p < 0.05

    def test_identical_groups_are_not_significant(self):
        p = permutation_diff_means([0.5] * 10, [0.5] * 10, 2000, 0)
        assert p == pytest.approx(1.0)

    def test_it_is_two_sided(self):
        """Direction must not change the p, or a result could be made significant by ordering."""
        a, b = [0.1, 0.2, 0.3, 0.2], [0.8, 0.9, 0.7, 0.85]
        assert permutation_diff_means(a, b, 3000, 0) == permutation_diff_means(b, a, 3000, 0)

    def test_an_empty_group_is_p_one(self):
        assert permutation_diff_means([], [0.5], 500, 0) == 1.0


class TestFisherExact:
    def test_a_perfectly_separated_table(self):
        assert fisher_exact_two_sided(2, 0, 0, 2) == pytest.approx(1 / 3)

    def test_a_table_with_no_association_is_p_one(self):
        assert fisher_exact_two_sided(1, 1, 1, 1) == pytest.approx(1.0)

    def test_stronger_separation_gives_a_smaller_p(self):
        assert fisher_exact_two_sided(10, 0, 0, 10) < fisher_exact_two_sided(6, 4, 4, 6)

    def test_an_empty_margin_is_not_a_division_by_zero(self):
        assert fisher_exact_two_sided(0, 0, 5, 5) == 1.0

    def test_it_never_exceeds_one(self):
        for t in ((3, 4, 5, 6), (1, 0, 0, 1), (7, 7, 7, 7), (2, 5, 3, 1)):
            assert 0.0 < fisher_exact_two_sided(*t) <= 1.0


class TestPooling:
    """Pooling exists for power: the LI-vs-SV contrast has 11 exclusive wins per arm on FATE-M and
    6 on ProofNet, and no test on 6 problems concludes anything."""

    def test_ids_are_namespaced_so_benchmarks_cannot_collide(self, tmp_path):
        # Both benchmarks number their problems from scratch, so an un-namespaced pool would treat
        # fate_m's "1" and proofnet's "1" as the same problem.
        a = write_run(tmp_path, "a", "sv", {"1": ["exact X"]})
        b = write_run(tmp_path, "b", "sv", {"1": ["exact Y"]})
        (b / "manifest.json").write_text(json.dumps({
            "run_id": "b", "config": {"benchmark": "proofnet_test", "arm": "sv"},
            "outcome": {"n_proved": 1},
        }), encoding="utf-8")
        solved, proofs, benches, n_shared = gather([a, b])
        assert benches == ["fate_m", "proofnet_test"]
        assert solved["sv"] == {"fate_m:1", "proofnet_test:1"}
        assert proofs["fate_m:1"] == ["exact X"]
        assert proofs["proofnet_test:1"] == ["exact Y"]
        assert n_shared == 2

    def test_only_problems_every_arm_reached_are_kept(self, tmp_path):
        a = write_run(tmp_path, "none_r", "none", {"1": None, "2": None})
        b = write_run(tmp_path, "sv_r", "sv", {"1": ["exact X"]})
        _, _, _, n_shared = gather([a, b])
        assert n_shared == 1, "problem 2 is missing from the sv run"

    def test_two_runs_of_the_same_arm_on_one_benchmark_are_refused(self, tmp_path):
        a = write_run(tmp_path, "sv1", "sv", {"1": ["exact X"]})
        b = write_run(tmp_path, "sv2", "sv", {"1": ["exact Y"]})
        with pytest.raises(SystemExit, match="pick one"):
            gather([a, b])

    def test_benchmarks_with_different_arms_are_refused(self, tmp_path):
        # Pooling a 3-arm benchmark with a 2-arm one would average different contrasts.
        a = write_run(tmp_path, "a", "sv", {"1": ["exact X"]})
        b = write_run(tmp_path, "b", "li", {"1": ["exact Y"]})
        (b / "manifest.json").write_text(json.dumps({
            "run_id": "b", "config": {"benchmark": "proofnet_test", "arm": "li"},
            "outcome": {"n_proved": 1},
        }), encoding="utf-8")
        with pytest.raises(SystemExit, match="same arms"):
            gather([a, b])


class TestEndToEnd:
    """Through the script, because the unit tests above would pass on a version that computed
    everything correctly and compared the wrong pair of arms."""

    def _fixture(self, tmp_path):
        # LI wins problems needing `Novel.lemma`; SV wins problems needing `Seen.lemma`. That is the
        # predecessor study's prediction, planted so the script has to recover it.
        split = write_split(tmp_path, [thm("T", [("exact Seen.lemma", ["Seen.lemma"])])])
        corpus = write_corpus(tmp_path, ["Seen.lemma", "Novel.lemma"])
        none = write_run(tmp_path, "none_run", "none", {f"p{i}": None for i in range(8)})
        sv = write_run(tmp_path, "sv_run", "sv", {
            **{f"p{i}": ["exact Seen.lemma"] for i in range(4)},
            **{f"p{i}": None for i in range(4, 8)},
        })
        li = write_run(tmp_path, "li_run", "li", {
            **{f"p{i}": None for i in range(4)},
            **{f"p{i}": ["exact Novel.lemma"] for i in range(4, 8)},
        }, n_candidates=50_000)
        return split, corpus, none, sv, li

    def _run(self, tmp_path, *extra):
        split, corpus, none, sv, li = self._fixture(tmp_path)
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "novel_premise_stratification.py"),
             "--seen-split", str(split), "--seen-cache", str(tmp_path / "seen.json"),
             "--corpus", str(corpus), "--run", str(none), "--run", str(sv), "--run", str(li),
             "--json-out", str(tmp_path / "out.json"), *extra],
            capture_output=True, text=True, cwd=REPO,
        )
        return p, tmp_path / "out.json"

    def test_it_recovers_a_planted_enrichment(self, tmp_path):
        p, out = self._run(tmp_path)
        assert p.returncode == 0, p.stderr
        payload = json.loads(out.read_text(encoding="utf-8"))

        contrast = next(c for c in payload["contrasts"] if c["contrast"] == "li@50k vs sv")
        # SV's exclusive wins cite only seen premises, LI's only unseen ones: the planted version
        # of the predecessor study's prediction.
        assert contrast["only_sv"]["mean_fraction_unseen"] == pytest.approx(0.0)
        assert contrast["only_li@50k"]["mean_fraction_unseen"] == pytest.approx(1.0)
        assert contrast["delta_mean_fraction_unseen"] == pytest.approx(1.0)
        assert contrast["p_permutation_fraction"] < 0.05
        assert contrast["significant"] is True
        # The saturated indicator is still reported, but it is not what decides significance.
        assert "p_fisher_two_sided_indicator" in contrast

    def test_the_arm_contrast_appears_without_being_asked_for(self, tmp_path):
        p, _ = self._run(tmp_path)
        assert "li@50k vs sv" in p.stdout

    def test_the_seen_cache_is_written_then_reused(self, tmp_path):
        p, _ = self._run(tmp_path)
        assert (tmp_path / "seen.json").exists()
        assert "cached to" in p.stdout
        p2, _ = self._run(tmp_path)
        assert "cached:" in p2.stdout, "a second run must not re-parse the 365 MB split"

    def test_a_missing_cache_without_a_split_is_refused(self, tmp_path):
        _, corpus, none, sv, li = self._fixture(tmp_path)
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "novel_premise_stratification.py"),
             "--corpus", str(corpus), "--run", str(sv), "--run", str(li),
             "--seen-cache", str(tmp_path / "absent.json")],
            capture_output=True, text=True, cwd=REPO,
        )
        assert p.returncode != 0
        assert "--seen-split" in p.stdout + p.stderr

    def test_one_run_is_refused(self, tmp_path):
        _, corpus, none, _, _ = self._fixture(tmp_path)
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "novel_premise_stratification.py"),
             "--corpus", str(corpus), "--run", str(none)],
            capture_output=True, text=True, cwd=REPO,
        )
        assert p.returncode != 0
        assert "at least twice" in p.stdout + p.stderr
