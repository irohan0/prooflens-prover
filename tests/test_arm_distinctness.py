"""Arm distinctness — the test that the reported tie is not one run counted twice.

Tier 1 reported 46 vs 46 on FATE-M and 26 vs 26 on ProofNet. Equal counts on two independent
benchmarks is exactly the shape a duplicated or mislabelled run would take, so the claim needs a
check rather than an assurance. These tests pin what that check can and cannot detect.

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

from prooflens_prover.eval.draws import load_draw  # noqa: E402
from verify_arm_distinctness import compare_pair, tie_probability  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def write_run(tmp_path, name, *, arm, retriever, index, solved, seed=0, latency_ms=None,
              n_candidates=None, attempted=None):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    cfg = {"benchmark": "fate_m", "arm": arm, "policy_kind": "vllm", "index": index,
           "policy_config": {"retriever": retriever}}
    if n_candidates is not None:
        cfg["n_candidates"] = n_candidates
    outcome = {"n_proved": len(solved)}
    if latency_ms is not None:
        outcome["retrieval"] = {"n_queries": 100, "mean_latency_ms": latency_ms}
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "seed": seed, "config": cfg, "outcome": outcome,
    }), encoding="utf-8")
    (d / "attempts.jsonl").write_text("\n".join(
        json.dumps({"problem_id": pid, "proved": pid in solved, "proof": solved.get(pid),
                    "status": "proved" if pid in solved else "exhausted"})
        for pid in (attempted if attempted is not None else solved)
    ), encoding="utf-8")
    return d


def sv_run(tmp_path, name, solved, **kw):
    return write_run(tmp_path, name, arm="sv", retriever="sv",
                     index="data/index/sv_ft_novel_lr3e6", solved=solved, latency_ms=37.6, **kw)


def li_run(tmp_path, name, solved, **kw):
    return write_run(tmp_path, name, arm="li", retriever="li",
                     index="data/index/li_ft_novel_bm25", solved=solved, latency_ms=929.6,
                     n_candidates=50_000, **kw)


def run_script(tmp_path, runs, *extra):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_arm_distinctness.py"),
         *sum([["--run", str(r)] for r in runs], []),
         "--json-out", str(tmp_path / "out.json"), *extra],
        capture_output=True, text=True, cwd=REPO,
    )


class TestTieProbability:
    def test_an_exact_tie_is_the_modal_outcome(self):
        """The crux of the answer: a tie is what equivalence looks like, not what a bug looks like.

        22 disagreements split 11-11 has probability 0.168 — higher than any other single split.
        """
        assert tie_probability(11, 11) == pytest.approx(0.1682, abs=1e-4)
        assert tie_probability(6, 6) == pytest.approx(0.2256, abs=1e-4)

    def test_larger_discordance_makes_an_exact_tie_less_likely(self):
        assert tie_probability(3, 3) > tie_probability(11, 11) > tie_probability(50, 50)

    def test_an_odd_discordance_cannot_tie(self):
        assert tie_probability(6, 5) is None

    def test_no_discordance_at_all_is_not_a_coin_flip_question(self):
        assert tie_probability(0, 0) is None


class TestPairComparison:
    def test_genuinely_different_arms_are_distinct(self, tmp_path):
        a = load_draw(sv_run(tmp_path, "sv", {"1": ["exact foo"], "2": ["simp"]}))
        b = load_draw(li_run(tmp_path, "li", {"1": ["exact bar"], "3": ["omega"]}))
        r = compare_pair(a, b)
        assert r["duplicate"] is False
        assert r["fraction_identical_proofs"] == 0.0
        assert r["discordant"] == 2

    def test_a_relabelled_copy_is_caught(self, tmp_path):
        """The failure this exists for: same proofs, zero discordance, so the counts tie exactly."""
        proofs = {str(i): [f"exact h{i}"] for i in range(10)}
        a = load_draw(sv_run(tmp_path, "sv", proofs))
        b = load_draw(li_run(tmp_path, "li_but_really_sv", proofs))
        r = compare_pair(a, b)
        assert r["duplicate"] is True
        assert r["fraction_identical_proofs"] == 1.0
        assert r["discordant"] == 0
        assert r["tie"] is True

    def test_a_tie_with_real_disagreement_is_not_a_duplicate(self, tmp_path):
        """The shape of the actual Tier 1 result: equal totals, different problems."""
        a = load_draw(sv_run(tmp_path, "sv", {"1": ["a"], "2": ["b"], "3": ["c"]}))
        b = load_draw(li_run(tmp_path, "li", {"1": ["z"], "4": ["y"], "5": ["x"]}))
        r = compare_pair(a, b)
        assert r["tie"] is True
        assert r["duplicate"] is False
        assert r["only_first"] == 2 and r["only_second"] == 2

    def test_the_recorded_retriever_and_latency_are_surfaced(self, tmp_path):
        a = load_draw(sv_run(tmp_path, "sv", {"1": ["a"]}))
        b = load_draw(li_run(tmp_path, "li", {"1": ["z"]}))
        r = compare_pair(a, b)
        assert r["retriever"] == ["sv", "li"]
        assert r["mean_latency_ms"] == [37.6, 929.6]
        assert r["index"][0] != r["index"][1]

    def test_arms_sharing_no_solved_problems_do_not_divide_by_zero(self, tmp_path):
        a = load_draw(sv_run(tmp_path, "sv", {"1": ["a"]}))
        b = load_draw(li_run(tmp_path, "li", {"2": ["z"]}))
        r = compare_pair(a, b)
        assert r["fraction_identical_proofs"] is None
        assert r["duplicate"] is False


class TestEndToEnd:
    def test_distinct_arms_pass_with_a_verdict(self, tmp_path):
        runs = [sv_run(tmp_path, "sv", {"1": ["a"], "2": ["b"], "3": ["c"]}),
                li_run(tmp_path, "li", {"1": ["z"], "4": ["y"], "5": ["x"]})]
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        assert "DISTINCT RUNS" in p.stdout
        assert "genuinely distinct run" in p.stdout
        payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert payload["pairs"][0]["duplicate"] is False

    def test_a_duplicate_exits_non_zero(self, tmp_path):
        """It must fail loudly: a silent pass here would certify an artefact as a result."""
        proofs = {str(i): [f"exact h{i}"] for i in range(10)}
        runs = [sv_run(tmp_path, "sv", proofs), li_run(tmp_path, "li_copy", proofs)]
        p = run_script(tmp_path, runs)
        assert p.returncode != 0
        assert "DUPLICATE" in p.stdout
        assert "artefact" in p.stdout + p.stderr

    def test_the_exact_tie_is_reported_with_its_probability(self, tmp_path):
        runs = [sv_run(tmp_path, "sv", {"1": ["a"], "2": ["b"], "3": ["c"]}),
                li_run(tmp_path, "li", {"1": ["z"], "4": ["y"], "5": ["x"]})]
        p = run_script(tmp_path, runs)
        assert "EXACT TIE" in p.stdout
        assert "modal outcome" in p.stdout

    def test_mixed_seeds_are_flagged_as_weakening_the_test(self, tmp_path):
        """Across seeds, differing proofs prove nothing — they are expected within one arm too."""
        runs = [sv_run(tmp_path, "sv0", {"1": ["a"]}, seed=0),
                li_run(tmp_path, "li1", {"1": ["z"]}, seed=1)]
        p = run_script(tmp_path, runs)
        assert p.returncode == 0, p.stderr
        assert "span seeds [0, 1]" in p.stdout
        assert "FIXED seed" in p.stdout

    def test_passing_the_same_directory_twice_is_refused(self, tmp_path):
        d = sv_run(tmp_path, "sv", {"1": ["a"]})
        p = run_script(tmp_path, [d, d])
        assert p.returncode != 0
        assert "passed twice" in p.stdout + p.stderr

    def test_one_run_is_refused(self, tmp_path):
        p = run_script(tmp_path, [sv_run(tmp_path, "sv", {"1": ["a"]})])
        assert p.returncode != 0
        assert "at least twice" in p.stdout + p.stderr
