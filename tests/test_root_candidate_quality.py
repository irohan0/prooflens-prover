"""Root-state candidate quality — the paired version of a comparison that cannot be paired.

`PolicyStats.mean_candidate_logprob` is a run-level average and the arms diverge after their first
tactic, so its ordering (li -0.810, sv -0.835, none -0.887 on FATE-M) is suggestive and not a
measurement: the three numbers describe three different distributions of proof states. At depth 0
every arm faces the identical benchmark statement, which is the one place the comparison is clean.

Hermetic: reads run directories written by the fixture below.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from root_candidate_quality import paired, root_candidates  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def write_run(tmp_path, name, arm, per_problem, n_candidates=None):
    """`per_problem` maps id -> [(depth, logprob), ...]."""
    d = tmp_path / name
    d.mkdir()
    cfg = {"benchmark": "fate_m", "arm": arm, "policy_kind": "vllm"}
    if n_candidates is not None:
        cfg["n_candidates"] = n_candidates
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "started_utc": "2026-08-09T00:00:00+00:00",
        "config": cfg, "outcome": {"n_proved": 0},
    }))
    rows = []
    for pid, steps in per_problem.items():
        rows.append({
            "problem_id": pid, "status": "exhausted", "proved": False, "proof": None,
            "trace": [{"depth": depth, "tactic": f"t{i}", "logprob": lp,
                       "outcome": "failed", "elapsed_s": 0.1, "error": None}
                      for i, (depth, lp) in enumerate(steps)],
        })
    (d / "attempts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


class TestRootExtraction:
    def test_it_takes_only_depth_zero(self, tmp_path):
        d = write_run(tmp_path, "r", "li", {"a": [(0, -0.5), (0, -1.5), (1, -9.0), (2, -9.0)]},
                      n_candidates=50_000)
        _, per_problem = root_candidates(d)
        assert per_problem["a"] == [-0.5, -1.5], "deeper states are not shared between arms"

    def test_the_label_carries_the_first_stage_budget(self, tmp_path):
        d = write_run(tmp_path, "r", "li", {"a": [(0, -0.5)]}, n_candidates=50_000)
        assert root_candidates(d)[0] == "li@50k"

    def test_an_arm_without_a_first_stage_is_unannotated(self, tmp_path):
        d = write_run(tmp_path, "r", "sv", {"a": [(0, -0.5)]})
        assert root_candidates(d)[0] == "sv"

    def test_problems_with_no_root_candidates_are_omitted(self, tmp_path):
        # A problem whose statement failed to elaborate has no expansions at all, and including it
        # as an empty list would make `max()` raise on the best-candidate statistic.
        d = write_run(tmp_path, "r", "sv", {"a": [(0, -0.5)], "b": []})
        assert set(root_candidates(d)[1]) == {"a"}

    def test_non_finite_logprobs_are_dropped(self, tmp_path):
        # `propose` maps NaN to -inf for ordering; either would poison a mean.
        d = write_run(tmp_path, "r", "sv",
                      {"a": [(0, -0.5), (0, float("-inf")), (0, float("nan"))]})
        assert root_candidates(d)[1]["a"] == [-0.5]


class TestPairedStatistic:
    def test_it_pairs_on_shared_problems_only(self):
        a = {"x": [-1.0], "y": [-1.0], "gone": [-1.0]}
        b = {"x": [-0.5], "y": [-0.5]}
        r = paired(a, b, lambda v: float(np.mean(v)), 500, 500, 0)
        assert r["n_problems"] == 2
        assert r["mean_difference"] == pytest.approx(0.5)

    def test_a_consistent_improvement_is_detected(self):
        a = {f"p{i}": [-1.0] for i in range(60)}
        b = {f"p{i}": [-0.6] for i in range(60)}
        r = paired(a, b, lambda v: float(np.mean(v)), 2000, 2000, 0)
        assert r["mean_difference"] == pytest.approx(0.4)
        assert r["significant"] is True

    def test_no_difference_is_not_significant(self):
        a = {f"p{i}": [-1.0] for i in range(60)}
        r = paired(a, dict(a), lambda v: float(np.mean(v)), 2000, 2000, 0)
        assert r["mean_difference"] == 0.0
        assert r["significant"] is False

    def test_the_best_candidate_statistic_differs_from_the_mean(self):
        """Best-first tries the top-scoring candidate first, so the max is its own question."""
        a = {"p": [-1.0, -1.0]}
        b = {"p": [-0.1, -3.0]}
        mean = paired(a, b, lambda v: float(np.mean(v)), 200, 200, 0)["mean_difference"]
        best = paired(a, b, max, 200, 200, 0)["mean_difference"]
        assert mean == pytest.approx(-0.55)
        assert best == pytest.approx(0.9)


class TestEndToEnd:
    """Through the actual script: the unit tests above would pass on a version that computed
    everything correctly and printed the wrong arm, which is the gap that let `--n-candidates` be
    read, echoed and never applied."""

    def test_it_reports_every_arm_and_both_contrasts(self, tmp_path):
        none = write_run(tmp_path, "none_run", "none", {f"p{i}": [(0, -1.0)] for i in range(12)})
        sv = write_run(tmp_path, "sv_run", "sv", {f"p{i}": [(0, -0.8)] for i in range(12)})
        li = write_run(tmp_path, "li_run", "li", {f"p{i}": [(0, -0.6)] for i in range(12)},
                       n_candidates=50_000)
        out = tmp_path / "q.json"
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "root_candidate_quality.py"),
             "--run", str(none), "--run", str(sv), "--run", str(li),
             "--n-boot", "300", "--n-perm", "300", "--json-out", str(out)],
            capture_output=True, text=True, cwd=REPO,
        )
        assert p.returncode == 0, p.stderr
        assert "sv vs none" in p.stdout
        assert "li@50k vs none" in p.stdout
        # The contrast that is the research question must appear without being asked for.
        assert "li@50k vs sv" in p.stdout

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert [a["arm"] for a in payload["arms"]] == ["none", "sv", "li@50k"]
        assert {c["contrast"] for c in payload["comparisons"]} == {
            "sv vs none", "li@50k vs none", "li@50k vs sv",
        }

    def test_one_run_is_refused(self, tmp_path):
        one = write_run(tmp_path, "only", "li", {"p0": [(0, -1.0)]})
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "root_candidate_quality.py"),
             "--run", str(one)],
            capture_output=True, text=True, cwd=REPO,
        )
        assert p.returncode != 0
        assert "at least twice" in p.stdout + p.stderr

    def test_runs_sharing_no_problems_are_refused(self, tmp_path):
        a = write_run(tmp_path, "a_run", "none", {"p0": [(0, -1.0)]})
        b = write_run(tmp_path, "b_run", "li", {"q0": [(0, -1.0)]})
        p = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "root_candidate_quality.py"),
             "--run", str(a), "--run", str(b)],
            capture_output=True, text=True, cwd=REPO,
        )
        assert p.returncode != 0
        assert "share no problem" in p.stdout + p.stderr
