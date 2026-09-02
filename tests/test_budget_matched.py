"""Pricing the ensemble at equal generations, which is what §3.4 never did.

The union-of-arms statistic that drove an entire phase of this project (+14 of 327) compares two
arms at one seed against one arm at one seed — twice the generations on one side.
`budget_matched.py` holds generations fixed and reports what the second *architecture* is worth.

What is guarded here is the arithmetic that would be wrong quietly: the estimator's endpoints, the
budget pairing itself (k against 2k, the single thing the script exists to get right), and the
refusals that stop an unequal comparison being reported as an equal one.

Hermetic: no GPU, no Lean, no model.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from budget_matched import by_seed, curve, passk  # noqa: E402
from test_passk_profile import write_run  # noqa: E402


@pytest.fixture
def eight_seeds(tmp_path):
    """Both benchmarks, both arms, eight seeds each — the shape the sweep produced."""
    for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
        for arm in ("li", "sv"):
            for seed in range(8):
                # Overlapping-but-not-identical solve sets, so the ensemble is not degenerate.
                base = {str(i) for i in range(10)}
                extra = {str(20 + seed)} if arm == "li" else {str(40 + seed)}
                write_run(tmp_path, f"{bench}_{arm}_{seed}", arm=arm, seed=seed,
                          solved=base | extra, benchmark=bench, n_problems=n)
    return tmp_path


# --- the estimator ------------------------------------------------------------------------------

def test_passk_endpoints():
    draws = [{"a"}, {"a"}, set(), set()]
    # Never solved -> 0 at every k; always solved -> 1 at every k.
    assert passk(draws, "zzz", 1) == 0.0
    assert passk(draws, "zzz", 4) == 0.0
    assert passk([{"a"}] * 4, "a", 1) == 1.0
    # Solved by half the draws: one draw is a coin flip, all four is certain.
    assert passk(draws, "a", 1) == pytest.approx(0.5)
    assert passk(draws, "a", 4) == pytest.approx(1.0)
    # Monotone in k — more draws can only help.
    vals = [passk(draws, "a", k) for k in range(1, 5)]
    assert vals == sorted(vals)


def test_passk_at_k_equals_K_is_the_union():
    """At k = K the estimator must equal the plain union, or the two halves of §7 disagree."""
    draws = [{"a"}, {"b"}, {"b", "c"}, set()]
    ids = ["a", "b", "c", "d"]
    assert curve(draws, ids, 4).sum() == pytest.approx(3.0)  # a, b, c solved by someone


def test_passk_rejects_k_outside_range():
    with pytest.raises(ValueError):
        passk([{"a"}, set()], "a", 3)
    with pytest.raises(ValueError):
        passk([{"a"}, set()], "a", 0)


def test_ensemble_draw_dominates_either_arm_at_the_same_k():
    """A draw of both arms solves whatever either solved — the ensemble curve can never be lower."""
    li = [{"a"}, {"a", "b"}]
    sv = [{"c"}, set()]
    ens = [a | b for a, b in zip(li, sv, strict=True)]
    ids = ["a", "b", "c"]
    for k in (1, 2):
        assert (curve(ens, ids, k) >= curve(li, ids, k)).all()
        assert (curve(ens, ids, k) >= curve(sv, ids, k)).all()


# --- the budget pairing, which is the whole point -------------------------------------------------

def test_ensemble_at_k_is_paired_against_single_at_2k(eight_seeds):
    """The header must say 16,384 generations for --k 4, and the arms must be li@8 / sv@8.

    If this pairing slips to k against k the script silently reproduces the very error it was
    written to correct — the ensemble compared at twice the budget.
    """
    out = run(eight_seeds, "--k", "4")
    assert "16,384 generations/problem" in out
    assert "ensemble@4" in out and "li@8" in out and "sv@8" in out
    assert "li@4" not in out and "ensemble@8" not in out


def test_k_needing_more_seeds_than_exist_is_refused(eight_seeds):
    """--k 5 wants 10 single-arm draws from 8 seeds. Clamping would unequalise the budget."""
    out = run(eight_seeds, "--k", "5", expect_fail=True)
    assert "only 8 seeds" in out


def test_a_duplicated_arm_seed_is_refused(eight_seeds):
    write_run(eight_seeds, "dupe", arm="li", seed=3, solved={"1"}, benchmark="fate_m",
              n_problems=141)
    out = run(eight_seeds, expect_fail=True)
    assert "duplicate" in out.lower()


def test_arms_at_different_seeds_are_refused(eight_seeds):
    """An ensemble draw is both arms at ONE seed; without the pairing there is no such object."""
    for p in eight_seeds.glob("fate_m_sv_7"):
        for f in p.iterdir():
            f.unlink()
        p.rmdir()
    out = run(eight_seeds, expect_fail=True)
    assert "different seeds" in out


def test_unverified_runs_are_refused_but_allowed_with_the_flag(eight_seeds):
    (eight_seeds / "fate_m_li_0" / "verification.json").unlink()
    out = run(eight_seeds, expect_fail=True)
    assert "not been verified" in out
    assert "ensemble@4" in run(eight_seeds, "--allow-unverified")


def test_a_short_benchmark_is_refused(eight_seeds):
    """A rate over the wrong denominator is not comparable to a published one.

    The check is on the *union* over runs, so every run of the benchmark has to load short before it
    fires — one truncated run is covered by its siblings, which is the correct behaviour and is what
    an earlier version of this test got wrong.
    """
    for d in eight_seeds.glob("fate_m_*"):
        rows = (d / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[:-1]
        (d / "attempts.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = run(eight_seeds, expect_fail=True)
    assert "140 problems, not 141" in out


def test_one_short_run_among_siblings_is_not_refused(eight_seeds):
    """The complement of the above: a resumed run missing a row must not fail the whole contrast."""
    d = eight_seeds / "fate_m_li_0"
    rows = (d / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[:-1]
    (d / "attempts.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert "POOLED  (327 problems" in run(eight_seeds)


# --- discounting --------------------------------------------------------------------------------

def test_a_rejected_proof_does_not_count_toward_any_curve(eight_seeds):
    """The discount is why this script disagreed with a hand-rolled one by exactly one problem."""
    before = run(eight_seeds)
    write_run(eight_seeds, "fate_m_sv_0", arm="sv", seed=0, solved={str(i) for i in range(10)} |
              {"40", "99"}, benchmark="fate_m", n_problems=141, rejected=["99"])
    after = run(eight_seeds)
    assert _solved(before, "sv@8", "fate_m") == _solved(after, "sv@8", "fate_m")


# --- reporting ----------------------------------------------------------------------------------

def test_by_seed_reads_arms_and_seeds(eight_seeds):
    dirs = sorted(p for p in eight_seeds.iterdir() if p.name.startswith("fate_m"))
    solved, attempted = by_seed(dirs)
    assert set(solved) == {"li", "sv"}
    assert sorted(solved["li"]) == list(range(8))
    assert len(attempted) == 141


def test_agreement_gate_is_printed(eight_seeds):
    """CI and permutation must be reported as agreeing or not — a silent check is no check."""
    assert "agree:" in run(eight_seeds)


def test_pooled_section_covers_both_benchmarks(eight_seeds):
    out = run(eight_seeds)
    assert "POOLED  (327 problems" in out


def test_identical_arms_give_a_zero_contrast(tmp_path):
    """If both arms solve the same problems, the ensemble adds nothing and must say +0.00."""
    for bench, n in (("proofnet_test", 186), ("fate_m", 141)):
        for arm in ("li", "sv"):
            for seed in range(8):
                write_run(tmp_path, f"{bench}_{arm}_{seed}", arm=arm, seed=seed,
                          solved={str(i) for i in range(10)}, benchmark=bench, n_problems=n)
    out = run(tmp_path)
    pooled = out[out.index("POOLED  ("):]
    assert "+0.00 problems" in pooled
    assert "permutation p = 1.0000" in pooled


# --- helpers --------------------------------------------------------------------------------------

def run(root, *extra, expect_fail=False):
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "budget_matched.py"),
         "--results-root", str(root), "--n-boot", "200", "--n-perm", "200", *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    if expect_fail:
        assert r.returncode != 0, f"expected a refusal, got:\n{r.stdout}"
    else:
        assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


def _solved(out, name, bench):
    section = out[out.index(bench):]
    line = next(x for x in section.splitlines() if x.strip().startswith(name))
    return float(line.split("expected solved:")[1].split("(")[0])


def test_numpy_is_actually_used():
    """`curve` must return an ndarray: the contrast helpers index it as one."""
    assert isinstance(curve([{"a"}], ["a"], 1), np.ndarray)
