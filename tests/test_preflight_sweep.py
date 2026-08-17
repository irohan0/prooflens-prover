"""The sweep preflight — a checker that cannot fail is not a checker.

Every test here asserts a **refusal**. The script's whole value is exiting non-zero on a
configuration that would have wasted a GPU allocation, so the positive path is the cheap half and
the negative paths are the point.

Each refusal corresponds to a failure this project has already paid for:

* a corpus-id mismatch — arms that were never ranking the same premise set;
* a benchmark path resolving to the wrong problem count — a rate against a different denominator;
* a budget whose projected wall clock exceeds the SLURM limit — a job killed part-way, leaving a
  manifest describing an experiment that never finished;
* a policy that raises on its first `propose` — after `import Mathlib` and a 15 GB model load.

Hermetic: no GPU, no model, no Lean, no index.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preflight_sweep import (  # noqa: E402
    BASELINE,
    EXPECTED_CORPUS_ID,
    Check,
    check_index,
    index_corpus_id,
    project_hours,
)

REPO = Path(__file__).resolve().parent.parent

#: A ProofNet-shaped statement. `load_benchmark` needs `id` and `formal_statement`.
STATEMENT = "theorem exercise_1_1 (G : Type*) [Group G] (a b : G) : a * b = b * a := sorry"


def write_index(tmp_path, name, corpus_id=EXPECTED_CORPUS_ID):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"n_docs": 276070, "corpus_id": corpus_id}),
                                 encoding="utf-8")
    return d


def write_benchmark(tmp_path, name="proofnet_test", n=186):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.jsonl").write_text("\n".join(
        json.dumps({"id": str(i), "formal_statement": f"import Mathlib\n\n{STATEMENT}"})
        for i in range(n)
    ), encoding="utf-8")
    return d


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "preflight_sweep.py"), *map(str, args)],
        capture_output=True, text=True, encoding="utf-8",
    )


# --- the index guard ----------------------------------------------------------------------------

def test_an_index_over_a_different_corpus_is_refused(tmp_path):
    c = Check()
    check_index(c, "index", write_index(tmp_path, "wrong", corpus_id="999:deadbeef"))
    assert c.failures and "corpus_id" in c.failures[0]


def test_a_missing_index_directory_is_refused(tmp_path):
    c = Check()
    check_index(c, "index", tmp_path / "nope")
    assert c.failures and "does not exist" in c.failures[0]


def test_an_index_with_no_metadata_is_refused(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    c = Check()
    check_index(c, "index", d)
    assert c.failures and "meta.json" in c.failures[0]


def test_the_right_corpus_passes(tmp_path):
    c = Check()
    check_index(c, "index", write_index(tmp_path, "right"))
    assert not c.failures


def test_index_corpus_id_reads_the_metadata(tmp_path):
    assert index_corpus_id(write_index(tmp_path, "i")) == EXPECTED_CORPUS_ID
    assert index_corpus_id(tmp_path / "absent") is None


# --- the wall-clock projection --------------------------------------------------------------

def test_the_baseline_is_the_published_budget_so_a_matching_request_reproduces_it():
    # The table is measured at 64 x 16. Asking for exactly that must return the measured total,
    # or every projection built on it is scaled from the wrong origin.
    retrieval, gen = BASELINE[("proofnet_test", "li")]
    assert project_hours("proofnet_test", "li", 64, 16, 186) == pytest.approx(retrieval + gen)


def test_doubling_samples_does_not_change_retrieval_cost():
    # One retrieval per expansion, so samples cannot move it. Getting this wrong would double the
    # projected cost of the late-interaction arm and rule out a budget that actually fits.
    retrieval, gen = BASELINE[("proofnet_test", "li")]
    assert project_hours("proofnet_test", "li", 64, 32, 186) == pytest.approx(retrieval + 2 * gen)


def test_doubling_expansions_scales_both_halves():
    retrieval, gen = BASELINE[("proofnet_test", "li")]
    assert project_hours("proofnet_test", "li", 128, 16, 186) == pytest.approx(
        2 * retrieval + 2 * gen)


def test_a_limited_run_is_projected_pro_rata():
    full = project_hours("fate_m", "sv", 64, 16, 141)
    assert project_hours("fate_m", "sv", 64, 16, 60) == pytest.approx(full * 60 / 141)


def test_fusion_is_projected_from_the_late_interaction_baseline():
    # It queries both retrievers, so it cannot be cheaper than the more expensive of them.
    assert project_hours("fate_m", "fusion", 64, 16, 141) >= project_hours(
        "fate_m", "li", 64, 16, 141)


def test_premise_free_mixing_is_charged_for():
    # A second prefill per expansion, measured at +6% in the pilot (2.32 h -> 2.46 h). Unpriced, the
    # projection would clear a limit the run then misses.
    plain = project_hours("proofnet_test", "li", 64, 32, 186)
    mixed = project_hours("proofnet_test", "li", 64, 32, 186, premise_free=0.25)
    assert mixed > plain
    assert mixed / plain == pytest.approx(1.06)


def test_an_unmeasured_benchmark_projects_nothing_rather_than_guessing():
    assert project_hours("minif2f_test", "li", 64, 16, 244) is None


# --- end to end ---------------------------------------------------------------------------------

def test_a_clean_configuration_exits_zero(tmp_path):
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path),
                  "--samples-per-step", 32, "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PREFLIGHT CLEAN" in out.stdout


def test_a_budget_that_would_be_killed_by_the_wall_clock_is_refused(tmp_path):
    # 128 x 32 on ProofNet projects past 8 hours. Discovering that at hour 8 costs the allocation
    # and leaves a manifest describing an experiment that never finished.
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path),
                  "--max-expansions", 128, "--samples-per-step", 32,
                  "--results-root", tmp_path)
    assert out.returncode == 1
    assert "wall clock" in out.stdout and "Shard with" in out.stdout


def test_a_limited_run_warns_that_its_projection_is_not_a_uniform_sample(tmp_path):
    # Measured: the budget pilot on ProofNet's first 60 was projected at 1.71 h and took 2.32 h.
    # A 36% under-estimate eats the whole default headroom, so a --limit projection must not be
    # presented with the same confidence as a full-benchmark one.
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path),
                  "--limit", 60, "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "not a uniform sample" in out.stdout


def test_a_full_run_projection_is_stated_without_that_caveat(tmp_path):
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path), "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "not a uniform sample" not in out.stdout


def test_a_wrong_problem_count_is_refused(tmp_path):
    # A mis-resolved --data-root that happens to contain a smaller file would otherwise produce a
    # rate against a different denominator and never say so.
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path, n=12),
                  "--results-root", tmp_path)
    assert out.returncode == 1
    assert "expected 186" in out.stdout


def test_a_mismatched_corpus_stops_the_submission(tmp_path):
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx", corpus_id="1:bad"),
                  "--data-root", write_benchmark(tmp_path),
                  "--results-root", tmp_path)
    assert out.returncode == 1
    assert "PREFLIGHT FAILED" in out.stdout


def test_fusion_without_both_indices_is_refused(tmp_path):
    out = run_cli("--benchmark", "fate_m", "--arm", "fusion",
                  "--index-li", write_index(tmp_path, "li"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--results-root", tmp_path)
    assert out.returncode == 1
    assert "index (sv)" in out.stdout


def test_the_none_arm_needs_no_index(tmp_path):
    out = run_cli("--benchmark", "fate_m", "--arm", "none",
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr


def test_a_model_path_that_is_not_a_model_directory_is_refused(tmp_path):
    out = run_cli("--benchmark", "fate_m", "--arm", "none",
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--model", tmp_path, "--results-root", tmp_path)
    assert out.returncode == 1
    assert "config.json" in out.stdout


def test_the_premise_free_split_is_exercised_rather_than_only_parsed(tmp_path):
    # The point of constructing the real policy: a bad fraction must fail here, not after a 15 GB
    # model load.
    out = run_cli("--benchmark", "fate_m", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--premise-free-fraction", 0.25, "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "premise_free_fraction=0.25" in out.stdout
