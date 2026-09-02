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
    EXTRAPOLATION_SAFETY,
    FUSION_SV_CPU_MS,
    SWEEP_QUERY_MS,
    Check,
    check_index,
    fusion_retrieval_factor,
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


def write_model(tmp_path, name="model"):
    """A directory the model check accepts, so `--model` stops being a warning of its own."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
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


def test_doubling_samples_scales_generation_and_carries_the_safety_factor():
    """The model still holds retrieval constant in `samples`. The sweep showed that is wrong.

    It was a reasonable belief — one query per expansion, and samples do not change the expansion
    cap — but more samples keep the frontier alive longer, so more expansions actually execute.
    Measured on ProofNet late interaction: 5,631 queries at 16 samples, 7,575 at 32.

    Rather than fit a curve to two points, the shortfall is absorbed by `EXTRAPOLATION_SAFETY`,
    which is applied to anything away from the point BASELINE was measured at. The guarantee that
    matters is the next test's: never project under what the sweep really took.
    """
    retrieval, gen = BASELINE[("proofnet_test", "li")]
    assert project_hours("proofnet_test", "li", 64, 32, 186) == pytest.approx(
        (retrieval + 2 * gen) * EXTRAPOLATION_SAFETY)


def test_the_measured_point_itself_carries_no_safety_factor():
    # At 64 x 16 with no mixing the projection is a measurement, not an extrapolation. Inflating it
    # there would make every baseline in this file disagree with the runs it was taken from.
    retrieval, gen = BASELINE[("proofnet_test", "li")]
    assert project_hours("proofnet_test", "li", 64, 16, 186) == pytest.approx(retrieval + gen)


# NB `bench`, not `benchmark`: pytest-benchmark owns a fixture of that name, and a parametrised
# argument shadowing it crashes the plugin during report generation rather than failing the test.
@pytest.mark.parametrize(("bench", "arm", "slowest_seed_hours"), [
    ("proofnet_test", "li", 6.89),
    ("proofnet_test", "sv", 4.54),
    ("fate_m", "li", 3.58),
    ("fate_m", "sv", 2.37),
])
def test_the_projection_never_comes_in_under_what_the_sweep_actually_took(
        bench, arm, slowest_seed_hours):
    """The guarantee this whole projection exists to provide.

    Before recalibration it failed on all four cells, by 9–18%, and worst on ProofNet late
    interaction — the job with the least headroom in the study. Under-projecting greenlights a job
    that then dies at the wall clock, which is exactly what a preflight is supposed to prevent.

    Compared against the **slowest** of the eight seeds, not the median: an array job is only as
    good as its worst task.
    """
    n = {"proofnet_test": 186, "fate_m": 141}[bench]
    projected = project_hours(bench, arm, 64, 32, n, premise_free=0.25)
    assert projected >= slowest_seed_hours, (
        f"{bench}/{arm} projects {projected:.2f} h against a measured {slowest_seed_hours:.2f} h"
    )


def test_doubling_expansions_scales_both_halves():
    retrieval, gen = BASELINE[("proofnet_test", "li")]
    assert project_hours("proofnet_test", "li", 128, 16, 186) == pytest.approx(
        (2 * retrieval + 2 * gen) * EXTRAPOLATION_SAFETY)


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
    assert "wall clock" in out.stdout and "shard it" in out.stdout


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


# --- pricing the fusion arm ----------------------------------------------------------------------

def test_fusion_is_priced_from_measured_latencies_not_a_guessed_multiplier():
    """It used to assume single-vector adds ~10% to a fused query, from its cost *on a GPU*.

    The fusion arm puts single-vector on the CPU by default, precisely so the two indices do not
    have to share a card with a 7B model. On ProofNet — the tightest job in the study — that
    difference is over half an hour, and half an hour is the whole margin.
    """
    li_ms = SWEEP_QUERY_MS[("proofnet_test", "li")]
    assert fusion_retrieval_factor("proofnet_test", 0.0) == pytest.approx(1.0)
    assert fusion_retrieval_factor("proofnet_test", li_ms) == pytest.approx(2.0)
    assert fusion_retrieval_factor("proofnet_test", 400.0) > 1.3


def test_a_slower_single_vector_projects_a_longer_run():
    n = 186
    fast = project_hours("proofnet_test", "fusion", 64, 32, n, 0.25, fusion_sv_ms=38.8)
    slow = project_hours("proofnet_test", "fusion", 64, 32, n, 0.25, fusion_sv_ms=400.0)
    assert slow > fast


def test_the_cpu_cost_is_the_measured_one_not_the_placeholder():
    """The pilot measured 512.65 and 505.37 ms/query on the two fusion modes.

    The placeholder it replaced was 400.0, called "deliberately far above any plausible value". It
    was below the truth, which is the dangerous direction: a projection built on it runs short and
    greenlights a job that dies at the wall clock. Pinned so it cannot drift back to a guess.
    """
    assert FUSION_SV_CPU_MS >= 512.65
    # and it is an order of magnitude above the same retriever on a GPU, which is the finding
    assert FUSION_SV_CPU_MS > 10 * max(SWEEP_QUERY_MS[("fate_m", "sv")],
                                       SWEEP_QUERY_MS[("proofnet_test", "sv")])


def test_the_gpu_alternative_is_available_for_the_tighter_benchmark():
    # `--fusion-sv-device cuda` trades 480 ms/query for ~1.5 GB beside a 7B model. On ProofNet that
    # is the difference between fitting an 8 h job and needing a resume, so the figure has to exist.
    from preflight_sweep import FUSION_SV_GPU_MS
    assert FUSION_SV_GPU_MS == SWEEP_QUERY_MS[("proofnet_test", "sv")]
    assert FUSION_SV_GPU_MS < FUSION_SV_CPU_MS / 10


def test_fusion_is_always_treated_as_an_extrapolation():
    # No fusion run has ever been timed at any budget, so even a request at exactly the point
    # BASELINE was measured at is a guess for this arm.
    retrieval, gen = BASELINE[("fate_m", "li")]
    plain = project_hours("fate_m", "li", 64, 16, 141)
    fused = project_hours("fate_m", "fusion", 64, 16, 141)
    assert plain == pytest.approx(retrieval + gen)
    assert fused > plain * EXTRAPOLATION_SAFETY * 0.99


# --- the three-way wall-clock gate ----------------------------------------------------------------

def test_a_run_over_the_headroom_but_under_the_limit_warns_instead_of_refusing(tmp_path):
    """A hard failure here would have blocked the pass@8 sweep, which worked.

    Its late-interaction jobs project past the 80% headroom and finished at 6.89 h under an 8 h
    limit. What deserves a refusal is a projection past the limit itself; between the two the run is
    resumable, every attempt is fsynced, and one re-queue costs far less than not running.
    """
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path),
                  "--samples-per-step", 32, "--premise-free-fraction", 0.25,
                  "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PREFLIGHT CLEAN" in out.stdout
    assert "ONE resume" in out.stdout


def test_the_fusion_projection_is_labelled_as_unmeasured(tmp_path):
    out = run_cli("--benchmark", "fate_m", "--arm", "fusion",
                  "--index-sv", write_index(tmp_path, "sv"),
                  "--index-li", write_index(tmp_path, "li"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--samples-per-step", 32, "--premise-free-fraction", 0.25,
                  "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "NOT measured" in out.stdout


def test_passing_a_measured_latency_says_so(tmp_path):
    out = run_cli("--benchmark", "fate_m", "--arm", "fusion",
                  "--index-sv", write_index(tmp_path, "sv"),
                  "--index-li", write_index(tmp_path, "li"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--samples-per-step", 32, "--premise-free-fraction", 0.25,
                  "--fusion-sv-ms", 142.0, "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "measured" in out.stdout and "142" in out.stdout


# --- the fusion arm's own guards, which a stub retriever cannot reach ---------------------------

def test_the_real_fusion_retriever_is_constructed_not_stubbed(tmp_path):
    """`FusionRetriever` has guards no stub can trigger, and they decide whether the arm runs.

    The policy check builds one `StubRetriever` standing in for the whole retriever, so an unknown
    mode or a fetch depth below the request would sail past it and surface only after the queue
    wait, the model load and `import Mathlib`.
    """
    out = run_cli("--benchmark", "fate_m", "--arm", "fusion",
                  "--index-sv", write_index(tmp_path, "sv"),
                  "--index-li", write_index(tmp_path, "li"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "fusion" in out.stdout
    assert "mode=rrf" in out.stdout
    assert "('sv', 'li')" in out.stdout


def test_a_fetch_depth_below_the_request_is_refused(tmp_path):
    # At fetch_k == top_k the two rankings are cut before they can disagree, and fusion becomes
    # whichever retriever happens to be listed first — reported as a fusion result.
    out = run_cli("--benchmark", "fate_m", "--arm", "fusion",
                  "--index-sv", write_index(tmp_path, "sv"),
                  "--index-li", write_index(tmp_path, "li"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--fusion-fetch-k", 4, "--top-k", 10, "--results-root", tmp_path)
    assert out.returncode == 1
    assert "degenerates to one retriever" in out.stdout


def test_an_unknown_fusion_mode_is_rejected_at_the_command_line(tmp_path):
    out = run_cli("--benchmark", "fate_m", "--arm", "fusion",
                  "--index-sv", write_index(tmp_path, "sv"),
                  "--index-li", write_index(tmp_path, "li"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--fusion-mode", "borda", "--results-root", tmp_path)
    assert out.returncode == 2                      # argparse, before anything else runs
    assert "borda" in out.stderr


def test_the_interleave_mode_also_constructs(tmp_path):
    # The alternative rule the pilot has to choose between. If it could not be preflighted, the
    # pilot would be one arm not two.
    out = run_cli("--benchmark", "fate_m", "--arm", "fusion",
                  "--index-sv", write_index(tmp_path, "sv"),
                  "--index-li", write_index(tmp_path, "li"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--fusion-mode", "interleave", "--results-root", tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "mode=interleave" in out.stdout


def test_the_preflight_defaults_match_the_runner_it_is_checking():
    """A default restated here that drifts from `prove_benchmark.py` makes this check a fiction.

    Not hypothetical: `--temperature` and `--top-p` were restated in this project and silently
    overrode REAL-Prover's 1.5/0.9 on every run, because an argparse default is always passed.
    """
    import prove_benchmark
    from prooflens_prover.retrieval.base import DEFAULT_TOP_K
    from prooflens_prover.retrieval.fusion import DEFAULT_FETCH_K

    src = (REPO / "scripts" / "preflight_sweep.py").read_text(encoding="utf-8")
    assert "default=DEFAULT_TOP_K" in src
    assert "default=DEFAULT_FETCH_K" in src
    assert "choices=sorted(FUSION_MODES)" in src
    # and the runner takes them from the same place
    runner = (REPO / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    assert "default=DEFAULT_FETCH_K" in runner
    assert prove_benchmark.DEFAULT_FETCH_K == DEFAULT_FETCH_K
    assert DEFAULT_TOP_K == 10


def test_a_warning_reaches_the_headline_not_a_footnote(tmp_path):
    """"PREFLIGHT CLEAN" followed by a footnote reads as a clean bill of health.

    The warnings this script emits are not trivia — a projection past the headroom means the job
    will need a resume, and a `--limit` subset means the projection does not scale pro rata. Both
    are decisions, and a decision printed under a line saying "safe to submit" is a decision nobody
    makes.
    """
    out = run_cli("--benchmark", "proofnet_test", "--arm", "li",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path),
                  "--samples-per-step", 32, "--premise-free-fraction", 0.25,
                  "--model", write_model(tmp_path), "--results-root", tmp_path)
    assert out.returncode == 0
    assert "WITH 1 WARNING" in out.stdout
    head = out.stdout[out.stdout.index("PREFLIGHT CLEAN"):]
    assert "ONE resume" in head, "the warning must be on or after the headline, not buried above it"


def test_a_configuration_with_nothing_to_flag_says_so_plainly(tmp_path):
    out = run_cli("--benchmark", "fate_m", "--arm", "sv",
                  "--index", write_index(tmp_path, "idx"),
                  "--data-root", write_benchmark(tmp_path, "fate_m", n=141),
                  "--model", write_model(tmp_path), "--results-root", tmp_path)
    assert out.returncode == 0
    assert "safe to submit." in out.stdout
    assert "WARNING" not in out.stdout
