"""Resuming a run must continue *that* run, not start a different experiment inside it.

`RunManifest.load` deliberately preserves the original `run_id`, `git_commit` and — the part that
matters here — the original `config` and `seed`. That is the right choice: a resumed run is the same
run, and rewriting its provenance to the restart time would misattribute its results. But it means a
mismatched resume is **invisible in the record**. The manifest describes configuration A, half the
attempts were produced under configuration B, and the reported rate is over all of them.

The seed is the sharpest case and the reason this file exists. `slurm/prove_benchmark_llm.sbatch`
defaults `SEED` to 0, so resuming a seed-6 run without repeating the seed appends seed-0 draws to
it. `passk_union.py` refuses duplicate seeds precisely to stop one draw being counted twice — and it
cannot catch this, because the only seed it can read is the one in the manifest, which still says 6.
Two draws would enter pass@8 as one, and every number downstream would be wrong by an unknown amount
in an unknown direction.

Hermetic: no GPU, no Lean, no model, no index.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.prover.search import SearchConfig  # noqa: E402
from prove_benchmark import resume_mismatches, same_index_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SBATCH = REPO_ROOT / "slurm" / "prove_benchmark_llm.sbatch"

#: The interrupted run this file keeps resuming: ProofNet / sv / seed 6 at the sweep's budget.
SWEEP = {
    "benchmark": "proofnet_test",
    "arm": "sv",
    "policy_kind": "vllm",
    "index": "data/index/sv_ft_novel_lr3e6",
    "n_candidates": None,
    "policy_config": {"premise_free_fraction": 0.25},
    "search": SearchConfig(max_expansions=64, samples_per_step=32).to_dict(),
}


def manifest(seed: int = 6, **overrides):
    return SimpleNamespace(run_id="proofnet_test_sv_vllm_x", seed=seed,
                           config={**SWEEP, **overrides})


def args(**overrides):
    base = dict(benchmark="proofnet_test", arm="sv", policy="vllm", seed=6,
                index=Path("data/index/sv_ft_novel_lr3e6"), premise_free_fraction=0.25)
    return SimpleNamespace(**{**base, **overrides})


def cfg(**overrides):
    return SearchConfig(**{"max_expansions": 64, "samples_per_step": 32, **overrides})


# --- the seed, which is why this exists ----------------------------------------------------------

def test_the_matching_resume_is_allowed():
    assert resume_mismatches(manifest(), args(), cfg(), None) == []


def test_resuming_a_seed_6_run_at_the_default_seed_is_refused():
    # The exact submission this guards: RESUME=... without SEED=6, since the sbatch defaults to 0.
    bad = resume_mismatches(manifest(seed=6), args(seed=0), cfg(), None)
    assert any("seed" in line for line in bad)
    assert any("6" in line and "0" in line for line in bad)


def test_the_seed_message_names_both_values_so_the_fix_is_readable():
    (line,) = [x for x in resume_mismatches(manifest(seed=6), args(seed=3), cfg(), None)
               if "seed" in x and "search" not in x]
    assert "the run has 6" in line and "you passed 3" in line


# --- the budget ---------------------------------------------------------------------------------

def test_resuming_at_the_wrong_sample_count_is_refused():
    # 32 -> 16 halves the generation budget for the unfinished half of one benchmark.
    bad = resume_mismatches(manifest(), args(), cfg(samples_per_step=16), None)
    assert any("search.samples_per_step" in line for line in bad)


def test_resuming_at_the_wrong_expansion_cap_is_refused():
    bad = resume_mismatches(manifest(), args(), cfg(max_expansions=128), None)
    assert any("search.max_expansions" in line for line in bad)


def test_the_whole_search_dict_is_compared_not_a_chosen_subset():
    # Every field in SearchConfig is part of the budget. A subset would let the next field added
    # drift silently, which is how `--n-candidates` was echoed but never applied for five hours.
    for field in SearchConfig().to_dict():
        was = SearchConfig(max_expansions=64, samples_per_step=32).to_dict()
        moved = {**was, field: (not was[field]) if isinstance(was[field], bool) else 999}
        bad = resume_mismatches(
            manifest(search=was),
            args(),
            SimpleNamespace(to_dict=lambda m=moved: m),
            None,
        )
        assert any(f"search.{field}" in line for line in bad), f"{field} can drift unnoticed"


def test_resuming_with_a_different_premise_free_fraction_is_refused():
    bad = resume_mismatches(manifest(), args(premise_free_fraction=0.0), cfg(), None)
    assert any("premise_free_fraction" in line for line in bad)


# --- the experiment identity ---------------------------------------------------------------------

def test_resuming_across_benchmarks_is_refused():
    bad = resume_mismatches(manifest(), args(benchmark="fate_m"), cfg(), None)
    assert any("benchmark" in line for line in bad)


def test_resuming_across_arms_is_refused():
    bad = resume_mismatches(manifest(), args(arm="li"), cfg(), None)
    assert any("arm" in line for line in bad)


def test_resuming_across_policies_is_refused():
    bad = resume_mismatches(manifest(), args(policy="repertoire"), cfg(), None)
    assert any("policy_kind" in line for line in bad)


def test_resuming_against_a_different_index_is_refused():
    other = args(index=Path("data/index/li_ft_novel_bm25"))
    assert any("index" in line for line in resume_mismatches(manifest(), other, cfg(), None))


def test_the_same_index_spelled_two_ways_is_not_a_mismatch():
    # The sbatch cds to the repo and passes a relative path; a hand run may pass the absolute one.
    # Refusing over the spelling would block a legitimate resume for no reason, and a manifest
    # written on another platform carries that platform's separator.
    assert same_index_path("data/index/sv_ft_novel_lr3e6",
                           "/home/x/prooflens-prover/data/index/sv_ft_novel_lr3e6")
    assert same_index_path("data/index/sv_ft_novel_lr3e6", "data\\index\\sv_ft_novel_lr3e6")
    assert same_index_path("data/index/sv_ft_novel_lr3e6/", "data/index/sv_ft_novel_lr3e6")


def test_two_genuinely_different_indices_are_still_caught_by_that_tolerance():
    assert not same_index_path("data/index/sv_ft_novel_lr3e6", "data/index/li_ft_novel_bm25")
    assert not same_index_path("data/index/sv_ft_novel_lr3e6", "data/index/sv_zeroshot")


def test_a_run_with_no_index_recorded_is_not_treated_as_a_mismatch():
    # The `none` and `fusion` arms both write index=None; so does an older manifest.
    assert same_index_path(None, "data/index/sv_ft_novel_lr3e6")
    assert same_index_path("data/index/sv_ft_novel_lr3e6", None)


def test_resuming_at_a_different_first_stage_budget_is_refused():
    # Measured recall@10 for late interaction: 0.443 at 1,000 candidates, 0.979 at 50,000. Two
    # halves of one run differing here are not the same retriever.
    bad = resume_mismatches(manifest(n_candidates=50000), args(), cfg(), 1000)
    assert any("n_candidates" in line for line in bad)


def test_every_mismatch_is_reported_at_once_rather_than_one_per_resubmission():
    bad = resume_mismatches(manifest(), args(seed=0, arm="li"), cfg(samples_per_step=16), None)
    assert len(bad) >= 3


# --- a field absent from an older manifest is not a mismatch -------------------------------------

def test_a_field_missing_from_an_older_manifest_does_not_block_a_resume():
    # Runs predating `premise_free_fraction` have no such key. Treating absent as 0.0-and-mismatched
    # would make every pre-sweep run unresumable, which is a different bug.
    older = manifest(policy_config={}, search={})
    assert resume_mismatches(older, args(), cfg(), None) == []


# --- the sbatch side -----------------------------------------------------------------------------

def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations, so a multi-line argument list reads as one line."""
    out: list[str] = []
    for raw in text.splitlines():
        if out and out[-1].endswith("\\"):
            out[-1] = out[-1][:-1].rstrip() + " " + raw.strip()
        else:
            out.append(raw)
    return out


def test_the_sbatch_passes_resume_through_to_the_script():
    lines = _logical_lines(SBATCH.read_text(encoding="utf-8"))
    invocation = next(x for x in lines if "prove_benchmark.py" in x and "--policy vllm" in x)
    assert '${RESUME:+--resume "$RESUME"}' in invocation, (
        "RESUME is documented but never reaches the script — the failure mode this project has "
        "already paid for once with --n-candidates"
    )


def test_resume_is_plumbed_rather_than_left_to_extra():
    # Via EXTRA it would work, and would take SEED's default of 0. The whole point is that the
    # script sees --seed alongside --resume so its guard can compare them.
    text = SBATCH.read_text(encoding="utf-8")
    assert 'RESUME="${RESUME:-}"' in text


def test_resume_inside_an_array_job_is_refused_by_the_sbatch():
    # Eight tasks appending to one attempts.jsonl records eight seeds as one run.
    text = SBATCH.read_text(encoding="utf-8")
    block = text[text.index('RESUME="${RESUME:-}"'):text.index('EXTRA="${EXTRA:-}"')]
    assert "SLURM_ARRAY_TASK_ID" in block and "exit 1" in block


def test_the_script_refuses_a_mismatched_resume_rather_than_warning():
    src = (REPO_ROOT / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    block = src[src.index("if args.resume is not None:"):src.index("done: set[str] = set()")]
    assert "resume_mismatches" in block and "raise SystemExit" in block


def test_the_refusal_reaches_the_user_before_the_lean_backend_is_built():
    # After `import Mathlib` it has already cost 158 s node-local, and the operator has walked away.
    src = (REPO_ROOT / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    assert src.index("resume_mismatches(manifest") < src.index("LeanInteractBackend(")


def test_the_manifest_json_of_a_real_run_carries_every_field_the_guard_compares(tmp_path):
    # The guard is only as good as the manifest. If a key it reads is not written, the check is a
    # no-op that reports nothing and passes everything.
    written = {"benchmark": "x", "arm": "sv", "policy_kind": "vllm", "index": "i",
               "n_candidates": None, "policy_config": {"premise_free_fraction": 0.25},
               "search": SearchConfig().to_dict()}
    src = (REPO_ROOT / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")
    create = src[src.index("manifest = RunManifest.create("):src.index("capture_lean=True")]
    for key in written:
        assert f'"{key}":' in create, f"the guard reads {key}, which no run writes"
    (tmp_path / "manifest.json").write_text(json.dumps({"config": written, "seed": 6}))


# --- output encoding, which has already cost this project a cycle --------------------------------

def test_both_sbatch_scripts_pin_the_output_encoding():
    """Lean is a unicode language and SLURM redirects stdout to a file.

    Python then takes its encoding from the locale, so a batch job with LANG unset gets ASCII and
    printing a goal state containing U+2115 raises UnicodeEncodeError — killing a job that may
    already have finished every problem. `prove_benchmark_llm.sbatch` opens by recording that an
    encoding traceback hid for a full cycle behind a split stderr; this is the same class.
    """
    for name in ("prove_benchmark_llm.sbatch", "verify_proofs.sbatch"):
        text = (REPO_ROOT / "slurm" / name).read_text(encoding="utf-8")
        assert 'export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"' in text, name
