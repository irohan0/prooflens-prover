"""Contamination audit — what counts as the benchmark's answer sitting in the retrieval corpus.

The premise corpus is all of Mathlib, and some benchmark theorems are restatements of lemmas
Mathlib already contains. When that happens the retriever can hand over the theorem itself and the
proof is one `exact`. That is a valid proof, not a `sorry`-style cheat, so nothing in the Lean
backend flags it — the only defence is measuring it.

These tests pin the definition, which is deliberately narrow. Every case below that returns None is
a way the audit is allowed to under-count, and under-counting is the safe direction for a figure
whose job is to bound a concern.

Hermetic: no corpus file, no runs, no Lean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contamination_audit import one_step_citation, read_run  # noqa: E402
from prooflens_prover.eval.premises import load_corpus, resolve  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

#: A miniature corpus. `Nat.succ_le_succ` exercises resolution by suffix; `fun` is here because a
#: real Mathlib name really does end in it, which is what made the first version of this audit
#: report `exact fun h => ...` as a corpus citation.
EXACT = {"Sylow.normalizer_normalizer", "Nat.succ_le_succ", "isClosed_le", "Order.fun"}
BY_SUFFIX = {
    "normalizer_normalizer": {"Sylow.normalizer_normalizer"},
    "succ_le_succ": {"Nat.succ_le_succ"},
    "isClosed_le": {"isClosed_le"},
    "fun": {"Order.fun"},
}


def cite(*tactics):
    return one_step_citation(list(tactics), EXACT, BY_SUFFIX)


def test_a_single_exact_naming_a_corpus_premise_is_a_corpus_answer():
    assert cite("exact Sylow.normalizer_normalizer P") == (
        "Sylow.normalizer_normalizer", "exact Sylow.normalizer_normalizer P")


def test_apply_counts_as_well_as_exact():
    assert cite("apply isClosed_le")[0] == "isClosed_le"


def test_an_abbreviated_name_resolves_through_the_suffix_index():
    # `open Nat` lets a tactic write only the tail; the corpus stores the qualified name.
    assert cite("exact succ_le_succ h")[0] == "succ_le_succ"


def test_a_multi_step_proof_is_not_a_corpus_answer():
    # The prover had to do work first, so the corpus did not contain the answer outright.
    assert cite("intro x", "exact Sylow.normalizer_normalizer x") is None


def test_a_term_building_completion_is_not_a_corpus_answer():
    # `fun` is the last component of a real corpus name, so resolution alone accepts it. The
    # keyword check is what rejects it, and without that the audit over-counted by 4 problems.
    assert cite("exact fun hx => hx.mul_rat h") is None


def test_an_anonymous_constructor_is_not_a_corpus_answer():
    assert cite("exact ⟨0, by simp⟩") is None


def test_a_local_hypothesis_is_not_a_corpus_answer():
    # `h₁` names something in scope, not a premise anyone retrieved.
    assert cite("exact h₁") is None


def test_a_tactic_that_is_not_exact_or_apply_is_not_a_corpus_answer():
    assert cite("simp [isClosed_le]") is None


def test_an_empty_proof_is_not_a_corpus_answer():
    assert cite() is None
    assert one_step_citation(None, EXACT, BY_SUFFIX) is None


def test_blank_steps_do_not_make_a_one_step_proof_look_multi_step():
    assert cite("", "exact isClosed_le hf hg", "  ")[0] == "isClosed_le"


def test_resolution_returns_every_name_an_abbreviation_could_denote():
    # The ambiguity is real and is why the audit reports the surface token alongside the tactic:
    # a reader can check which premise was meant, and the script never has to guess.
    assert resolve("succ_le_succ", EXACT, BY_SUFFIX) == {"Nat.succ_le_succ"}
    assert resolve("nope", EXACT, BY_SUFFIX) is None


def test_load_corpus_indexes_both_the_full_name_and_its_last_component(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        '{"name": "Sylow.normalizer_normalizer"}\n{"name": "isClosed_le"}\n\n',
        encoding="utf-8")
    exact, by_suffix = load_corpus(corpus)
    assert exact == {"Sylow.normalizer_normalizer", "isClosed_le"}
    assert by_suffix["normalizer_normalizer"] == {"Sylow.normalizer_normalizer"}


def test_read_run_keeps_only_solved_problems(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "run_id": "run", "config": {"benchmark": "fate_m", "arm": "li", "n_candidates": 50000},
    }), encoding="utf-8")
    (d / "attempts.jsonl").write_text("\n".join([
        json.dumps({"problem_id": "1", "proved": True, "proof": ["exact isClosed_le"]}),
        json.dumps({"problem_id": "2", "proved": False, "proof": None}),
    ]), encoding="utf-8")
    benchmark, arm, solved = read_run(d)
    assert (benchmark, arm) == ("fate_m", "li@50k")
    assert set(solved) == {"1"}


@pytest.mark.parametrize("flag", ["--corpus", "--run"])
def test_the_script_requires_a_corpus_and_at_least_one_run(flag):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "contamination_audit.py"), flag, "x"],
        capture_output=True, text=True)
    assert proc.returncode != 0
