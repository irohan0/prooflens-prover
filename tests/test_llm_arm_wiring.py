"""The plumbing between `--policy vllm` and the code, and between runs and Table 1.

Two failures this guards against, both of the shape that has already cost this project cluster time:

1. A flag that is parsed, echoed and never applied. `--n-candidates` did exactly that for a
   five-hour run.
2. A table that cannot tell two incomparable experiments apart. A 7B language model and a 19-tactic
   repertoire both produce runs for `arm=li` on `fate_m`; keyed on (benchmark, arm) alone, the newer
   would silently take the cell, under a caption claiming the generator was held fixed.

Hermetic: no vLLM, no weights, no Lean, no cluster.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_table1 import discover  # noqa: E402
from prooflens_prover.data.informal import coverage, load_informal_names  # noqa: E402
from prooflens_prover.prover.vllm_policy import SamplingConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_run(root: Path, name: str, benchmark: str, arm: str, n_proved: int,
              policy_kind: str | None = "repertoire", n_total: int = 10,
              started: str = "2026-08-08T00:00:00+00:00") -> Path:
    d = root / name
    d.mkdir(parents=True)
    cfg = {"benchmark": benchmark, "arm": arm}
    if policy_kind is not None:
        cfg["policy_kind"] = policy_kind
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "started_utc": started, "config": cfg,
        "outcome": {"n_proved": n_proved},
    }))
    (d / "attempts.jsonl").write_text("\n".join(
        json.dumps({"problem_id": str(i), "proved": i < n_proved,
                    "status": "proved" if i < n_proved else "exhausted",
                    "proof": ["aesop"] if i < n_proved else None})
        for i in range(n_total)
    ))
    return d


class TestTableDoesNotMixPolicies:
    def test_llm_and_model_free_runs_do_not_compete_for_one_cell(self, tmp_path):
        write_run(tmp_path, "old_repertoire", "fate_m", "li", 31, "repertoire",
                  started="2026-08-06T00:00:00+00:00")
        write_run(tmp_path, "new_vllm", "fate_m", "li", 80, "vllm",
                  started="2026-08-09T00:00:00+00:00")

        # The vllm run is newer and scores far higher. It must not appear in the model-free table.
        rep = discover(tmp_path, "repertoire")
        assert list(rep) == [("fate_m", "li")]
        assert rep[("fate_m", "li")].name == "old_repertoire"

        llm = discover(tmp_path, "vllm")
        assert llm[("fate_m", "li")].name == "new_vllm"

    def test_runs_predating_the_field_count_as_model_free(self, tmp_path):
        # Every run made before `policy_kind` existed used the repertoire policy — that is what
        # it was, so treating the absent field as `repertoire` is a fact, not a default.
        write_run(tmp_path, "legacy", "fate_m", "li", 31, policy_kind=None)
        assert ("fate_m", "li") in discover(tmp_path, "repertoire")
        assert discover(tmp_path, "vllm") == {}

    def test_unfinished_runs_are_still_ignored(self, tmp_path):
        d = write_run(tmp_path, "partial", "fate_m", "li", 5, "vllm")
        m = json.loads((d / "manifest.json").read_text())
        m["outcome"] = None
        (d / "manifest.json").write_text(json.dumps(m))
        assert discover(tmp_path, "vllm") == {}


class TestProveBenchmarkExposesThePolicy:
    """Static checks on the script, because the argument plumbing is what broke before."""

    @staticmethod
    def _src() -> str:
        return (REPO_ROOT / "scripts" / "prove_benchmark.py").read_text(encoding="utf-8")

    def test_policy_flag_exists_with_both_choices(self):
        src = self._src()
        assert '"--policy"' in src
        assert 'POLICY_TAGS = {"repertoire": "repertoire", "vllm": "vllm"}' in src

    def test_run_name_carries_the_policy_tag(self):
        # `fate_m_li_repertoire_...` vs `fate_m_li_vllm_...`: without this the run id alone cannot
        # tell you which system produced a result.
        assert 'name=f"{args.benchmark}_{args.arm}_{POLICY_TAGS[args.policy]}"' in self._src()

    def test_manifest_records_the_policy_kind_and_config(self):
        src = self._src()
        assert '"policy_kind": args.policy' in src
        assert '"policy_config": policy.config()' in src

    def test_resume_refuses_to_cross_policies(self):
        src = self._src()
        block = src[src.index("if args.resume is not None:"):src.index("else:\n        manifest")]
        assert "policy_kind" in block, (
            "resume checks the arm but not the policy; a 7B model and a repertoire sharing one "
            "attempts.jsonl would produce a pass rate belonging to neither"
        )

    def test_model_is_required_for_the_vllm_policy_and_rejected_otherwise(self):
        src = self._src()
        assert "--policy vllm requires --model" in src
        assert "--model applies to --policy vllm only" in src

    def test_every_new_knob_reaches_build_policy_or_the_generator(self):
        """Generalises the `--n-candidates` failure to the flags added for this arm."""
        src = self._src()
        body = src[src.index("def build_policy"):src.index("def main()")]
        for flag in ("temperature", "top_p", "max_tokens", "prompt_limit", "informal_names",
                     "strip_echo", "dtype", "gpu_memory_utilization", "max_model_len"):
            assert f"args.{flag}" in body, f"--{flag.replace('_', '-')} never reaches build_policy"

    def test_the_decoding_defaults_are_derived_not_restated(self):
        """A flag reaching `build_policy` is not enough — its *default* has to be right too.

        `SamplingConfig` carried REAL-Prover's temperature 1.5 / top_p 0.9 while argparse declared
        `default=1.0` for both. argparse passes its default whether or not the operator names the
        flag, so the corrected constants were dead code and every run decoded at 1.0/1.0. Deriving
        the defaults is the only version of this that cannot drift apart again.
        """
        src = self._src()
        for field in ("temperature", "top_p", "max_tokens"):
            assert f"default=SamplingConfig.{field}" in src, (
                f"--{field.replace('_', '-')} restates its default instead of taking it from "
                "SamplingConfig, which is how 1.0/1.0 silently overrode 1.5/0.9"
            )

    def test_the_derived_defaults_are_real_provers_values(self):
        """Pins the numbers themselves, so deriving from a *wrong* SamplingConfig still fails."""
        assert (SamplingConfig.temperature, SamplingConfig.top_p) == (1.5, 0.9)
        assert SamplingConfig.max_tokens == 256

    def test_logprobs_are_requested(self):
        """vLLM v1 returns `cumulative_logprob=None` without this, and the search ranks on it."""
        assert SamplingConfig.logprobs == 1

    def test_manifest_records_the_policy_health_counters(self):
        """`PolicyStats` existed, was populated, and was never written anywhere.

        The pass rate cannot tell you that one arm got fewer usable candidates per expansion than
        the other — which would mean the arms ran different search budgets and would present as a
        retrieval effect. These two lines are what make the smoke run judgeable at all.
        """
        src = self._src()
        assert "policy_stats=policy_stats" in src
        assert "generator_stats=generator_stats" in src


class TestSbatchForTheLlmArm:
    @staticmethod
    def _src() -> str:
        path = REPO_ROOT / "slurm" / "prove_benchmark_llm.sbatch"
        return path.read_text(encoding="utf-8")

    def test_it_exists_and_selects_the_vllm_policy(self):
        assert "--policy vllm" in self._src()

    @staticmethod
    def _invocation(src: str) -> str:
        """Just the `prove_benchmark.py` command, not everything after it.

        Sliced to end-of-file, this region swallowed the closing `echo` block — so a variable named
        only in the help text printed at the end would have counted as "passed to the script". The
        footer now mentions `$SAMPLES` while explaining the health check, which is exactly the shape
        of that false pass. Bounded to the command's own backslash continuations instead.
        """
        lines = src.splitlines()
        start = next(i for i, ln in enumerate(lines) if "scripts/prove_benchmark.py" in ln)
        out = [lines[start]]
        while lines[start].rstrip().endswith("\\"):
            start += 1
            out.append(lines[start])
        return "\n".join(out)

    @staticmethod
    def _reaching(src: str, invocation: str) -> set[str]:
        """Variables that affect the run, following one level of indirection.

        A knob does not have to appear in the invocation itself: `ENFORCE_EAGER` is read by a test
        that sets `EAGER_FLAG`, and it is `$EAGER_FLAG` that gets passed. Excluding such variables
        from the check would have been the easy fix and the wrong one — this is the check that
        caught `--top-p` being declared nowhere and defaulting silently, so it should get *smarter*,
        not narrower.

        A line that both reads `$VAR` and assigns a variable already known to reach the invocation
        counts as `VAR` reaching it. Iterated so a two-step derivation still resolves.
        """
        reaching = set(re.findall(r"\$\{?([A-Z_]+)", invocation))
        for _ in range(3):
            for line in src.splitlines():
                assigned = set(re.findall(r"^\s*([A-Z_]+)=", line))
                assigned |= set(re.findall(r"&&\s*([A-Z_]+)=", line))
                if assigned & reaching:
                    reaching |= set(re.findall(r"\$\{?([A-Z_]+)", line))
        return reaching

    def test_every_declared_knob_reaches_the_script(self):
        src = self._src()
        declared = set(re.findall(r'^([A-Z_]+)="\$\{\1:-', src, flags=re.MULTILINE))
        internal = {"REPO", "VENV", "VLLM_VENV", "STAGE_LEAN"}
        missing = sorted(declared - internal - self._reaching(src, self._invocation(src)))
        assert not missing, f"declared but never passed to prove_benchmark.py: {missing}"

    def test_the_indirection_check_still_catches_a_dead_knob(self):
        """Guards the guard: a knob wired to nothing must still fail.

        Following indirection risks making the check accept anything at all, so this feeds it a
        variable that is declared and genuinely unused and requires that it is still reported.
        """
        src = self._src() + '\nDEAD_KNOB="${DEAD_KNOB:-0}"\n'
        assert "DEAD_KNOB" not in self._reaching(src, self._invocation(src))

    def test_a_knob_named_only_in_the_help_text_does_not_count_as_passed(self):
        """The false pass the unbounded slice allowed."""
        src = self._src() + '\nECHOED_ONLY="${ECHOED_ONLY:-0}"\necho "see $ECHOED_ONLY"\n'
        assert "ECHOED_ONLY" not in self._reaching(src, self._invocation(src))

    def test_it_uses_the_vllm_virtualenv_not_the_retrieval_one(self):
        # vLLM pins torch hard enough to need its own env; running this under the retrieval venv
        # would fail at `import vllm` after paying the Mathlib import.
        assert "VLLM_VENV" in self._src()

    def test_the_sampling_defaults_match_real_provers_shipped_config(self):
        """The sbatch is the last place a correct default can be overridden, and it was.

        `SAMPLES=32` doubled their generation budget and `TEMPERATURE=1.0` was a guess, so a run
        launched this way differed from REAL-Prover in three ways while the header claimed it
        differed only in the retriever. Their `conf/config.py`: NUM_SAMPLES=16, and
        PROVER_MODEL_PARAMS {"temperature": 1.5, "top_p": 0.9, "max_tokens": 256}.
        """
        src = self._src()
        expected = {
            "SAMPLES": "16",
            "TEMPERATURE": str(SamplingConfig.temperature),
            "TOP_P": str(SamplingConfig.top_p),
            "MAX_TOKENS": str(SamplingConfig.max_tokens),
        }
        for name, value in expected.items():
            assert f'{name}="${{{name}:-{value}}}"' in src, (
                f"{name} does not default to {value}; the sbatch would override the corrected value"
            )

    def test_the_sampling_knobs_are_actually_passed(self):
        """`--top-p` was never on the command line at all, so argparse's default won regardless.

        The generic declared-but-not-passed check could not catch it: the variable was not declared
        either, so there was nothing to notice was missing.
        """
        invocation = self._invocation(self._src())
        for flag in ("--temperature", "--top-p", "--max-tokens", "--samples-per-step"):
            assert flag in invocation, f"{flag} is never passed to prove_benchmark.py"

    def test_it_leaves_gpu_headroom_for_the_query_encoder(self):
        """Three consumers share the GPU, and one of them allocates after vLLM has taken its share.

        The retrieval child runs the query encoder on `--device cuda` (which is what the model-free
        runs did, and what keeps this arm's retrieval bit-identical to theirs). It loads before vLLM
        profiles, so the weights are accounted for — but its cuBLAS workspaces are allocated on the
        first real query, after vLLM has claimed its fraction. 0.90 left too little for that.
        """
        src = self._src()
        assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"' in src

    def test_the_chat_template_defaults_to_the_one_the_checkpoint_ships(self):
        """The sbatch is the last place this can be overridden, and the wrong value cost a run.

        REAL-Prover's code hard-codes `deepseek`; their weights ship a ChatML `chat_template`.
        Sending deepseek produced multilingual token salad and one progress step across five
        problems, where the model-free repertoire made 290 on one of them.
        """
        src = self._src()
        assert 'TEMPLATE="${TEMPLATE:-qwen_chatml}"' in src
        assert "--template" in self._invocation(src)

    def test_the_jit_escape_hatch_is_reachable_without_editing_code(self):
        """A GPU node has the CUDA driver and no toolkit, so a self-compiling kernel fails there.

        FlashInfer's sampler was the first to do it and is disabled in `ENGINE_ENV`. If a different
        component does the same, `ENFORCE_EAGER=1` turns off CUDA-graph capture and torch.compile
        from the submit line — which is the difference between one round-trip and two.
        """
        src = self._src()
        assert 'ENFORCE_EAGER="${ENFORCE_EAGER:-0}"' in src
        assert 'EAGER_FLAG="--enforce-eager"' in src
        assert "$EAGER_FLAG" in self._invocation(src), "the flag is built but never passed"

    def test_the_header_does_not_claim_the_retrieval_child_avoids_the_gpu(self):
        """It said exactly that for two revisions while passing `--device cuda`.

        A comment that contradicts the code is worse than no comment: the GPU-memory arithmetic in
        that header was derived from the false claim.
        """
        src = self._src()
        assert "retrieval child does not touch the GPU" not in src


class TestInformalNamesLoader:
    def test_loads_a_mapping(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join([
            json.dumps({"formal_name": "mul_comm", "informal_name": "multiplication commutes"}),
            json.dumps({"formal_name": "add_zero", "informal_name": "adding zero"}),
        ]), encoding="utf-8")
        assert load_informal_names(f) == {
            "mul_comm": "multiplication commutes", "add_zero": "adding zero",
        }

    def test_accepts_a_directory_and_finds_the_jsonl(self, tmp_path):
        (tmp_path / "data.jsonl").write_text(
            json.dumps({"name": "mul_comm", "informal": "x"}), encoding="utf-8")
        assert load_informal_names(tmp_path) == {"mul_comm": "x"}

    def test_blank_glosses_are_dropped_not_stored(self, tmp_path):
        # A gloss of "" renders identically to an absent one; storing it would overstate coverage.
        f = tmp_path / "d.jsonl"
        f.write_text("\n".join([
            json.dumps({"formal_name": "a", "informal_name": "  "}),
            json.dumps({"formal_name": "b", "informal_name": "real"}),
        ]), encoding="utf-8")
        assert load_informal_names(f) == {"b": "real"}

    def test_unknown_schema_fails_loudly_naming_the_actual_keys(self, tmp_path):
        # Silently yielding {} is indistinguishable from passing no dataset at all.
        f = tmp_path / "d.jsonl"
        f.write_text(json.dumps({"lemma": "a", "gloss": "b"}), encoding="utf-8")
        with pytest.raises(SystemExit, match="record keys.*gloss.*lemma"):
            load_informal_names(f)

    def test_keys_can_be_overridden(self, tmp_path):
        f = tmp_path / "d.jsonl"
        f.write_text(json.dumps({"lemma": "a", "gloss": "b"}), encoding="utf-8")
        assert load_informal_names(f, "lemma", "gloss") == {"a": "b"}

    def test_an_all_blank_file_is_an_error(self, tmp_path):
        f = tmp_path / "d.jsonl"
        f.write_text(json.dumps({"formal_name": "a", "informal_name": ""}), encoding="utf-8")
        with pytest.raises(SystemExit, match="no usable informal names"):
            load_informal_names(f)

    def test_coverage_measures_the_documented_bias(self):
        m = {"a": "x", "b": "y"}
        assert coverage(m, ["a", "b", "c", "d"]) == {
            "n_premises": 4, "n_with_gloss": 2, "coverage": 0.5, "n_mapping": 2,
        }

    def test_coverage_of_an_empty_corpus_is_zero_not_a_crash(self):
        assert coverage({}, [])["coverage"] == 0.0


class TestNamesAreStoredAsPathLists:
    """`mathlib_informal_v4.16.0` stores names as path-component LISTS, not dotted strings.

        {"name": ["CoxeterSystem", "length_simple_mul_ne"], "informal_name": "..."}

    Stringifying one yields `"['CoxeterSystem', 'length_simple_mul_ne']"` — a key no premise in the
    corpus has. The mapping would be fully populated, pass the `not mapping` check, and produce
    exactly the blank glosses it was added to fix, with no error anywhere.
    """

    def test_path_lists_join_to_dotted_names(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({
            "name": ["CoxeterSystem", "length_simple_mul_ne"],
            "informal_name": "Left Multiplication by Simple Reflection Changes Length",
        }), encoding="utf-8")
        m = load_informal_names(f)
        assert list(m) == ["CoxeterSystem.length_simple_mul_ne"]
        assert "[" not in next(iter(m))

    def test_a_single_component_name_has_no_dot(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"name": ["mul_comm"], "informal_name": "x"}), encoding="utf-8")
        assert list(load_informal_names(f)) == ["mul_comm"]

    def test_plain_string_names_still_work(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"name": "mul_comm", "informal_name": "x"}), encoding="utf-8")
        assert list(load_informal_names(f)) == ["mul_comm"]

    def test_the_real_record_shape_parses(self, tmp_path):
        """One verbatim record from the dataset, including the fields we ignore."""
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({
            "module_name": ["Mathlib", "GroupTheory", "Coxeter", "Length"],
            "kind": "theorem",
            "name": ["CoxeterSystem", "length_simple_mul_ne"],
            "signature": " (w : W) (i : B) : ℓ(s i * w) ≠ ℓ w",
            "type": "∀ {B : Type u_1} ...",
            "value": ":= by\n  convert cs.length_mul_simple_ne w⁻¹ i using 1",
            "docstring": None,
            "informal_name": "Left Multiplication by Simple Reflection Changes Length",
            "informal_description": "For any element $w$ in a Coxeter group ...",
        }), encoding="utf-8")
        m = load_informal_names(f)
        # `informal_name` must win over `informal_description`: REAL-Prover's prompt field is the
        # short name, and the description is several sentences of LaTeX.
        assert m == {
            "CoxeterSystem.length_simple_mul_ne":
                "Left Multiplication by Simple Reflection Changes Length"
        }

    def test_docstring_null_does_not_become_the_gloss(self, tmp_path):
        # `docstring` is in INFORMAL_KEYS but sits after informal_name, and is often null.
        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps(
            {"name": ["a"], "docstring": None, "informal_name": "real gloss"}), encoding="utf-8")
        assert load_informal_names(f) == {"a": "real gloss"}


class TestCoverageGuard:
    """The check that would have caught the path-list bug."""

    def test_a_populated_mapping_matching_nothing_is_an_error(self):
        from prooflens_prover.data.informal import check_coverage

        # Exactly the failure: 300 entries, all keyed on stringified lists.
        broken = {f"['Foo', 'bar_{i}']": "gloss" for i in range(300)}
        with pytest.raises(SystemExit, match="schema mismatch"):
            check_coverage(broken, [f"Foo.bar_{i}" for i in range(300)])

    def test_the_error_shows_example_keys_so_the_cause_is_visible(self):
        from prooflens_prover.data.informal import check_coverage

        with pytest.raises(SystemExit, match=r"example mapping keys.*\['Foo'"):
            check_coverage({"['Foo', 'bar']": "g"}, ["Foo.bar", "Foo.baz"])

    def test_good_coverage_passes_and_reports(self):
        from prooflens_prover.data.informal import check_coverage

        m = {"a": "x", "b": "y", "c": "z"}
        assert check_coverage(m, ["a", "b", "c", "d"])["coverage"] == 0.75

    def test_genuinely_sparse_coverage_can_be_allowed_deliberately(self):
        from prooflens_prover.data.informal import check_coverage

        result = check_coverage({"a": "x"}, [f"p{i}" for i in range(100)], minimum=0.0)
        assert result["coverage"] == 0.0

    def test_an_empty_corpus_does_not_trip_the_guard(self):
        from prooflens_prover.data.informal import check_coverage

        assert check_coverage({"a": "x"}, [])["n_premises"] == 0


class TestRealProverSamplingParameters:
    """Pinned to their `conf/config.py` PROVER_MODEL_PARAMS, not to plausible guesses."""

    def test_defaults_match_their_config_verbatim(self):
        from prooflens_prover.prover.vllm_policy import SamplingConfig

        s = SamplingConfig()
        assert (s.temperature, s.top_p, s.max_tokens) == (1.5, 0.9, 256)

    def test_top_k_matches_their_num_querys(self):
        from prooflens_prover.prover.vllm_policy import VLLMPolicy

        assert VLLMPolicy.top_k == 10

    def test_search_defaults_match_their_shipped_config(self):
        from prooflens_prover.prover.search import SearchConfig

        c = SearchConfig()
        assert (c.samples_per_step, c.max_depth, c.max_expansions) == (16, 32, 64)
        assert c.length_penalty == 0.5

    def test_our_guard_bans_at_least_what_theirs_does(self):
        """Their `ABANDON_IF_CONTAIN = ["sorry", "admit", "apply?"]`."""
        from prooflens_prover.lean.backend import TacticPolicy as TacticGuard

        guard = TacticGuard()
        for banned in ("exact sorry", "simp; admit", "apply?"):
            assert guard.reject_reason(banned) is not None, banned


class TestEnvironmentCheck:
    """`check_env.py` exists to end the one-traceback-per-cluster-job loop.

    Four submit-wait-diagnose cycles were spent discovering pylate's transitive dependencies one
    at a time (`scipy`, then `datasets`). The reason they were invisible is that this package defers
    `torch`, `pylate`, `lean_interact` and `vllm` to function-level imports, deliberately, to keep
    the suite hermetic — so `import prooflens_prover` proves nothing about whether a run will start.
    """

    def test_submodules_of_from_imports_are_recorded_separately(self, tmp_path):
        """The exact blind spot: `import pylate` succeeds, `import pylate.models` does not."""
        from check_env import third_party_imports

        src = tmp_path / "src" / "pkg"
        src.mkdir(parents=True)
        (src / "m.py").write_text("def f():\n    from pylate import models\n", encoding="utf-8")
        targets = third_party_imports(tmp_path / "src")
        assert "pylate" in targets
        assert "pylate.models" in targets, (
            "recording only the parent reproduces the blind spot the script exists to close"
        )

    def test_function_level_imports_are_found(self, tmp_path):
        from check_env import third_party_imports

        src = tmp_path / "src"
        src.mkdir()
        (src / "m.py").write_text(
            "def load():\n"
            "    import torch\n"
            "    if True:\n"
            "        from lean_interact import AutoLeanServer\n",
            encoding="utf-8",
        )
        targets = third_party_imports(src)
        assert {"torch", "lean_interact", "lean_interact.AutoLeanServer"} <= set(targets)

    def test_stdlib_and_first_party_are_excluded(self, tmp_path):
        from check_env import third_party_imports

        src = tmp_path / "src"
        src.mkdir()
        (src / "m.py").write_text(
            "import json\nimport pathlib\nfrom prooflens_prover.utils import io\nimport numpy\n",
            encoding="utf-8",
        )
        targets = third_party_imports(src)
        assert set(targets) == {"numpy"}

    def test_an_attribute_is_not_reported_as_a_missing_module(self):
        """`from numpy import ndarray` must not report a missing `numpy.ndarray` module."""
        from check_env import try_import

        assert try_import("json.dumps") is None          # attribute of a real module
        assert try_import("json") is None
        assert try_import("definitely_not_installed_xyz") is not None

    def test_a_genuinely_missing_submodule_is_reported(self):
        from check_env import try_import

        # Reported as an absent *attribute* of a module that imports fine, which is what it is —
        # more useful than repeating the outer "no module named json.no_such_submodule".
        reason = try_import("json.no_such_submodule")
        assert reason is not None
        assert "has no attribute 'no_such_submodule'" in reason

    def test_a_broken_parent_reports_the_real_cause_not_the_outer_error(self, monkeypatch):
        """`from vllm import LLM` resolves `LLM` through a lazy module `__getattr__`.

        When vLLM is broken by an incompatible transformers, the useful error is raised *there*. An
        earlier version discarded it and reported a nonexistent `vllm.LLM` module, which sent me
        looking for a missing package instead of a version conflict.
        """
        import check_env

        real = check_env.importlib.import_module

        def fake(name, *a, **kw):
            if name == "fakepkg":
                raise ImportError("fakepkg requires transformers>=5.5.3")
            if name == "fakepkg.Thing":
                raise ModuleNotFoundError("No module named 'fakepkg.Thing'")
            return real(name, *a, **kw)

        monkeypatch.setattr(check_env.importlib, "import_module", fake)
        reason = check_env.try_import("fakepkg.Thing")
        assert "transformers>=5.5.3" in reason, reason

    def test_star_imports_do_not_produce_a_bogus_target(self, tmp_path):
        from check_env import third_party_imports

        src = tmp_path / "src"
        src.mkdir()
        (src / "m.py").write_text("from numpy import *\n", encoding="utf-8")
        assert set(third_party_imports(src)) == {"numpy"}


class TestBootstrapUsesConstraintsNotNoDeps:
    """`--no-deps` is a promise to supply the dependency tree by hand, and I could not."""

    @staticmethod
    def _src() -> str:
        return (REPO_ROOT / "scripts" / "bootstrap_llm.sh").read_text(encoding="utf-8")

    def test_pylate_is_installed_with_dependencies(self):
        """Checked in the pip *invocation*, not the file.

        A first version of this test searched the whole script and failed on the comment explaining
        why `--no-deps` was abandoned — the same mistake as grepping a job script for a variable
        name and calling it verified.
        """
        commands = [
            ln for ln in self._src().splitlines()
            if "pip install" in ln and not ln.lstrip().startswith("#")
        ]
        assert commands, "no pip install command found in bootstrap_llm.sh"
        offenders = [ln for ln in commands if "--no-deps" in ln]
        assert not offenders, (
            f"--no-deps skipped scipy and datasets, costing two cluster round-trips: {offenders}. "
            "A constraints file protects torch without hiding real requirements."
        )

    def test_a_constraints_file_protects_torch(self):
        src = self._src()
        assert "CONSTRAINTS=" in src
        assert '-c "$CONSTRAINTS"' in src
        assert "torch" in src

    def test_it_verifies_the_protected_versions_did_not_move(self):
        # A resolver that honours the constraint is the expectation; verifying is the guarantee.
        assert "did not move" in self._src()

    def test_it_runs_check_env_rather_than_ad_hoc_imports(self):
        assert "check_env.py" in self._src()

    def test_it_resolves_before_installing(self):
        """A dry run costs seconds and reports a conflict without touching the environment.

        Pinning six packages produced `ResolutionImpossible` — vLLM installs transformers 5.x and
        sentence-transformers requires <5. Discovering that from a real install leaves the
        environment half-modified; from `--dry-run` it is untouched, and the
        constraint set can be iterated on a login node in seconds rather than in cluster jobs.
        """
        src = self._src()
        assert "--dry-run" in src
        assert src.index("--dry-run") < src.index('=== installing ==='), (
            "the dry run must come before the real install, or it proves nothing"
        )

    def test_only_torch_and_vllm_are_constrained(self):
        """Over-constraining is what made resolution impossible.

        torch cannot move (vLLM ships kernels compiled against it) and vllm must not be replaced
        by a resolver "solving" a conflict. Everything else vLLM declares a range for, and anything
        that range is supported by definition.
        """
        src = self._src()
        line = next(ln for ln in src.splitlines() if ln.startswith("PROTECTED="))
        assert line == 'PROTECTED="torch vllm"', line

    def test_the_prover_environment_does_not_install_pylate(self):
        """Installing pylate in the prover venv is what created the conflict, and bought nothing.

        `vllm` needs `transformers>=5.5.3`, `pylate` needs `<=5.3.0`. Retrieval runs in a child
        process under a different interpreter, so the prover never encodes a query and has no use
        for pylate at all. Job 18340441 died on
        `ImportError: cannot import name 'ALLOWED_LAYER_TYPES'` — vLLM 0.26 against the
        transformers 5.3.0 that pylate had dragged it down to.
        """
        src = self._src()
        # Scoped to the WANTED array, not the whole file. Three earlier versions of tests in this
        # module matched prose in a comment — the same mistake as grepping a job script for a
        # variable name and calling the flag verified.
        start = src.index("WANTED=(")
        wanted = src[start:src.index(")", start)]
        requested = [
            ln.strip().strip('"').split("#")[0].strip()
            for ln in wanted.splitlines()[1:] if ln.strip().startswith('"')
        ]
        assert not [r for r in requested if "pylate" in r or "sentence-transformers" in r], (
            f"the prover environment must not install a retrieval package: {requested}"
        )
        assert any(r.startswith("transformers>=5.5.3") for r in requested), (
            f"vLLM's transformers floor must be asserted, or pip leaves it downgraded: {requested}"
        )

    def test_it_removes_a_previously_installed_pylate(self):
        # The vLLM venv already has one from the attempts to share an environment, and it is what
        # holds transformers below vLLM's floor. Not installing it is not enough.
        src = self._src()
        assert "pip uninstall" in src
        assert "for pkg in pylate sentence-transformers" in src

    def test_it_validates_both_environments(self):
        # A perfect prover environment plus a retrieval environment that cannot import pylate
        # still cannot run an arm. Checking only the half you stand in cost four cycles.
        src = self._src()
        assert "--require vllm,vllm.LLM" in src
        assert "--require pylate,pylate.models" in src
        assert "$RETRIEVAL_PYTHON" in src
