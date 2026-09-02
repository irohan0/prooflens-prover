"""The first-stage budget must be overridable at run time, and must land in the manifest.

Measured recall@10 of LI's two-stage path against exact full-corpus MaxSim, on 141 real FATE-M
queries (`results/tables/li_recall_fate_m.json`):

    n_candidates    % of corpus    recall@10    lossless queries
           1,000          0.36%        0.443            9/141
           5,000          1.81%        0.696           32/141
          20,000          7.24%        0.888           81/141
          50,000         18.11%        0.979          124/141

Every LI benchmark result produced before this was measured ran at 1,000 — seeing **under half** of
its true top-10. Two LI runs differing only in this number are not the same experiment, so the
budget has to be (a) settable without rebuilding the 5.5 GB index and (b) recorded in the run
manifest. Neither was true.

Hermetic: no torch, no checkpoints, no cluster.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prooflens_prover.data.premises import PremiseRecord  # noqa: E402
from prooflens_prover.retrieval.base import RetrievalStats  # noqa: E402
from prooflens_prover.retrieval.dense import (  # noqa: E402
    EncoderSpec,
    LateInteractionIndex,
    LateInteractionRetriever,
    SingleVectorIndex,
    SingleVectorRetriever,
    l2_normalise,
)
from prove_benchmark import build_retriever, effective_n_candidates  # noqa: E402

DIM = 8


def _index(n_docs=40, n_candidates=1000):
    rng = np.random.default_rng(0)
    lengths = rng.integers(2, 6, size=n_docs)
    offsets = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    tokens = l2_normalise(rng.standard_normal((int(offsets[-1]), DIM)).astype(np.float32))
    pooled = l2_normalise(
        np.stack([tokens[offsets[i]:offsets[i + 1]].mean(axis=0) for i in range(n_docs)])
    )
    recs = [PremiseRecord(name=f"L{i}", kind="theorem", statement="s", module="M", is_prop=True)
            for i in range(n_docs)]
    spec = EncoderSpec(kind="li", checkpoint="t", base_model="t", dim=DIM)
    return LateInteractionIndex(recs, tokens, offsets, pooled, spec, n_candidates)


class TestEffectiveNCandidates:
    def test_reads_the_budget_off_a_late_interaction_retriever(self):
        r = LateInteractionRetriever(_index(n_candidates=777), lambda s: None, RetrievalStats())
        assert effective_n_candidates(r) == 777

    def test_is_none_for_single_vector(self):
        # SV ranks the whole corpus exactly; reporting a first-stage budget for it would imply an
        # approximation it does not make.
        spec = EncoderSpec(kind="sv", checkpoint="t", base_model="t", dim=DIM)
        recs = [PremiseRecord(name="L", kind="theorem", statement="s", module="M", is_prop=True)]
        idx = SingleVectorIndex(recs, l2_normalise(np.ones((1, DIM), dtype=np.float32)), spec)
        assert effective_n_candidates(SingleVectorRetriever(idx, lambda s: None)) is None

    def test_is_none_for_a_retriever_without_an_index(self):
        from prooflens_prover.retrieval.base import NullRetriever

        assert effective_n_candidates(NullRetriever()) is None


class TestBuildRetrieverOverride:
    """`build_retriever` is where the override has to bite — the retriever reads the index's stored
    value at query time, so setting it anywhere later would be too late."""

    def test_override_reaches_the_index(self, monkeypatch, tmp_path):
        idx = _index(n_candidates=1000)
        monkeypatch.setattr(
            "prooflens_prover.retrieval.dense.load_retriever",
            lambda arm, d, checkpoint=None, device=None, stats=None: LateInteractionRetriever(
                idx, lambda s: None, stats or RetrievalStats()
            ),
        )
        r = build_retriever("li", tmp_path, RetrievalStats(), n_candidates=50_000)
        assert r.index.n_candidates == 50_000
        assert effective_n_candidates(r) == 50_000

    def test_absent_override_leaves_the_stored_value(self, monkeypatch, tmp_path):
        idx = _index(n_candidates=1000)
        monkeypatch.setattr(
            "prooflens_prover.retrieval.dense.load_retriever",
            lambda arm, d, checkpoint=None, device=None, stats=None: LateInteractionRetriever(
                idx, lambda s: None, stats or RetrievalStats()
            ),
        )
        assert build_retriever("li", tmp_path, RetrievalStats()).index.n_candidates == 1000

    def test_rejected_for_the_sv_arm(self, monkeypatch, tmp_path):
        spec = EncoderSpec(kind="sv", checkpoint="t", base_model="t", dim=DIM)
        recs = [PremiseRecord(name="L", kind="theorem", statement="s", module="M", is_prop=True)]
        sv = SingleVectorIndex(recs, l2_normalise(np.ones((1, DIM), dtype=np.float32)), spec)
        monkeypatch.setattr(
            "prooflens_prover.retrieval.dense.load_retriever",
            lambda arm, d, checkpoint=None, device=None, stats=None: SingleVectorRetriever(
                sv, lambda s: None, stats or RetrievalStats()
            ),
        )
        # Silently accepting it would imply SV had been given a matching handicap, which it has not.
        with pytest.raises(SystemExit, match="li arm only"):
            build_retriever("sv", tmp_path, RetrievalStats(), n_candidates=50_000)


class TestBudgetChangesRetrieval:
    """If widening the budget could not change what is returned, the whole H1 experiment is moot."""

    def test_a_wider_budget_can_change_the_result(self):
        idx = _index(n_docs=400, n_candidates=3)
        q = l2_normalise(np.random.default_rng(1).standard_normal((4, DIM)).astype(np.float32))
        tight = [i for i, _ in idx.topk(q, k=10)]
        idx.n_candidates = 400
        wide = [i for i, _ in idx.topk(q, k=10)]
        assert tight != wide

    def test_full_budget_equals_exact(self):
        idx = _index(n_docs=200, n_candidates=200)
        q = l2_normalise(np.random.default_rng(2).standard_normal((4, DIM)).astype(np.float32))
        assert [i for i, _ in idx.topk(q, k=10)] == [
            i for i, _ in idx.exact_topk_chunked(q, k=10, chunk=37)
        ]


class TestSbatchActuallyPassesTheFlag:
    """A job script that reads a variable, echoes it, and then never passes it is the worst case.

    `N_CANDIDATES=50000 sbatch slurm/prove_benchmark.sbatch` ran for five hours, printed
    `n_cand : 50000` in its header, and produced a run whose manifest recorded
    `n_candidates: 1000` — because the edit that added the flag to the python invocation silently
    failed to match (CRLF line endings vs an `\n` pattern), while the two edits that added the
    variable and the echo succeeded. Grepping for the variable name found two hits and looked fine.

    The plumbing between an sbatch knob and the script is not otherwise covered by any test: the
    unit tests above prove `build_retriever` honours `--n-candidates`, which was true the whole
    time and was not the problem.
    """

    @staticmethod
    def _sbatch() -> str:
        path = Path(__file__).resolve().parent.parent / "slurm" / "prove_benchmark.sbatch"
        return path.read_text(encoding="utf-8")

    def test_flag_appears_in_the_python_invocation(self):
        src = self._sbatch()
        start = src.index('scripts/prove_benchmark.py')
        end = src.index("$EXTRA", start)
        invocation = src[start:end]
        assert "--n-candidates" in invocation, (
            "slurm/prove_benchmark.sbatch defines N_CANDIDATES and echoes it but never passes it "
            "to prove_benchmark.py — the run would silently use the index's stored budget"
        )

    def test_flag_is_conditional_so_an_unset_value_is_not_passed_as_empty(self):
        # A bare `--n-candidates "$N_CANDIDATES"` with N_CANDIDATES unset sends an empty string,
        # which argparse rejects with `invalid int value: ''` after the Mathlib staging has run.
        assert '${N_CANDIDATES:+--n-candidates "$N_CANDIDATES"}' in self._sbatch()

    def test_every_declared_knob_reaches_the_script(self):
        """Generalises the above: each `X="${X:-...}"` knob must appear later in the invocation.

        Catches the same failure for any future variable, not just this one.
        """
        import re

        src = self._sbatch()
        start = src.index('scripts/prove_benchmark.py')
        invocation = src[start:]
        declared = set(re.findall(r'^([A-Z_]+)="\$\{\1:-', src, flags=re.MULTILINE))
        # Knobs consumed by the job script itself rather than forwarded as arguments.
        internal = {"REPO", "VENV", "STAGE_LEAN", "N_QUERIES", "CHUNK"}
        missing = sorted(
            v for v in declared - internal
            if f'${v}' not in invocation and f'${{{v}' not in invocation
        )
        assert not missing, f"declared but never passed to prove_benchmark.py: {missing}"


class TestBudgetIsVisibleInTheTable:
    """Table 1 keeps the newest run per (benchmark, arm), so the li column's budget can change
    without the table changing shape. At 1k the first stage retained 0.443 of its own top-10 and at
    50k 0.979 — a difference larger than any effect the table reports — so an unlabelled li number
    is ambiguous between two different experiments."""

    def test_arm_load_reads_the_budget_from_the_manifest(self, tmp_path):
        from prooflens_prover.eval.compare import Arm

        d = tmp_path / "r"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "run_id": "r", "config": {"benchmark": "fate_m", "arm": "li", "n_candidates": 50_000},
        }))
        (d / "attempts.jsonl").write_text(
            json.dumps({"problem_id": "1", "status": "proved", "proved": True, "proof": ["aesop"]})
        )
        assert Arm.load(d).n_candidates == 50_000

    def test_budget_is_none_when_the_run_predates_the_field(self, tmp_path):
        from prooflens_prover.eval.compare import Arm

        d = tmp_path / "r"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "run_id": "r", "config": {"benchmark": "fate_m", "arm": "li"},
        }))
        (d / "attempts.jsonl").write_text(
            json.dumps({"problem_id": "1", "status": "exhausted", "proved": False, "proof": None})
        )
        # Not defaulted to 1000: an unrecorded value must read as unknown, since silently filling it
        # in is what made the 0.443-recall runs look interchangeable with the corrected ones.
        assert Arm.load(d).n_candidates is None

    @pytest.mark.parametrize("n,expected", [
        (1000, "1k"), (50_000, "50k"), (20_000, "20k"), (None, "?"), (1500, "1500"), (999, "999"),
    ])
    def test_budget_rendering(self, n, expected):
        from prooflens_prover.eval.compare import format_budget

        assert format_budget(n) == expected

    def test_table_annotates_the_li_cell_and_warns_on_mixed_budgets(self, tmp_path):
        """End-to-end through the actual script. The unit tests above would all pass on a
        `build_table1.py` that computed the annotation and never printed it — the same shape of gap
        that let `N_CANDIDATES` be read, echoed, and never passed."""
        import subprocess

        root = tmp_path / "logs"
        root.mkdir()
        for bench, arm, n_cand, n_proved in [
            ("fate_m", "none", None, 2), ("fate_m", "li", 50_000, 5),
            ("proofnet_test", "none", None, 1), ("proofnet_test", "li", 1_000, 2),
        ]:
            d = root / f"{bench}_{arm}"
            d.mkdir()
            cfg = {"benchmark": bench, "arm": arm}
            if n_cand is not None:
                cfg["n_candidates"] = n_cand
            (d / "manifest.json").write_text(json.dumps({
                "run_id": d.name, "started_utc": "2026-08-06T00:00:00+00:00",
                "config": cfg, "outcome": {"n_proved": n_proved},
            }))
            (d / "attempts.jsonl").write_text("\n".join(
                json.dumps({"problem_id": str(i),
                            "status": "proved" if i < n_proved else "exhausted",
                            "proved": i < n_proved,
                            "proof": ["aesop"] if i < n_proved else None})
                for i in range(10)
            ))

        repo = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(repo / "src")}
        p = subprocess.run(
            [sys.executable, str(repo / "scripts" / "build_table1.py"),
             "--results-root", str(root), "--out-dir", str(tmp_path / "out"),
             "--n-boot", "50", "--n-perm", "50"],
            capture_output=True, text=True, env=env, cwd=repo,
        )
        assert p.returncode == 0, p.stderr
        table = (tmp_path / "out" / "table1.md").read_text(encoding="utf-8")
        assert "5/10 (50.0%) @50k" in table, table
        assert "2/10 (20.0%) @1k" in table, table
        assert "the LI column is not one system" in p.stdout, p.stdout
        rendered = json.loads((tmp_path / "out" / "table1.json").read_text(encoding="utf-8"))
        assert rendered["li_n_candidates"] == {"fate_m": 50_000, "proofnet_test": 1_000}


class TestTableCaptionMatchesThePolicy:
    """The caption is not decoration; under the wrong policy it contradicts the table.

    It said unconditionally that "the rows above use a model-free tactic policy with no language
    model at all, so the rates are not comparable" — true for Track A' and flatly false for Tier 1,
    where the rows *are* a frozen 7B and the published numbers *are* comparable in kind. A reader
    seeing 32.6% beside a published 56.7% needs the 4,000x budget gap named, not a disclaimer that
    the comparison is invalid.
    """

    @staticmethod
    def build(tmp_path, policy):
        import subprocess

        root = tmp_path / "logs"
        root.mkdir(exist_ok=True)
        for bench, arm, n_cand, n_proved in [
            ("fate_m", "none", None, 2), ("fate_m", "sv", None, 4),
            ("fate_m", "li", 50_000, 5),
        ]:
            d = root / f"{bench}_{arm}_{policy}"
            d.mkdir()
            cfg = {"benchmark": bench, "arm": arm, "policy_kind": policy}
            if n_cand is not None:
                cfg["n_candidates"] = n_cand
            (d / "manifest.json").write_text(json.dumps({
                "run_id": d.name, "started_utc": "2026-08-09T00:00:00+00:00",
                "config": cfg, "outcome": {"n_proved": n_proved},
            }))
            (d / "attempts.jsonl").write_text("\n".join(
                json.dumps({"problem_id": str(i),
                            "status": "proved" if i < n_proved else "exhausted",
                            "proved": i < n_proved,
                            "proof": ["aesop"] if i < n_proved else None})
                for i in range(10)
            ))
        repo = Path(__file__).resolve().parent.parent
        out = tmp_path / f"out_{policy}"
        p = subprocess.run(
            [sys.executable, str(repo / "scripts" / "build_table1.py"),
             "--results-root", str(root), "--out-dir", str(out),
             "--policy", policy, "--n-boot", "50", "--n-perm", "50"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(repo / "src")}, cwd=repo,
        )
        assert p.returncode == 0, p.stderr
        return (out / "table1.md").read_text(encoding="utf-8")

    def test_the_llm_table_does_not_claim_to_be_model_free(self, tmp_path):
        table = self.build(tmp_path, "vllm")
        assert "model-free" not in table.replace("model-free repertoire", "")
        assert "REAL-Prover-v1 (7B) frozen" in table

    def test_the_llm_table_names_the_budget_gap(self, tmp_path):
        """32.6% beside a published 56.7% is a budget difference before it is a system one."""
        table = self.build(tmp_path, "vllm")
        assert "1/4,000" in table
        assert "Pass@64x64" in table

    def test_the_llm_table_explains_the_withheld_premise_metric(self, tmp_path):
        table = self.build(tmp_path, "vllm")
        assert "not reported" in table
        assert "premise-needed rates" not in table

    def test_the_model_free_table_keeps_its_own_caption(self, tmp_path):
        table = self.build(tmp_path, "repertoire")
        assert "model-free" in table
        assert "premise-needed rates" in table
        assert "1/4,000" not in table

    def test_each_title_names_its_tier(self, tmp_path):
        assert "Track A'" in self.build(tmp_path, "repertoire")
        assert "Tier 1" in self.build(tmp_path, "vllm")


class TestTableRefusesIncomparableCells:
    """Two ways `results/exported/logs` builds a plausible, wrong Table 1.

    `discover` takes the most recent finalised run per (benchmark, arm). Against the private
    `results/logs` that was unambiguous, because only Tier 1 lived there. Against the published
    export it is not: the pass@8 sweep (64 x 32, eight seeds) and the budget pilot (64 x 16, the
    first 60 ProofNet problems) are both *newer* than the Tier 1 runs whose cells they take. Both
    produce a table that looks entirely normal and is not the published one.

    The refusals are the point of these tests -- a checker that cannot fail is not a checker.
    """

    @staticmethod
    def _run(root, tmp_path, *extra):
        import subprocess
        repo = Path(__file__).resolve().parent.parent
        return subprocess.run(
            [sys.executable, str(repo / "scripts" / "build_table1.py"),
             "--policy", "vllm", "--results-root", str(root),
             "--out-dir", str(tmp_path / "out"), "--n-boot", "50", "--n-perm", "50", *extra],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(repo / "src")}, cwd=repo,
        )

    @staticmethod
    def _write(root, name, *, bench, arm, started, samples, n_problems, n_proved):
        d = root / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({
            "run_id": name, "started_utc": started,
            "config": {"benchmark": bench, "arm": arm, "policy_kind": "vllm",
                       "n_problems": n_problems,
                       "search": {"max_expansions": 64, "samples_per_step": samples}},
            "outcome": {"n_proved": n_proved},
        }))
        (d / "attempts.jsonl").write_text("\n".join(
            json.dumps({"problem_id": str(i), "proved": i < n_proved,
                        "status": "proved" if i < n_proved else "exhausted",
                        "proof": ["aesop"] if i < n_proved else None})
            for i in range(n_problems)
        ))

    def test_refuses_cells_built_at_different_search_budgets(self, tmp_path):
        root = tmp_path / "logs"
        self._write(root, "a", bench="fate_m", arm="none", started="2026-08-01T00:00:00+00:00",
                    samples=16, n_problems=10, n_proved=3)
        self._write(root, "b", bench="fate_m", arm="sv", started="2026-08-09T00:00:00+00:00",
                    samples=32, n_problems=10, n_proved=6)
        p = self._run(root, tmp_path)
        assert p.returncode != 0
        assert "different search budgets" in p.stderr, p.stderr
        assert "64 x 16" in p.stderr and "64 x 32" in p.stderr, p.stderr

    def test_the_match_filter_resolves_it(self, tmp_path):
        root = tmp_path / "logs"
        self._write(root, "a", bench="fate_m", arm="none", started="2026-08-01T00:00:00+00:00",
                    samples=16, n_problems=10, n_proved=3)
        self._write(root, "b", bench="fate_m", arm="sv", started="2026-08-09T00:00:00+00:00",
                    samples=32, n_problems=10, n_proved=6)
        self._write(root, "c", bench="fate_m", arm="sv", started="2026-08-02T00:00:00+00:00",
                    samples=16, n_problems=10, n_proved=5)
        p = self._run(root, tmp_path, "--match", "search.samples_per_step=16")
        assert p.returncode == 0, p.stderr
        assert "5/10" in (tmp_path / "out" / "table1.md").read_text(encoding="utf-8")

    def test_refuses_a_sixty_problem_cell_beside_a_full_benchmark(self, tmp_path):
        root = tmp_path / "logs"
        self._write(root, "full", bench="proofnet_test", arm="none",
                    started="2026-08-01T00:00:00+00:00", samples=16, n_problems=186, n_proved=24)
        self._write(root, "pilot", bench="proofnet_test", arm="li",
                    started="2026-08-15T00:00:00+00:00", samples=16, n_problems=60, n_proved=7)
        p = self._run(root, tmp_path)
        assert p.returncode != 0
        assert "different numbers of problems" in p.stderr, p.stderr
        assert "60 problems" in p.stderr and "186 problems" in p.stderr, p.stderr

    def test_full_benchmarks_drops_the_pilot(self, tmp_path):
        root = tmp_path / "logs"
        self._write(root, "full", bench="proofnet_test", arm="none",
                    started="2026-08-01T00:00:00+00:00", samples=16, n_problems=186, n_proved=24)
        self._write(root, "tier1", bench="proofnet_test", arm="li",
                    started="2026-08-10T00:00:00+00:00", samples=16, n_problems=186, n_proved=28)
        self._write(root, "pilot", bench="proofnet_test", arm="li",
                    started="2026-08-15T00:00:00+00:00", samples=16, n_problems=60, n_proved=7)
        p = self._run(root, tmp_path, "--full-benchmarks")
        assert p.returncode == 0, p.stderr
        table = (tmp_path / "out" / "table1.md").read_text(encoding="utf-8")
        assert "28/186" in table, table
        assert "7/60" not in table, table

    def test_a_uniformly_small_run_set_is_allowed(self, tmp_path):
        """Raggedness is the defect, not size. Fixture-scale tables must still build."""
        root = tmp_path / "logs"
        self._write(root, "a", bench="fate_m", arm="none", started="2026-08-01T00:00:00+00:00",
                    samples=16, n_problems=3, n_proved=1)
        self._write(root, "b", bench="fate_m", arm="sv", started="2026-08-02T00:00:00+00:00",
                    samples=16, n_problems=3, n_proved=2)
        p = self._run(root, tmp_path)
        assert p.returncode == 0, p.stderr
