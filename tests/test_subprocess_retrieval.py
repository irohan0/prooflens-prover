"""Retrieval in a second interpreter: mostly tested without one, and once with one.

    vllm 0.26.0   requires transformers>=5.5.3
    pylate 1.6.0  requires transformers<=5.3.0

Disjoint ranges, so no pin resolves it and the two stacks cannot share a process. This client is the
alternative, and it has to be right for a reason beyond installability: the prover's retrieval must
be *bit-identical* to the model-free arm's, or a Track A'-to-Tier 1 difference could be a difference
in retrieval rather than in the generator.

The protocol tests use an in-memory transport. The last class spawns a real subprocess — a stub
server needing no heavy dependencies — because a protocol both sides of which I wrote can agree
with itself and still not survive a pipe.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from prooflens_prover.retrieval.base import RetrievalStats
from prooflens_prover.retrieval.subprocess_client import (
    ProcessTransport,
    SubprocessRetriever,
    spawn_retrieval_server,
)
from prooflens_prover.retrieval.testing import StubTransport

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestHandshake:
    def test_index_metadata_reaches_the_manifest(self):
        """A subprocess arm recording less than an in-process one would not be comparable."""
        r = SubprocessRetriever(StubTransport(), arm="li")
        assert r.index.corpus_id == "276070:31db61c63a9b7ee1"
        assert r.index.n_docs == 276070
        assert r.index.n_candidates == 50000
        assert r.index.encoder == {"kind": "li", "checkpoint": "stub", "dim": 128}

    def test_effective_n_candidates_works_unchanged(self):
        """`prove_benchmark` reads the budget off `retriever.index.n_candidates`."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from prove_benchmark import effective_n_candidates

        assert effective_n_candidates(SubprocessRetriever(StubTransport(), arm="li")) == 50000

    def test_a_server_serving_the_wrong_arm_is_refused(self):
        # The worst possible silent failure: every number attributed to the wrong retriever.
        with pytest.raises(RuntimeError, match="serving arm='sv' but this run is arm='li'"):
            SubprocessRetriever(StubTransport(arm="sv"), arm="li")

    def test_name_is_the_arm_not_the_transport(self):
        # Two runs of one arm must not look like different arms because of how retrieval was hosted.
        assert SubprocessRetriever(StubTransport(), arm="li").name == "li"


class TestRetrieve:
    def test_premises_round_trip(self):
        r = SubprocessRetriever(StubTransport(), arm="li")
        premises = r.retrieve("⊢ a * b = b * a", k=2)
        assert [p.formal_name for p in premises] == ["mul_comm", "add_zero"]
        assert premises[0].formal_statement == "∀ a b, a * b = b * a"
        assert premises[0].score == pytest.approx(0.9)

    def test_k_is_forwarded(self):
        t = StubTransport()
        SubprocessRetriever(t, arm="li").retrieve("q", k=1)
        assert t.sent[-1] == {"query": "q", "k": 1}

    def test_stats_are_recorded_in_seconds_and_the_right_order(self):
        """`RetrievalStats.record(latency_s, n_returned)`.

        The first version of this client called it `record(n_returned, latency_ms)` — reversed and
        in the wrong unit, which would have reported ~2 ms mean latency for every arm and a premise
        count in the hundreds. Both plausible enough to publish.
        """
        stats = RetrievalStats()
        r = SubprocessRetriever(StubTransport(), arm="li", stats=stats)
        r.retrieve("q", k=2)
        d = stats.to_dict()
        assert d["n_queries"] == 1
        assert d["mean_premises_returned"] == 2.0
        assert 0.0 <= d["mean_latency_ms"] < 1000.0

    def test_unicode_survives_the_pipe(self):
        # Lean goals are full of it, and this is JSON over a text stream.
        t = StubTransport(premises=[
            {"formal_name": "foo", "formal_statement": "∀ (α : Type u) (x : α), x = x",
             "score": 1.0}
        ])
        [p] = SubprocessRetriever(t, arm="li").retrieve("⊢ ∑ i ∈ s, f i = 0", k=1)
        assert p.formal_statement == "∀ (α : Type u) (x : α), x = x"
        assert t.sent[-1]["query"] == "⊢ ∑ i ∈ s, f i = 0"

    def test_informal_name_is_carried_when_present(self):
        t = StubTransport(premises=[
            {"formal_name": "mul_comm", "formal_statement": "s",
             "informal_name": "multiplication commutes", "score": 1.0}
        ])
        [p] = SubprocessRetriever(t, arm="li").retrieve("q", k=1)
        assert p.informal_name == "multiplication commutes"

    def test_missing_informal_name_defaults_to_empty(self):
        [p] = SubprocessRetriever(StubTransport(), arm="li").retrieve("q", k=1)
        assert p.informal_name == ""


class TestFailureModes:
    def test_an_error_payload_raises_rather_than_returning_nothing(self):
        """Returning zero premises silently looks exactly like a retriever that found nothing."""
        t = StubTransport(scripted=[
            json.dumps({"arm": "li", "n_docs": 1, "corpus_id": "c"}),
            json.dumps({"error": "index file is truncated"}),
        ])
        r = SubprocessRetriever(t, arm="li")
        with pytest.raises(RuntimeError, match="index file is truncated"):
            r.retrieve("q")

    def test_non_json_output_names_what_it_received(self):
        t = StubTransport(scripted=[
            json.dumps({"arm": "li", "n_docs": 1, "corpus_id": "c"}),
            "Traceback (most recent call last):",
        ])
        r = SubprocessRetriever(t, arm="li")
        with pytest.raises(RuntimeError, match="not JSON.*Traceback"):
            r.retrieve("q")

    def test_names_are_fetched_lazily_and_only_once(self):
        t = StubTransport(names=["a", "b", "c"])
        r = SubprocessRetriever(t, arm="li")
        assert [x["cmd"] for x in t.sent if "cmd" in x] == ["hello"]
        assert [rec.name for rec in r.index.records] == ["a", "b", "c"]
        assert [rec.name for rec in r.index.records] == ["a", "b", "c"]
        assert [x["cmd"] for x in t.sent if "cmd" in x].count("names") == 1

    def test_close_shuts_the_server_down(self):
        t = StubTransport()
        SubprocessRetriever(t, arm="li").close()
        assert t.closed

    def test_context_manager_closes(self):
        t = StubTransport()
        with SubprocessRetriever(t, arm="li"):
            pass
        assert t.closed

    def test_a_missing_interpreter_fails_with_a_useful_message(self, tmp_path):
        with pytest.raises(SystemExit, match="must point at the virtualenv that has pylate"):
            spawn_retrieval_server(tmp_path / "no_such_python", "li", None, REPO_ROOT)


STUB_SERVER = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "hello":
            out = {"arm": "li", "corpus_id": "c", "n_docs": 2, "n_candidates": 7,
                   "encoder": {"kind": "li"}}
        elif req.get("cmd") == "names":
            out = {"names": ["mul_comm", "add_zero"]}
        elif req.get("cmd") == "shutdown":
            sys.stdout.write(json.dumps({"ok": True}) + "\\n"); sys.stdout.flush(); break
        else:
            out = {"premises": [{"formal_name": "mul_comm",
                                 "formal_statement": "\\u2200 a b, a * b = b * a",
                                 "score": 0.5}][: req.get("k", 10)]}
        sys.stdout.write(json.dumps(out) + "\\n")
        sys.stdout.flush()
''')


class TestOverARealPipe:
    """A protocol both ends of which one person wrote can agree with itself and still not survive a
    pipe: buffering, encoding, and a child that dies are all invisible to an in-memory double."""

    @pytest.fixture
    def server_script(self, tmp_path):
        path = tmp_path / "stub_server.py"
        path.write_text(STUB_SERVER, encoding="utf-8")
        return path

    def test_end_to_end_through_a_subprocess(self, server_script):
        transport = ProcessTransport([sys.executable, str(server_script)])
        try:
            r = SubprocessRetriever(transport, arm="li")
            assert r.index.n_docs == 2
            assert r.index.n_candidates == 7
            [p] = r.retrieve("⊢ a * b = b * a", k=1)
            assert p.formal_name == "mul_comm"
            assert p.formal_statement == "∀ a b, a * b = b * a"
            assert [rec.name for rec in r.index.records] == ["mul_comm", "add_zero"]
        finally:
            transport.close()

    def test_many_queries_do_not_desynchronise(self, server_script):
        """One stray write to stdout would be read as a response and shift every later reply."""
        transport = ProcessTransport([sys.executable, str(server_script)])
        try:
            r = SubprocessRetriever(transport, arm="li")
            for _ in range(25):
                assert [p.formal_name for p in r.retrieve("q", k=1)] == ["mul_comm"]
        finally:
            transport.close()

    def test_a_server_that_exits_gives_a_clear_error_not_a_hang(self, tmp_path):
        script = tmp_path / "dies.py"
        script.write_text("import sys; sys.exit(3)", encoding="utf-8")
        with pytest.raises(RuntimeError, match="closed its output|exited with code"):
            SubprocessRetriever(ProcessTransport([sys.executable, str(script)]), arm="li")

    def test_the_real_server_script_at_least_parses_and_shows_help(self):
        # Not a functional test — the real server needs a 5.5 GB index. This catches a syntax error
        # or a bad import path, which is the failure that would otherwise appear inside a GPU job.
        p = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "retrieval_server.py"), "--help"],
            capture_output=True, text=True,
        )
        assert p.returncode == 0, p.stderr
        assert "--arm" in p.stdout
