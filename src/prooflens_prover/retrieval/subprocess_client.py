"""Run retrieval in another interpreter, because two stacks cannot share one.

## The conflict this exists to make impossible

    vllm 0.26.0   requires transformers>=5.5.3
    pylate 1.6.0  requires transformers<=5.3.0

Disjoint ranges. No pin resolves that — one of the two packages has to move to a version whose range
overlaps, or they have to stop sharing a process. Six attempts at the first approach cost a day;
this is the second.

`SubprocessRetriever` implements the same `Retriever` protocol as the in-process retrievers and
forwards each query to `scripts/retrieval_server.py`, launched under the **retrieval** virtualenv —
the environment that produced every verified Track A' result and that nothing here needs to modify.
The prover keeps vLLM's environment. Neither has to know the other exists.

## Why this is better than a version compromise even if a compromise were available

The retrieval environment is the one the published numbers came from. Leaving it untouched means the
LLM arm queries *bit-identically* the same retriever as the model-free arm, so a difference between
Track A' and Tier 1 cannot be a difference in retrieval. A negotiated `transformers` version, even a
working one, would silently change the query encoder's dependencies underneath a comparison.

## Protocol

Newline-delimited JSON over stdin/stdout, because the alternative — a socket, a port, a service to
supervise — is more failure modes for no benefit at one query at a time. stderr is left attached to
the parent so the server's logging lands in the SLURM log rather than in a pipe nobody reads.

    -> {"query": "...", "k": 10}
    <- {"premises": [{"formal_name": ..., "formal_statement": ..., "score": ...}],
        "latency_ms": 1.2}

A handshake on startup carries what the run manifest needs (`corpus_id`, `n_docs`, `n_candidates`,
the encoder spec), so the manifest is as complete as an in-process run's. Without that, a subprocess
arm would be unauditable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from prooflens_prover.retrieval.base import DEFAULT_TOP_K, Premise, RetrievalStats
from prooflens_prover.utils.logging import get_logger

log = get_logger(__name__)

#: The server loads a 5.5 GB index and a query encoder before it answers the handshake. Measured at
#: 77 s on CSF3 with the LI index. Generous because the alternative — a timeout that fires on a slow
#: NFS day — kills a job for no reason.
DEFAULT_STARTUP_TIMEOUT_S = 900.0


class LineTransport(Protocol):
    """Newline-delimited request/response. Injectable so the client is testable without a
    subprocess."""

    def send(self, line: str) -> None:
        ...

    def recv(self) -> str:
        ...

    def close(self) -> None:
        ...


class ProcessTransport:
    """A child interpreter, talking over its stdin/stdout."""

    def __init__(self, argv: list[str], cwd: Path | None = None):
        self.argv = argv
        # stderr is deliberately NOT captured: the server's logging belongs in the job's log, and a
        # pipe nobody drains is also a deadlock waiting for a large enough traceback.
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(cwd) if cwd else None,
        )

    def send(self, line: str) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"retrieval server exited with code {self.proc.returncode} before this request; "
                "its traceback is in the job's stderr"
            )
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def recv(self) -> str:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.poll()
            raise RuntimeError(
                f"retrieval server closed its output (exit code {code}); see the job's stderr"
            )
        return line

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.send(json.dumps({"cmd": "shutdown"}))
            self.proc.wait(timeout=30)
        except Exception:                       # noqa: BLE001 — best effort; we kill next
            self.proc.kill()
            self.proc.wait(timeout=30)


@dataclass
class RemoteIndexInfo:
    """What the run manifest needs about an index it never loaded.

    Mirrors the attribute names of the in-process indices (`corpus_id`, `n_docs`, `n_candidates`,
    `encoder`) so `prove_benchmark`'s manifest code and `effective_n_candidates` work unchanged. A
    subprocess arm whose manifest recorded less than an in-process one would not be comparable
    to it, which is the whole point of the arm.
    """

    corpus_id: str | None = None
    n_docs: int = 0
    n_candidates: int | None = None
    encoder: dict[str, Any] | None = None
    _names: list[str] | None = field(default=None, repr=False)
    _fetch_names: Any = field(default=None, repr=False)

    @property
    def records(self) -> list[Any]:
        """Premise names, as objects with a `.name`, fetched on first use.

        Only the gloss-coverage check needs these, and it needs all 276,070, so they are pulled
        once on demand rather than shipped in the handshake of every run that does not.
        """
        if self._names is None:
            self._names = list(self._fetch_names()) if self._fetch_names else []
        return [_NamedOnly(n) for n in self._names]


@dataclass(frozen=True)
class _NamedOnly:
    name: str


class SubprocessRetriever:
    """A `Retriever` whose work happens in another interpreter."""

    def __init__(
        self,
        transport: LineTransport,
        arm: str,
        stats: RetrievalStats | None = None,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    ):
        self.transport = transport
        self.arm = arm
        self.stats = stats or RetrievalStats()
        self.index = RemoteIndexInfo(_fetch_names=self._fetch_names)
        self._handshake(startup_timeout_s)

    @property
    def name(self) -> str:
        # Deliberately the same name an in-process retriever of this arm reports. The manifest
        # records `retrieval_transport` separately; the arm's identity must not depend on how it
        # was hosted, or two runs of one arm would look like two different arms.
        return self.arm

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.transport.send(json.dumps(payload))
        raw = self.transport.recv()
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"retrieval server sent something that is not JSON: {raw[:200]!r}"
            ) from exc
        if "error" in response:
            raise RuntimeError(f"retrieval server error: {response['error']}")
        return response

    def _handshake(self, timeout_s: float) -> None:
        t0 = time.perf_counter()
        info = self._request({"cmd": "hello"})
        took = time.perf_counter() - t0
        if took > timeout_s:
            log.warning("retrieval server took %.0fs to start (limit %.0fs)", took, timeout_s)
        self.index.corpus_id = info.get("corpus_id")
        self.index.n_docs = int(info.get("n_docs", 0))
        self.index.n_candidates = info.get("n_candidates")
        self.index.encoder = info.get("encoder")
        served = info.get("arm")
        if served != self.arm:
            # A server serving a different arm than the manifest will record is the single worst
            # possible outcome here: every number would be attributed to the wrong retriever.
            raise RuntimeError(
                f"retrieval server is serving arm={served!r} but this run is arm={self.arm!r}"
            )
        log.info("retrieval server ready in %.1fs: arm=%s n_docs=%d corpus_id=%s n_candidates=%s",
                 took, served, self.index.n_docs, self.index.corpus_id, self.index.n_candidates)

    def _fetch_names(self) -> list[str]:
        log.info("fetching %d premise names from the retrieval server", self.index.n_docs)
        return list(self._request({"cmd": "names"})["names"])

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Premise]:
        t0 = time.perf_counter()
        response = self._request({"query": query, "k": k})
        premises = [
            Premise(
                formal_name=p["formal_name"],
                formal_statement=p["formal_statement"],
                informal_name=p.get("informal_name", ""),
                score=float(p.get("score", 0.0)),
            )
            for p in response["premises"]
        ]
        # Timed on this side, so the number includes the IPC the arm actually pays. Reporting the
        # server's own `latency_ms` would understate the cost of the design.
        self.stats.record(time.perf_counter() - t0, len(premises))
        return premises

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> SubprocessRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def spawn_retrieval_server(
    python: str | Path,
    arm: str,
    index_dir: Path | None,
    repo_root: Path,
    n_candidates: int | None = None,
    checkpoint: str | None = None,
    device: str | None = None,
    stats: RetrievalStats | None = None,
) -> SubprocessRetriever:
    """Launch `scripts/retrieval_server.py` under `python` and return a client for it."""
    argv = [
        str(python), str(repo_root / "scripts" / "retrieval_server.py"),
        "--arm", arm,
    ]
    if index_dir is not None:
        argv += ["--index", str(index_dir)]
    if n_candidates is not None:
        argv += ["--n-candidates", str(n_candidates)]
    if checkpoint is not None:
        argv += ["--checkpoint", checkpoint]
    if device is not None:
        argv += ["--device", device]
    log.info("spawning retrieval server: %s", " ".join(argv))
    if not Path(python).exists():
        raise SystemExit(
            f"no interpreter at {python}. --retrieval-python must point at the virtualenv that has "
            "pylate installed (the one that produced the model-free results), not this one."
        )
    return SubprocessRetriever(ProcessTransport(argv, cwd=repo_root), arm=arm, stats=stats)


def _selftest() -> int:                        # pragma: no cover - developer convenience
    """`python -m prooflens_prover.retrieval.subprocess_client` round-trips against a stub."""
    from prooflens_prover.retrieval.testing import StubTransport

    r = SubprocessRetriever(StubTransport(), arm="li")
    print(r.index)
    print(r.retrieve("⊢ a * b = b * a", k=2))
    return 0


if __name__ == "__main__":                     # pragma: no cover
    sys.exit(_selftest())
