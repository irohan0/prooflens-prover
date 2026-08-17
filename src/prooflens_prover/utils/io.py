"""JSONL / YAML I/O and env-var config expansion.

JSONL is the project's audit format: one proof attempt per line, appended as the run proceeds so a
killed job keeps everything it had already produced. `orjson` is used when available because the
search loop writes a record per attempt and json.dumps shows up in profiles at PutnamBench scale.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

try:
    import orjson

    def _dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)
except ImportError:  # pragma: no cover - orjson is a hard dep; this keeps tests runnable without it
    orjson = None

    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj, default=str).encode("utf-8")


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file. Blank lines are skipped (a truncated final line from a killed job is
    the common case, and it must not poison an otherwise-complete run's aggregation)."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl_tolerant(path: Path | str) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Every parseable row, plus the `(line number, byte length)` of each one that is not.

    **Why tolerance, when `read_jsonl` deliberately raises.** A corrupt row must not be able to
    throw away a run that has already finished its work. Measured: ProofNet / sv / seed 6 of the
    pass@8 sweep completed all 186 problems in 4 h 59 m, then died in its own reporting block —
    `json.loads` over the file it had just written — because line 55 was a truncated 118 KB record.
    Every proof was on disk and the exit status said the job had failed.

    `attempts.jsonl` is appended with `O_APPEND` and an fsync per row, which makes it durable
    against a SLURM kill but *not* against NFS, where an append that size is not atomic and
    `results/logs` lives. So a partial row mid-file is a real state this reader has to survive.

    A skipped row means one problem's outcome is unknown, which is not the same as unsolved. Callers
    are handed the line numbers rather than a count so they can say which, and must not present a
    rate over the remainder as if the denominator were whole. `scripts/repair_attempts.py`
    quarantines the bad rows so a resume can re-attempt exactly those problems.
    """
    rows: list[dict[str, Any]] = []
    bad: list[tuple[int, int]] = []
    with open(path, "rb") as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                bad.append((n, len(raw)))
    return rows, bad


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows to JSONL, creating parent directories. Returns the row count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "wb") as f:
        for row in rows:
            f.write(_dumps(row))
            f.write(b"\n")
            n += 1
    return n


class JsonlAppender:
    """Append-and-flush JSONL writer for long-running search jobs.

    Flushes every record. That costs throughput but means a job killed by a SLURM walltime limit
    loses **zero** completed attempts — the property that makes a partial run still reportable, and
    the one the plan's "resumable from last checkpoint" requirement rests on.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: TextIO | None = None
        self.n_written = 0

    def __enter__(self) -> JsonlAppender:
        self._f = open(self.path, "ab")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def append(self, row: dict[str, Any]) -> None:
        if self._f is None:
            raise RuntimeError("JsonlAppender must be used as a context manager")
        self._f.write(_dumps(row))
        self._f.write(b"\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.n_written += 1


def expand_env(obj: Any) -> Any:
    """Recursively expand `${VAR}` in strings within a nested config structure.

    Mirrors the predecessor project's config convention (`${SCRATCH}/...`, `${DATA_ROOT}/...`) so
    the same YAML works unchanged on a laptop and on CSF3, where those roots differ.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    return obj


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config with `${VAR}` expansion applied."""
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(cfg).__name__}")
    return expand_env(cfg)
