"""In-memory doubles for the subprocess retrieval transport.

Lives in the package rather than in `tests/` so the client's `_selftest` and the wiring tests share
one implementation. A second, subtly different stub is how a protocol test starts passing against a
protocol the real server does not speak.
"""

from __future__ import annotations

import json
from typing import Any


class StubTransport:
    """Answers the protocol without a subprocess.

    `scripted` replaces the default replies; each entry is the response to the next request, so a
    test can inject a malformed line or an error payload.
    """

    def __init__(
        self,
        arm: str = "li",
        premises: list[dict[str, Any]] | None = None,
        scripted: list[str] | None = None,
        names: list[str] | None = None,
    ):
        self.arm = arm
        self.premises = premises if premises is not None else [
            {"formal_name": "mul_comm", "formal_statement": "∀ a b, a * b = b * a", "score": 0.9},
            {"formal_name": "add_zero", "formal_statement": "∀ a, a + 0 = a", "score": 0.7},
        ]
        self.names = names if names is not None else ["mul_comm", "add_zero"]
        self.scripted = list(scripted or [])
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._pending: str | None = None

    def send(self, line: str) -> None:
        payload = json.loads(line)
        self.sent.append(payload)
        if self.scripted:
            self._pending = self.scripted.pop(0)
            return
        cmd = payload.get("cmd")
        if cmd == "hello":
            self._pending = json.dumps({
                "arm": self.arm, "corpus_id": "276070:31db61c63a9b7ee1", "n_docs": 276070,
                "n_candidates": 50000,
                "encoder": {"kind": self.arm, "checkpoint": "stub", "dim": 128},
            })
        elif cmd == "names":
            self._pending = json.dumps({"names": self.names})
        elif cmd == "shutdown":
            self._pending = json.dumps({"ok": True})
        else:
            k = int(payload.get("k", 10))
            self._pending = json.dumps({
                "premises": self.premises[:k], "latency_ms": 1.0,
            })

    def recv(self) -> str:
        if self._pending is None:
            raise AssertionError("recv() without a preceding send()")
        line, self._pending = self._pending, None
        return line

    def close(self) -> None:
        self.closed = True
