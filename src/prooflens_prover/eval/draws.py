"""A *draw*: one run read as one sample of one arm, and the set operations that compare two.

Two analyses need the same primitives from opposite directions, which is why they live here rather
than inside either script:

* `scripts/replication_variance.py` — the **same** arm at different seeds, to measure how much a
  re-run moves on its own (the noise floor).
* `scripts/verify_arm_distinctness.py` — **different** arms at the same seed, to prove they are
  genuinely different runs and not one run counted twice.

The second exists because Tier 1 reported an exact tie — 46 vs 46 on FATE-M, 26 vs 26 on
ProofNet — which is the shape a duplicated or mislabelled run would take. The distinguishing
evidence is `identical_proof_fraction`: at a fixed seed the vLLM engine is deterministic, so two
runs of the same arm agree on *every* shared proof, character for character. Two genuinely
different arms do not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from prooflens_prover.eval.compare import format_budget

#: Separator joining a proof's tactics into one comparable string. Chosen because it cannot occur
#: inside a Lean tactic, so a join can never make two different proofs compare equal.
TACTIC_JOIN = "\n;;\n"


@dataclass
class Draw:
    """One run: a single sampling draw of one arm on one benchmark."""

    run_id: str
    benchmark: str
    arm: str
    seed: int
    config: dict
    attempted: set[str] = field(default_factory=set)
    solved: set[str] = field(default_factory=set)
    proofs: dict[str, str] = field(default_factory=dict)

    @property
    def retriever(self) -> str | None:
        """The retriever the policy actually used, as recorded rather than inferred from the arm."""
        return (self.config.get("policy_config") or {}).get("retriever")

    @property
    def index(self) -> str | None:
        return self.config.get("index")

    @property
    def retrieval(self) -> dict:
        """`outcome.retrieval` — query count and latency. Empty for the no-retrieval control."""
        return self._retrieval

    _retrieval: dict = field(default_factory=dict, repr=False)


def load_draw(run_dir: Path) -> Draw:
    """Read one run directory into a `Draw`, with problem ids namespaced by benchmark.

    Namespacing matters as soon as two benchmarks are pooled: FATE-M and ProofNet both number their
    problems from scratch, so an un-namespaced union would silently merge unrelated problems.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config", {})
    n_candidates = cfg.get("n_candidates")
    arm = cfg.get("arm", "?")

    draw = Draw(
        run_id=manifest.get("run_id", run_dir.name),
        benchmark=cfg.get("benchmark", "?"),
        arm=f"{arm}@{format_budget(n_candidates)}" if n_candidates else arm,
        seed=int(manifest.get("seed", 0)),
        config=cfg,
        _retrieval=(manifest.get("outcome") or {}).get("retrieval") or {},
    )
    for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = f"{draw.benchmark}:{row['problem_id']}"
        draw.attempted.add(pid)
        if row.get("proved"):
            draw.solved.add(pid)
            draw.proofs[pid] = TACTIC_JOIN.join(row.get("proof") or ())
    return draw


def identical_proof_fraction(a: Draw, b: Draw) -> tuple[int, float | None]:
    """`(problems both solved, fraction whose proofs are byte-identical)`.

    The load-bearing statistic for both callers, in opposite directions.

    Comparing two draws of one arm, 1.0 means the seed never reached the sampler and the
    "replicate" is the same draw — every variance computed from it would be zero, which reads as an
    exceptionally stable result rather than an absent measurement.

    Comparing two different arms, 1.0 means one run was counted twice or mislabelled, and any
    difference reported between them is an artefact. Well below 1.0 means the arms genuinely
    explored different proofs.
    """
    shared = a.solved & b.solved
    if not shared:
        return 0, None
    return len(shared), sum(1 for p in shared if a.proofs.get(p) == b.proofs.get(p)) / len(shared)


def discordance(a: set[str], b: set[str]) -> tuple[int, int]:
    """`(|a \\ b|, |b \\ a|)` — the problems exactly one of the two solved."""
    return len(a - b), len(b - a)


def union_gain(a: set[str], b: set[str]) -> int:
    """How many problems either-of-two solves beyond the better single set."""
    return len(a | b) - max(len(a), len(b))


def solve_rate_map(draws: list[Draw]) -> dict[str, float]:
    """`{problem: fraction of draws that solved it}` over problems every draw attempted."""
    shared = set.intersection(*(d.attempted for d in draws))
    return {p: sum(1 for d in draws if p in d.solved) / len(draws) for p in shared}
