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
from prooflens_prover.utils.io import read_jsonl

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

    #: Problems this run claimed and `verify_proofs.py` rejected. Held rather than just subtracted,
    #: so a report can state how many claims were discounted instead of leaving the difference
    #: between `n_proved` in the manifest and the rate here unexplained.
    discounted: set[str] = field(default_factory=set)


def failed_verification(run_dir: Path) -> set[str]:
    """Problem ids whose claimed proof did NOT re-elaborate, from this run's `verification.json`.

    Empty when the report is absent — callers that require verification check for that separately.
    This function answers only "which claims did the independent re-check reject?".

    **Why a claim can fail while the search was honest.** Measured on ProofNet / sv / seed 6: the
    recorded proof begins with a bare `let`. During search each tactic is applied to a proof state
    on its own, so `let` was accepted as one step; verification joins the steps with newlines into a
    single tactic block, where `let` swallows the following line as its binder name and the block
    stops parsing. The rejection is still correct — a proof that does not elaborate is not a proof —
    but it is a *serialisation* failure, not a `sorry` and not an unsound step.
    """
    vf = run_dir / "verification.json"
    if not vf.exists():
        return set()
    report = json.loads(vf.read_text(encoding="utf-8"))
    return {str(f["problem_id"]) for f in report.get("failures") or ()}


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
    rejected = failed_verification(run_dir)
    for row in read_jsonl(run_dir / "attempts.jsonl"):
        pid = f"{draw.benchmark}:{row['problem_id']}"
        draw.attempted.add(pid)
        if not row.get("proved"):
            continue
        # A claimed proof that does not re-elaborate is not a proof, so it does not enter `solved`.
        # This is the minimal correct adjustment, and deliberately not "discard the run": on the one
        # run where it fired, 34 of 35 proofs verified and that run held the joint-highest count of
        # its arm, so dropping it entirely would have removed a high seed and biased the arm
        # downward — a larger error than the one being corrected. Discounting can only ever lower a
        # reported rate.
        if str(row["problem_id"]) in rejected:
            draw.discounted.add(pid)
            continue
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
