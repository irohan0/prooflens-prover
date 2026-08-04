"""Benchmark loading — one `Problem` schema for FATE-M, ProofNet, miniF2F and PutnamBench.

## Where the data comes from, and why

FATE-M / ProofNet / miniF2F are read from **REAL-Prover's own repository**
(`Realprover/data/*.jsonl`). Using their files rather than re-deriving our own removes a whole
class of confound: a row of ours is then comparable to a row of theirs without an argument about
whether the two harnesses were even given the same problems. It also settles a real discrepancy —
the standalone FATE repo ships a refactored FATE-M of **150** problems, while REAL-Prover's paper
reports on **141**. Their file is the one their 56.7 / 44.7 numbers refer to.

Verified counts: FATE-M 141, ProofNet-test 186, miniF2F-test 244.

## The parsing problem

A `formal_statement` is not a bare declaration. It bundles imports, `open`s, `set_option`s,
`variable` bindings, comments and the declaration together::

    import Mathlib
    import Aesop
    set_option maxHeartbeats 0
    open BigOperators Real Nat Topology Rat

    theorem mathd_algebra_478 (b h v : ℝ) ... : v = 65 := sorry

Those parts have to go to *different* places:

- **imports** define the REPL environment. `import Mathlib` costs ~86 s and ~4 GB, so environments
  are cached per distinct import set — and since almost every problem uses the same one or two
  imports, that cost is paid once or twice per worker rather than per problem.
- **everything else** (`open`, `set_option`, `variable`, comments) *scopes* the declaration and
  must be sent in the same command as it. Dropping the `open`s would leave half the identifiers in
  the statement unresolvable, and the problem would fail to elaborate for a reason that has nothing
  to do with the prover.

`split_statement` does that separation, and is unit-tested against a real example of each form.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prooflens_prover.utils.logging import get_logger

log = get_logger("data.benchmarks")

#: Declaration keywords that open a provable statement. `example` matters: FATE-M uses it
#: throughout, and a loader that only recognised `theorem` would silently drop every FATE-M
#: problem — the primary discriminating benchmark.
_DECL_KEYWORDS = ("theorem", "lemma", "example", "def", "instance", "abbrev")
_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|nonrec\s+)*"
    r"(" + "|".join(_DECL_KEYWORDS) + r")\b",
    re.MULTILINE,
)
_IMPORT_RE = re.compile(r"^\s*import\s+\S+", re.MULTILINE)


@dataclass(frozen=True)
class Problem:
    """One benchmark problem, normalised across sources."""

    id: str
    source: str                       # "fate_m" | "proofnet_test" | "minif2f_test" | "putnambench"
    imports: str                      # the `import ...` lines; defines the REPL environment
    preamble: str                     # open / set_option / variable / comments; scopes the decl
    declaration: str                  # the `theorem`/`example ... := sorry` itself
    informal_statement: str | None = None
    reference_proof: str | None = None
    raw: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def statement(self) -> str:
        """Preamble + declaration — what gets sent to the REPL as one command.

        Excludes imports, which are elaborated separately and cached (see module docstring).
        """
        if not self.preamble:
            return self.declaration
        return f"{self.preamble}\n\n{self.declaration}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "source": self.source, "imports": self.imports,
            "preamble": self.preamble, "declaration": self.declaration,
            "informal_statement": self.informal_statement, "meta": self.meta,
        }


def split_statement(text: str) -> tuple[str, str, str]:
    """Split a benchmark `formal_statement` into (imports, preamble, declaration).

    The declaration is taken from the **first** line that starts a declaration keyword through to
    the end of the text, so multi-line statements (the common case) stay intact. Everything before
    it that is not an `import` becomes the preamble.

    Raises `ValueError` when no declaration is found: a silently-empty declaration would elaborate
    to nothing and score as an unprovable problem, quietly depressing every arm equally and
    masking a loader bug as a hard benchmark.
    """
    imports = "\n".join(m.group(0).strip() for m in _IMPORT_RE.finditer(text))
    without_imports = _IMPORT_RE.sub("", text)

    m = _DECL_RE.search(without_imports)
    if m is None:
        raise ValueError(f"no declaration keyword found in statement:\n{text[:300]}")

    preamble = without_imports[: m.start()].strip()
    declaration = without_imports[m.start():].strip()
    return imports, preamble, declaration


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_realprover_jsonl(path: Path | str, source: str) -> list[Problem]:
    """Load one of REAL-Prover's benchmark files.

    Schema: `id`, `formal_statement`, and optionally `formal_proof` / `informal_statement`
    (FATE-M carries both; ProofNet and miniF2F carry neither).
    """
    path = Path(path)
    problems: list[Problem] = []
    for row in _load_jsonl(path):
        raw = row["formal_statement"]
        imports, preamble, declaration = split_statement(raw)
        proof = (row.get("formal_proof") or "").strip() or None
        problems.append(Problem(
            id=str(row["id"]),
            source=source,
            imports=imports,
            preamble=preamble,
            declaration=declaration,
            informal_statement=(row.get("informal_statement") or "").strip() or None,
            reference_proof=proof,
            raw=raw,
        ))
    log.info("loaded %d problems from %s (%s)", len(problems), path.name, source)
    return problems


def load_putnambench(lean4_src_dir: Path | str) -> list[Problem]:
    """Load PutnamBench from its per-problem `.lean` files (`lean4/src/putnam_*.lean`).

    PutnamBench ships one file per problem rather than a JSONL, and many carry a separate
    `abbrev ..._solution` that supplies the numeric answer. Those are kept in the preamble so the
    **answer-given** task variant is what runs — the standard, easier setting that published
    PutnamBench numbers use. Any run must state which variant it used; they are not comparable.
    """
    d = Path(lean4_src_dir)
    files = sorted(d.glob("putnam_*.lean"))
    problems: list[Problem] = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        try:
            imports, preamble, declaration = split_statement(raw)
        except ValueError:
            log.warning("skipping %s: no declaration found", f.name)
            continue
        problems.append(Problem(
            id=f.stem, source="putnambench", imports=imports, preamble=preamble,
            declaration=declaration, raw=raw,
            meta={"year": _putnam_year(f.stem), "problem": _putnam_index(f.stem)},
        ))
    log.info("loaded %d PutnamBench problems from %s", len(problems), d)
    return problems


def _putnam_year(stem: str) -> int | None:
    m = re.search(r"putnam_(\d{4})_", stem)
    return int(m.group(1)) if m else None


def _putnam_index(stem: str) -> str | None:
    """`a1`/`b3` etc — lets results be stratified by position, since A1/B1 are reliably the
    easiest problems in each session and an aggregate hides that structure entirely."""
    m = re.search(r"putnam_\d{4}_([ab]\d+)", stem)
    return m.group(1) if m else None


#: Filenames as shipped in REAL-Prover's repo, with their verified problem counts. The counts are
#: asserted at load time by `load_benchmark`: a silent change in row count would alter every
#: reported rate while still "working".
REALPROVER_BENCHMARKS: dict[str, tuple[str, int]] = {
    "fate_m": ("fate_m.jsonl", 141),
    "proofnet_test": ("proofnet_test.jsonl", 186),
    "proofnet_valid": ("proofnet_valid.jsonl", 185),
    "minif2f_test": ("minif2f_test.jsonl", 244),
    "minif2f_valid": ("minif2f_valid.jsonl", 244),
}


def load_benchmark(
    name: str, data_root: Path | str, check_count: bool = True
) -> list[Problem]:
    """Load a benchmark by name.

    `data_root` is REAL-Prover's `Realprover/data` directory for the JSONL benchmarks, or
    PutnamBench's `lean4/src` for `putnambench`.
    """
    if name == "putnambench":
        return load_putnambench(data_root)

    if name not in REALPROVER_BENCHMARKS:
        raise ValueError(
            f"unknown benchmark {name!r}; have {sorted(REALPROVER_BENCHMARKS)} + 'putnambench'"
        )
    filename, expected = REALPROVER_BENCHMARKS[name]
    problems = load_realprover_jsonl(Path(data_root) / filename, name)
    if check_count and len(problems) != expected:
        raise ValueError(
            f"{name}: expected {expected} problems, loaded {len(problems)}. The published "
            f"comparison assumes the documented count; a different one silently changes every "
            f"reported rate. Pass check_count=False only if you intend a different subset."
        )
    return problems
