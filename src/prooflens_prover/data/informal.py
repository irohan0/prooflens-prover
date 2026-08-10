"""Informal (natural-language) names for Mathlib premises.

Source: the `FrenzyMath/mathlib_informal_v4.16.0` dataset — **the same Mathlib version this project
pins**, which is why the join is worth doing at all rather than approximating across versions.

REAL-Prover's premise blocks carry an `Informal name` beside the formal name and statement, served
from a dataset they describe but never released under that path. This one matches, so the field can
be filled instead of left blank.

## Why this only touches the prompt

The glosses are joined at prompt-construction time, never into the retrieval index:

* the retrievers encode **formal statements**, and were trained that way;
* `corpus_id` hashes premise **names only**, so it does not change;
* the dense index stores its own copy of the records, so a corpus file edit would not reach it
  anyway — but nothing here needs it to.

So adding glosses invalidates no index, needs no rebuild of the 5.5 GB LI index, and cannot alter
any retrieval result. It changes exactly one thing: what the model reads.

## Missing glosses stay missing

A premise absent from the mapping keeps an empty gloss. It does **not** get a humanised declaration
name ("mul comm" from `mul_comm`): that is invented content in a field the model was trained to read
as a human-written description, and a plausible fabrication is worse than a visible blank.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from prooflens_prover.utils.logging import get_logger

log = get_logger(__name__)

#: Key names seen in this family of datasets, most specific first. Tried in order; the first key
#: present in the first record wins, and the choice is logged so it appears in the job output.
FORMAL_KEYS = ("formal_name", "name", "Formal name", "decl_name", "declaration")
INFORMAL_KEYS = (
    "informal_name", "Informal name", "informal", "informal_statement",
    "informal_description", "description", "docstring",
)


#: Below this, the mapping is assumed to be keyed on something other than Lean declaration names.
#: A mapping of 300k entries that matches 0.1% of the corpus is not a near-miss — it is a schema
#: mismatch, and it renders identically to having no glosses at all. `mathlib_informal_v4.16.0`
#: against Mathlib v4.16.0 should cover most theorems; 5% is far below any plausible real value.
MIN_PLAUSIBLE_COVERAGE = 0.05


def _join_name(value: Any) -> str:
    """Normalise a formal name to Lean's dotted form.

    `mathlib_informal_v4.16.0` stores names as **path component lists** —
    `["CoxeterSystem", "length_simple_mul_ne"]`, not `"CoxeterSystem.length_simple_mul_ne"`. Passing
    that through `str()` yields `"['CoxeterSystem', 'length_simple_mul_ne']"`, which matches no
    premise in the corpus. The mapping would be fully populated, contain no usable key, and produce
    exactly the blank glosses it was added to fix — with no error anywhere.
    """
    if isinstance(value, (list, tuple)):
        return ".".join(str(part) for part in value)
    return str(value)


def _pick_key(record: dict[str, Any], candidates: Iterable[str], role: str) -> str:
    for key in candidates:
        if key in record:
            return key
    raise SystemExit(
        f"cannot find the {role} field in the informal-names dataset.\n"
        f"  tried: {list(candidates)}\n"
        f"  record keys: {sorted(record)}\n"
        f"Pass --informal-formal-key / --informal-gloss-key to name it explicitly."
    )


def load_informal_names(
    path: Path | str,
    formal_key: str | None = None,
    gloss_key: str | None = None,
) -> dict[str, str]:
    """Load `formal name -> informal name` from a JSONL file.

    Field names are detected from the first record rather than hard-coded, because this dataset's
    schema is not documented and a wrong guess would silently yield an empty mapping — which renders
    identically to having no dataset at all. Detection failure is a hard error naming the actual
    keys.

    Blank glosses are dropped rather than stored: an entry mapping to `""` is indistinguishable in
    the prompt from an absent one, and counting it as present would overstate coverage.
    """
    p = Path(path)
    if p.is_dir():                       # `snapshot_download` target rather than the file itself
        candidates = sorted(f for f in p.glob("*.jsonl") if f.is_file())
        if not candidates:
            raise SystemExit(f"no .jsonl file in {p}")
        p = candidates[0]

    mapping: dict[str, str] = {}
    n_lines = n_blank = n_dupe = 0
    fk, gk = formal_key, gloss_key

    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_lines += 1
            if fk is None:
                fk = _pick_key(record, FORMAL_KEYS, "formal-name")
                gk = _pick_key(record, INFORMAL_KEYS, "informal-name")
                log.info("informal-names schema: formal=%r informal=%r", fk, gk)
            raw_name = record.get(fk)
            gloss = record.get(gk)
            if not raw_name or not gloss or not str(gloss).strip():
                n_blank += 1
                continue
            name, gloss = _join_name(raw_name), str(gloss).strip()
            if name in mapping and mapping[name] != gloss:
                n_dupe += 1
            mapping[name] = gloss

    log.info("informal names: %d usable of %d records (%d blank, %d conflicting dupes) from %s",
             len(mapping), n_lines, n_blank, n_dupe, p)
    if not mapping:
        raise SystemExit(
            f"{p} yielded no usable informal names. The mapping being empty is indistinguishable "
            "from not passing --informal-names at all, so this is an error rather than a warning."
        )
    return mapping


def coverage(mapping: dict[str, str], names: Iterable[str]) -> dict[str, Any]:
    """How much of a premise corpus the mapping actually covers.

    Belongs in the run manifest. Coverage is the size of the documented bias: at 100% the prompts
    match REAL-Prover's shape exactly, and at 40% most premises still render a blank gloss to a
    model trained to expect one.
    """
    names = list(names)
    hit = sum(1 for n in names if mapping.get(n))
    return {
        "n_premises": len(names),
        "n_with_gloss": hit,
        "coverage": round(hit / len(names), 4) if names else 0.0,
        "n_mapping": len(mapping),
    }


def check_coverage(
    mapping: dict[str, str],
    names: Iterable[str],
    minimum: float = MIN_PLAUSIBLE_COVERAGE,
) -> dict[str, Any]:
    """Report coverage, and refuse to continue if it implies a schema mismatch.

    The `not mapping` check in `load_informal_names` cannot catch the failure that actually happened
    here: names stored as `["A", "b"]` produced a **fully populated** mapping whose every key was a
    stringified Python list, matching nothing. Non-empty, no error, and blank glosses throughout.

    Only a join against the real corpus detects that, so the check lives here: the first point at
    which both sides are available.
    """
    result = coverage(mapping, names)
    if result["n_premises"] and result["coverage"] < minimum:
        sample = sorted(mapping)[:3]
        raise SystemExit(
            f"informal-name coverage is {100 * result['coverage']:.2f}% "
            f"({result['n_with_gloss']}/{result['n_premises']} premises) from a mapping of "
            f"{result['n_mapping']} entries.\n"
            f"  example mapping keys: {sample}\n"
            "That is a schema mismatch, not a sparse dataset: a populated mapping matching almost "
            "nothing renders exactly like having no glosses at all, so it fails here rather than "
            "producing a run whose prompts are quietly wrong.\n"
            "Check --informal-formal-key, or pass --min-gloss-coverage 0 to proceed deliberately."
        )
    log.info("informal-name coverage: %d/%d premises (%.1f%%) from %d mapping entries",
             result["n_with_gloss"], result["n_premises"],
             100 * result["coverage"], result["n_mapping"])
    return result
