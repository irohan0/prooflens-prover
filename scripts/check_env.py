#!/usr/bin/env python
"""Import everything this project imports, and report every failure at once.

    python scripts/check_env.py
    python scripts/check_env.py --protected torch,vllm,transformers,numpy

## Why this exists

Setting up the vLLM environment cost four cluster round-trips, each one a two-minute job that failed
on a single missing transitive dependency: first `scipy`, then `datasets`. The loop had no bound
because I was reading one traceback at a time.

The reason those imports were invisible is that this package defers them. `torch`, `pylate`,
`lean_interact` and `vllm` are all imported *inside functions*, deliberately — it keeps the hermetic
test suite runnable without a GPU, a Lean toolchain, or a 10 GB install. The cost is that
`import prooflens_prover` proves almost nothing about whether a run will start.

So this walks the source with `ast`, finds every third-party import at **any** nesting depth, and
imports them all. It is generated from the code at run time rather than from a hand-written list,
so it cannot drift out of date the way a list would.

## What it does not do

It does not import *transitive* dependencies by name — it cannot know them. It does not need to:
importing `pylate.models` is what pulled in `scipy` and `datasets`, and that import either works or
names what is missing. One run, every failure.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prooflens_prover.utils.logging import ensure_utf8_output  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

#: This project deliberately spans **two** environments, because vLLM requires
#: `transformers>=5.5.3` and pylate requires `transformers<=5.3.0` — disjoint, so no single
#: environment can host both:
#:
#:   * the *retrieval* venv has pylate and serves `scripts/retrieval_server.py`;
#:   * the *prover* venv has vLLM and spawns that server as a child process.
#:
#: So "missing" is environment-dependent, and a package absent from the right environment is correct
#: rather than broken. Each side names what it requires with `--require`; anything else in this map
#: is reported as `skipped`. Without `--require`, a check would pass on an environment missing the
#: one thing it exists to provide.
OPTIONAL_IN: dict[str, str] = {
    "vllm": "the prover environment only; pass --require vllm there",
    "pylate": "the retrieval environment only; pass --require pylate.models there",
    "sentence_transformers": "pulled in by pylate, so retrieval-environment only",
    "bm25s": "the bm25 arm only, and it lives with retrieval",
}


def third_party_imports(root: Path) -> dict[str, set[str]]:
    """Every non-stdlib, non-first-party import target in `root`, at any nesting depth.

    `from X import Y` yields **both** `X` and `X.Y`. That distinction is the point: the failure
    that cost four round-trips was `from pylate import models`, where `import pylate` succeeds — it
    does almost nothing — and `import pylate.models` is what pulls in scipy and datasets. Recording
    only the parent reproduces the blind spot this script exists to close.

    `X.Y` where `Y` is an attribute rather than a submodule (`from numpy import ndarray`) is
    resolved by `hasattr` at import time, not here, so there are no false failures.
    """
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}

    def record(target: str, path: Path) -> None:
        top = target.split(".")[0]
        if top in stdlib or top == "prooflens_prover" or top == "__future__":
            return
        try:
            shown = str(path.relative_to(REPO_ROOT))
        except ValueError:               # `--src` outside the repo, or a test fixture in tmp_path
            shown = str(path)
        found.setdefault(target, set()).add(shown.replace("\\", "/"))

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:                        # a file that cannot parse is its own bug
            print(f"  !! {path}: {exc}")
            continue
        for node in ast.walk(tree):                      # ast.walk descends into function bodies
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record(alias.name, path)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                record(node.module, path)
                for alias in node.names:
                    if alias.name != "*":
                        record(f"{node.module}.{alias.name}", path)
    return found


def try_import(target: str) -> str | None:
    """Import `target`, returning None on success or a reason string on failure.

    A dotted target may name a submodule or an attribute of its parent. Both are legitimate results
    of `from X import Y`, so `X.Y` counts as satisfied if `Y` is an attribute of `X` — otherwise
    `from numpy import ndarray` would report a missing `numpy.ndarray` module.
    """
    try:
        importlib.import_module(target)
        return None
    except ModuleNotFoundError as exc:
        if "." not in target:
            return f"{type(exc).__name__}: {exc}"
        parent, _, leaf = target.rpartition(".")
        try:
            if hasattr(importlib.import_module(parent), leaf):
                return None                              # an attribute, not a submodule
            return f"{parent} imports, but has no attribute {leaf!r}"
        except Exception as inner:                       # noqa: BLE001 — reporting, not handling
            # Report the REAL failure, not the outer "no module named X.Y". `from vllm import LLM`
            # resolves `LLM` through a lazy module `__getattr__`, so when vLLM is broken by an
            # incompatible transformers the useful error is raised here — and an earlier version of
            # this function discarded it, reporting a nonexistent `vllm.LLM` module instead. That
            # sent me looking for the wrong problem.
            return f"resolving {leaf!r} on {parent}: {type(inner).__name__}: {inner}"
    except Exception as exc:                             # noqa: BLE001 — reporting, not handling
        return f"{type(exc).__name__}: {exc}"


def main() -> int:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=REPO_ROOT / "src")
    ap.add_argument("--protected", default="",
                    help="comma-separated packages whose installed versions to print. A dependency "
                         "install that moved one of these has broken something silently.")
    ap.add_argument("--require", default="",
                    help="comma-separated import targets that must succeed even if OPTIONAL_IN "
                         "lists them, e.g. --require vllm on the LLM environment")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))

    required = {t for t in args.require.split(",") if t}
    targets = third_party_imports(args.src)
    if not targets:
        print(f"no third-party imports found under {args.src} — is the path right?")
        return 1

    print(f"=== {len(targets)} third-party import targets found in {args.src} ===")
    failed: list[tuple[str, str, set[str]]] = []
    skipped: list[str] = []
    for target in sorted(targets):
        top = target.split(".")[0]
        reason = try_import(target)
        if reason is None:
            print(f"  {target:<26} OK")
        elif top in OPTIONAL_IN and target not in required and top not in required:
            print(f"  {target:<26} skipped ({OPTIONAL_IN[top]})")
            skipped.append(target)
        else:
            print(f"  {target:<26} FAILED  {reason}")
            failed.append((target, reason, targets[target]))

    if args.protected:
        print("\n=== installed versions of protected packages ===")
        for name in args.protected.split(","):
            name = name.strip()
            if not name:
                continue
            try:
                print(f"  {name:<26} {importlib.metadata.version(name)}")
            except importlib.metadata.PackageNotFoundError:
                print(f"  {name:<26} NOT INSTALLED")

    print()
    if failed:
        print(f"BROKEN: {len(failed)} import target(s) failed.\n")
        for target, reason, where in failed:
            print(f"  {target}\n      {reason}\n      needed by: {', '.join(sorted(where))}")
        print(
            "\nThese are the imports the code performs, so every one of them is on the path to a "
            "run.\nFix them together — each is otherwise one cluster job and one traceback."
        )
        return 1

    print(f"ENV OK — {len(targets) - len(skipped)} imports succeeded"
          + (f", {len(skipped)} skipped as optional" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
