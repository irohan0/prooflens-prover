"""Guards that the repository is complete as a *clone*, not merely as a working directory.

Every other test in this suite passes against the files on disk. That is exactly the blind spot
that let `src/prooflens_prover/data/` go uncommitted for four commits: `.gitignore` contained
`data/`, which git matches at any depth, so the benchmark and premise-corpus loaders were silently
excluded from every commit. Locally everything imported. A fresh clone raised
`ModuleNotFoundError: No module named 'prooflens_prover.data'`, and only a cluster job found it.

The project's stated goal is that anyone can clone from GitHub and reproduce the results. These
tests are the cheapest possible check of that claim, and they run in milliseconds on every commit
rather than once per cluster submission.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directories whose contents are source and must always be committed in full.
SOURCE_DIRS = ["src", "tests", "scripts", "slurm", "configs"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout


@pytest.fixture(scope="module")
def in_git_repo() -> bool:
    try:
        _git("rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git repository, or git unavailable")
    return True


def test_no_source_file_is_gitignored(in_git_repo):
    """The exact check that would have caught the `data/` bug at commit time."""
    out = _git(
        "ls-files", "--others", "--ignored", "--exclude-standard", "--", *SOURCE_DIRS
    )
    offenders = [
        line for line in out.splitlines()
        if line.strip() and "__pycache__" not in line and not line.endswith((".pyc", ".pyo"))
    ]
    assert not offenders, (
        "these source files are excluded by .gitignore and would be MISSING from a fresh clone:\n  "
        + "\n  ".join(offenders)
        + "\n\nA bare directory pattern like `data/` matches at every depth. Anchor it with a "
          "leading slash (`/data/`) so it only applies at the repository root."
    )


def test_every_python_source_file_is_tracked(in_git_repo):
    """Belt and braces: an untracked source file is as broken as an ignored one."""
    tracked = set(_git("ls-files", "--", "src").splitlines())
    on_disk = {
        p.relative_to(REPO_ROOT).as_posix()
        for p in (REPO_ROOT / "src").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, (
        "Python sources present on disk but not tracked by git:\n  " + "\n  ".join(missing)
    )


def test_every_package_directory_has_an_init(in_git_repo):
    """A directory without `__init__.py` is not an importable subpackage."""
    pkg_root = REPO_ROOT / "src" / "prooflens_prover"
    missing = [
        d.relative_to(REPO_ROOT).as_posix()
        for d in pkg_root.rglob("*")
        if d.is_dir() and "__pycache__" not in d.parts and not (d / "__init__.py").exists()
    ]
    assert not missing, "package directories without __init__.py:\n  " + "\n  ".join(missing)


def test_large_generated_artefacts_stay_ignored(in_git_repo):
    """The fix must not overshoot: the premise corpus and indices are ~260 MB and regenerable."""
    for path in ("data/premises/mathlib_v4160.jsonl", "data/index/bm25_mathlib_v4160/index.npz"):
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", path],
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"{path} is NOT ignored — large generated data must never be committed"
        )


class TestGitignoreSyntax:
    """`.gitignore` has silently excluded needed files twice; both were syntax subtleties.

    1. `data/` (no leading slash) matches a directory named `data` at ANY depth, which excluded
       `src/prooflens_prover/data/` from every commit.
    2. A trailing `#` comment is NOT a comment — git only honours `#` at the start of a line — so
       `!results/exported/**/*.jsonl   # keep these` becomes a pattern ending in the comment text
       and matches nothing. That silently dropped the exported run records from the commit meant to
       publish them, and had already made the `tests/fixtures` whitelist dead on arrival.

    Neither failure produces an error anywhere. Both are cheap to detect.
    """

    def test_no_trailing_comments_on_pattern_lines(self):
        bad = []
        for n, raw in enumerate((REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines(),
                                start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # An unescaped `#` after the start of a pattern is literal text, not a comment.
            if "#" in line.replace(r"\#", ""):
                bad.append(f"  line {n}: {raw}")
        assert not bad, (
            "`.gitignore` patterns with a trailing `#` comment — git treats the comment as part of "
            "the pattern, so these rules match nothing:\n" + "\n".join(bad)
        )

    def test_negation_patterns_actually_reinclude(self, in_git_repo):
        """Every `!` rule must un-ignore at least one path, or it is dead weight pretending to work.

        Checked with `git check-ignore`, which is git's own resolver — reimplementing the matching
        rules here would just reproduce whatever misunderstanding caused the bug.
        """
        if not in_git_repo:
            pytest.skip("not a git repo")
        negations = [
            ln.strip()[1:] for ln in
            (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("!")
        ]
        for pattern in negations:
            # Resolve the pattern to concrete files, skipping rules whose target does not exist yet.
            base = pattern.split("*")[0].rstrip("/")
            root = REPO_ROOT / base
            if not root.exists():
                continue
            suffix = pattern.rsplit(".", 1)[-1]
            matches = list(root.rglob(f"*.{suffix}"))
            if not matches:
                continue
            rel = matches[0].relative_to(REPO_ROOT).as_posix()
            ignored = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", rel],
            ).returncode == 0
            assert not ignored, (
                f"`!{pattern}` does not re-include anything — {rel} is still ignored. "
                "A negation that matches nothing is worse than no rule: it reads as protection."
            )
