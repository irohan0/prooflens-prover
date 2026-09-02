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


class TestNonAsciiOutputIsSafe:
    """A script must not be able to die on its own output after the work is finished.

    Nearly every script here prints an em-dash or a Δ, and `prove_benchmark.py` prints Lean goals
    full of `α`/`⊢`. Python takes the stdout encoding from the locale, so under a POSIX or `cp1252`
    locale `print()` raises `UnicodeEncodeError` — after the results are on disk, with a non-zero
    exit status that stops anything chained with `&&`. This project has already lost one debugging
    cycle to an encoding traceback hidden in a separate stderr stream.

    Static, because the alternative — re-running each script under a non-UTF-8 locale — needs the
    cluster's data. Static is enough: the failure mode is a *missing* call, not a subtle one.
    """

    @staticmethod
    def _prints(text: str) -> list[str]:
        import re

        return re.findall(r"print\((?:[^()]|\([^()]*\))*\)", text, flags=re.S)

    @pytest.mark.parametrize(
        "script",
        sorted(p.name for p in (REPO_ROOT / "scripts").glob("*.py")),
    )
    def test_script_printing_non_ascii_forces_utf8(self, script):
        text = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
        offenders = [s for s in self._prints(text) if any(ord(c) > 127 for c in s)]
        if not offenders:
            return
        assert "ensure_utf8_output()" in text, (
            f"scripts/{script} prints non-ASCII ({offenders[0][:60]!r}...) but never calls "
            "ensure_utf8_output(); it will raise UnicodeEncodeError under a non-UTF-8 locale, "
            "after doing all of its work"
        )

    def test_the_helper_is_idempotent_and_survives_a_stream_without_reconfigure(self):
        from prooflens_prover.utils.logging import ensure_utf8_output

        # Called twice, and under pytest's capture objects (which have no `reconfigure`). Neither
        # may raise: a hardening helper that itself throws is worse than the problem.
        ensure_utf8_output()
        ensure_utf8_output()


class TestScriptsRunFromAFreshClone:
    """`python scripts/<x>.py` must work with no install and no exported PYTHONPATH.

    The sbatch files export `PYTHONPATH="$REPO/src"`, so cluster *jobs* were always fine. The
    analysis scripts, though, get run by hand on a login node — and there `build_table1.py` died
    with `ModuleNotFoundError: No module named 'prooflens_prover'` in the middle of collecting
    results. Second time this class of friction cost a round trip; the first was an
    `export_results.py` invoked from the wrong directory.

    The README's claim is that a clone reproduces the results. A clone that needs an undocumented
    environment variable does not.
    """

    SCRIPTS = sorted(p.name for p in (REPO_ROOT / "scripts").glob("*.py"))

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_package_import_is_preceded_by_a_path_bootstrap(self, script):
        text = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
        if "from prooflens_prover" not in text:
            return                              # standalone by design, e.g. export_results.py
        assert "sys.path.insert" in text, (
            f"scripts/{script} imports prooflens_prover but never puts src/ on sys.path; it only "
            "runs where PYTHONPATH happens to be set"
        )
        assert text.index("sys.path.insert") < text.index("from prooflens_prover"), (
            f"scripts/{script} inserts src/ on sys.path *after* importing the package — the import "
            "has already failed by then"
        )

    def test_the_bootstrap_actually_works(self, tmp_path):
        """One end-to-end proof the mechanism is real, not just present.

        `build_table1.py` because it is the script that failed and the only analysis script with no
        torch import, so this stays fast. PYTHONPATH is cleared and the cwd is elsewhere.
        """
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        p = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "build_table1.py"), "--help"],
            capture_output=True, text=True, env=env, cwd=tmp_path,
        )
        assert p.returncode == 0, f"stdout={p.stdout}\nstderr={p.stderr}"
        assert "ModuleNotFoundError" not in p.stderr


class TestPublishAllowlist:
    """`publish.sh` mirrors a filtered tree to a private GitHub repo a supervisor reads.

    It is an allowlist so that a working-notes file added later defaults to *not* published. The
    inverse default has already failed on this project: `.gitignore` silently excluded
    `src/prooflens_prover/data/` for four commits, and the working copy looked correct throughout.

    The invariant worth testing statically is that the allowlist cannot sweep in something the
    forbidden list is meant to stop. `publish.sh` re-checks at run time and refuses to push, but by
    then the tree is built and the failure is confusing.

    Skipped where `publish.sh` is absent. The script belongs to the private tree and is not itself
    allowlisted, so a clone of the *published* repo has these tests and not their subject — and a
    fresh clone must have a green suite, or "the tests pass" stops meaning anything to a reader.
    """

    @staticmethod
    def _array(name: str) -> list[str]:
        import re

        script = REPO_ROOT / "publish.sh"
        if not script.exists():
            pytest.skip("publish.sh is not part of the published tree")
        src = script.read_text(encoding="utf-8")
        body = re.search(rf"^{name}=\((.*?)^\)", src, flags=re.S | re.M)
        assert body, f"{name}=( ... ) not found in publish.sh"
        return [w.strip().strip("'\"") for w in body.group(1).split() if not w.startswith("#")]

    def test_no_allowlisted_path_contains_a_forbidden_one(self):
        allowed = self._array("PATHS")
        forbidden = self._array("FORBIDDEN")
        clashes = [
            (a, f) for a in allowed for f in forbidden
            if f == a or f.startswith(a.rstrip("/") + "/")
        ]
        assert not clashes, (
            f"publish.sh would copy a forbidden path: {clashes}. Narrow the allowlisted entry — "
            "`results/exported` rather than `results`, for example."
        )

    def test_every_allowlisted_path_exists(self):
        missing = [p for p in self._array("PATHS") if not (REPO_ROOT / p).exists()]
        assert not missing, f"publish.sh allowlists paths that do not exist: {missing}"

    def test_working_notes_are_forbidden(self):
        """Named explicitly, because these are the files whose leak would matter."""
        forbidden = set(self._array("FORBIDDEN"))
        for f in ("CLAUDE.md", ".claude", "LEARNINGS.md", "MEETING_SCRIPT.md", "PLAN.md"):
            assert f in forbidden, f"{f} is not in publish.sh's FORBIDDEN list"

    def test_root_markdown_other_than_readme_is_never_allowlisted(self):
        allowed = self._array("PATHS")
        stray = [p for p in allowed if p.endswith(".md") and p != "README.md"]
        assert not stray, (
            f"publish.sh allowlists a root markdown file other than README.md: {stray}"
        )
