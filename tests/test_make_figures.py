"""The dissertation figures: registry invariants and what must not reach a plot.

Rendering is not tested here -- a plot that draws is not a plot that is *right*, and the numbers it
draws are already covered by the tests on `budget_matched.py`, `passk_union.py` and
`passk_profile.py`, which are the same code paths `figures_data.py` calls.

What is tested is what breaks silently: a figure count that drifts from the seventeen the report
references, a duplicate name, or LaTeX markup inside a string matplotlib will render character for
character. `16{,}384` looked correct in the source and shipped as literal braces in two bar labels.

Hermetic: no GPU, no Lean, no model, no results tree.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("matplotlib", reason="figures need matplotlib")

from make_figures import FIGURES  # noqa: E402

#: The report references figures by number, so the count is part of the contract with the text.
EXPECTED_FIGURES = 17

#: Matplotlib calls whose string arguments are drawn onto the canvas.
TEXT_CALLS = frozenset({
    "annotate", "text", "set_title", "set_xlabel", "set_ylabel", "set_xticks", "set_yticks",
    "suptitle", "supxlabel", "supylabel", "set_label",
})


def _literal(node) -> str | None:
    """The string a node renders to, with `{...}` standing in for interpolated values.

    An f-string is a `JoinedStr` of alternating constants and expressions, so walking for bare
    `Constant` nodes sees `f"mean $+{x:.2f}$"` as the two fragments `'mean $+'` and `'$'` -- each
    with an odd number of `$`. Reconstructing first is what makes a balance check meaningful.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append("{}")
        return "".join(out)
    return None


def _drawn_strings():
    """Every string that reaches a matplotlib text call, with its line number."""
    tree = ast.parse((REPO / "scripts" / "make_figures.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in TEXT_CALLS:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.JoinedStr):
                yield node.func.attr, sub.lineno, _literal(sub)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                # Skip constants that belong to an enclosing f-string; those are covered above.
                if not any(sub in ast.walk(j) for j in ast.walk(node)
                           if isinstance(j, ast.JoinedStr)):
                    yield node.func.attr, sub.lineno, sub.value


# --- the registry ---------------------------------------------------------------------------------

def test_there_are_exactly_the_figures_the_report_references():
    assert len(FIGURES) == EXPECTED_FIGURES


def test_names_are_unique_and_numbered_in_order():
    names = list(FIGURES)
    assert len(set(names)) == len(names)
    numbers = [int(re.match(r"fig(\d+)_", n).group(1)) for n in names]
    assert numbers == list(range(1, len(names) + 1)), (
        "figure numbers must be contiguous and in dissertation order: the number IS the "
        f"report position. Got {numbers}"
    )


def test_every_figure_names_a_dissertation_section():
    for name, meta in FIGURES.items():
        assert re.fullmatch(r"\d+(\.\d+)?(-\d+(\.\d+)?)?", meta["section"]), name


def test_every_figure_says_what_it_shows():
    for name, meta in FIGURES.items():
        assert len(meta["what"]) > 25, name


# --- what must not reach a plot -----------------------------------------------------------------

def test_latex_thousands_separator_never_reaches_a_plot():
    """`{,}` belongs in prose typeset by LaTeX and never in an annotation.

    matplotlib is not LaTeX: it renders `16{,}384` character for character. Two figures shipped
    bar labels reading `16{,}384` before this was checked.
    """
    offenders = [(c, ln, t) for c, ln, t in _drawn_strings() if "{,}" in t and "$" not in t]
    assert not offenders, (
        "LaTeX thousands separator drawn into a plot: "
        + "; ".join(f"line {ln} in .{c}(): {t!r}" for c, ln, t in offenders)
    )


def test_no_drawn_string_uses_a_latex_only_command():
    r"""`\texttt`, `\emph`, `\ref` render literally in matplotlib; only mathtext is interpreted.

    `\%` is deliberately absent from this list: matplotlib's mathtext does support it, and an
    earlier version of this test flagged a correct label because of that.
    """
    bad = ("\\texttt", "\\emph", "\\ref", "\\citep", "\\textbf", "\\begin")
    offenders = [(c, ln, t) for c, ln, t in _drawn_strings() if any(b in t for b in bad)]
    assert not offenders, (
        "LaTeX-only command drawn into a plot: "
        + "; ".join(f"line {ln} in .{c}(): {t!r}" for c, ln, t in offenders)
    )


def test_mathtext_dollars_are_balanced():
    """An odd `$` renders the rest of the label as maths, which is silent and ugly."""
    offenders = [(c, ln, t) for c, ln, t in _drawn_strings() if t.count("$") % 2]
    assert not offenders, (
        "unbalanced mathtext delimiters: "
        + "; ".join(f"line {ln} in .{c}(): {t!r}" for c, ln, t in offenders)
    )


# --- the transcribed predecessor numbers --------------------------------------------------------

def test_predecessor_constants_match_their_source_document():
    """`figures_data` transcribes `prooflens_results.md`; a typo there is invisible in a plot.

    Checked by finding each value in the source document rather than by restating it here, so this
    test fails if either side drifts.

    Skipped when the source is absent: it belongs to the predecessor project and is not part of
    this repository, so a fresh clone has the constants but not the document to check them against.
    """
    import figures_data as D
    doc = REPO / "prooflens_results.md"
    if not doc.exists():
        pytest.skip("prooflens_results.md is the predecessor's file and is not in this repo")
    src = doc.read_text(encoding="utf-8")
    for system, ((r_m, _), (n_m, _)) in D.PL_CROSSOVER.items():
        for value in (r_m, n_m):
            assert f"{value:.2f}" in src, f"{system}: {value} not in prooflens_results.md"
    for _seed, delta, _p in D.PL_PER_SEED:
        assert f"+{delta:.2f}" in src, f"per-seed delta {delta} not in prooflens_results.md"
    for split, row in D.PL_DOWNSTREAM.items():
        for cond, value in row.items():
            assert f"{value:.2f}" in src, f"{split}/{cond}: {value} not in prooflens_results.md"


def test_predecessor_crossover_has_the_direction_the_thesis_claims():
    """SV must fall random->novel and LI-IDF must not. If this flips, the figure is mislabelled."""
    import figures_data as D
    (sv_r, _), (sv_n, _) = D.PL_CROSSOVER["FT-SV (control)"]
    (li_r, _), (li_n, _) = D.PL_CROSSOVER["FT-LI (IDF)"]
    assert sv_n < sv_r, "single-vector should degrade on novel premises"
    assert li_n > li_r, "late interaction with IDF weighting should not degrade"


def test_every_per_seed_delta_favours_late_interaction():
    """The claim is 5/5 seeds. A sign error here would invert the most-cited figure in Part I."""
    import figures_data as D
    assert all(d > 0 for _, d, _ in D.PL_PER_SEED)
    assert all(p < 0.05 for _, _, p in D.PL_PER_SEED)
