#!/usr/bin/env python3
"""The seventeen dissertation figures, regenerated from the records.

    python scripts/make_figures.py                       # all of them, into figures/
    python scripts/make_figures.py --only fig13 fig17    # just these
    python scripts/make_figures.py --list                # names and the section each belongs to

PNG at 300 dpi is the only artefact: high enough for print inclusion, and one format means one
thing to keep in step with the text. Captions live beside each reference in the write-up.

Numbering follows the dissertation's own order, so `figNN` is also its position in the report:

    01-03  the predecessor retrieval study (transcribed from prooflens_results.md)
    04     the architecture: the fixed prover loop and the four retrievers
    05-10  method and Tier 1
    11-15  scaling to pass@8
    16-17  fusion and the budget-matched control

Figure 04 is the one schematic: it draws no data, but every constant on it is imported from the
module it documents rather than typed, so it cannot drift from the code the way a hand-drawn
diagram does.

Nothing here invents a number. Everything for this project's own results comes from
`results/tables/*.json` (produced by committed analysis scripts) or from the exported run records
through `eval/draws.py` -- the same path `passk_union.py` and `budget_matched.py` use, so a figure
and the sentence beside it cannot drift apart. Predecessor values are transcribed constants in
`figures_data.py`, each tagged with the section of `prooflens_results.md` it came from.

Figures are sized for inclusion at `width=\\textwidth` with **no further scaling**; scaling a
figure down leaves its labels smaller than the body text, which is the usual way a thesis figure
becomes unreadable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from matplotlib import pyplot as plt  # noqa: E402

import figures_data as D  # noqa: E402
from figures_style import (  # noqa: E402
    ARM,
    ARM_LABEL,
    HALF_WIDTH,
    STATUS,
    STATUS_LABEL,
    TEXT_WIDTH,
    bar_labels,
    pct_axis,
    save,
    use_style,
)

BENCH_LABEL = {"proofnet_test": "ProofNet-test ($n=186$)", "fate_m": "FATE-M ($n=141$)"}
BENCH_SHORT = {"proofnet_test": "ProofNet-test", "fate_m": "FATE-M"}

#: Predecessor arms keep their own colours: they are retrievers evaluated offline, not the prover
#: arms, and reusing the prover palette would imply they are the same objects.
PL = {"FT-SV (control)": "#0072B2", "FT-LI (no weighting)": "#E69F00", "FT-LI (IDF)": "#D55E00",
      "BM25": "#999999", "none": "#BBBBBB"}

FIGURES: dict[str, dict] = {}


def figure(name: str, section: str, what: str):
    def deco(fn):
        FIGURES[name] = {"fn": fn, "section": section, "what": what}
        return fn
    return deco


# =================================================================================================
# Part I -- the predecessor retrieval study
# =================================================================================================

@figure("fig01_prooflens_crossover", "3.3",
        "The matched control: SV degrades random->novel, LI does not; per-seed deltas")
def fig01(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    xs = [0, 1]
    for system, ((r_m, r_s), (n_m, n_s)) in D.PL_CROSSOVER.items():
        ax.errorbar(xs, [r_m, n_m], yerr=[r_s, n_s], marker="o", capsize=3,
                    color=PL[system], label=system,
                    lw=2.0 if "control" in system else 1.6)
        ax.annotate(f"{n_m:.2f}", (1.06, n_m), fontsize=8, color=PL[system], va="center")
    ax.annotate("$-5.78$\n($-17.9\\%$)", (0.5, 29.6), ha="center", fontsize=8.5,
                color=PL["FT-SV (control)"])
    ax.annotate("$+0.95$", (0.5, 28.72), ha="center", fontsize=8.5, color=PL["FT-LI (IDF)"])
    ax.set_xticks(xs, ["random\n($n=2{,}811$)", "novel premises\n($n=4{,}357$)"])
    ax.set_xlim(-0.28, 1.30)
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title("held-out split", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    seeds = [s for s, _, _ in D.PL_PER_SEED]
    deltas = [d for _, d, _ in D.PL_PER_SEED]
    ax.bar(range(len(seeds)), deltas, 0.5, color=PL["FT-LI (IDF)"])
    for i, (d, (_, _, p)) in enumerate(zip(deltas, D.PL_PER_SEED, strict=True)):
        ax.annotate(f"{d:+.2f}", (i, d + 0.08), ha="center", fontsize=8)
        ax.annotate(f"$p={p:.4f}$", (i, 0.12), ha="center", fontsize=6.8, rotation=90,
                    color="white")
    ax.axhline(0, color="#444444", lw=1.0)
    mean = sum(deltas) / len(deltas)
    ax.axhline(mean, color="#B22222", ls="--", lw=1.2)
    ax.annotate(f"mean $+{mean:.2f}$", (len(seeds) - 0.5, mean + 0.12), ha="right", fontsize=8,
                color="#B22222")
    ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_xlabel("training seed")
    ax.set_ylabel("R@10, LI $-$ SV (pts)")
    ax.set_title("novel premises, paired per seed", fontsize=9)
    ax.set_ylim(0, 4.1)
    ax.grid(axis="x", visible=False)
    return save(fig, out, "fig01_prooflens_crossover")


@figure("fig02_prooflens_mechanism", "3.5-3.6",
        "Where LI's edge comes from (lexical, not structural) and the leakage that hid it")
def fig02(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8))

    ax = axes[0]
    x = np.arange(len(D.PL_STRATIFIED))
    w = 0.34
    li = [r[2] for r in D.PL_STRATIFIED]
    sv = [r[3] for r in D.PL_STRATIFIED]
    b1 = ax.bar(x - w / 2, li, w, color=PL["FT-LI (IDF)"], label="FT-LI")
    b2 = ax.bar(x + w / 2, sv, w, color=PL["FT-SV (control)"], label="FT-SV")
    bar_labels(ax, b1, "{:.1f}", dy=0.5)
    bar_labels(ax, b2, "{:.1f}", dy=0.5)
    for i, r in enumerate(D.PL_STRATIFIED):
        sig = "significant" if r[5] > 0 else "tied"
        ax.annotate(f"$\\Delta={r[4]:+.1f}$\n[{r[5]:+.1f}, {r[6]:+.1f}]\n{sig}",
                    (i, 36.5), ha="center", va="top", fontsize=7.5,
                    color="#B22222" if r[5] > 0 else "#555555")
    ax.set_xticks(x, [f"{r[0]}\n{r[1]:,} ({100 * r[1] / 4357:.0f}%)" for r in D.PL_STRATIFIED])
    ax.set_ylabel("Recall@10 on novel (%)")
    ax.set_ylim(0, 40)
    ax.set_title("the advantage is lexical", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    labels = [r[0] for r in D.PL_LEAKAGE]
    vals = [r[2] for r in D.PL_LEAKAGE]
    bars = ax.bar([0, 1], vals, 0.5, color=["#B22222", "#009E73"])
    bar_labels(ax, bars, "{:.2f}", dy=1.4)
    lo, hi = D.PL_LEAKAGE[1][3], D.PL_LEAKAGE[1][4]
    ax.errorbar([1], [vals[1]], yerr=[[vals[1] - lo], [hi - vals[1]]], fmt="none",
                ecolor="#009E73", elinewidth=1.5, capsize=4)
    ax.axhline(D.PL_LEAKAGE_PUBLISHED, color="#444444", ls="--", lw=1.2)
    ax.annotate(f"published {D.PL_LEAKAGE_PUBLISHED}%", (1.42, D.PL_LEAKAGE_PUBLISHED + 1.5),
                ha="right", fontsize=8)
    ax.set_xticks([0, 1], [f"{lab}\n$n={r[1]:,}$" for lab, r in zip(labels, D.PL_LEAKAGE,
                                                                    strict=True)])
    ax.set_ylabel("Recall@10 (%)")
    ax.set_ylim(0, 78)
    ax.set_title("public checkpoint, novel split", fontsize=9)
    ax.grid(axis="x", visible=False)
    return save(fig, out, "fig02_prooflens_mechanism")


@figure("fig03_prooflens_downstream", "3.8",
        "Retrieval helps a fixed generator, but LI and SV tie -- the result this project tests")
def fig03(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), sharey=True)
    order = ["none", "BM25", "FT-LI", "FT-SV"]
    colours = [PL["none"], PL["BM25"], PL["FT-LI (IDF)"], PL["FT-SV (control)"]]
    titles = {"novel_premises": "novel premises ($n=4{,}357$)", "random": "random ($n=2{,}811$)"}
    for ax, split in zip(axes, ("novel_premises", "random"), strict=True):
        vals = [D.PL_DOWNSTREAM[split][k] for k in order]
        bars = ax.bar(np.arange(4), vals, 0.55, color=colours)
        bar_labels(ax, bars, "{:.2f}", dy=0.7)
        ax.set_xticks(np.arange(4), ["no\npremises", "BM25", "FT-LI", "FT-SV"])
        ax.set_title(titles[split], fontsize=9)
        ax.grid(axis="x", visible=False)
        # The bracket that carries the finding: LI and SV are indistinguishable.
        ax.plot([2, 3], [max(vals[2:]) + 4.5] * 2, color="#B22222", lw=1.2)
        ax.annotate("tie (ns)", (2.5, max(vals[2:]) + 5.2), ha="center", fontsize=8,
                    color="#B22222")
        ax.plot([0, 3], [max(vals) + 10.5] * 2, color="#444444", lw=1.0)
        ax.annotate("$p=0.0001$ vs floor", (1.5, max(vals) + 11.2), ha="center", fontsize=8)
    axes[0].set_ylabel("premise_name@8 (%)")
    axes[0].set_ylim(0, 56)
    return save(fig, out, "fig03_prooflens_downstream")


# =================================================================================================
# Part II -- method and Tier 1
# =================================================================================================

def _box(ax, x, y, w, h, text, *, fc="white", ec="#444444", lw=1.0, ls="solid", fs=7.2,
         weight="normal", tc="black", pad=0.006):
    """One rounded box with centred text, in axes coordinates (the axes spans the whole figure)."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.008",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2, clip_on=False))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight,
            color=tc, zorder=3, linespacing=1.4)


def _arrow(ax, xy_from, xy_to, *, color="#555555", lw=1.1, head=True):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=lw,
                                shrinkA=1.0, shrinkB=1.0), zorder=4)


@figure("fig04_architecture", "4.1",
        "System architecture: the fixed prover loop, and the four interchangeable retrievers")
def fig04(root, out):
    """Drawn rather than measured -- but every constant on it is imported from the module it
    documents, so a change in code is a change in the figure. The claim the picture makes is the
    dashed box: exactly one component differs between arms, and panel (b) is the complete list of
    what that component is ever set to.
    """
    from prooflens_prover.prover.search import SearchConfig
    from prooflens_prover.prover.vllm_policy import SamplingConfig
    from prooflens_prover.retrieval.base import DEFAULT_TOP_K, PROMPT_PREMISE_LIMIT
    from prooflens_prover.retrieval.dense import (
        LI_DOCUMENT_LENGTH,
        LI_QUERY_LENGTH,
        SV_MAX_SEQ_LENGTH,
    )
    from prooflens_prover.retrieval.fusion import DEFAULT_FETCH_K, RRF_K

    sc, sm = SearchConfig(), SamplingConfig()
    n_prem, n_cand = 276070, 50000

    fig = plt.figure(figsize=(TEXT_WIDTH, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    # --- (a) the loop that never changes ---------------------------------------------------------
    ax.text(0.0, 0.972, "(a)   the prover, held identical across every arm",
            fontsize=9.5, weight="bold", va="bottom")

    x, w = 0.25, 0.50
    rows = [
        (0.895, 0.055, "Lean 4 proof state, pretty-printed\n"
                       "(the retriever's query and the prompt's goal are one string)", "white"),
        (0.798, 0.064, "RETRIEVER  —  the only component that varies\n"
                       f"top-{DEFAULT_TOP_K} requested, the first "
                       f"{PROMPT_PREMISE_LIMIT} formatted into the prompt", "#FDF0E6"),
        (0.706, 0.052, "REAL-Prover prompt, transcribed verbatim\n"
                       "premise block (worst-ranked first), then the goal", "white"),
        (0.600, 0.070, "REAL-Prover-v1  ·  7B  ·  frozen  ·  vLLM\n"
                       f"$T = {sm.temperature}$,  top-$p$ = {sm.top_p},  "
                       f"{sm.max_tokens} max tokens,  logprobs on\n"
                       "$n = 16$ (Tier 1) or $32$ (pass@8) samples per state", "#EAF3F8"),
        (0.508, 0.052, "clean · dedupe · score\n"
                       r"$\mathrm{score} = \sum \log p \;/\; \mathrm{depth}^{0.5}$", "white"),
        (0.410, 0.058, "Lean 4 REPL (LeanInteract)  +  admissibility guard\n"
                       "no sorry / admit / native_decide, and no goals may remain", "white"),
    ]
    for y, h, txt, fc in rows:
        varies = txt.startswith("RETRIEVER")
        _box(ax, x, y, w, h, txt, fc=fc, ec=ARM["li"] if varies else "#444444",
             lw=1.7 if varies else 1.0, ls=(0, (5, 3)) if varies else "solid",
             fs=7.2, weight="bold" if varies else "normal")
    for i in range(len(rows) - 1):
        _arrow(ax, (0.5, rows[i][0]), (0.5, rows[i + 1][0] + rows[i + 1][1]))

    y_first = rows[0][0] + rows[0][1] / 2
    y_last = rows[-1][0] + rows[-1][1] / 2
    lx = 0.163
    _arrow(ax, (x, y_last), (lx, y_last), head=False)
    _arrow(ax, (lx, y_last), (lx, y_first), head=False)
    _arrow(ax, (lx, y_first), (x, y_first))
    ax.text(0.075, 0.700, "goals remain:\npush to the\nbest-first frontier",
            ha="center", va="center", fontsize=7.0, color="#555555", linespacing=1.5)
    ax.text(0.075, 0.575, f"$\\leq$ {sc.max_expansions} expansions\n"
                          f"$\\leq$ depth {sc.max_depth}\n"
                          f"$\\leq$ {sc.wall_clock_s:.0f} s wall clock",
            ha="center", va="center", fontsize=7.0, color="#555555", linespacing=1.5)

    _arrow(ax, (x + w, y_last), (0.805, y_last), color="#009E73")
    _box(ax, 0.815, y_last - 0.029, 0.170, 0.058, "no goals left:\nPROVED",
         fc="#EAF7F2", ec="#009E73", lw=1.3, fs=7.2, weight="bold", tc="#00654A")

    ax.plot([0.0, 1.0], [0.372, 0.372], color="#CCCCCC", lw=0.8, clip_on=False)

    # --- (b) the complete list of what that component is set to -----------------------------------
    ax.text(0.0, 0.340, "(b)   the four settings of that component, and what each computes",
            fontsize=9.5, weight="bold", va="bottom")

    cols = [
        ("none", "no-retrieval control", [
            ("no retriever object\nis called at all:\n"
             r"$k = 0$ short-circuits" "\nthe policy", "#F6F6F6"),
            ("premise block: empty\n(the goal is unchanged)", "#F6F6F6"),
        ], "no index  ·  0 ms\nexact by vacuity"),
        ("sv", "single-vector", [
            (f"state $\\rightarrow$ one 768-d\nvector\n"
             f"({SV_MAX_SEQ_LENGTH} tokens, pooled)", "white"),
            (f"cosine against all\n{n_prem:,} premises\n"
             "$\\bf{exact}$: one matrix-\nvector product", "#EAF3F8"),
        ], "943 MB  ·  42 ms\nranks 100% of the corpus"),
        ("li", "late interaction", [
            (f"state $\\rightarrow$ {LI_QUERY_LENGTH} $\\times$ 128-d\ntoken vectors\n"
             f"(premises at {LI_DOCUMENT_LENGTH} tok)", "white"),
            (f"stage 1: mean-pooled\nshortlist, top {n_cand:,}\n"
             f"({100 * n_cand / n_prem:.1f}% of the corpus)", "#FDF0E6"),
            ("stage 2: exact MaxSim\non that shortlist\n"
             r"$\sum_i \max_j\, q_i \cdot d_j$", "#FDF0E6"),
        ], "5.5 GB  ·  930 ms\nrecall 0.979 vs exact"),
        ("fusion", "both, merged", [
            ("sv (CPU) and li (GPU),\neach asked for\n"
             f"fetch_k = {DEFAULT_FETCH_K}", "white"),
            (f"rrf: $\\sum_r 1/({RRF_K} + \\mathrm{{rank}}_r)$\n"
             "or interleave:\n3 from each, in turn", "#E7F5F0"),
        ], "6.4 GB  ·  both latencies\n(they run in sequence)"),
    ]
    gap = 0.020
    cw = (1.0 - gap * (len(cols) - 1)) / len(cols)
    # Every column spans the same vertical extent, so the footers line up and the eye reads the
    # extra box in `li` as the one structural difference rather than as a longer column.
    span, bgap, y_top, y_foot = 0.238, 0.026, 0.268, 0.018
    for i, (arm, title, boxes, foot) in enumerate(cols):
        cx0 = i * (cw + gap)
        mid = cx0 + cw / 2
        bh = (span - (len(boxes) - 1) * bgap) / len(boxes)
        ax.text(mid, 0.302, arm, ha="center", va="bottom", fontsize=9.0, weight="bold",
                color=ARM[arm], family="monospace")
        ax.text(mid, 0.283, title, ha="center", va="bottom", fontsize=7.2, color=ARM[arm])
        y = y_top
        for j, (txt, fc) in enumerate(boxes):
            _box(ax, cx0, y - bh, cw, bh, txt, fc=fc, ec=ARM[arm], lw=1.0, fs=6.6)
            if j < len(boxes) - 1:
                _arrow(ax, (mid, y - bh), (mid, y - bh - bgap), color=ARM[arm], lw=1.0)
            y -= bh + bgap
        ax.text(mid, y_foot, foot, ha="center", va="top", fontsize=6.5, color="#555555",
                linespacing=1.5)
    return save(fig, out, "fig04_architecture")


@figure("fig05_li_recall_curve", "4.4",
        "The two-stage approximation late interaction is forced into, measured on real queries")
def fig05(root, out):
    t = D.table("li_recall_fate_m")
    pts = sorted((int(k), v["recall"]) for k, v in t["recall_by_n_candidates"].items())
    xs = [p[0] for p in pts]
    ys = [100 * p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.8, 2.5))
    ax.plot(xs, ys, marker="o", color=ARM["li"])
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(f"{y:.1f}", (x, y - 5.5), ha="center", fontsize=7.5)
    ax.axvline(1000, color="#999999", ls=":", lw=1.2)
    ax.annotate("conventional\ndefault (1,000)", (1120, 8), fontsize=7.5, color="#666666")
    ax.axvline(50000, color=ARM["li"], ls=":", lw=1.2)
    ax.annotate("used throughout\n(50,000)", (44000, 8), fontsize=7.5, color=ARM["li"], ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("first-stage candidates re-ranked by exact MaxSim")
    ax.set_ylabel("recall@10 vs exact\nfull-corpus MaxSim (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="x", visible=True)
    return save(fig, out, "fig05_li_recall_curve")


def _forest(ax, rows, show_labels: bool):
    """rows: (label, effect, lo, hi, colour, emphasised, p). Effects in percentage points.

    `show_labels` exists because these panels share a y axis: with tick labels drawn on every
    panel, each panel's labels land on top of its neighbour's data. Only the leftmost keeps them.
    """
    ys = np.arange(len(rows))[::-1]
    for y, (_label, eff, lo, hi, colour, bold, p) in zip(ys, rows, strict=True):
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=colour, lw=1.6, solid_capstyle="butt")
            for e in (lo, hi):
                ax.plot([e, e], [y - 0.11, y + 0.11], color=colour, lw=1.3)
        ax.plot([eff], [y], "o", color=colour, ms=6.5 if bold else 4.5,
                mec="black" if bold else colour, mew=0.9 if bold else 0)
        if p is not None:
            ax.annotate(f"$p={p:.3f}$", (eff, y + 0.22), ha="center", va="bottom", fontsize=7.5,
                        color="#333333")
    ax.axvline(0, color="#444444", lw=1.0, ls="--")
    # With a shared y axis the ticks are one object: setting empty labels on a later panel blanks
    # the first panel too. Hide the labels on the panel instead of replacing them.
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(labelleft=show_labels, left=False)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.2)


@figure("fig06_tier1_effects", "5.1",
        "Tier 1 paired contrasts: retrieval helps pooled, the architectures do not differ")
def fig06(root, out):
    t1 = D.table("table1")["comparisons"]
    pooled = {
        "sv_vs_none": D.table("pooled_sv_vs_none")["pooled"]["primary"],
        "li_vs_none": D.table("pooled_li50k_vs_none")["pooled"]["primary"],
        "li_vs_sv": D.table("pooled_li50k_vs_sv")["pooled"]["primary"],
    }
    contrasts = [("sv_vs_none", "single-vector $-$ none", ARM["sv"]),
                 ("li_vs_none", "late interaction $-$ none", ARM["li"]),
                 ("li_vs_sv", "late int. $-$ single-vector", ARM["li"])]

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), sharex=True, sharey=True)
    for ax, (key, title, colour) in zip(axes, contrasts, strict=True):
        rows = []
        for bench in ("proofnet_test", "fate_m"):
            c = t1[f"{bench}:{key}"]
            lo, hi = c["ci95"]
            rows.append((BENCH_SHORT[bench], 100 * c["delta_rate"], 100 * lo, 100 * hi,
                         colour, False, c["p_permutation"]))
        p = pooled[key]
        ci = p.get("ci95")
        rows.append(("pooled", 100 * p["delta_rate"],
                     100 * ci[0] if ci else None, 100 * ci[1] if ci else None,
                     colour, True, p["p_permutation"]))
        _forest(ax, rows, show_labels=ax is axes[0])
        ax.set_title(title, fontsize=8.5)
    axes[0].set_xlim(-9, 13)
    fig.supxlabel("paired effect (percentage points)", fontsize=9, y=-0.03)
    return save(fig, out, "fig06_tier1_effects")


@figure("fig07_union_complementarity", "5.4",
        "Equal counts, different theorems: 28 of 88 solved by exactly one architecture")
def fig07(root, out):
    disc = {b: D.table("discordance_" + ("proofnet" if b == "proofnet_test" else b))
            for b in ("proofnet_test", "fate_m")}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.2))
    labels, both, only_sv, only_li = [], [], [], []
    for bench in ("proofnet_test", "fate_m"):
        d = disc[bench]
        labels.append(BENCH_SHORT[bench])
        only_sv.append(d["only_sv"]["n"])
        only_li.append(d["only_li@50k"]["n"])
        both.append(d["solved"]["sv"] - d["only_sv"]["n"])
    labels.append("pooled")
    only_sv.append(sum(only_sv))
    only_li.append(sum(only_li))
    both.append(sum(both))

    y = np.arange(len(labels))[::-1]
    h = 0.62
    b1 = ax.barh(y, both, h, color="#BBBBBB", label="solved by both")
    b2 = ax.barh(y, only_sv, h, left=both, color=ARM["sv"], label="single-vector only")
    b3 = ax.barh(y, only_li, h, left=np.array(both) + np.array(only_sv), color=ARM["li"],
                 label="late interaction only")
    for bars in (b1, b2, b3):
        for bar in bars:
            w = bar.get_width()
            if w:
                ax.annotate(f"{int(w)}", (bar.get_x() + w / 2, bar.get_y() + h / 2),
                            ha="center", va="center", fontsize=8,
                            color="black" if bars is b1 else "white")
    for yi, b, s, li_ in zip(y, both, only_sv, only_li, strict=True):
        ax.annotate(f"union {b + s + li_}", (b + s + li_ + 1.8, yi), va="center", fontsize=8.5)
    ax.set_yticks(y, labels)
    ax.set_xlabel("problems solved")
    ax.set_xlim(0, 106)
    ax.grid(axis="y", visible=False)
    h_, lab_ = ax.get_legend_handles_labels()
    fig.legend(h_, lab_, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncols=3)
    return save(fig, out, "fig07_union_complementarity")


@figure("fig08_terminal_status", "5.5",
        "How every search ended at 16 samples per expansion")
def fig08(root, out):
    order = ["proved", "no_candidates", "exhausted", "wall_clock", "error"]
    arms = ["none", "sv", "li"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7), sharey=True)
    for ax, bench in zip(axes, ("proofnet_test", "fate_m"), strict=True):
        counts = {a: D.tier1_status_counts(root, bench, a) for a in arms}
        x = np.arange(len(arms))
        bottom = np.zeros(len(arms))
        for st in order:
            vals = np.array([100 * counts[a].get(st, 0) / sum(counts[a].values()) for a in arms])
            if vals.sum() == 0:
                continue
            ax.bar(x, vals, 0.55, bottom=bottom, color=STATUS[st], label=STATUS_LABEL[st])
            for xi, (v, b) in enumerate(zip(vals, bottom, strict=True)):
                if v >= 6:
                    ax.annotate(f"{v:.0f}", (xi, b + v / 2), ha="center", va="center",
                                fontsize=7.5, color="white")
            bottom += vals
        ax.set_xticks(x, ["no\nretrieval", "single-\nvector", "late\ninteraction"])
        ax.set_title(BENCH_SHORT[bench])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of attempts (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=2)
    return save(fig, out, "fig08_terminal_status")


@figure("fig09_cost", "5.6",
        "What each retriever costs: query latency (log) and index footprint")
def fig09(root, out):
    # (label, ms/query, colour). Ranges are the two benchmarks; the mean of the two is plotted and
    # the range is annotated, because a single number would hide that ProofNet is consistently
    # slower -- its states are longer, so the MaxSim rerank has more query tokens to score.
    rows = [
        ("BM25", 9.8, ARM["none"], "169 MB"),
        ("single-vector (GPU)", 40.8, ARM["sv"], "943 MB"),
        ("late interaction @1k (GPU)", 81.4, "#F0A860", "5.5 GB"),
        ("single-vector (CPU)", 475.0, "#5FA8D3", "943 MB"),
        ("late interaction @50k (GPU)", 1029.5, ARM["li"], "5.5 GB"),
        ("fusion: SV(CPU) + LI@50k(GPU)", 1568.5, ARM["fusion"], "6.4 GB"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7),
                             gridspec_kw={"width_ratios": [2.0, 1]})

    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], 0.62, color=[r[2] for r in rows])
    for yi, r in zip(y, rows, strict=True):
        ax.annotate(f"{r[1]:,.1f} ms", (r[1] * 1.15, yi), va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(5, 6000)
    ax.set_xlabel("retrieval latency (ms / query, log scale)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    ax = axes[1]
    # Short labels: this panel is a third of the figure width and the full arm names collide.
    idx = [("BM25", 0.169, ARM["none"]), ("SV", 0.943, ARM["sv"]), ("LI", 5.5, ARM["li"])]
    bars = ax.bar(np.arange(3), [i[1] for i in idx], 0.55, color=[i[2] for i in idx])
    for b, i in zip(bars, idx, strict=True):
        ax.annotate(f"{i[1]:.2f} GB" if i[1] >= 1 else f"{1000 * i[1]:.0f} MB",
                    (b.get_x() + b.get_width() / 2, b.get_height() + 0.14),
                    ha="center", fontsize=8)
    ax.set_xticks(np.arange(3), [i[0] for i in idx])
    ax.set_ylabel("index size on disk (GB)")
    ax.set_ylim(0, 6.8)
    ax.grid(axis="x", visible=False)
    ax.set_title("footprint", fontsize=9)
    return save(fig, out, "fig09_cost")


@figure("fig10_root_candidate_quality", "6.2",
        "Late interaction improves the mean candidate, not the best one -- the mechanism")
def fig10(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6))
    for ax, (bench, tab) in zip(
            axes,
            (("proofnet_test", "root_quality_proofnet"), ("fate_m", "root_quality_fate_m")),
            strict=True):
        arms = D.table(tab)["arms"]
        base = next(a for a in arms if a["arm"] == "none")
        names, mean_d, best_d, colours = [], [], [], []
        for a in arms:
            if a["arm"] == "none":
                continue
            key = "li" if a["arm"].startswith("li") else a["arm"]
            names.append(ARM_LABEL[key].replace(" ", "\n"))
            mean_d.append(a["mean_logprob"] - base["mean_logprob"])
            best_d.append(a["mean_best_logprob"] - base["mean_best_logprob"])
            colours.append(ARM[key])
        x = np.arange(len(names))
        w = 0.34
        ax.bar(x - w / 2, mean_d, w, color=colours)
        ax.bar(x + w / 2, best_d, w, color=colours, alpha=0.42)
        for xi, (m, b) in enumerate(zip(mean_d, best_d, strict=True)):
            ax.annotate(f"{m:+.3f}", (xi - w / 2, m), ha="center",
                        va="bottom" if m >= 0 else "top", fontsize=7.5)
            ax.annotate(f"{b:+.3f}", (xi + w / 2, b), ha="center",
                        va="bottom" if b >= 0 else "top", fontsize=7.5)
        ax.axhline(0, color="#444444", lw=1.0)
        ax.set_xticks(x, names)
        ax.set_title(BENCH_SHORT[bench])
        ax.grid(axis="x", visible=False)
        if ax is axes[0]:
            ax.set_ylabel("$\\Delta$ log-probability\nvs no retrieval")
    axes[0].bar([np.nan], [np.nan], color="#666666", label="mean candidate")
    axes[0].bar([np.nan], [np.nan], color="#666666", alpha=0.42, label="best candidate")
    axes[0].legend(loc="upper left")
    return save(fig, out, "fig10_root_candidate_quality")


# =================================================================================================
# Part III -- scaling to pass@8
# =================================================================================================

@figure("fig11_headline_vs_published", "8.2",
        "The pass@8 ensemble against published systems")
def fig11(root, out):
    ours = {"proofnet_test": (44, 186), "fate_m": (72, 141)}
    benches = ["proofnet_test", "fate_m"]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.0))
    x = np.arange(2)
    w = 0.26

    ours_pct = [100 * ours[b][0] / ours[b][1] for b in benches]
    rp = [D.REALPROVER[b] for b in benches]
    rp_nr = [D.REALPROVER_NO_RETRIEVAL[b] for b in benches]

    b1 = ax.bar(x - w, ours_pct, w, color=ARM["ensemble"],
                label="this work, pass@8 ensemble (1/128 budget)")
    b2 = ax.bar(x, rp, w, color=ARM["published"], label="REAL-Prover-v1, Pass@$64{\\times}64$")
    b3 = ax.bar(x + w, rp_nr, w, color=ARM["published"], alpha=0.40,
                label="REAL-Prover-v1, no retrieval")

    for bars, vals in ((b1, [f"{ours[b][0]}/{ours[b][1]}" for b in benches]),
                       (b2, [f"{v:.1f}%" for v in rp]),
                       (b3, [f"{v:.1f}%" for v in rp_nr])):
        for bar, lab in zip(bars, vals, strict=True):
            ax.annotate(lab, (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8),
                        ha="center", va="bottom", fontsize=8)

    ax.axhline(D.REPROVER["proofnet_test"], xmin=0.03, xmax=0.44, color="#666666", ls=":",
               lw=1.2, zorder=4)
    ax.annotate("ReProver\n13.8%", (0.44, D.REPROVER["proofnet_test"]), fontsize=7.5,
                color="#666666", ha="left", va="center", zorder=6)
    ax.set_xticks(x, [BENCH_LABEL[b] for b in benches])
    pct_axis(ax, top=70)
    ax.legend(loc="upper left")
    return save(fig, out, "fig11_headline_vs_published")


@figure("fig12_budget_vs_performance", "8.2",
        "Solve rate against generation budget, four orders of magnitude")
def fig12(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))
    for ax, bench in zip(axes, ("proofnet_test", "fate_m"), strict=True):
        ids = D.problem_ids(root, bench)
        draws = D.sweep_draws(root, bench, arms=("li", "sv"))
        li = [draws[("li", s)] for s in range(8)]
        sv = [draws[("sv", s)] for s in range(8)]
        ens = [li[s] | sv[s] for s in range(8)]
        n = len(ids)
        for arm, pool, per_draw in (("li", li, 2048), ("sv", sv, 2048), ("ensemble", ens, 4096)):
            ks = range(1, 9)
            xs = [k * per_draw for k in ks]
            ys = [100 * D.curve(pool, ids, k) / n for k in ks]
            ax.plot(xs, ys, marker="o", color=ARM[arm], label=ARM_LABEL[arm],
                    lw=1.8 if arm == "ensemble" else 1.4,
                    ms=4.5 if arm == "ensemble" else 3.5)
        ax.scatter([D.REALPROVER_GENERATIONS], [D.REALPROVER[bench]], marker="*", s=170,
                   color=ARM["published"], zorder=5, label="REAL-Prover-v1")
        ax.set_xscale("log")
        ax.set_xlim(1.2e3, 1.4e7)
        ax.set_xlabel("generations per problem")
        ax.set_title(BENCH_LABEL[bench])
        ax.grid(axis="x", visible=True)
        if ax is axes[0]:
            ax.set_ylabel("problems solved (%)")
    axes[0].legend(loc="lower right")
    return save(fig, out, "fig12_budget_vs_performance")


@figure("fig13_coverage_curve", "8.3",
        "pass@k coverage curves by the unbiased estimator")
def fig13(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))
    for ax, bench in zip(axes, ("proofnet_test", "fate_m"), strict=True):
        ids = D.problem_ids(root, bench)
        draws = D.sweep_draws(root, bench, arms=("li", "sv"))
        li = [draws[("li", s)] for s in range(8)]
        sv = [draws[("sv", s)] for s in range(8)]
        ens = [li[s] | sv[s] for s in range(8)]
        n = len(ids)
        ks = list(range(1, 9))
        for arm, pool in (("ensemble", ens), ("li", li), ("sv", sv)):
            ys = [100 * D.curve(pool, ids, k) / n for k in ks]
            ax.plot(ks, ys, marker="o", color=ARM[arm], label=ARM_LABEL[arm],
                    lw=1.8 if arm == "ensemble" else 1.4,
                    ms=4.5 if arm == "ensemble" else 3.5)
        ax.axhline(D.REALPROVER[bench], color=ARM["published"], ls="--", lw=1.2)
        ax.annotate(f"REAL-Prover-v1 {D.REALPROVER[bench]}%", (1.05, D.REALPROVER[bench] + 0.6),
                    fontsize=7.5, color=ARM["published"])
        ax.set_xlabel("$k$ (independent seeds)")
        ax.set_xticks(ks)
        ax.set_title(BENCH_LABEL[bench])
        if ax is axes[0]:
            ax.set_ylabel("problems solved (%)")
    axes[0].set_ylim(12, 27)
    axes[1].set_ylim(34, 60)
    axes[1].legend(loc="lower right")
    return save(fig, out, "fig13_coverage_curve")


@figure("fig14_null_at_scale", "8.4",
        "The architecture null at one seed and at eight, 16x the budget")
def fig14(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7), sharey=True)
    t1 = D.table("table1")["comparisons"]
    x = np.arange(2)
    w = 0.34
    benches = ("proofnet_test", "fate_m")

    ax = axes[0]
    for i, arm in enumerate(("sv", "li")):
        side = "baseline" if arm == "sv" else "treatment"
        vals = [t1[f"{b}:li_vs_sv"][side]["proved"] for b in benches]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=ARM[arm], label=ARM_LABEL[arm])
        bar_labels(ax, bars, "{:.0f}", dy=0.8)
    ax.set_xticks(x, [BENCH_SHORT[b] for b in benches])
    ax.set_title("pass@1: 64$\\times$16, one seed", fontsize=9)
    ax.set_ylabel("problems solved")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.annotate("$p=1.0000$", (0, 33), ha="center", fontsize=8.5)
    ax.annotate("$p=1.0000$", (1, 51), ha="center", fontsize=8.5)

    ax = axes[1]
    for i, arm in enumerate(("sv", "li")):
        vals = []
        for bench in benches:
            draws = D.sweep_draws(root, bench, arms=(arm,))
            vals.append(len(set().union(*(draws[(arm, s)] for s in range(8)))))
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=ARM[arm])
        bar_labels(ax, bars, "{:.0f}", dy=0.8)
    ax.set_xticks(x, [BENCH_SHORT[b] for b in benches])
    ax.set_title("pass@8: 64$\\times$32, eight seeds", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.annotate("$p=1.0000$", (0, 44), ha="center", fontsize=8.5)
    ax.annotate("$p=0.7539$", (1, 73), ha="center", fontsize=8.5)
    ax.set_ylim(0, 84)
    return save(fig, out, "fig14_null_at_scale")


@figure("fig15_sample_budget_correction", "8.5",
        "A sample budget masquerading as a mechanism: no_candidates at 16 vs 32 samples")
def fig15(root, out):
    arms = ["sv", "li"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), sharey=True)
    for ax, bench in zip(axes, ("proofnet_test", "fate_m"), strict=True):
        x = np.arange(len(arms))
        w = 0.34
        series = (("16 samples (Tier 1)",
                   lambda a, b=bench: D.tier1_status_counts(root, b, a)),
                  ("32 samples (sweep)",
                   lambda a, b=bench: D.status_counts(root, b, a)))
        for i, (label, getter) in enumerate(series):
            vals = []
            for arm in arms:
                c = getter(arm)
                vals.append(100 * c.get("no_candidates", 0) / sum(c.values()))
            bars = ax.bar(x + (i - 0.5) * w, vals, w, color=[ARM[a] for a in arms],
                          alpha=1.0 if i else 0.45, hatch="" if i else "///",
                          edgecolor="white", linewidth=0)
            bar_labels(ax, bars, "{:.1f}", dy=0.9)
            if ax is axes[0]:
                ax.bar([np.nan], [np.nan], color="#666666", alpha=1.0 if i else 0.45,
                       hatch="" if i else "///", edgecolor="white", label=label)
        ax.set_xticks(x, [ARM_LABEL[a].replace(" ", "\n") for a in arms])
        ax.set_title(BENCH_SHORT[bench])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("attempts ending in\nno_candidates (%)")
    axes[0].set_ylim(0, 50)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncols=2)
    return save(fig, out, "fig15_sample_budget_correction")


# =================================================================================================
# Part IV -- fusion
# =================================================================================================

@figure("fig16_fusion_passk", "9.2",
        "The fusion arm against both incumbents at equal generation budget")
def fig16(root, out):
    draws = D.sweep_draws(root, "fate_m")
    arms = ["li", "sv", "fusion"]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.74, 2.9))
    unions, per_seed = [], {}
    for a in arms:
        seeds = [draws[(a, s)] for s in range(4)]
        unions.append(len(set().union(*seeds)))
        per_seed[a] = [len(s) for s in seeds]
    ens = set().union(*(draws[("li", s)] for s in range(4)),
                      *(draws[("sv", s)] for s in range(4)))
    unions.append(len(ens))

    x = np.arange(4)
    bars = ax.bar(x, unions, 0.55, color=[ARM[a] for a in arms] + [ARM["ensemble"]])
    bars[-1].set_alpha(0.75)          # this bar buys its height with twice the generations
    bar_labels(ax, bars, "{:.0f}", dy=1.0)
    for i, a in enumerate(arms):
        ax.scatter([i] * 4, per_seed[a], facecolor="white", edgecolor="#333333", zorder=5, s=24,
                   linewidths=0.9)
    ax.axhline(69, color="#B22222", ls="--", lw=1.3)
    ax.annotate("threshold to replace the ensemble: 69", (-0.44, 71.5), fontsize=8,
                color="#B22222")
    ax.set_xticks(x, ["late\ninteraction\n8,192 gen", "single-\nvector\n8,192 gen",
                      "fusion (RRF)\n8,192 gen", "ensemble\nSV $\\cup$ LI\n16,384 gen"])
    ax.set_ylabel("problems solved, pass@4 (of 141)")
    ax.set_ylim(0, 92)
    ax.grid(axis="x", visible=False)
    return save(fig, out, "fig16_fusion_passk")


@figure("fig17_budget_matched", "9.4",
        "The union ceiling priced at equal generations: +14 becomes +2.87, not significant")
def fig17(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))

    ax = axes[0]
    bars = ax.bar([0, 1, 2], [74, 74, 88], 0.5,
                  color=[ARM["sv"], ARM["li"], ARM["ensemble"]])
    bars[-1].set_alpha(0.75)
    bar_labels(ax, bars, "{:.0f}", dy=1.6)
    # The budget goes in the tick label rather than an annotation: it is the property that makes
    # this panel's comparison unfair, so it belongs on the axis, not floating over a bar.
    ax.set_xticks([0, 1, 2], ["single-\nvector\n2,048 gen", "late\ninteraction\n2,048 gen",
                              "union\n4,096 gen"])
    ax.set_ylabel("problems solved (of 327)")
    ax.set_ylim(0, 140)
    ax.grid(axis="x", visible=False)
    ax.set_title("as reported: unequal budget", fontsize=9)
    ax.annotate("$+14$ over either arm,\non twice the generations", (1.0, 120), ha="center",
                va="center", fontsize=8.5, color="#B22222")

    ax = axes[1]
    tot = {}
    for name in ("li", "sv", "ens"):
        s = 0.0
        for bench in ("proofnet_test", "fate_m"):
            ids = D.problem_ids(root, bench)
            d = D.sweep_draws(root, bench, arms=("li", "sv"))
            li = [d[("li", i)] for i in range(8)]
            sv = [d[("sv", i)] for i in range(8)]
            pool, k = {"li": (li, 8), "sv": (sv, 8),
                       "ens": ([li[i] | sv[i] for i in range(8)], 4)}[name]
            s += D.curve(pool, ids, k)
        tot[name] = s

    bars = ax.bar([0, 1, 2], [tot["sv"], tot["li"], tot["ens"]], 0.5,
                  color=[ARM["sv"], ARM["li"], ARM["ensemble"]])
    bars[-1].set_alpha(0.75)
    bar_labels(ax, bars[:2], "{:.2f}", dy=1.6)
    lo, hi = tot["li"] - 2.34, tot["li"] + 8.54
    ax.errorbar([2], [tot["ens"]], yerr=[[tot["ens"] - lo], [hi - tot["ens"]]],
                fmt="none", ecolor="#B22222", elinewidth=1.5, capsize=4, zorder=6)
    ax.annotate(f"{tot['ens']:.2f}", (2, hi + 2.0), ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1, 2], ["single-\nvector\n8 seeds", "late\ninteraction\n8 seeds",
                              "ensemble\n4 seeds"])
    ax.set_ylim(0, 140)
    ax.grid(axis="x", visible=False)
    ax.set_title("equal budget: 16,384 gen/problem", fontsize=9)
    ax.annotate("ensemble $-$ late interaction\n$+2.87$,  CI $[-2.34, +8.54]$,  $p=0.33$",
                (1.0, 130), ha="center", va="center", fontsize=8.5, color="#B22222")
    return save(fig, out, "fig17_budget_matched")


# =================================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=REPO / "results" / "exported" / "logs")
    ap.add_argument("--out", type=Path, default=REPO / "figures")
    ap.add_argument("--only", nargs="*", default=None,
                    help="figure name prefixes, e.g. --only fig13 fig17")
    ap.add_argument("--list", action="store_true", help="names and dissertation sections")
    args = ap.parse_args()

    if args.list:
        for name, meta in FIGURES.items():
            print(f"{name:34} sec {meta['section']:<7} {meta['what']}")
        return

    use_style()
    args.out.mkdir(parents=True, exist_ok=True)

    # Stated rather than assumed: a run with no verification.json is counted without the discount
    # its verified rivals receive, so a figure built on one is slightly generous to that arm.
    if missing := D.unverified(args.results_root):
        print(f"WARNING - {len(missing)} run(s) in this figure set have no verification.json, so "
              "their claims are counted without the discount a verified run receives:")
        for m in missing:
            print(f"    {m}")
        print()

    names = [n for n in FIGURES if not args.only or any(n.startswith(p) for p in args.only)]
    if not names:
        raise SystemExit(f"--only {args.only} matched no figure. Try --list.")

    for name in names:
        written = FIGURES[name]["fn"](args.results_root, args.out)
        print(f"  {name:34} -> {', '.join(written)}")

    print(f"\n{len(names)} figure(s) -> {args.out}")


if __name__ == "__main__":
    main()
