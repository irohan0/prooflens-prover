"""Shared look for every dissertation figure, so a LaTeX report can mix them on one page.

Matched to a 12pt `report` on A4 with default margins: the text block is about 6.3 in wide, so a
full-width figure is 6.3 and a half-width one is 3.05. Font sizes are set for figures included at
`width=\textwidth` **without further scaling** — a figure scaled down in LaTeX has smaller labels
than the body text, which is the most common way a thesis figure becomes unreadable.

Colours are Okabe-Ito, which stays distinguishable under the three common colour-vision deficiencies
and, unlike the matplotlib default cycle, survives greyscale printing in the order used here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Okabe-Ito. One arm keeps one colour in every figure — an arm that changes colour between two
#: plots on facing pages is read as two different things.
ARM = {
    "none": "#999999",
    "sv": "#0072B2",
    "li": "#D55E00",
    "li@50k": "#D55E00",
    "fusion": "#009E73",
    "ensemble": "#000000",
    "published": "#CC79A7",
}
ARM_LABEL = {
    "none": "no retrieval",
    "sv": "single-vector",
    "li": "late interaction",
    "li@50k": "late interaction",
    "fusion": "fusion (RRF)",
    "ensemble": "ensemble (SV $\\cup$ LI)",
}
#: Terminal statuses, in the order they are stacked. `proved` first so the bars read left-to-right
#: as "succeeded, then the two ways of failing".
STATUS = {
    "proved": "#009E73",
    "exhausted": "#E69F00",
    "no_candidates": "#0072B2",
    "wall_clock": "#CC79A7",
    "error": "#999999",
}
STATUS_LABEL = {
    "proved": "proved",
    "exhausted": "exhausted (hit expansion cap)",
    "no_candidates": "no candidates (frontier emptied)",
    "wall_clock": "wall clock",
    "error": "error",
}

TEXT_WIDTH = 6.3
HALF_WIDTH = 3.05


def use_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType, so the PDF is editable and searchable
        "ps.fonttype": 42,
    })


def save(fig, out_dir, name: str) -> list[str]:
    """PNG at 300 dpi — high enough for print inclusion, and the only artefact the report needs."""
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return [path.name]


def pct_axis(ax, top: float | None = None) -> None:
    ax.set_ylabel("problems solved (%)")
    if top is not None:
        ax.set_ylim(0, top)
    ax.grid(axis="x", visible=False)


def bar_labels(ax, bars, fmt="{:.0f}", dy=0.6, **kw) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h + dy),
                    ha="center", va="bottom", fontsize=8, **kw)
