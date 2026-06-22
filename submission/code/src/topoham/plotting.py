"""Figure generation from results artifacts, styled to Nature Machine
Intelligence (NMI) display conventions.

Design rules applied here (see Nature Portfolio artwork & formatting guidance):
  * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
  * Sans-serif typeface (Arial/Helvetica family), 5--8 pt range.
  * No in-panel titles -- every description lives in the LaTeX caption.
  * Bold lower-case panel labels (a, b, ...) for multi-panel figures.
  * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
    2011): safe under deuteranopia/protanopia, avoids the red--green trap.
  * Uncertainty is shown wherever a mean is plotted (shaded 95% CI band on the
    frontier, error bars on the family bars); the caption states n and that the
    interval is a 95% confidence interval.
  * Top/right spines removed for an uncluttered Nature-style frame.

The figures are produced purely from ``results/summary.json`` -- the single
source of truth written by the experiment runner -- so they regenerate
bit-for-bit from fixed seeds.
"""
from __future__ import annotations

import os

# Determinism: pin the build epoch BEFORE importing matplotlib so its PDF
# backend stamps a fixed CreationDate -> byte-identical figure PDFs across runs
# (the plotted numbers are already deterministic: they are read from the
# fixed-seed results/summary.json, the single source of truth).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

# --- Okabe-Ito colour-blind-safe qualitative palette ------------------------
NMI_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Column widths in inches (Nature: single column 89 mm, double column 183 mm).
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20

# Human-readable, fixed ordering and display names for legends/axes. This is the
# canonical left-to-right order in every figure so colours are stable.
ORDERING_ORDER = ["random", "coefficient", "locality", "commutator", "learned"]
ORDERING_LABELS = {
    "random": "random",
    "coefficient": "coefficient",
    "locality": "locality",
    "commutator": "commutator",
    "learned": "learned router",
}


def apply_nmi_style() -> None:
    """Install NMI-conforming matplotlib defaults (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable
            "ps.fonttype": 42,
            "svg.hashsalt": "topoham",  # deterministic element IDs across runs
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",  # keep in-figure math sans-serif
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.0,
            "legend.frameon": False,
            "axes.prop_cycle": cycler(color=NMI_PALETTE),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        }
    )


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.02) -> None:
    """Bold lower-case panel label in the upper-left, Nature convention."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _color_for(ordering: str) -> str:
    return NMI_PALETTE[ORDERING_ORDER.index(ordering) % len(NMI_PALETTE)]


def fig_frontier(summary: Dict, out: Path) -> Path:
    """Trotter fidelity vs gate-budget proxy, per term ordering.

    The headline efficiency view: mean Trotter fidelity (over all 120 instances)
    as the gate proxy L*r grows along the step grid. A shaded 95% confidence
    band is drawn for orderings whose summary curve carries a CI half-width as a
    third element of each (x, y[, ci]) point. No in-plot title -- the description
    lives in the caption.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 2.9))
    frontier = summary.get("frontier", {})
    for ordering in ORDERING_ORDER:
        curve = frontier.get(ordering)
        if not curve:
            continue
        xs = np.array([pt[0] for pt in curve])
        ys = np.array([pt[1] for pt in curve])
        cis = np.array([pt[2] if len(pt) > 2 else 0.0 for pt in curve])
        color = _color_for(ordering)
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=2.5,
            color=color,
            label=ORDERING_LABELS.get(ordering, ordering),
        )
        if np.any(cis > 0):
            ax.fill_between(xs, ys - cis, ys + cis, color=color, alpha=0.15,
                            linewidth=0)
    ax.set_xlabel(r"gate-budget proxy $L\cdot r$ (terms $\times$ Trotter steps)")
    ax.set_ylabel("Trotter fidelity vs exact")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", handlelength=1.4)
    ax.grid(True, axis="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_family_bars(summary: Dict, out: Path) -> Path:
    """Held-out Trotter fidelity by Hamiltonian family, per ordering, with 95% CIs.

    Grouped bar chart; error bars are the 95% confidence interval
    (1.96 s / sqrt(n)) over the n instances in each family, taken directly from
    ``summary['by_family']``.
    """
    apply_nmi_style()
    by_family = summary.get("by_family", {})
    families = sorted(by_family.keys())
    orderings = [o for o in ORDERING_ORDER
                 if any(o in by_family[f] for f in families)]

    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 3.0))
    width = 0.8 / max(1, len(orderings))
    x = np.arange(len(families))
    for i, ordng in enumerate(orderings):
        vals = [by_family[f].get(ordng, {}).get("mean", 0.0) for f in families]
        errs = [by_family[f].get(ordng, {}).get("ci95", 0.0) for f in families]
        ax.bar(
            x + i * width,
            vals,
            width,
            yerr=errs,
            capsize=2,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            color=_color_for(ordng),
            label=ORDERING_LABELS.get(ordng, ordng),
        )
    ax.set_xticks(x + width * (len(orderings) - 1) / 2)
    ax.set_xticklabels(
        [f.replace("_", " ") for f in families], rotation=20, ha="right"
    )
    ax.set_ylabel("mean Trotter fidelity")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=len(orderings), loc="upper center", bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.0, handlelength=1.2)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _box(ax, xy, w, h, text, fc, ec="#222222"):
    """Draw a rounded method-schematic box with centred wrapped text."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        zorder=5,
    )
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                        shrinkA=2, shrinkB=2),
    )


def fig_schematic(summary: Dict, out: Path) -> Path:
    """Method-overview schematic (the NMI 'Figure 1' convention).

    A left-to-right pipeline: a structured Hamiltonian is decomposed into Pauli
    terms; the anticommutation (commutator) graph is built; a greedy
    commutator-graph ordering -- or a learned router over graph features -- fixes
    the term order; a matched-cost first-order Trotter schedule is evaluated by
    an exact statevector simulator; fidelity is scored against the exact
    propagator at an identical gate budget.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, green, orange, purple, grey = (
        "#D6E6F2",
        "#D6EFE3",
        "#FBE6D4",
        "#ECDCE9",
        "#ECECEC",
    )
    y = 0.40
    h = 0.34
    boxes = [
        (0.010, 0.150, "Hamiltonian\n$H=\\sum_j c_j P_j$\n3 families", blue),
        (0.205, 0.165, "commutator graph\n$\\mathcal{G}(H)$\n(edge = anticommute)", green),
        (0.415, 0.165, "term ordering $\\pi$\ngreedy graph / router\n(no extra gates)", orange),
        (0.625, 0.165, "matched-cost\nTrotter $U_{\\mathrm{Trot}}(\\pi)$\nbudget $L\\cdot r$", purple),
        (0.835, 0.155, "fidelity vs exact\n$e^{-iHt}$\nstatevector", grey),
    ]
    rights = []
    lefts = []
    for x0, w, text, fc in boxes:
        r, l = _box(ax, (x0, y), w, h, text, fc)
        rights.append(r)
        lefts.append(l)
    for i in range(len(boxes) - 1):
        _arrow(ax, rights[i], lefts[i + 1])

    # Annotate the two internal guarantees that underpin the benchmark.
    ax.text(0.4975, 0.07,
            "Pauli rotation $\\leftrightarrow$ dense expm agree to $10^{-9}$"
            "   $\\bullet$   gate proxy invariant under ordering",
            ha="center", va="center", fontsize=6.6, color="#555555")
    ax.text(0.4975, 0.95,
            "leading Trotter error $\\propto$ adjacent anticommuting pairs",
            ha="center", va="center", fontsize=6.8, color="#333333")
    fig.savefig(out)
    plt.close(fig)
    return out
