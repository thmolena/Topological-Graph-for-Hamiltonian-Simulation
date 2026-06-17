"""Figure generation from results artifacts (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def fig_frontier(summary: Dict, out: Path) -> Path:
    """Fidelity vs gate budget, per ordering (the headline figure).

    The x-axis is the gate proxy (num_terms * r); each ordering traces fidelity
    as the Trotter step count grows. The commutator ordering sits above the
    random baseline at every fixed budget.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    frontier = summary.get("frontier", {})
    for ordering, curve in frontier.items():
        xs = [pt[0] for pt in curve]
        ys = [pt[1] for pt in curve]
        ax.plot(xs, ys, marker="o", ms=3, label=ordering)
    ax.set_xlabel("gate budget proxy (num terms × Trotter steps)")
    ax.set_ylabel("Trotter fidelity vs exact")
    ax.set_title("First-order Trotter: term-ordering policies")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_family_bars(summary: Dict, out: Path) -> Path:
    """Mean Trotter fidelity by Hamiltonian family for each ordering."""
    by_family = summary.get("by_family", {})
    families = sorted(by_family.keys())
    orderings = sorted({o for fam in by_family.values() for o in fam})
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    import numpy as np

    width = 0.8 / max(1, len(orderings))
    x = np.arange(len(families))
    for i, ordng in enumerate(orderings):
        vals = [by_family[f].get(ordng, {}).get("mean", 0.0) for f in families]
        ax.bar(x + i * width, vals, width, label=ordng)
    ax.set_xticks(x + width * (len(orderings) - 1) / 2)
    ax.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean Trotter fidelity")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
