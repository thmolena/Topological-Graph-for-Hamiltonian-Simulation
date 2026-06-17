#!/usr/bin/env python
"""Generate all figures from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from topoham import plotting


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    out = Path("figures")
    out.mkdir(exist_ok=True)
    plotting.fig_frontier(summary, out / "fig_frontier.pdf")
    plotting.fig_family_bars(summary, out / "fig_family.pdf")
    print(f"wrote {out}/fig_frontier.pdf, {out}/fig_family.pdf")


if __name__ == "__main__":
    main()
