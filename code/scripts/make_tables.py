#!/usr/bin/env python
"""Generate LaTeX tables from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    tables = Path("results")
    tables.mkdir(exist_ok=True)
    gp = summary.get("gate_proxy_fixed", summary["headline"]["gate_proxy"])

    rows = []
    for ordng, m in summary["orderings"].items():
        f = m["fidelity"]
        rows.append((ordng, f["mean"], f["ci95"], m["infidelity_mean"],
                     m["adjacent_anticommutations_mean"], gp))

    # LaTeX
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"ordering & fidelity & $\pm$95\% & infidelity & adj.\ anticomm. & gate proxy \\",
             r"\midrule"]
    for ordng, mean, ci, inf, adj, g in rows:
        lines.append(f"{ordng} & {mean:.4f} & {ci:.4f} & {inf:.4f} & {adj:.2f} & {g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "main_results.tex").write_text("\n".join(lines) + "\n")

    print("wrote results/main_results.tex")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
