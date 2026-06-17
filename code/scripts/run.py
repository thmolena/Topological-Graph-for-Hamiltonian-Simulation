#!/usr/bin/env python
"""Run the Trotter term-ordering experiment and write results/summary.json."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from topoham.config import Config
from topoham.runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    summary = run(cfg, Path(args.out))
    h = summary["headline"]
    print(f"[{cfg.name}] instances={h['n_instances']}  gate-proxy={h['gate_proxy']}  "
          f"commutator fidelity={h['commutator_fidelity_mean']:.4f}  "
          f"random fidelity={h['random_fidelity_mean']:.4f}  "
          f"infidelity reduction vs random={h['infidelity_reduction_vs_random']}x  "
          f"runtime={summary['provenance']['runtime_sec']}s")


if __name__ == "__main__":
    main()
