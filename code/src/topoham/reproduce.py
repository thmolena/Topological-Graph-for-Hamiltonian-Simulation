"""Installed reproduction command for the topoham artifact.

Exposes the ``topoham-reproduce`` console script (see the ``[project.scripts]``
table in ``pyproject.toml``). A single invocation deterministically regenerates
every reported artifact from one seeded run: the experiment runner
(``scripts/run.py`` -> ``results/summary.json``), the LaTeX/Markdown tables
(``scripts/make_tables.py``), the figure PDFs (``scripts/make_figures.py``), and
the readiness-gate audit (``scripts/audit_claims.py``). Each table, figure, and
macro therefore traces to a single run.

The pipeline is driven through the repository's existing scripts so that the
installed command and ``make full-run`` execute identical experiment logic. The
package locates that script tree relative to the source checkout; the command is
intended to run from a clone of the repository.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _code_dir() -> Path:
    """Return the repository ``code/`` directory that holds ``scripts/``."""
    # src/topoham/reproduce.py -> parents[2] == code/
    return Path(__file__).resolve().parents[2]


def _run(*args: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, *args], cwd=_code_dir(), env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full.yaml",
        help="experiment configuration (default: configs/full.yaml, reported scale)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="regenerate tables/figures from an existing results/summary.json",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="skip the readiness-gate audit step",
    )
    args = parser.parse_args(argv)

    if not args.skip_run:
        _run("scripts/run.py", "--config", args.config, "--out", "results")
    _run("scripts/make_tables.py")
    _run("scripts/make_figures.py")
    if not args.skip_audit:
        _run("scripts/audit_claims.py")


if __name__ == "__main__":
    main()
