#!/usr/bin/env python
"""Readiness-gate audit: traceable numbers, required phrase, no forbidden claims.

Exits non-zero if any check fails, so it can gate CI / `make audit`.
"""
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from topoham import audit


def main() -> int:
    problems = []

    summary_path = Path("results/summary.json")
    if not summary_path.exists():
        print("FAIL: results/summary.json missing -- run `make demo` first")
        return 1
    summary = json.loads(summary_path.read_text())

    headline = summary.get("headline", {})
    if not headline:
        problems.append("summary headline is empty")

    # README must carry the mandated phrase and avoid forbidden claims.
    readme = Path("../README.md")
    if readme.exists():
        text = readme.read_text()
        problems += audit.audit_reproducibility_phrase(text)
        problems += audit.audit_forbidden_claims(text)

    # Every headline macro must be present (traceable to this run).
    for key in ("commutator_fidelity_mean", "random_fidelity_mean",
                "infidelity_reduction_vs_random", "n_instances"):
        if key not in headline:
            problems.append(f"headline macro {key!r} not generated")

    # Integrity flag: the fast Pauli kernel must have matched dense expm.
    if not summary.get("pauli_algebra_verified", False):
        problems.append("pauli algebra verification flag is not set")

    if problems:
        print("AUDIT FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("AUDIT PASSED: all numbers traceable, phrase present, no forbidden claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
