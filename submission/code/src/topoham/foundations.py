"""From-scratch foundations for commutator-graph Hamiltonian simulation."""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Hamiltonian simulation",
        "The exact evolution exp(-i H t) is approximated by a Trotter product "
        "over Pauli terms.  The code evaluates small systems exactly so it can "
        "measure fidelity and ordering effects without treating dense matrices "
        "as the scalable proposal.",
    ),
    (
        "Pauli algebra",
        "pauli.py implements tensor-product Pauli strings and commutators.  Two "
        "terms that commute can be swapped without first-order commutator error; "
        "non-commuting neighbours define the graph structure used by the "
        "ordering policies.",
    ),
    (
        "Commutator graph",
        "commutator_graph.py builds a graph whose vertices are Hamiltonian terms "
        "and whose edges mark non-zero commutators.  Ordering terms is then a "
        "graph problem: reduce adjacent conflicts at matched gate budget.",
    ),
    (
        "Policies and baselines",
        "policies.py compares random ordering, graph heuristics and learned or "
        "topology-aware choices.  The manuscript reports the clear effect of "
        "structured ordering versus random, while avoiding unsupported claims "
        "that the non-random orderings are separated at this scale.",
    ),
    (
        "Metrics",
        "metrics.py computes fidelity, error and confidence intervals across a "
        "seeded instance family.  Figures show frontier curves and family-level "
        "breakdowns because those display types are standard and interpretable "
        "for NMI-style benchmark papers.",
    ),
    (
        "Audit discipline",
        "runner.py writes results/summary.json as the single source of truth.  "
        "make_tables.py and make_figures.py render only from that artifact, and "
        "audit_claims.py prevents unsupported superiority language.",
    ),
    (
        "Reproduction path",
        "Run scripts/reproduce_all.sh full from the code folder or use the "
        "topoham-reproduce entry point.  Dependencies are in pyproject.toml; "
        "no requirements text file is needed.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
