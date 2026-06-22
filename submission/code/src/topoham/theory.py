"""From-principles guide for commutator-graph Hamiltonian simulation.

Hamiltonian simulation
----------------------
An n-qubit state is a vector in C**(2**n). A Hamiltonian H is a Hermitian
operator, and exact dynamics are exp(-i H t). The code represents H as a sum of
Pauli strings with real coefficients, avoiding dense matrices except for small
reference checks.

Pauli computation
-----------------
pauli.py implements two central facts. First, two Pauli strings commute if the
number of positions with anticommuting single-qubit factors is even. Second,
because P**2 = I, a single Pauli rotation has the closed form

    exp(-i theta P) psi = cos(theta) psi - i sin(theta) P psi.

This gives an exact statevector Trotter kernel without forming expm for every
term.

Commutator graph
----------------
commutator_graph.py builds a graph with one node per Hamiltonian term and an
edge between anticommuting terms. First-order Trotter error is controlled by
commutators, so this graph exposes which adjacent term pairs are expensive.

Policies and machine learning
-----------------------------
policies.py compares random, coefficient, locality, commutator-aware and learned
ordering policies at the same gate proxy. The learned policy is a small
logistic-regression router over invariant commutator-graph features; it selects
among human-readable orderings instead of inventing an opaque permutation.

Evaluation
----------
env.py compares each Trotterized state with an exact reference and reports
fidelity, infidelity, observable error and gate proxy. runner.py aggregates over
Hamiltonian families, writes results/summary.json, and plotting/table scripts
derive all manuscript artifacts from that single source.

Reproduction
------------
From code/:

    export PYTHONPATH=src
    make test
    bash scripts/reproduce_all.sh full
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
