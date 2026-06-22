# topoham — code artifact

CPU reference implementation for **Lie-Topological Graph Policies for Biomolecular
Hamiltonian Simulation**. First-order Trotter evolution is computed with an exact
numpy statevector simulator built from single-Pauli rotations
(`cos θ · ψ − i sin θ · Pψ`), and validated against dense `scipy.linalg.expm` and
Krylov `scipy.sparse.linalg.expm_multiply` references. The ordering policies use
networkx (commutator graph) and scikit-learn (the learned router); an optional
PyTorch GNN backend can be dropped in. No heavyweight quantum dependency.

## Install
```bash
conda env create -f environment.yml && conda activate topoham
export PYTHONPATH=src
```

## Reproduce
```bash
make test        # Pauli-rotation ↔ expm cross-check, Trotter convergence, core claim
make demo        # smoke config (~1-4 s) -> results/summary.json
make tables      # results/main_results.tex
make figures     # figures/fig_frontier.pdf, figures/fig_family.pdf
make audit       # readiness gate
make full-run    # reported-scale config (minutes)
# or, one command:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```
> macOS: the Makefile/scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because conda and
> pip-PyTorch can both ship an OpenMP runtime. No effect on results.

## Layout
```
src/topoham/
  pauli.py             Pauli strings · commute()/anticommute() · fast e^{-iθP} rotation
  hamiltonians.py      TFIM · Heisenberg · molecular_like (random local 2-body) families
  commutator_graph.py  anticommutation graph · greedy commuting-group colouring · features
  env.py               first-order Trotter statevector + exact expm / Krylov references
  policies.py          random / coefficient / locality / commutator / learned orderings
  metrics.py           fidelity · observable error · gate proxy · summaries
  runner.py            end-to-end protocol -> results/summary.json
  audit.py             forbidden-claims + traceable-number checks
  plotting.py          fidelity-vs-gate-budget frontier + per-family figures
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     pauli↔expm · Trotter convergence · commutator-beats-random · metrics bounds
```

## What is computed
`results/summary.json` holds, per term-ordering policy at a **matched gate budget**
(`num_terms × Trotter steps`): the mean Trotter fidelity with 95% CIs, mean
infidelity, mean Z₀ observable error, mean number of anticommuting adjacencies, the
fidelity-vs-gate-budget frontier curve, and the per-family breakdown. The integrity
flag `pauli_algebra_verified` records that the fast Pauli-rotation kernel matched
dense `expm` before any fidelity number was produced. It is the single source of
truth for every table, figure and macro.

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
A from-basics explanation of the Pauli, commutator-graph, and Trotter objects is in `src/topoham/foundations.py`.
