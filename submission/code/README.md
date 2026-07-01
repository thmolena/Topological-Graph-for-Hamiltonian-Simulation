# SpecOps-CGS — Commutator-Graph Scheduling for Trotter Simulation

Reproducibility package for the paper

> **Zero-Overhead Cancellation of the Leading Trotter Error via Commutator-Graph Scheduling**
> Molena Huynh (2026). Part of the **spectral-truncation operators (SpecOps)** program.

For a Hamiltonian in Pauli form `H = Σ_j c_j P_j`, a first-order Lie–Trotter step costs
exactly one rotation per term, so the gate count `L·r` is invariant under the term ordering `π`.
This package builds the **commutator error form** — the exact leading-order Trotter error
operator `E_π = Σ_{a<b} [h_{π(a)}, h_{π(b)}]`, assembled matrix-free over the commutator graph
and validated against dense matrix exponentials — and shows two things:

1. **Ordering impotence.** Reordering only re-signs the commutators in `E_π`; its
   Hilbert–Schmidt norm is ordering-invariant, so no fixed ordering escapes the `O(t²/r)` rate.
2. **Free second order.** A per-step step-alternating **antithetic** schedule cancels `E_π`
   exactly and converges at `O(t³/r²)` at the **identical** `L·r` rotation count.

A learned logistic router over ten commutator-graph features selects the gate-optimal schedule
per instance. One seeded command regenerates every figure, table, and quoted number in the paper.

The Python import name is `topoham` (historical); the pip distribution name is `specops-cgs`.

---

## Install

From this directory (`submission/code`):

```bash
pip install .
# or, for development:
pip install -e .
# with the optional learned-GPU / test extras:
pip install ".[gpu,dev]"
```

Requires Python ≥ 3.9. Core dependencies: numpy, scipy, networkx, scikit-learn,
matplotlib, pyyaml.

On macOS with conda + pip OpenMP runtimes, export `KMP_DUPLICATE_LIB_OK=TRUE`
(the reproduce entrypoint sets this for its subprocesses automatically).

---

## Reproduce the paper

The console entrypoint runs the full pipeline (experiment → tables → figures) and
syncs artifacts to the manuscript:

```bash
cd submission/code
cgs-reproduce                          # full config (configs/full.yaml), ~2 min CPU
cgs-reproduce --config configs/smoke.yaml   # fast demo, a few seconds
cgs-reproduce --skip-run               # rebuild tables/figures from an existing run
```

This writes:

* `results/summary.json` — every number, with provenance (seed, platform, runtime, peak memory);
* `results/macros.tex` — `\newcommand` for each number the manuscript prose quotes;
* `results/tab_*.tex` — the five LaTeX table fragments the manuscript `\input`s;
* `figures/fig_*.pdf` — the six manuscript figures;

and copies the figures/tables into `submission/figures` and `submission/tables`.
The manuscript `\input{code/results/macros.tex}` and `\includegraphics` from
`code/figures/`, so **no number is entered by hand**.

Equivalent Make targets: `make full-run && make tables && make figures` (and `make demo`,
`make test`, `make audit`).

---

## Extend / tweak

All experiment parameters live in a YAML config (`configs/full.yaml`, `configs/smoke.yaml`)
loaded into `topoham.config.Config`. Copy a config, edit fields, and run
`cgs-reproduce --config configs/my.yaml`. Every field:

| field               | type        | meaning |
|---------------------|-------------|---------|
| `name`              | str         | run label, stored in the summary provenance |
| `seed`              | int         | master RNG seed (fully determines instances) |
| `families`          | list[str]   | Hamiltonian families to sweep: `tfim`, `heisenberg`, `molecular_like` |
| `sizes`             | list[int]   | qubit counts `n` to sweep |
| `n_per_family`      | int         | number of seeded instances per (family, size) |
| `time`              | float       | evolution time `t` |
| `steps`             | int         | reference (matched-budget) Trotter step count `r` |
| `steps_grid`        | list[int]   | step counts for the gate-budget / convergence frontier |
| `schedules`         | list[str]   | schedules compared: `random`, `coefficient`, `commutator`, `antithetic`, `symmetric`, `learned` |
| `targets`           | list[float] | target fidelities for the gates-to-target tables |
| `target_ref`        | float       | target fidelity the learned router optimises for |
| `impotence_samples` | int         | random orderings per instance for the impotence sweep (Theorem 1) |
| `impotence_max_n`   | int         | cap `n` for the exact dense spectral-norm impotence sweep; larger `n` use the matrix-free HS surrogate |
| `reference_backend` | str         | exact reference: `expm` (dense) or `krylov` (matrix-free, scales to larger `n`) |

Any field omitted from the YAML falls back to the dataclass default in
`src/topoham/config.py`.

### Add a new Hamiltonian family

Add a builder in `src/topoham/hamiltonians.py` returning the Pauli-term list
`[(coeff, pauli_string), ...]`, register its name in the family dispatch there,
then list the name under `families:` in your config.

### Add a new schedule

Implement it in `src/topoham/schedules.py` (map an ordering + step index to the
per-step term order / sign pattern), add its name to `schedules:` in the config, and add a
display label in `scripts/make_tables.py` (the `DISP` dict) so it appears in the tables.

### Add a new config parameter

Add a field to the `Config` dataclass in `src/topoham/config.py` (with a default),
read it wherever it is used in `src/topoham/runner.py`, and set it in your YAML.

### Use the library in your own project

```python
from topoham.config import Config
from topoham.runner import run
from pathlib import Path

cfg = Config(families=["tfim"], sizes=[6], n_per_family=4, time=1.0, steps=8)
summary = run(cfg, Path("my_results"))
print(summary["headline"]["gate_speedup_vs_first_order"])
```

Lower-level building blocks are importable directly: `topoham.error_form`
(the matrix-free commutator error operator), `topoham.commutator_graph`,
`topoham.hamiltonians`, `topoham.schedules`, `topoham.policies` (learned router),
and `topoham.metrics`.

---

## Cite this work

If you use this software or its results, please cite the paper:

```bibtex
@article{huynh2026topoham,
  title   = {Zero-Overhead Cancellation of the Leading Trotter Error via Commutator-Graph Scheduling},
  author  = {Huynh, Molena},
  year    = {2026},
  note    = {Part of the spectral-truncation operators (SpecOps) program},
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## Reproducibility

All numbers, tables, and figures are script-generated from a single seeded run;
`results/summary.json` records full provenance (seed, command, platform, package
versions, runtime, peak memory). All experiments run on commodity CPU hardware.
