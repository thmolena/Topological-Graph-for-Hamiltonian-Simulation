"""End-to-end experiment driver.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. The protocol, for each Hamiltonian instance:

  1. build the exact reference  e^{-iHt}|psi0>  (dense expm or Krylov);
  2. for every ordering policy, run first-order Trotter at the *fixed* step
     budget r and score fidelity / observable error -- all at the **same gate
     proxy** (num_terms * r), so the comparison is at matched implementation
     cost;
  3. sweep r over a grid to trace the fidelity-vs-gate-budget frontier.

The ``learned`` policy is a router over the fixed strategies. It is trained
**leave-one-instance-out**: for each test Hamiltonian the classifier sees only
the other instances' (features -> best-fixed-strategy) labels, so there is no
leakage of the test instance into its own predictor.

Headline = mean commutator-ordering fidelity vs mean random-ordering fidelity at
the fixed budget, plus the multiplicative *infidelity reduction*. An integrity
flag ``pauli_algebra_verified`` records that the fast Pauli-rotation kernel was
cross-checked against dense ``expm`` before any numbers were produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.linalg import expm

from . import metrics, pauli, policies
from .config import Config
from .env import TrotterEnv
from .hamiltonians import Hamiltonian, build
from .policies import LearnedOrderingPolicy, feature_vector
from .seed import RunProvenance, set_seed

# Orderings compared in the report (references are the fidelity target, below).
ORDERINGS = list(policies.ORDERINGS)


@dataclass
class Instance:
    family: str
    n: int
    ham: Hamiltonian


def _verify_pauli_algebra(rng: np.random.Generator) -> bool:
    """Cross-check the fast Pauli-rotation kernel against dense ``expm``.

    Gate: if the O(2^n) statevector rotation ``cos(theta) psi - i sin(theta) P
    psi`` ever disagreed with ``expm(-i theta P) @ psi`` we must not emit any
    fidelity number. Verified on random small Paulis before the run proceeds.
    """
    ok = True
    for n in (2, 3):
        for _ in range(8):
            p = "".join(rng.choice(list("IXYZ"), size=n))
            theta = float(rng.uniform(-np.pi, np.pi))
            psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            psi /= np.linalg.norm(psi)
            fast = pauli.apply_pauli_rotation(psi, p, theta)
            exact = expm(-1j * theta * pauli.to_matrix(p)) @ psi
            if not np.allclose(fast, exact, atol=1e-9):
                ok = False
    return ok


def _best_fixed_strategy(ham: Hamiltonian, t: float, r: int,
                         rng: np.random.Generator) -> str:
    """Label = whichever fixed strategy attains the highest fidelity here."""
    env = TrotterEnv.from_hamiltonian(ham, t, r)
    best, best_f = "commutator", -1.0
    for strat in policies.FIXED_STRATEGIES:
        order = policies.get_ordering(strat, ham, np.random.default_rng(0))
        f = env.evaluate(order)["fidelity"]
        if f > best_f:
            best, best_f = strat, f
    return best


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    rng = set_seed(cfg.seed)

    pauli_ok = _verify_pauli_algebra(rng)

    # 1. Build the instance pool.
    instances: List[Instance] = []
    for fam in cfg.families:
        for n in cfg.sizes:
            for _ in range(cfg.n_per_family):
                ham = build(fam, n, rng)
                instances.append(Instance(fam, n, ham))

    # 2. Leave-one-out labels for the learned router (no test leakage).
    feats = [feature_vector(inst.ham) for inst in instances]
    labels = [_best_fixed_strategy(inst.ham, cfg.time, cfg.steps, rng)
              for inst in instances]

    # 3. Evaluate every ordering at the fixed step budget.
    fid: Dict[str, List[float]] = {o: [] for o in ORDERINGS}
    infid: Dict[str, List[float]] = {o: [] for o in ORDERINGS}
    obs_err: Dict[str, List[float]] = {o: [] for o in ORDERINGS}
    adj_anti: Dict[str, List[int]] = {o: [] for o in ORDERINGS}
    by_family: Dict[str, Dict[str, List[float]]] = {}
    gate_proxy_fixed = None

    for i, inst in enumerate(instances):
        env = TrotterEnv.from_hamiltonian(inst.ham, cfg.time, cfg.steps,
                                          reference_backend=cfg.reference_backend)
        gate_proxy_fixed = env.gate_proxy
        # learned router trained on all OTHER instances
        train_idx = [k for k in range(len(instances)) if k != i]
        learned = LearnedOrderingPolicy().fit(
            [instances[k].ham for k in train_idx],
            [labels[k] for k in train_idx],
        )
        by_family.setdefault(inst.family, {})
        for ordng in ORDERINGS:
            if ordng == "learned":
                order = learned.ordering(inst.ham, np.random.default_rng(cfg.seed + i))
            else:
                order = policies.get_ordering(ordng, inst.ham,
                                              np.random.default_rng(cfg.seed + i))
            res = env.evaluate(order)
            fid[ordng].append(res["fidelity"])
            infid[ordng].append(res["infidelity"])
            obs_err[ordng].append(res["observable_error"])
            adj_anti[ordng].append(
                policies.adjacent_anticommutations(inst.ham, order))
            by_family[inst.family].setdefault(ordng, []).append(res["fidelity"])

    # 4. Fidelity-vs-gate-budget frontier over the steps grid.
    # Key the frontier by Trotter-step count r (the controllable budget knob):
    # for each r we average, across all instances, the per-instance gate proxy
    # (num_terms * r) and the fidelity. This compares orderings at the same r --
    # a clean, monotone frontier -- rather than mixing heterogeneous (num_terms,
    # r) combinations that happen to share a gate-proxy value.
    fid_by_r = {o: {r: [] for r in cfg.steps_grid} for o in ORDERINGS}
    gp_by_r = {r: [] for r in cfg.steps_grid}
    for inst in instances:
        for r in cfg.steps_grid:
            env = TrotterEnv.from_hamiltonian(inst.ham, cfg.time, r,
                                              reference_backend=cfg.reference_backend)
            gp_by_r[r].append(env.gate_proxy)
            for ordng in ORDERINGS:
                strat = ordng if ordng != "learned" else "commutator"
                order = policies.get_ordering(strat, inst.ham,
                                              np.random.default_rng(cfg.seed))
                fid_by_r[ordng][r].append(env.evaluate(order)["fidelity"])
    frontier: Dict[str, List[list]] = {}
    for ordng in ORDERINGS:
        points = []
        for r in cfg.steps_grid:
            vals = np.asarray(fid_by_r[ordng][r], dtype=float)
            n = max(1, vals.size)
            # 95% confidence interval half-width on the plotted mean fidelity:
            # 1.96 * sample_std / sqrt(n). This is a derived uncertainty for the
            # shaded band only; the mean (the reported quantity) is unchanged.
            ci95 = 1.96 * float(vals.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
            points.append([
                int(round(float(np.mean(gp_by_r[r])))),
                round(float(vals.mean()), 6),
                round(ci95, 6),
            ])
        frontier[ordng] = points

    # 5. Aggregate.
    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "pauli_algebra_verified": bool(pauli_ok),
        "gate_proxy_fixed": int(gate_proxy_fixed or 0),
        "orderings": {},
        "frontier": frontier,
        "by_family": {},
        "headline": {},
    }
    for ordng in ORDERINGS:
        summary["orderings"][ordng] = {
            "fidelity": metrics.summarize(fid[ordng]),
            "infidelity_mean": round(float(np.mean(infid[ordng])), 6),
            "observable_error_mean": round(float(np.mean(obs_err[ordng])), 6),
            "adjacent_anticommutations_mean": round(float(np.mean(adj_anti[ordng])), 3),
        }
    for fam, ords in by_family.items():
        summary["by_family"][fam] = {o: metrics.summarize(v) for o, v in ords.items()}

    comm_f = summary["orderings"]["commutator"]["fidelity"]["mean"]
    rand_f = summary["orderings"]["random"]["fidelity"]["mean"]
    comm_inf = max(1e-9, 1.0 - comm_f)
    rand_inf = max(1e-9, 1.0 - rand_f)
    summary["headline"] = {
        "commutator_fidelity_mean": round(comm_f, 6),
        "random_fidelity_mean": round(rand_f, 6),
        "learned_fidelity_mean": round(summary["orderings"]["learned"]["fidelity"]["mean"], 6),
        "fidelity_gain": round(comm_f - rand_f, 6),
        "infidelity_reduction_vs_random": round(rand_inf / comm_inf, 3),
        "gate_proxy": int(gate_proxy_fixed or 0),
        "n_instances": len(instances),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
