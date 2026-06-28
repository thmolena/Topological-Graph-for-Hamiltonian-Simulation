"""End-to-end experiment driver for the scheduling study.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. The protocol, for each Hamiltonian instance:

  1. build the exact reference  e^{-iHt}|psi0>  (dense expm or Krylov);
  2. for every schedule and every step count r in the grid, run the product
     formula and score fidelity at the *realised rotation count* (the gate cost);
  3. from the (gates, fidelity) curves compute the gates-to-target-fidelity (the
     headline matched-cost efficiency metric) and the fidelity-vs-gate frontier;
  4. quantify ordering impotence -- how little any reordering moves the leading
     commutator-error operator norm (Theorem 1) -- and fit the convergence rate
     of each schedule (first-order ~1/r^2 vs folded ~1/r^4, Theorem 2).

Schedules generalise term orderings: ``random``/``coefficient``/``commutator`` are
first-order; ``antithetic`` is the zero-overhead step-alternating fold (identical
L*r rotations) and ``symmetric`` is the Strang half-step fold. The ``learned``
policy is a router over the fixed schedules, trained leave-one-instance-out.

Two integrity flags are recorded before any number is emitted: the fast
Pauli-rotation kernel matches dense ``expm`` (``pauli_algebra_verified``) and the
matrix-free commutator error form matches the dense pair-commutator sum
(``error_form_verified``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import expm

from . import error_form, metrics, pauli, policies, schedules
from .config import Config
from .env import TrotterEnv
from .hamiltonians import Hamiltonian, build
from .policies import LearnedSchedulePolicy, feature_vector
from .seed import RunProvenance, set_seed

FIXED = list(schedules.FIXED_SCHEDULES)
ALL_SCHEDULES = FIXED + ["learned"]


@dataclass
class Instance:
    family: str
    n: int
    ham: Hamiltonian


def _verify_pauli_algebra(rng: np.random.Generator) -> bool:
    """Cross-check the fast Pauli-rotation kernel against dense ``expm``."""
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


def _verify_reference_backend(backend: str, rng: np.random.Generator,
                              tol: float = 1e-9) -> bool:
    """Cross-check the scalable matrix-free reference against dense ``expm``.

    The Krylov reference (``expm_multiply`` on a matrix-free sparse Hamiltonian)
    is what extends the exact benchmark past the dense-``expm`` ceiling; this gate
    confirms it reproduces dense ``expm`` to ``tol`` on small systems before it is
    trusted at scale (records ``reference_backend_verified``).
    """
    if backend == "expm":
        return True
    from .env import reference_state
    from .hamiltonians import build, FAMILIES
    ok = True
    for fam in FAMILIES:
        for n in (3, 4, 5):
            ham = build(fam, n, rng)
            psi0 = pauli.plus_state(n)
            a = reference_state(ham, 1.0, psi0, backend="expm")
            b = reference_state(ham, 1.0, psi0, backend=backend)
            if not np.allclose(a, b, atol=tol):
                ok = False
    return ok


def _gates_to_target(curve: Dict[int, tuple], target: float) -> Optional[int]:
    """First (smallest-r) rotation count at which mean fidelity reaches ``target``.

    ``curve`` maps r -> (mean_gates, mean_fidelity). Scanning by ascending r keeps
    the matched-cost reading monotone in the controllable budget knob.
    """
    for r in sorted(curve):
        g, f = curve[r]
        if f >= target:
            return int(round(g))
    return None


def _fit_slope(rs: List[int], infids: List[float], even_only: bool = True) -> float:
    """Log-log convergence rate -d(log infidelity)/d(log r) over the asymptotic
    (upper) part of the grid: first-order ~2, second-order folds ~4. Restricting
    to even r keeps the antithetic fold (whose step-pairs are palindromic only for
    even r) in its clean second-order regime."""
    pts = [(r, i) for r, i in zip(rs, infids)
           if i > 1e-11 and i < 0.5 and r > 1 and (r % 2 == 0 or not even_only)]
    pts = pts[len(pts) // 3:] if len(pts) >= 4 else pts
    if len(pts) < 2:
        return float("nan")
    x = np.log(np.array([r for r, _ in pts], dtype=float))
    y = np.log(np.array([i for _, i in pts], dtype=float))
    return float(-np.polyfit(x, y, 1)[0])


def _mean_instance_slope(per_inst, idxs, grid, s):
    """Mean over instances of the per-instance convergence rate (robust to mixing
    exactly-solved instances, which are skipped)."""
    vals = []
    for i in idxs:
        infids = [1.0 - per_inst[i][s][r][0] for r in grid]
        sl = _fit_slope(grid, infids)
        if not math.isnan(sl):
            vals.append(sl)
    return round(float(np.mean(vals)), 3) if vals else float("nan")


def _best_schedule_label(rec: Dict[str, Dict[int, tuple]], target: float) -> str:
    """The fixed schedule reaching ``target`` at fewest rotations on this instance
    (or, if none reach it, the one with the highest attained fidelity).

    ``rec[s]`` maps r -> (fidelity, gates) for a single instance.
    """
    best, best_key = FIXED[0], (2, float("inf"), float("inf"))
    for s in FIXED:
        curve = rec[s]
        reached = None
        for r in sorted(curve):
            f, gates = curve[r]
            if f >= target:
                reached = gates
                break
        if reached is not None:
            key = (0, int(reached), 0.0)
        else:
            max_fid = max(f for f, _ in curve.values())
            min_gates = min(g for _, g in curve.values())
            key = (1, -max_fid, int(min_gates))
        if key < best_key:
            best, best_key = s, key
    return best


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    rng = set_seed(cfg.seed)

    pauli_ok = _verify_pauli_algebra(rng)
    form_ok = error_form.verify_error_form(np.random.default_rng(cfg.seed + 1))
    ref_ok = _verify_reference_backend(cfg.reference_backend,
                                       np.random.default_rng(cfg.seed + 2))

    # 1. Build the instance pool.
    instances: List[Instance] = []
    for fam in cfg.families:
        for n in cfg.sizes:
            for _ in range(cfg.n_per_family):
                ham = build(fam, n, rng)
                instances.append(Instance(fam, n, ham))

    grid = list(cfg.steps_grid)
    r0 = cfg.steps

    # 2. Per-instance (schedule, r) -> (fidelity, gates); plus impotence + features.
    per_inst: List[Dict[str, Dict[int, tuple]]] = []
    impotence_rows: List[Dict[str, float]] = []
    inst_runtime: List[float] = []
    feats = [feature_vector(inst.ham) for inst in instances]
    for i, inst in enumerate(instances):
        t_inst = time.perf_counter()
        env = TrotterEnv.from_hamiltonian(inst.ham, cfg.time, 1,
                                          reference_backend=cfg.reference_backend)
        rec: Dict[str, Dict[int, tuple]] = {s: {} for s in FIXED}
        for s in FIXED:
            for r in grid:
                res = env.evaluate_schedule(s, r, np.random.default_rng(cfg.seed + i))
                rec[s][r] = (res["fidelity"], res["gates"])
        per_inst.append(rec)
        inst_runtime.append(time.perf_counter() - t_inst)
        # The exact (dense) spectral-norm impotence sweep is O((2^n)^3) over O(L^2)
        # dense pair commutators; restrict it to the small-n subset and rely on the
        # fully matrix-free collision structure + HS surrogate (both O(L^2 n)) to
        # carry Theorem 1 at larger n.
        if inst.n <= cfg.impotence_max_n:
            impotence_rows.append(
                error_form.ordering_impotence(
                    inst.ham, np.random.default_rng(cfg.seed + 7 + i),
                    n_samples=cfg.impotence_samples))

    # 3. Leave-one-instance-out learned router over schedules.
    labels = [_best_schedule_label(rec, cfg.target_ref) for rec in per_inst]
    learned_choice: List[str] = []
    for i in range(len(instances)):
        train_idx = [k for k in range(len(instances)) if k != i]
        router = LearnedSchedulePolicy().fit(
            [instances[k].ham for k in train_idx], [labels[k] for k in train_idx])
        learned_choice.append(router.predict_schedule(instances[i].ham))

    # Attach the learned schedule's own curve to each instance record.
    for i, rec in enumerate(per_inst):
        rec["learned"] = dict(rec[learned_choice[i]])

    # 4. Aggregate (schedule, r) -> mean gates, mean/ci fidelity, mean infidelity.
    def agg(idxs: List[int]):
        out: Dict[str, Dict[int, dict]] = {s: {} for s in ALL_SCHEDULES}
        for s in ALL_SCHEDULES:
            for r in grid:
                fids = np.array([per_inst[i][s][r][0] for i in idxs], dtype=float)
                gates = np.array([per_inst[i][s][r][1] for i in idxs], dtype=float)
                n = max(1, fids.size)
                ci = 1.96 * float(fids.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
                out[s][r] = {
                    "gates": float(gates.mean()),
                    "fid": float(fids.mean()), "fid_ci": ci,
                    "infid": float((1.0 - fids).mean()),
                }
        return out

    all_idx = list(range(len(instances)))
    agg_all = agg(all_idx)
    by_fam_idx: Dict[str, List[int]] = {}
    for i, inst in enumerate(instances):
        by_fam_idx.setdefault(inst.family, []).append(i)
    agg_fam = {fam: agg(idxs) for fam, idxs in by_fam_idx.items()}

    def curve_of(a, s):  # r -> (gates, fid) for gates-to-target
        return {r: (a[s][r]["gates"], a[s][r]["fid"]) for r in grid}

    # Headline: gates-to-target-fidelity over the aggregate curve. Computed for
    # the FIXED schedules only -- they have a consistent rotation count at each r,
    # so the matched-cost reading is well defined. The learned router (whose pick
    # varies by instance) is reported per-instance below instead.
    gates_to_target = {
        "targets": list(cfg.targets),
        "overall": {s: {f"{tg}": _gates_to_target(curve_of(agg_all, s), tg)
                        for tg in cfg.targets} for s in FIXED},
        "by_family": {fam: {s: {f"{tg}": _gates_to_target(curve_of(a, s), tg)
                                for tg in cfg.targets} for s in FIXED}
                      for fam, a in agg_fam.items()},
    }

    # Scaling with system size n: the central substance claim is that the
    # convergence-order doubling (first-order p~2 -> antithetic p~4) and the
    # matched-cost speedup are *size independent*, while the matrix-free reference
    # keeps the exact benchmark tractable. We aggregate slope, gates-to-target and
    # wall-clock per size.
    by_size_idx: Dict[int, List[int]] = {}
    for i, inst in enumerate(instances):
        by_size_idx.setdefault(inst.n, []).append(i)
    sizes_sorted = sorted(by_size_idx)
    agg_size = {nq: agg(by_size_idx[nq]) for nq in sizes_sorted}

    def _g2t_size(nq: int, s: str) -> Optional[int]:
        return _gates_to_target(curve_of(agg_size[nq], s), cfg.target_ref)

    by_size: Dict = {
        "target": cfg.target_ref,
        "sizes": sizes_sorted,
        "n_per_size": {nq: len(by_size_idx[nq]) for nq in sizes_sorted},
        "slope_first": {nq: _mean_instance_slope(per_inst, by_size_idx[nq], grid, "coefficient")
                        for nq in sizes_sorted},
        "slope_antithetic": {nq: _mean_instance_slope(per_inst, by_size_idx[nq], grid, "antithetic")
                             for nq in sizes_sorted},
        "slope_symmetric": {nq: _mean_instance_slope(per_inst, by_size_idx[nq], grid, "symmetric")
                            for nq in sizes_sorted},
        "gates_first": {nq: _g2t_size(nq, "coefficient") for nq in sizes_sorted},
        "gates_antithetic": {nq: _g2t_size(nq, "antithetic") for nq in sizes_sorted},
        "gates_symmetric": {nq: _g2t_size(nq, "symmetric") for nq in sizes_sorted},
        "runtime_sec": {nq: round(float(np.mean([inst_runtime[i] for i in by_size_idx[nq]])), 4)
                        for nq in sizes_sorted},
    }

    def _speedup_size(nq: int) -> Optional[float]:
        gf = by_size["gates_first"][nq]
        folded = [g for g in (by_size["gates_antithetic"][nq],
                              by_size["gates_symmetric"][nq]) if g is not None]
        gb = min(folded) if folded else None
        return round(gf / gb, 2) if (gf and gb) else None

    by_size["speedup"] = {nq: _speedup_size(nq) for nq in sizes_sorted}

    # Per-instance gates-to-target at the reference target: the fair way to score
    # the instance-adaptive learned router against the per-instance oracle and the
    # fixed schedules, on the subset every fixed schedule reaches.
    def inst_g2t(rec_s):
        for r in grid:
            f, g = rec_s[r]
            if f >= cfg.target_ref:
                return int(g)
        return None
    ig = {s: [inst_g2t(per_inst[i][s]) for i in all_idx]
          for s in ALL_SCHEDULES}
    oracle = []
    for k in range(len(instances)):
        cand = [ig[s][k] for s in FIXED if ig[s][k] is not None]
        oracle.append(min(cand) if cand else None)
    common = [k for k in range(len(instances)) if all(ig[s][k] is not None for s in FIXED)]
    per_instance_g2t = {
        "target": cfg.target_ref,
        "n_instances": len(instances),
        "common_subset_n": len(common),
        "mean_gates_common": {
            s: round(float(np.mean([ig[s][k] for k in common])), 1) if common else None
            for s in ALL_SCHEDULES},
        "oracle_mean_gates_common": round(float(np.mean([oracle[k] for k in common])), 1)
        if common else None,
        "reach_rate": {
            s: round(float(np.mean([ig[s][k] is not None for k in all_idx])), 3)
            for s in ALL_SCHEDULES},
    }

    # Convergence (infidelity vs r) + fitted convergence rates (Theorem 2).
    # The reported rate is the mean per-instance log-log slope (robust to mixing
    # exactly-solved instances); the aggregate curve is kept for the figure.
    convergence = {"r_grid": grid, "schedules": {}}
    slopes = {}
    slopes_by_family = {}
    for s in ALL_SCHEDULES:
        rows = [{"r": r, "gates": round(agg_all[s][r]["gates"], 2),
                 "fid": round(agg_all[s][r]["fid"], 6),
                 "fid_ci": round(agg_all[s][r]["fid_ci"], 6),
                 "infid": round(agg_all[s][r]["infid"], 8)} for r in grid]
        convergence["schedules"][s] = rows
        slopes[s] = _mean_instance_slope(per_inst, all_idx, grid, s)
        slopes_by_family[s] = {fam: _mean_instance_slope(per_inst, idxs, grid, s)
                               for fam, idxs in by_fam_idx.items()}

    # Frontier (fidelity vs realised total rotation count).
    frontier = {s: [[int(round(agg_all[s][r]["gates"])),
                     round(agg_all[s][r]["fid"], 6),
                     round(agg_all[s][r]["fid_ci"], 6)] for r in grid]
                for s in ALL_SCHEDULES}

    # Matched-budget snapshot at the reference step count r0.
    sched_table = {}
    for s in ALL_SCHEDULES:
        fids = np.array([per_inst[i][s][r0][0] for i in all_idx], dtype=float)
        gates = np.array([per_inst[i][s][r0][1] for i in all_idx], dtype=float)
        sched_table[s] = {
            "fidelity": metrics.summarize(fids.tolist()),
            "infidelity_mean": round(float((1.0 - fids).mean()), 6),
            "gates_mean": round(float(gates.mean()), 2),
            "slope": slopes[s],
        }

    by_family = {fam: {s: metrics.summarize([per_inst[i][s][r0][0] for i in idxs])
                       for s in ALL_SCHEDULES}
                 for fam, idxs in by_fam_idx.items()}

    # Ordering impotence (Theorem 1): how little reordering moves ||E_pi||.
    def col(key):
        return np.array([row[key] for row in impotence_rows], dtype=float)
    imp = {
        "ratio_coeff_to_min_mean": round(float(col("ratio_coeff_to_min").mean()), 4),
        "ratio_median_to_min_mean": round(float(col("ratio_median_to_min").mean()), 4),
        "ratio_coeff_to_min_max": round(float(col("ratio_coeff_to_min").max()), 4),
        "hs_ratio_max_to_min_mean": round(float(col("hs_ratio_max_to_min").mean()), 4),
        "n_instances": len(impotence_rows),
        "max_n": cfg.impotence_max_n,
    }
    # Collision structure per family (the resource that *is* ordering-reducible).
    collisions_by_family = {}
    for fam, idxs in by_fam_idx.items():
        cs = [error_form.collision_stats(instances[i].ham) for i in idxs]
        collisions_by_family[fam] = {
            "n_anti_pairs": round(float(np.mean([c["n_anti_pairs"] for c in cs])), 2),
            "colliding_pairs": round(float(np.mean([c["colliding_pairs"] for c in cs])), 2),
            "irreducible_frac": round(float(np.mean([c["irreducible_frac"] for c in cs])), 4),
        }

    # Learned router agreement with the per-instance optimum.
    learned_acc = round(float(np.mean([learned_choice[i] == labels[i]
                                       for i in all_idx])), 4)

    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "pauli_algebra_verified": bool(pauli_ok),
        "error_form_verified": bool(form_ok),
        "reference_backend_verified": bool(ref_ok),
        "schedules_table": sched_table,
        "gates_to_target": gates_to_target,
        "by_size": by_size,
        "per_instance_gates_to_target": per_instance_g2t,
        "convergence": convergence,
        "slopes": slopes,
        "slopes_by_family": slopes_by_family,
        "frontier": frontier,
        "by_family": by_family,
        "ordering_impotence": imp,
        "collisions_by_family": collisions_by_family,
        "learned": {
            "label_distribution": {s: int(labels.count(s)) for s in FIXED},
            "choice_distribution": {s: int(learned_choice.count(s)) for s in FIXED},
            "leave_one_out_accuracy": learned_acc,
        },
        "headline": {},
    }

    # Headline macros.
    tgt = cfg.target_ref
    g_first = gates_to_target["overall"]["coefficient"][f"{tgt}"]
    g_anti = gates_to_target["overall"]["antithetic"][f"{tgt}"]
    g_sym = gates_to_target["overall"]["symmetric"][f"{tgt}"]
    best_folded = min([g for g in (g_anti, g_sym) if g is not None], default=None)
    speedup = (round(g_first / best_folded, 2)
               if (g_first and best_folded) else None)
    summary["headline"] = {
        "n_instances": len(instances),
        "time": cfg.time,
        "target": tgt,
        "gates_to_target_first_order": g_first,
        "gates_to_target_antithetic": g_anti,
        "gates_to_target_symmetric": g_sym,
        "gate_speedup_vs_first_order": speedup,
        "learned_mean_gates_common": per_instance_g2t["mean_gates_common"].get("learned"),
        "oracle_mean_gates_common": per_instance_g2t["oracle_mean_gates_common"],
        "symmetric_mean_gates_common": per_instance_g2t["mean_gates_common"].get("symmetric"),
        "learned_reach_rate": per_instance_g2t["reach_rate"].get("learned"),
        "slope_first_order": slopes["coefficient"],
        "slope_antithetic": slopes["antithetic"],
        "slope_symmetric": slopes["symmetric"],
        "ordering_impotence_ratio": imp["ratio_median_to_min_mean"],
        "error_form_verified": bool(form_ok),
        "reference_backend": cfg.reference_backend,
        "reference_backend_verified": bool(ref_ok),
        "min_n": int(min(sizes_sorted)),
        "max_n": int(max(sizes_sorted)),
        "n_sizes": len(sizes_sorted),
        "slope_first_maxn": by_size["slope_first"][max(sizes_sorted)],
        "slope_antithetic_maxn": by_size["slope_antithetic"][max(sizes_sorted)],
        "speedup_maxn": by_size["speedup"][max(sizes_sorted)],
        "impotence_max_n": cfg.impotence_max_n,
        "impotence_n_instances": len(impotence_rows),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
