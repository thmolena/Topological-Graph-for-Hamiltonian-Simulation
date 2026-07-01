"""First-order Trotter simulation environment.

Given a Hamiltonian ``H = sum_j c_j P_j``, an evolution time ``t``, a step count
``r`` and an **ordering** ``pi`` of the terms, the first-order (Lie-Trotter)
approximation of :math:`e^{-iHt}` is

.. math::

   U_{\\mathrm{trot}}(\\pi)=\\Big(\\prod_{j\\in\\pi} e^{-i\\,c_j P_j\\, t/r}\\Big)^{\\,r},

applied to an initial state by :func:`trotter_state` using the exact single-Pauli
rotation kernel (``pauli.apply_pauli_rotation``) -- **no dense matrix of**
:math:`U` **is ever formed**. The implementation cost (the *gate proxy*) is
``len(terms) * r`` and is **independent of the ordering**, so comparing orderings
at fixed ``r`` is a comparison at matched cost.

The exact reference :func:`reference_state` uses ``scipy.linalg.expm`` (dense, the
ground truth for small ``n``) or ``scipy.sparse.linalg.expm_multiply`` (a Krylov
matrix-free reference). The reward is the fidelity of the Trotter state against
that reference -- a number in ``[0, 1]`` that the ordering policies maximise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from . import metrics, pauli
from .hamiltonians import Hamiltonian


def _popcount(x: np.ndarray) -> np.ndarray:
    """Per-element population count for a non-negative int64 array.

    ``np.bitwise_count`` only exists in NumPy >= 2.0; this SWAR fallback gives
    identical results on any supported NumPy (>= 1.24). Used to compute the Pauli
    ``Z``-string parity :math:`(-1)^{\\text{popcount}(x)}` when assembling the
    matrix-free sparse Hamiltonian.
    """
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = x.astype(np.int64, copy=True)
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


# ---------------------------------------------------------------------------
# Exact / reference evolution
# ---------------------------------------------------------------------------
def dense_hamiltonian(ham: Hamiltonian) -> np.ndarray:
    """Assemble the dense :math:`2^n \\times 2^n` Hermitian matrix of ``H``.

    Only used to *build references* on small ``n``; the Trotter path never
    touches it.
    """
    dim = 1 << ham.n
    H = np.zeros((dim, dim), dtype=complex)
    for c, p in ham.terms:
        H += c * pauli.to_matrix(p)
    return H


def sparse_pauli_matrix(p: str) -> sp.csr_matrix:
    r"""Sparse :math:`2^n\times2^n` matrix of a Pauli string, built matrix-free.

    A Pauli string is a signed permutation: it sends each computational basis
    state :math:`|x\rangle` to a unit-modulus phase times :math:`|x\oplus f\rangle`,
    so its matrix has exactly one nonzero per column. We construct that pattern
    directly from index arithmetic in :math:`O(2^n)` time and memory -- **never
    forming a dense** :math:`2^n\times2^n` **array** -- which is what lets the
    Krylov reference scale past the dense-``expm`` ceiling. ``X`` and ``Y`` flip
    the qubit bit; ``Y`` and ``Z`` contribute a :math:`(-1)^{x_q}` phase, and each
    ``Y`` contributes a global factor of :math:`i`.
    """
    n = len(p)
    dim = 1 << n
    flip = 0            # bits flipped by X and Y
    zy = 0              # bits carrying a (-1)^{x_q} phase (Y and Z)
    n_y = 0             # number of Y factors (global i^{n_y})
    for q, ch in enumerate(p):
        bit = 1 << (n - 1 - q)          # qubit q is bit (n-1-q), big-endian
        if ch == "X":
            flip |= bit
        elif ch == "Y":
            flip |= bit
            zy |= bit
            n_y += 1
        elif ch == "Z":
            zy |= bit
    cols = np.arange(dim, dtype=np.int64)
    rows = cols ^ flip
    parity = _popcount(cols & zy) & 1
    vals = ((1j) ** n_y) * np.where(parity == 0, 1.0, -1.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))


def sparse_hamiltonian(ham: Hamiltonian) -> sp.csr_matrix:
    """Sparse CSR Hermitian matrix of ``H``, assembled matrix-free.

    Each Pauli term is a one-nonzero-per-column signed permutation
    (:func:`sparse_pauli_matrix`); summing ``c_j P_j`` gives ``H`` without ever
    forming a dense matrix, so the Krylov reference (``expm_multiply``) scales to
    large ``n`` where dense ``expm`` is intractable.
    """
    dim = 1 << ham.n
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for c, p in ham.terms:
        H = H + c * sparse_pauli_matrix(p)
    return H


def reference_state(ham: Hamiltonian, t: float, psi0: np.ndarray,
                    backend: str = "expm") -> np.ndarray:
    r"""Exact :math:`e^{-iHt}\,|\psi_0\rangle` via dense ``expm`` or Krylov.

    ``backend="expm"``  -> ``scipy.linalg.expm`` (dense, exact).
    ``backend="krylov"`` -> ``scipy.sparse.linalg.expm_multiply`` (matrix-free).
    Both agree to machine precision; ``krylov`` is the scalable path.
    """
    if backend == "expm":
        U = expm(-1j * t * dense_hamiltonian(ham))
        return U @ psi0
    if backend == "krylov":
        return expm_multiply(-1j * t * sparse_hamiltonian(ham), psi0)
    raise ValueError(f"unknown reference backend {backend!r}")


# ---------------------------------------------------------------------------
# Trotter evolution under a term ordering
# ---------------------------------------------------------------------------
def trotter_state(ham: Hamiltonian, t: float, r: int, ordering: Sequence[int],
                  psi0: np.ndarray) -> np.ndarray:
    r"""Apply the first-order Trotter approximant under ``ordering`` to ``psi0``.

    ``ordering`` is a permutation of ``range(len(ham))``: the sequence in which
    the per-term rotations :math:`e^{-i c_j P_j t/r}` are applied within each of
    the ``r`` steps. Returns the evolved statevector.
    """
    if sorted(ordering) != list(range(len(ham))):
        raise ValueError("ordering must be a permutation of the term indices")
    dt = t / r
    psi = np.asarray(psi0, dtype=complex).copy()
    terms = ham.terms
    for _ in range(r):
        for j in ordering:
            c, p = terms[j]
            psi = pauli.apply_pauli_rotation(psi, p, c * dt)
    return psi


@dataclass
class TrotterEnv:
    """Holds a Hamiltonian, an evolution time/step budget and a probe state.

    One :meth:`evaluate` call runs a Trotter schedule under a given ordering and
    scores it against the cached exact reference. The gate proxy is fixed by
    ``(n_terms, r)`` and reported alongside every evaluation so orderings are
    always compared at matched implementation cost.
    """

    ham: Hamiltonian
    t: float
    r: int
    psi0: np.ndarray
    reference_backend: str = "expm"
    _ref: np.ndarray = field(default=None, repr=False)

    @classmethod
    def from_hamiltonian(cls, ham: Hamiltonian, t: float, r: int,
                         psi0: np.ndarray | None = None,
                         reference_backend: str = "expm") -> "TrotterEnv":
        if psi0 is None:
            psi0 = pauli.plus_state(ham.n)
        env = cls(ham=ham, t=t, r=r, psi0=np.asarray(psi0, dtype=complex),
                  reference_backend=reference_backend)
        env._ref = reference_state(ham, t, env.psi0, reference_backend)
        return env

    @property
    def reference(self) -> np.ndarray:
        if self._ref is None:
            self._ref = reference_state(self.ham, self.t, self.psi0,
                                        self.reference_backend)
        return self._ref

    @property
    def gate_proxy(self) -> int:
        return metrics.gate_proxy(len(self.ham), self.r)

    def evaluate(self, ordering: Sequence[int]) -> dict:
        """Run the Trotter schedule under ``ordering`` and score it.

        Returns ``fidelity`` (the reward), ``infidelity``, ``observable_error``
        (on :math:`Z_0`) and the fixed ``gate_proxy``.
        """
        psi = trotter_state(self.ham, self.t, self.r, ordering, self.psi0)
        return {
            "fidelity": metrics.fidelity(self.reference, psi),
            "infidelity": metrics.infidelity(self.reference, psi),
            "observable_error": metrics.observable_error(self.reference, psi),
            "gate_proxy": self.gate_proxy,
        }

    def evaluate_schedule(self, name: str, r: int,
                          rng: "np.random.Generator | None" = None) -> dict:
        """Run a named *schedule* at step count ``r`` and score it.

        Schedules generalise orderings: the per-step ordering may alternate
        direction (``antithetic``) or fold symmetrically (``symmetric``). The
        cost reported is the realised rotation count ``gates`` (the matched-cost
        x-axis), not a proxy. Uses the cached exact reference.
        """
        from . import schedules
        if rng is None:
            rng = np.random.default_rng(0)
        flat, _ = schedules.build_flat(name, self.ham, r, rng)
        psi, gates = schedules.apply_flat(self.ham, self.t, r, flat, self.psi0)
        return {
            "fidelity": metrics.fidelity(self.reference, psi),
            "infidelity": metrics.infidelity(self.reference, psi),
            "observable_error": metrics.observable_error(self.reference, psi),
            "gates": int(gates),
        }
