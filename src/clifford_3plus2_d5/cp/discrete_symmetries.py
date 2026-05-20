"""Discrete P, T, C operators in the chiral Dirac basis.

In the chiral basis used by ``spacetime_qca.dirac`` (signature ``(+---)``,
``gamma^0`` off-diagonal, ``gamma^i = gamma^0 alpha_i``):

- ``gamma^0`` is symmetric and real.
- ``gamma^1``, ``gamma^3`` are antisymmetric and real.
- ``gamma^2`` is symmetric and pure imaginary (``(gamma^2)* = -gamma^2``).

The conjugation conditions for the discrete symmetries:

- **P** (unitary): ``P gamma^0 P^{-1} = gamma^0``, ``P gamma^i P^{-1} = -gamma^i``.
  Solution: ``P_spinor = gamma^0``.

- **T** (antiunitary, time-reversing): the antiunitary action is ``ψ → T_spinor ψ*``.
  Requiring T-invariance of the Dirac equation gives
  ``T_spinor (gamma^mu)* T_spinor^{-1} = (-1)^{mu_0} gamma^mu``.
  In our basis: anticommutes with ``gamma^0`` and ``gamma^2``, commutes
  with ``gamma^1`` and ``gamma^3``.  Solution: ``T_spinor = gamma^2 gamma^0``.

- **C** (antiunitary, charge conjugation): the standard charge-conjugation
  matrix in the chiral basis is ``C_spinor = i gamma^2`` (acting as
  ``ψ → i gamma^2 ψ*``).  This is structurally distinct from T: C's
  conjugation pattern is ``(-1, -1, -1, -1)`` (anticommutes with every
  γ^μ after the antiunitary conjugation), while T's is ``(-1, 1, 1, 1)``.

**Important convention note** (2026-05-20 audit):

  The earlier version of this module used ``C_spinor = gamma^2 gamma^0``
  (same matrix as T).  That was a Bloch-level particle-hole-like
  operator, NOT standard physical charge conjugation.  This module
  now uses the standard ``i gamma^2`` for C; the previous matrix is
  still available as ``bloch_particle_hole_spinor`` for backward
  compatibility.

## Hamiltonian-sign convention

For each operator ``A``, ``A H(k) A^{-1} = s · H(k_image)`` where the
sign ``s ∈ {+1, -1}`` is the **hamiltonian_sign** field.  The walk-
symmetry test compares ``A · U(k) · A^{-1}`` to ``U(k_image)`` if
``XNOR(A.antiunitary, s = +1)`` is False (no dagger), or to
``U(k_image)†`` if XNOR is True (dagger).  This unifies the four
cases of (unitary/antiunitary) × (+H/−H) action on the Hamiltonian.

For the BCC Dirac walk with massless H = α·k, all named operators
turn out to have ``s = +1`` (the −H case appears for massive
Dirac with a β m term, but not in this audit).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.cp.reuse import (
    gamma0,
    gamma_spatial_matrices,
    same_matrix,
)


def parity_spinor() -> sp.Matrix:
    """Return the unitary parity spinor matrix ``P = gamma^0``."""

    return gamma0()


def time_reversal_spinor() -> sp.Matrix:
    """Return the antiunitary T spinor matrix ``T = gamma^2 gamma^0``."""

    _, gamma2, _ = gamma_spatial_matrices()
    return (gamma2 * gamma0()).applyfunc(sp.simplify)


def charge_conjugation_spinor() -> sp.Matrix:
    """Return the antiunitary C spinor matrix ``C = i gamma^2``.

    Standard chiral-basis charge conjugation: ``ψ → i gamma^2 ψ*``.
    The matrix satisfies ``(i gamma^2)^2 = I``, ``C gamma^μ C^{-1} =
    -(gamma^μ)^T`` (standard QFT condition).
    """

    _, gamma2, _ = gamma_spatial_matrices()
    return (sp.I * gamma2).applyfunc(sp.simplify)


def bloch_particle_hole_spinor() -> sp.Matrix:
    """Return the Bloch-level particle-hole operator ``M = gamma^2 gamma^0``.

    This was the pre-2026-05-20 implementation of `charge_conjugation_spinor`.
    Kept under a semantically explicit name for backward compatibility
    and audit traceability.  It is NOT standard physical charge
    conjugation; structurally it's a second T-like operator with the
    same matrix as T.
    """

    _, gamma2, _ = gamma_spatial_matrices()
    return (gamma2 * gamma0()).applyfunc(sp.simplify)


@dataclass(frozen=True)
class SymmetryOperator:
    """Discrete spacetime symmetry packaged for walk-commutation tests.

    Fields:
        name:               short identifier such as ``"P"`` or ``"CPT"``.
        spinor_matrix:      4x4 chiral-basis matrix acting on spinor components.
        antiunitary:        True if the operator includes complex conjugation.
        time_reverse:       True if the operator reverses walk direction.
        momentum_flip:      True if the operator flips momentum.
        hamiltonian_sign:   ±1; ``A H A^{-1} = sign · H(k_image)``.
        kind:               semantic label, e.g. ``"physical_P"``,
                            ``"physical_C"``, ``"composite_CP"``.

    Composite operators (``PT``, ``CP``, ``CT``, ``CPT``) are constructed
    by ``compose(...)`` rather than by direct instantiation.

    The walk-symmetry test (``walk_symmetries.walk_respects_symmetry``)
    uses the XNOR(antiunitary, hamiltonian_sign==+1) criterion to decide
    whether to compare to ``B(k_image)`` or ``B(k_image)†``.
    """

    name: str
    spinor_matrix: sp.Matrix
    antiunitary: bool
    time_reverse: bool
    momentum_flip: bool
    hamiltonian_sign: int
    kind: str


def parity_operator() -> SymmetryOperator:
    """Standard physical parity P, with M = γ⁰; A H A⁻¹ = +H(-k)."""

    return SymmetryOperator(
        name="P",
        spinor_matrix=parity_spinor(),
        antiunitary=False,
        time_reverse=False,
        momentum_flip=True,
        hamiltonian_sign=+1,
        kind="physical_P",
    )


def time_reversal_operator() -> SymmetryOperator:
    """Standard physical T, with M = γ²γ⁰; A H A⁻¹ = +H(-k)."""

    return SymmetryOperator(
        name="T",
        spinor_matrix=time_reversal_spinor(),
        antiunitary=True,
        time_reverse=True,
        momentum_flip=True,
        hamiltonian_sign=+1,
        kind="physical_T",
    )


def charge_conjugation_operator() -> SymmetryOperator:
    """Standard physical C, with M = i·γ²; A H A⁻¹ = +H(+k) for massless Dirac.

    For massive Dirac, the mass term β·m would flip sign under C,
    giving a mixed-sign Hamiltonian action.  For the massless walk
    treated here, sign = +1.
    """

    return SymmetryOperator(
        name="C",
        spinor_matrix=charge_conjugation_spinor(),
        antiunitary=True,
        time_reverse=False,
        momentum_flip=False,
        hamiltonian_sign=+1,
        kind="physical_C",
    )


def compose(first: SymmetryOperator, second: SymmetryOperator, name: str) -> SymmetryOperator:
    """Compose ``second ∘ first`` (first applied first, then second).

    The composition rules:

    - antiunitary flags XOR (one K cancels another K to give unitary);
    - momentum flips XOR (two flips return to original momentum);
    - time-reverse flags XOR (two reversals cancel);
    - hamiltonian_sign multiplies (signs compose multiplicatively);
    - matrices multiply as ``M_second · M_first`` (with complex
      conjugation of the inner matrix if ``second`` is antiunitary).
    """

    if second.antiunitary:
        inner = first.spinor_matrix.applyfunc(sp.conjugate)
    else:
        inner = first.spinor_matrix
    matrix = (second.spinor_matrix * inner).applyfunc(sp.simplify)
    return SymmetryOperator(
        name=name,
        spinor_matrix=matrix,
        antiunitary=first.antiunitary ^ second.antiunitary,
        time_reverse=first.time_reverse ^ second.time_reverse,
        momentum_flip=first.momentum_flip ^ second.momentum_flip,
        hamiltonian_sign=first.hamiltonian_sign * second.hamiltonian_sign,
        kind=f"composite_{name}",
    )


def cpt_operator() -> SymmetryOperator:
    return compose(
        compose(charge_conjugation_operator(), parity_operator(), name="CP"),
        time_reversal_operator(),
        name="CPT",
    )


def cp_operator() -> SymmetryOperator:
    return compose(charge_conjugation_operator(), parity_operator(), name="CP")


def ct_operator() -> SymmetryOperator:
    return compose(charge_conjugation_operator(), time_reversal_operator(), name="CT")


def pt_operator() -> SymmetryOperator:
    return compose(parity_operator(), time_reversal_operator(), name="PT")


def all_seven_operators() -> tuple[SymmetryOperator, ...]:
    """Return the seven physically distinct discrete symmetries."""

    p = parity_operator()
    t = time_reversal_operator()
    c = charge_conjugation_operator()
    return (
        p,
        t,
        c,
        pt_operator(),
        cp_operator(),
        ct_operator(),
        cpt_operator(),
    )


def conjugation_pattern(operator: SymmetryOperator) -> tuple[int, int, int, int]:
    """Return ``(s_0, s_1, s_2, s_3)`` with ``s_μ = +1`` if the operator
    commutes with ``γ^μ`` (after the antiunitary conjugation if applicable)
    and ``-1`` if it anticommutes.
    """

    gammas = (gamma0(), *gamma_spatial_matrices())
    matrix = operator.spinor_matrix
    inverse = matrix.inv().applyfunc(sp.simplify)
    pattern: list[int] = []
    for gamma in gammas:
        target = gamma.applyfunc(sp.conjugate) if operator.antiunitary else gamma
        conjugated = (matrix * target * inverse).applyfunc(sp.simplify)
        if same_matrix(conjugated, gamma):
            pattern.append(1)
        elif same_matrix(conjugated, -gamma):
            pattern.append(-1)
        else:
            raise RuntimeError(
                f"operator {operator.name} does not produce ±gamma^mu under "
                f"conjugation; got {conjugated}",
            )
    return tuple(pattern)  # type: ignore[return-value]
