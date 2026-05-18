"""Discrete P, T, C operators in the chiral Dirac basis.

In the chiral basis used by ``spacetime_qca.dirac`` (signature ``(+---)``,
``gamma^0`` off-diagonal, ``gamma^i = gamma^0 alpha_i``):

- ``gamma^0`` is symmetric and real.
- ``gamma^1``, ``gamma^3`` are antisymmetric and real.
- ``gamma^2`` is symmetric and pure imaginary (``(gamma^2)* = -gamma^2``).

The conjugation conditions for the discrete symmetries reduce to:

- **P** (unitary): ``P gamma^0 P^{-1} = gamma^0``, ``P gamma^i P^{-1} = -gamma^i``.
  Solution: ``P_spinor = gamma^0``.

- **T** (antiunitary, includes time reversal): the antiunitary action is
  ``ψ → T_spinor · ψ*``.  Requiring T-invariance of the Dirac equation
  gives ``T_spinor (gamma^mu)* T_spinor^{-1} = (-1)^{mu_0} gamma^mu``.
  In our basis: anticommutes with ``gamma^0`` and ``gamma^2``, commutes
  with ``gamma^1`` and ``gamma^3``.  Solution: ``T_spinor = gamma^2 gamma^0``.

- **C** (antiunitary, no time reversal): the standard charge-conjugation
  condition is ``C gamma^mu C^{-1} = -(gamma^mu)^T``.  Computing transposes
  in our basis gives the same per-component conjugation pattern as T, so
  ``C_spinor = gamma^2 gamma^0`` as a matrix.  C and T differ in how they
  act on the walk (C commutes with U, T conjugates U to U^†).

This module exposes the spinor matrices, the conjugation-condition
predicates, and a ``SymmetryOperator`` dataclass that bundles (matrix,
antiunitary flag, time-reversal flag) for downstream walk-commutation
tests.
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
    """Return the antiunitary C spinor matrix ``C = gamma^2 gamma^0``.

    Same matrix as T in this chiral basis; the operational difference is
    the time-reversal flag, not the spinor matrix itself.
    """

    _, gamma2, _ = gamma_spatial_matrices()
    return (gamma2 * gamma0()).applyfunc(sp.simplify)


@dataclass(frozen=True)
class SymmetryOperator:
    """Discrete spacetime symmetry packaged for walk-commutation tests.

    Fields:
        name:           short identifier such as ``"P"`` or ``"CPT"``.
        spinor_matrix:  4x4 chiral-basis matrix acting on spinor components.
        antiunitary:    True if the operator includes complex conjugation.
        time_reverse:   True if the operator reverses walk direction.
        momentum_flip:  True if the operator flips momentum.

    Composite operators (``PT``, ``CP``, ``CT``, ``CPT``) are constructed
    by ``compose(...)`` rather than by direct instantiation.
    """

    name: str
    spinor_matrix: sp.Matrix
    antiunitary: bool
    time_reverse: bool
    momentum_flip: bool


def parity_operator() -> SymmetryOperator:
    return SymmetryOperator(
        name="P",
        spinor_matrix=parity_spinor(),
        antiunitary=False,
        time_reverse=False,
        momentum_flip=True,
    )


def time_reversal_operator() -> SymmetryOperator:
    return SymmetryOperator(
        name="T",
        spinor_matrix=time_reversal_spinor(),
        antiunitary=True,
        time_reverse=True,
        momentum_flip=True,
    )


def charge_conjugation_operator() -> SymmetryOperator:
    return SymmetryOperator(
        name="C",
        spinor_matrix=charge_conjugation_spinor(),
        antiunitary=True,
        time_reverse=False,
        momentum_flip=True,
    )


def compose(first: SymmetryOperator, second: SymmetryOperator, name: str) -> SymmetryOperator:
    """Compose ``second ∘ first`` (first applied first, then second).

    The composition rules:

    - antiunitary flags XOR (one K cancels another K to give unitary);
    - momentum flips XOR (two flips return to original momentum);
    - time-reverse flags XOR (two reversals cancel);
    - matrices multiply as ``M_second · M_first`` (but with complex
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
