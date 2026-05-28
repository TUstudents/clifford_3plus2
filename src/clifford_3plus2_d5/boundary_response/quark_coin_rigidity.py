"""V15 primitive quark coin rigidity theorem.

The V11 flat quark coin uses one even channel and five Clifford-odd channels.
V15 proves the exact one-parameter family behind that construction:

    B(r) = (I + i r Gamma_q) / sqrt(1 + 5 r^2)

with ``Gamma_q^2 = 5 I``.  The positive-branch phase is

    delta(r) = atan(r sqrt(5)).

Thus the CKM phase ``atan(sqrt(5))`` is equivalent to the flat ratio ``r = 1``.
It is not forced by unitarity and Clifford closure alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_coin,
    quark_boundary_phase_angle,
    quark_gamma_sum,
)


def isotropic_quark_coin(ratio: sp.Expr) -> sp.Matrix:
    """Return the exact isotropic even/odd quark coin ``B(r)``."""

    r = sp.sympify(ratio)
    gamma_sum = quark_gamma_sum()
    normalizer = sp.sqrt(1 + 5 * r**2)
    return sp.simplify((sp.eye(gamma_sum.rows) + sp.I * r * gamma_sum) / normalizer)


def isotropic_quark_phase_factor(ratio: sp.Expr) -> sp.Expr:
    """Return the positive-branch scalar phase factor for ``B(r)``."""

    r = sp.sympify(ratio)
    return sp.simplify((1 + sp.I * r * sp.sqrt(5)) / sp.sqrt(1 + 5 * r**2))


def isotropic_quark_phase_angle(ratio: sp.Expr) -> sp.Expr:
    """Return the positive-branch phase angle ``delta(r)``."""

    return sp.atan(sp.simplify(sp.sympify(ratio) * sp.sqrt(5)))


def flatness_ratio_for_phase(phase_angle: sp.Expr) -> sp.Expr:
    """Return the even/odd ratio required to produce ``phase_angle``."""

    return sp.simplify(sp.tan(phase_angle) / sp.sqrt(5))


def gamma_sum_square_is_five() -> bool:
    """Return true when ``Gamma_q^2 = 5I`` exactly."""

    gamma_sum = quark_gamma_sum()
    residual = gamma_sum * gamma_sum - 5 * sp.eye(gamma_sum.rows)
    return all(sp.simplify(entry) == 0 for entry in residual)


def isotropic_coin_is_unitary(ratio: sp.Expr) -> bool:
    """Return true when ``B(r)`` is exactly unitary for a real ratio."""

    coin = isotropic_quark_coin(ratio)
    residual = coin.conjugate().T * coin - sp.eye(coin.rows)
    return all(sp.simplify(entry) == 0 for entry in residual)


@dataclass(frozen=True)
class QuarkCoinRigidityAuditPayload:
    """Verdict payload for the V15 quark coin rigidity theorem."""

    final_verdict: str
    symbolic_ratio: sp.Symbol
    symbolic_phase_angle: sp.Expr
    flat_ratio: sp.Expr
    flat_phase_angle: sp.Expr
    gamma_sum_square_matches: bool
    symbolic_coin_unitary: bool
    v11_coin_is_flat_specialization: bool
    nonflat_controls_unitary: bool
    nonflat_controls_change_phase: bool
    flat_ergodicity_required: bool
    ckm_phase_forced_by_unitarity_alone: bool
    interpretation: str


def quark_coin_rigidity_audit_payload() -> QuarkCoinRigidityAuditPayload:
    """Return the V15 primitive quark coin rigidity verdict."""

    ratio = sp.Symbol("r", real=True)
    flat_ratio = flatness_ratio_for_phase(quark_boundary_phase_angle())
    flat_coin = isotropic_quark_coin(1)
    v11_coin = quark_boundary_coin()
    flat_residual = flat_coin - v11_coin

    nonflat_ratios = (sp.Rational(1, 2), sp.Integer(2))
    nonflat_controls_unitary = all(isotropic_coin_is_unitary(value) for value in nonflat_ratios)
    nonflat_controls_change_phase = all(
        sp.simplify(isotropic_quark_phase_angle(value) - quark_boundary_phase_angle()) != 0
        for value in nonflat_ratios
    )
    v11_coin_is_flat = all(sp.simplify(entry) == 0 for entry in flat_residual)

    checks_pass = (
        gamma_sum_square_is_five()
        and isotropic_coin_is_unitary(ratio)
        and sp.simplify(isotropic_quark_phase_angle(ratio) - sp.atan(ratio * sp.sqrt(5))) == 0
        and sp.simplify(flat_ratio - 1) == 0
        and v11_coin_is_flat
        and nonflat_controls_unitary
        and nonflat_controls_change_phase
    )

    if checks_pass:
        final_verdict = "QUARK_COIN_RIGIDITY_THEOREM_PASS"
        interpretation = (
            "The isotropic quark coin is exactly unitary for any real even/odd "
            "ratio r because Gamma_q^2 = 5I, and its positive-branch phase is "
            "atan(r sqrt(5)). Therefore the V11 phase atan(sqrt(5)) is "
            "equivalent to the flat primitive ratio r=1. Unitarity and "
            "Clifford closure alone leave r free."
        )
    else:
        final_verdict = "QUARK_COIN_RIGIDITY_THEOREM_KILL"
        interpretation = (
            "The one-parameter unitary coin theorem, flat specialization, or "
            "non-flat controls failed."
        )

    return QuarkCoinRigidityAuditPayload(
        final_verdict=final_verdict,
        symbolic_ratio=ratio,
        symbolic_phase_angle=isotropic_quark_phase_angle(ratio),
        flat_ratio=flat_ratio,
        flat_phase_angle=quark_boundary_phase_angle(),
        gamma_sum_square_matches=gamma_sum_square_is_five(),
        symbolic_coin_unitary=isotropic_coin_is_unitary(ratio),
        v11_coin_is_flat_specialization=v11_coin_is_flat,
        nonflat_controls_unitary=nonflat_controls_unitary,
        nonflat_controls_change_phase=nonflat_controls_change_phase,
        flat_ergodicity_required=sp.simplify(flat_ratio - 1) == 0,
        ckm_phase_forced_by_unitarity_alone=False,
        interpretation=interpretation,
    )
