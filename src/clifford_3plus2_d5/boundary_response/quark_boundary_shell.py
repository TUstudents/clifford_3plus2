"""V11 quark primitive boundary-shell gate.

The CKM texture in the boundary-response note requires a primitive quark shell

    S_q = 1_even + 5_odd = 1_direct + (2_BCC + 3_color).

V11 audits only that first quark-sector assumption and the associated flat
Clifford coin.  It does not assemble CKM magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


EVEN = "even"
ODD = "odd"
DIRECT = "direct"
BCC = "bcc"
COLOR = "color"


@dataclass(frozen=True)
class QuarkBoundaryChannel:
    """Primitive channel in the quark boundary shell."""

    name: str
    parity: str
    sector: str


def quark_primitive_channels() -> tuple[QuarkBoundaryChannel, ...]:
    """Return the proposed primitive quark shell channels."""

    return (
        QuarkBoundaryChannel(name="direct_even_return", parity=EVEN, sector=DIRECT),
        QuarkBoundaryChannel(name="bcc_odd_quadrature_1", parity=ODD, sector=BCC),
        QuarkBoundaryChannel(name="bcc_odd_quadrature_2", parity=ODD, sector=BCC),
        QuarkBoundaryChannel(name="color_odd_red", parity=ODD, sector=COLOR),
        QuarkBoundaryChannel(name="color_odd_green", parity=ODD, sector=COLOR),
        QuarkBoundaryChannel(name="color_odd_blue", parity=ODD, sector=COLOR),
    )


def quark_shell_dimension_breakdown(
    channels: tuple[QuarkBoundaryChannel, ...] | None = None,
) -> dict[str, int]:
    """Return exact shell counts by parity and sector."""

    selected = quark_primitive_channels() if channels is None else channels
    even_direct = sum(1 for channel in selected if channel.parity == EVEN and channel.sector == DIRECT)
    bcc_odd = sum(1 for channel in selected if channel.parity == ODD and channel.sector == BCC)
    color_odd = sum(1 for channel in selected if channel.parity == ODD and channel.sector == COLOR)
    odd_total = sum(1 for channel in selected if channel.parity == ODD)
    return {
        "even_direct": even_direct,
        "bcc_odd": bcc_odd,
        "color_odd": color_odd,
        "odd_total": odd_total,
        "total": len(selected),
    }


def _pauli_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return exact Pauli matrices."""

    sigma_1 = sp.Matrix([[0, 1], [1, 0]])
    sigma_2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sigma_3 = sp.Matrix([[1, 0], [0, -1]])
    return sigma_1, sigma_2, sigma_3


def quark_odd_clifford_generators() -> tuple[sp.Matrix, ...]:
    """Return a fixed exact Hermitian representation of ``Cl_5``."""

    sigma_1, sigma_2, sigma_3 = _pauli_matrices()
    identity_2 = sp.eye(2)
    return (
        sp.kronecker_product(sigma_1, identity_2),
        sp.kronecker_product(sigma_2, identity_2),
        sp.kronecker_product(sigma_3, sigma_1),
        sp.kronecker_product(sigma_3, sigma_2),
        sp.kronecker_product(sigma_3, sigma_3),
    )


def quark_gamma_sum(generators: tuple[sp.Matrix, ...] | None = None) -> sp.Matrix:
    """Return ``Gamma_q = sum_A gamma_A``."""

    selected = quark_odd_clifford_generators() if generators is None else generators
    return sum(selected, sp.zeros(4, 4))


def quark_boundary_coin(generators: tuple[sp.Matrix, ...] | None = None) -> sp.Matrix:
    """Return the flat primitive quark boundary coin."""

    gamma_sum = quark_gamma_sum(generators)
    return sp.simplify((sp.eye(gamma_sum.rows) + sp.I * gamma_sum) / sp.sqrt(6))


def quark_boundary_phase_factor() -> sp.Expr:
    """Return the scalar positive-branch phase from the flat coin."""

    return (1 + sp.I * sp.sqrt(5)) / sp.sqrt(6)


def quark_boundary_phase_angle() -> sp.Expr:
    """Return the scalar positive-branch phase angle."""

    return sp.atan(sp.sqrt(5))


def nonflat_quark_boundary_phase_angle(ratio: sp.Expr) -> sp.Expr:
    """Return the phase if odd channels carry relative amplitude ``ratio``."""

    return sp.atan(sp.simplify(ratio * sp.sqrt(5)))


def missing_color_channel_control() -> tuple[QuarkBoundaryChannel, ...]:
    """Return a shell with one color odd port removed."""

    return tuple(channel for channel in quark_primitive_channels() if channel.name != "color_odd_blue")


def commuting_odd_generators_control() -> tuple[sp.Matrix, ...]:
    """Return a bad odd-generator control with commuting matrices."""

    return tuple(sp.eye(4) for _ in range(5))


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices agree after simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def clifford_relations_hold(generators: tuple[sp.Matrix, ...] | None = None) -> bool:
    """Return true when the supplied generators satisfy exact ``Cl_5`` relations."""

    selected = quark_odd_clifford_generators() if generators is None else generators
    identity = sp.eye(selected[0].rows)
    for index, gamma in enumerate(selected):
        if not _matrix_equal(gamma * gamma, identity):
            return False
        for other in selected[index + 1 :]:
            if not _matrix_equal(gamma * other + other * gamma, sp.zeros(gamma.rows, gamma.cols)):
                return False
    return True


@dataclass(frozen=True)
class QuarkBoundaryShellAuditPayload:
    """Verdict payload for the V11 quark primitive shell gate."""

    final_verdict: str
    shell_breakdown: dict[str, int]
    gamma_sum_square_matches: bool
    coin_unitary: bool
    phase_factor: sp.Expr
    phase_angle: sp.Expr
    missing_color_control_rejected: bool
    nonflat_control_rejected: bool
    commuting_control_rejected: bool
    ckm_parked: bool
    interpretation: str


def quark_boundary_shell_audit_payload() -> QuarkBoundaryShellAuditPayload:
    """Return the V11 quark primitive shell verdict."""

    generators = quark_odd_clifford_generators()
    gamma_sum = quark_gamma_sum(generators)
    coin = quark_boundary_coin(generators)
    identity = sp.eye(4)

    breakdown = quark_shell_dimension_breakdown()
    expected_breakdown = {
        "even_direct": 1,
        "bcc_odd": 2,
        "color_odd": 3,
        "odd_total": 5,
        "total": 6,
    }
    gamma_sum_square_matches = _matrix_equal(gamma_sum * gamma_sum, 5 * identity)
    coin_unitary = _matrix_equal(coin.conjugate().T * coin, identity)

    missing_breakdown = quark_shell_dimension_breakdown(missing_color_channel_control())
    missing_color_rejected = missing_breakdown != expected_breakdown
    nonflat_rejected = (
        sp.simplify(nonflat_quark_boundary_phase_angle(sp.Integer(2)) - quark_boundary_phase_angle())
        != 0
    )
    commuting_sum = quark_gamma_sum(commuting_odd_generators_control())
    commuting_rejected = not _matrix_equal(commuting_sum * commuting_sum, 5 * identity)

    checks_pass = (
        breakdown == expected_breakdown
        and clifford_relations_hold(generators)
        and gamma_sum_square_matches
        and coin_unitary
        and missing_color_rejected
        and nonflat_rejected
        and commuting_rejected
    )

    if checks_pass:
        final_verdict = "QUARK_BOUNDARY_SHELL_Q1_PASS"
        interpretation = (
            "The proposed primitive quark shell has exactly one even direct "
            "channel and five odd channels split as 2_BCC + 3_color. A fixed "
            "Cl_5 representation gives Gamma_q^2 = 5I, the flat coin is "
            "unitary, and the positive-branch scalar phase is atan(sqrt(5)). "
            "Missing-color, non-flat, and commuting-generator controls are "
            "rejected. CKM magnitudes remain parked."
        )
    else:
        final_verdict = "QUARK_BOUNDARY_SHELL_Q1_KILL"
        interpretation = (
            "The shell counts, Cl_5 relations, flat coin, or negative controls "
            "failed. CKM magnitudes remain parked."
        )

    return QuarkBoundaryShellAuditPayload(
        final_verdict=final_verdict,
        shell_breakdown=breakdown,
        gamma_sum_square_matches=gamma_sum_square_matches,
        coin_unitary=coin_unitary,
        phase_factor=quark_boundary_phase_factor(),
        phase_angle=quark_boundary_phase_angle(),
        missing_color_control_rejected=missing_color_rejected,
        nonflat_control_rejected=nonflat_rejected,
        commuting_control_rejected=commuting_rejected,
        ckm_parked=True,
        interpretation=interpretation,
    )
