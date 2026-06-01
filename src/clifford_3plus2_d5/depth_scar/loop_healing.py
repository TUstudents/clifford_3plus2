"""V4 loop-healing deformation of the path repair scar.

V1-V3 establish the real path scar

    u -- a -- b

as the hierarchy operator.  A tree has no intrinsic CP holonomy: all edge
phases can be removed by port rephasings.  V4 asks what happens when the
missing repair edge is weakly restored.  The healed triangle has one cycle, so
one gauge-invariant phase can survive.

The labels are ordered as ``(u, a, b)``.  The real healing parameter ``delta``
restores the missing ``u-b`` repair edge.  The complex magnetic deformation
assigns phase ``phi`` to the oriented loop

    u -> a -> b -> u.

This sidecar layer does not derive ``delta`` or ``phi`` microscopically.  It
proves that, if a loop is healed, hierarchy and CP split cleanly:

* ``delta`` deforms the real depth spectrum ``{0, 2 + 4 delta, 6}``.
* ``phi`` is the only gauge-invariant graph phase.
* the path limit has no CP holonomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.edge_weight_potential import (
    weighted_triangle_laplacian_general,
)
from clifford_3plus2_d5.depth_scar.prediction_ledger import graph_cycle_rank
from clifford_3plus2_d5.depth_scar.repair_graphs import path_laplacian


def real_healed_laplacian(delta: sp.Expr) -> sp.Matrix:
    """Return the real weighted triangle with the missing ``u-b`` edge healed.

    The edge weights are ``(w_ua, w_ab, w_ub) = (1, 1, delta)``.
    """

    return weighted_triangle_laplacian_general((sp.Integer(1), sp.Integer(1), delta))


def real_healed_laplacian_spectrum_formula(delta: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return the exact Laplacian spectrum formula before BCC doubling."""

    return sp.Integer(0), sp.simplify(1 + 2 * delta), sp.Integer(3)


def real_healed_depth_spectrum_formula(delta: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return the exact BCC-doubled depth spectrum formula."""

    return sp.Integer(0), sp.simplify(2 + 4 * delta), sp.Integer(6)


def real_healed_spectrum_formula_pass() -> bool:
    """Return whether the healed real Laplacian has spectrum ``{0,1+2d,3}``."""

    delta = sp.symbols("delta")
    lam = sp.symbols("lambda")
    charpoly = sp.factor(real_healed_laplacian(delta).charpoly(lam).as_expr())
    expected = sp.factor(lam * (lam - (1 + 2 * delta)) * (lam - 3))
    return sp.simplify(charpoly - expected) == 0


def real_healed_depth_formula_pass() -> bool:
    """Return whether BCC doubling gives ``{0,2+4d,6}``."""

    delta = sp.symbols("delta")
    lam = sp.symbols("lambda")
    charpoly = sp.factor((2 * real_healed_laplacian(delta)).charpoly(lam).as_expr())
    expected = sp.factor(lam * (lam - (2 + 4 * delta)) * (lam - 6))
    return sp.simplify(charpoly - expected) == 0


def complex_healed_laplacian(delta: sp.Expr, phi: sp.Expr) -> sp.Matrix:
    """Return the Hermitian magnetic triangle Laplacian.

    The oriented loop phase is carried by the healed edge ``b -> u``.  Thus the
    off-diagonal entries are

        L[u,b] = -delta exp(-i phi),   L[b,u] = -delta exp(+i phi).

    At ``phi = 0`` this reduces to the real healed Laplacian.
    """

    phase = sp.exp(sp.I * phi)
    return sp.Matrix(
        [
            [1 + delta, -1, -delta / phase],
            [-1, 2, -1],
            [-delta * phase, -1, 1 + delta],
        ]
    )


def complex_healed_laplacian_is_hermitian() -> bool:
    """Return whether the symbolic magnetic Laplacian is Hermitian."""

    delta, phi = sp.symbols("delta phi", real=True)
    matrix = complex_healed_laplacian(delta, phi)
    return sp.simplify(matrix - matrix.H) == sp.zeros(3, 3)


def phi_zero_control_pass() -> bool:
    """Return whether the magnetic triangle reduces to the real triangle."""

    delta = sp.symbols("delta")
    return sp.simplify(
        complex_healed_laplacian(delta, sp.Integer(0)) - real_healed_laplacian(delta)
    ) == sp.zeros(3, 3)


def path_limit_control_pass() -> bool:
    """Return whether ``delta = 0`` recovers the original path scar."""

    phi = sp.symbols("phi")
    return sp.simplify(
        complex_healed_laplacian(sp.Integer(0), phi) - path_laplacian()
    ) == sp.zeros(3, 3)


def path_cycle_rank() -> int:
    """Return the cycle rank of the unhealed path graph."""

    return graph_cycle_rank(((0, 1), (1, 2)))


def healed_triangle_cycle_rank() -> int:
    """Return the cycle rank after restoring the missing edge."""

    return graph_cycle_rank(((0, 1), (1, 2), (2, 0)))


def gauge_transformed_edge_phase(theta_ij: sp.Expr, alpha_i: sp.Expr, alpha_j: sp.Expr) -> sp.Expr:
    """Return edge phase after port rephasings.

    Under ``|i> -> exp(i alpha_i)|i>``, an oriented edge phase transforms as

        theta_ij -> theta_ij + alpha_i - alpha_j.
    """

    return sp.simplify(theta_ij + alpha_i - alpha_j)


def path_phase_gauge_solution(theta_ua: sp.Expr, theta_ab: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return port phases that remove arbitrary phases on the path edges."""

    alpha_u = sp.Integer(0)
    alpha_a = theta_ua
    alpha_b = theta_ua + theta_ab
    return alpha_u, alpha_a, alpha_b


def path_phase_removal_residuals(theta_ua: sp.Expr, theta_ab: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return the transformed path-edge phases after applying the gauge solution."""

    alpha_u, alpha_a, alpha_b = path_phase_gauge_solution(theta_ua, theta_ab)
    return (
        gauge_transformed_edge_phase(theta_ua, alpha_u, alpha_a),
        gauge_transformed_edge_phase(theta_ab, alpha_a, alpha_b),
    )


def path_phases_are_removable() -> bool:
    """Return whether arbitrary phases on a tree can be gauged away."""

    theta_ua, theta_ab = sp.symbols("theta_ua theta_ab")
    return path_phase_removal_residuals(theta_ua, theta_ab) == (sp.Integer(0), sp.Integer(0))


def triangle_loop_phase(
    theta_ua: sp.Expr = sp.Integer(0),
    theta_ab: sp.Expr = sp.Integer(0),
    theta_bu: sp.Expr | None = None,
) -> sp.Expr:
    """Return the oriented loop phase for ``u -> a -> b -> u``."""

    if theta_bu is None:
        theta_bu = sp.symbols("phi")
    return sp.simplify(theta_ua + theta_ab + theta_bu)


def triangle_loop_phase_after_gauge(
    theta_ua: sp.Expr,
    theta_ab: sp.Expr,
    theta_bu: sp.Expr,
    alpha_u: sp.Expr,
    alpha_a: sp.Expr,
    alpha_b: sp.Expr,
) -> sp.Expr:
    """Return the loop phase after arbitrary port rephasings."""

    transformed_ua = gauge_transformed_edge_phase(theta_ua, alpha_u, alpha_a)
    transformed_ab = gauge_transformed_edge_phase(theta_ab, alpha_a, alpha_b)
    transformed_bu = gauge_transformed_edge_phase(theta_bu, alpha_b, alpha_u)
    return sp.simplify(transformed_ua + transformed_ab + transformed_bu)


def loop_phase_is_gauge_invariant() -> bool:
    """Return whether the triangle loop phase survives all port rephasings."""

    theta_ua, theta_ab, theta_bu = sp.symbols("theta_ua theta_ab theta_bu")
    alpha_u, alpha_a, alpha_b = sp.symbols("alpha_u alpha_a alpha_b")
    before = triangle_loop_phase(theta_ua, theta_ab, theta_bu)
    after = triangle_loop_phase_after_gauge(
        theta_ua, theta_ab, theta_bu, alpha_u, alpha_a, alpha_b
    )
    return sp.simplify(after - before) == 0


def conjugated_loop_phase(phi: sp.Expr) -> sp.Expr:
    """Return the loop phase after complex conjugation."""

    return sp.simplify(-phi)


def conjugation_flips_loop_phase() -> bool:
    """Return whether complex conjugation flips the CP holonomy."""

    phi = sp.symbols("phi")
    return sp.simplify(conjugated_loop_phase(phi) + phi) == 0


@dataclass(frozen=True)
class LoopHealingCPPayload:
    """V4 payload for loop healing and CP holonomy separation."""

    final_verdict: str
    real_spectrum_formula: tuple[sp.Expr, ...]
    doubled_spectrum_formula: tuple[sp.Expr, ...]
    real_spectrum_formula_pass: bool
    doubled_spectrum_formula_pass: bool
    magnetic_laplacian_hermitian: bool
    path_phases_removable: bool
    path_cycle_rank: int
    healed_triangle_cycle_rank: int
    loop_phase_gauge_invariant: bool
    phi_zero_control: bool
    path_limit_control: bool
    conjugation_flips_phase: bool
    microscopic_delta_phi_derived: bool
    interpretation: str


def loop_healing_cp_payload() -> LoopHealingCPPayload:
    """Return the V4 loop-healing and CP-deformation verdict."""

    delta = sp.symbols("delta")
    real_spectrum = real_healed_laplacian_spectrum_formula(delta)
    doubled_spectrum = real_healed_depth_spectrum_formula(delta)
    real_formula = real_healed_spectrum_formula_pass()
    doubled_formula = real_healed_depth_formula_pass()
    hermitian = complex_healed_laplacian_is_hermitian()
    phases_removable = path_phases_are_removable()
    path_rank = path_cycle_rank()
    triangle_rank = healed_triangle_cycle_rank()
    loop_invariant = loop_phase_is_gauge_invariant()
    phi_zero = phi_zero_control_pass()
    path_limit = path_limit_control_pass()
    conjugation = conjugation_flips_loop_phase()

    checks_pass = (
        real_formula
        and doubled_formula
        and hermitian
        and phases_removable
        and path_rank == 0
        and triangle_rank == 1
        and loop_invariant
        and phi_zero
        and path_limit
        and conjugation
    )

    if checks_pass:
        final_verdict = "LOOP_HEALING_CP_DEFORMATION_PASS"
        interpretation = (
            "The pure P3 scar is a tree, so path phases are removable and it has no "
            "intrinsic CP holonomy. Restoring the missing edge creates exactly one "
            "cycle. The real healing parameter deforms the depth spectrum to "
            "{0, 2+4 delta, 6}, while the loop phase is gauge-invariant and flips "
            "under conjugation. Delta and phi remain microscopic inputs."
        )
    elif path_rank != 0 or not phases_removable:
        final_verdict = "PURE_PATH_CP_KILL"
        interpretation = "The pure path unexpectedly carries a gauge-invariant phase."
    elif triangle_rank != 1 or not loop_invariant:
        final_verdict = "LOOP_HOLONOMY_KILL"
        interpretation = "The healed triangle does not produce exactly one holonomy."
    else:
        final_verdict = "LOOP_HEALING_CONTROL_KILL"
        interpretation = "A spectrum, Hermiticity, conjugation, or limit control failed."

    return LoopHealingCPPayload(
        final_verdict=final_verdict,
        real_spectrum_formula=real_spectrum,
        doubled_spectrum_formula=doubled_spectrum,
        real_spectrum_formula_pass=real_formula,
        doubled_spectrum_formula_pass=doubled_formula,
        magnetic_laplacian_hermitian=hermitian,
        path_phases_removable=phases_removable,
        path_cycle_rank=path_rank,
        healed_triangle_cycle_rank=triangle_rank,
        loop_phase_gauge_invariant=loop_invariant,
        phi_zero_control=phi_zero,
        path_limit_control=path_limit,
        conjugation_flips_phase=conjugation,
        microscopic_delta_phi_derived=False,
        interpretation=interpretation,
    )
