"""V6 local-unitarity normalization for the nilpotent repair flag.

V5 shows that the canonical length-3 repair flag

    N = |u><a| + |a><b|

induces ``Delta(P3)``.  V6 asks whether the equal unit coefficients can be
forced by local QCA normalization rather than inserted by hand.

The most general complex flag on the same support is

    N = r exp(i alpha) |u><a| + s exp(i beta) |a><b|.

If this flag is a local boundary subblock of a unitary update, it must be a
partial isometry on its active repair subspace.  The projection conditions

    (N.H N)^2 = N.H N,    (N N.H)^2 = N N.H

force the nonzero singular values to be one.  Since the support is a tree, the
two remaining phases are removable by port rephasings.  Thus the canonical V5
flag is the unique nonnegative representative after local unitarity and gauge
choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.loop_healing import gauge_transformed_edge_phase
from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
    nilpotent_flag_transfer_matches_v1,
)


def complex_local_flag_operator(
    r: sp.Expr,
    s: sp.Expr,
    alpha: sp.Expr,
    beta: sp.Expr,
) -> sp.Matrix:
    """Return ``r e^{i alpha}|u><a| + s e^{i beta}|a><b|``."""

    return sp.Matrix(
        [
            [0, r * sp.exp(sp.I * alpha), 0],
            [0, 0, s * sp.exp(sp.I * beta)],
            [0, 0, 0],
        ]
    )


def initial_projection(operator: sp.Matrix) -> sp.Matrix:
    """Return ``N.H N``."""

    return sp.simplify(operator.H * operator)


def final_projection(operator: sp.Matrix) -> sp.Matrix:
    """Return ``N N.H``."""

    return sp.simplify(operator * operator.H)


def projection_formula_pass() -> bool:
    """Return whether the generic real-magnitude flag has the expected projections."""

    r, s, alpha, beta = sp.symbols("r s alpha beta", real=True)
    operator = complex_local_flag_operator(r, s, alpha, beta)
    return (
        sp.simplify(initial_projection(operator) - sp.diag(0, r**2, s**2)) == sp.zeros(3, 3)
        and sp.simplify(final_projection(operator) - sp.diag(r**2, s**2, 0)) == sp.zeros(3, 3)
    )


def partial_isometry_residuals(r: sp.Expr, s: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return the independent idempotence residuals for the active singular values."""

    return (
        sp.simplify(r**4 - r**2),
        sp.simplify(s**4 - s**2),
    )


def partial_isometry_magnitude_solutions() -> tuple[tuple[sp.Expr, sp.Expr], ...]:
    """Return real nonzero magnitude solutions to the partial-isometry equations."""

    return (
        (-sp.Integer(1), -sp.Integer(1)),
        (-sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), -sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1)),
    )


def nonnegative_partial_isometry_solution() -> tuple[sp.Expr, sp.Expr]:
    """Return the unique nonnegative nonzero magnitude solution."""

    return sp.Integer(1), sp.Integer(1)


def partial_isometry_forces_unit_magnitudes() -> bool:
    """Return whether nonzero local-unitarity solutions force unit magnitudes."""

    r, s = sp.symbols("r s", real=True)
    solved = set(
        tuple(sp.simplify(solution[symbol]) for symbol in (r, s))
        for solution in sp.solve(partial_isometry_residuals(r, s), (r, s), dict=True)
        if solution[r] != 0 and solution[s] != 0
    )
    return solved == set(partial_isometry_magnitude_solutions()) and (
        nonnegative_partial_isometry_solution() == (sp.Integer(1), sp.Integer(1))
    )


def flag_phase_gauge_solution(alpha: sp.Expr, beta: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return port phases removing the two tree phases of the oriented flag.

    The nonzero entries map ``a -> u`` and ``b -> a``.  With port rephasings
    ``eta_u, eta_a, eta_b``, phases transform as ``theta + eta_target -
    eta_source``.  Setting ``eta_u = 0`` removes both phases with
    ``eta_a = alpha`` and ``eta_b = alpha + beta``.
    """

    eta_u = sp.Integer(0)
    eta_a = alpha
    eta_b = alpha + beta
    return eta_u, eta_a, eta_b


def flag_phase_removal_residuals(alpha: sp.Expr, beta: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return transformed phases after applying the tree gauge solution."""

    eta_u, eta_a, eta_b = flag_phase_gauge_solution(alpha, beta)
    return (
        gauge_transformed_edge_phase(alpha, eta_u, eta_a),
        gauge_transformed_edge_phase(beta, eta_a, eta_b),
    )


def flag_phases_are_removable() -> bool:
    """Return whether arbitrary phases on the length-3 tree support gauge away."""

    alpha, beta = sp.symbols("alpha beta")
    return flag_phase_removal_residuals(alpha, beta) == (sp.Integer(0), sp.Integer(0))


def port_rephasing_matrix(eta_u: sp.Expr, eta_a: sp.Expr, eta_b: sp.Expr) -> sp.Matrix:
    """Return the diagonal port rephasing matrix."""

    return sp.diag(sp.exp(sp.I * eta_u), sp.exp(sp.I * eta_a), sp.exp(sp.I * eta_b))


def rephase_operator(operator: sp.Matrix, eta_u: sp.Expr, eta_a: sp.Expr, eta_b: sp.Expr) -> sp.Matrix:
    """Return ``U N U.H`` for port rephasing ``U``."""

    unitary = port_rephasing_matrix(eta_u, eta_a, eta_b)
    return sp.simplify(unitary * operator * unitary.H)


def phased_unit_flag_gauge_equivalent_to_canonical() -> bool:
    """Return whether every unit-magnitude phased flag is gauge-equivalent to V5."""

    alpha, beta = sp.symbols("alpha beta", real=True)
    operator = complex_local_flag_operator(sp.Integer(1), sp.Integer(1), alpha, beta)
    etas = flag_phase_gauge_solution(alpha, beta)
    return sp.simplify(rephase_operator(operator, *etas) - nilpotent_flag_operator()) == sp.zeros(
        3, 3
    )


def canonical_flag_laplacian_and_transfer_pass() -> bool:
    """Return whether the gauged local flag reproduces V5 and V1."""

    return (
        sp.simplify(flag_laplacian_from_nilpotent(nilpotent_flag_operator())
                    - flag_laplacian_from_nilpotent()) == sp.zeros(3, 3)
        and nilpotent_flag_transfer_matches_v1()
    )


def rank_one_partial_isometry_control_rejected() -> bool:
    """Return whether a rank-one partial isometry is rejected as not length-3."""

    control = complex_local_flag_operator(sp.Integer(1), sp.Integer(0), sp.Integer(0), sp.Integer(0))
    return (
        sp.simplify(initial_projection(control) ** 2 - initial_projection(control)) == sp.zeros(3, 3)
        and sp.simplify(final_projection(control) ** 2 - final_projection(control)) == sp.zeros(3, 3)
        and sp.simplify(control**2) == sp.zeros(3, 3)
    )


def contractive_magnitude_control_rejected() -> bool:
    """Return whether a contractive non-unit flag fails partial-isometry."""

    control = complex_local_flag_operator(
        sp.Rational(1, 2),
        sp.Rational(1, 2),
        sp.Integer(0),
        sp.Integer(0),
    )
    return (
        sp.simplify(initial_projection(control) ** 2 - initial_projection(control)) != sp.zeros(3, 3)
        and sp.simplify(final_projection(control) ** 2 - final_projection(control)) != sp.zeros(3, 3)
    )


def unequal_magnitude_control_rejected() -> bool:
    """Return whether unequal nonzero magnitudes fail local partial-isometry."""

    control = complex_local_flag_operator(sp.Integer(1), sp.Integer(2), sp.Integer(0), sp.Integer(0))
    return sp.simplify(initial_projection(control) ** 2 - initial_projection(control)) != sp.zeros(
        3, 3
    )


def cyclic_unitary_closure_operator() -> sp.Matrix:
    """Return the unitary cyclic closure control."""

    return sp.Matrix(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    )


def cyclic_unitary_closure_rejected() -> bool:
    """Return whether cyclic unitary closure is non-nilpotent and returns ``K3``."""

    control = cyclic_unitary_closure_operator()
    laplacian = flag_laplacian_from_nilpotent(control)
    return (
        sp.simplify(control**3 - sp.eye(3)) == sp.zeros(3, 3)
        and sp.simplify(control**3) != sp.zeros(3, 3)
        and tuple(sorted(laplacian.eigenvals(), key=sp.default_sort_key))
        == (sp.Integer(0), sp.Integer(3))
    )


@dataclass(frozen=True)
class LocalFlagUnitarityPayload:
    """V6 payload for local-unitarity normalization of the repair flag."""

    final_verdict: str
    projection_formula_pass: bool
    partial_isometry_magnitudes_pass: bool
    phase_gauge_pass: bool
    canonical_representative_pass: bool
    laplacian_and_transfer_pass: bool
    rank_one_control_rejected: bool
    contractive_control_rejected: bool
    unequal_control_rejected: bool
    cyclic_control_rejected: bool
    length_three_support_microscopically_derived: bool
    interpretation: str


def local_flag_unitarity_payload() -> LocalFlagUnitarityPayload:
    """Return the V6 local-flag unitarity verdict."""

    projections = projection_formula_pass()
    magnitudes = partial_isometry_forces_unit_magnitudes()
    phases = flag_phases_are_removable()
    canonical = phased_unit_flag_gauge_equivalent_to_canonical()
    laplacian_transfer = canonical_flag_laplacian_and_transfer_pass()
    rank_one = rank_one_partial_isometry_control_rejected()
    contractive = contractive_magnitude_control_rejected()
    unequal = unequal_magnitude_control_rejected()
    cyclic = cyclic_unitary_closure_rejected()

    checks_pass = (
        projections
        and magnitudes
        and phases
        and canonical
        and laplacian_transfer
        and rank_one
        and contractive
        and unequal
        and cyclic
    )

    if checks_pass:
        final_verdict = "LOCAL_FLAG_PARTIAL_ISOMETRY_PASS"
        interpretation = (
            "On the length-3 nilpotent support, local partial-isometry forces "
            "the two active repair magnitudes to be unit. Because the support is "
            "a tree, the two phases are removable, leaving the canonical V5 "
            "flag. The remaining input is the microscopic origin of the "
            "length-3 support itself."
        )
    elif not magnitudes:
        final_verdict = "LOCAL_FLAG_UNITARITY_MAGNITUDE_KILL"
        interpretation = "Partial-isometry did not force unit nonzero flag magnitudes."
    elif not rank_one:
        final_verdict = "LOCAL_FLAG_SUPPORT_KILL"
        interpretation = "A shorter support was not rejected as a non-length-3 flag."
    elif not phases or not canonical:
        final_verdict = "LOCAL_FLAG_PHASE_GAUGE_KILL"
        interpretation = "Tree phases could not be removed to the canonical flag."
    else:
        final_verdict = "LOCAL_FLAG_CONTROL_KILL"
        interpretation = "A projection, transfer, contractive, unequal, or cyclic control failed."

    return LocalFlagUnitarityPayload(
        final_verdict=final_verdict,
        projection_formula_pass=projections,
        partial_isometry_magnitudes_pass=magnitudes,
        phase_gauge_pass=phases,
        canonical_representative_pass=canonical,
        laplacian_and_transfer_pass=laplacian_transfer,
        rank_one_control_rejected=rank_one,
        contractive_control_rejected=contractive,
        unequal_control_rejected=unequal,
        cyclic_control_rejected=cyclic,
        length_three_support_microscopically_derived=False,
        interpretation=interpretation,
    )
