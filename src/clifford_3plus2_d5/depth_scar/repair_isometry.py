"""V10 active repair isometry saturation.

V9 proves that one-tick locality fixes the active repair support but leaves a
weighted path unless the repair block saturates to an isometry.  V10 proves the
exact algebraic condition for that saturation.

Let ``A = span{|a>, |b>}`` be the active defect domain and
``R = span{|u>, |a>}`` the repaired range.  For a full microscopic unitary
``U``, define

    N = P_R U P_A,
    L = (I - P_R) U P_A.

Then unitarity gives ``N.H N + L.H L = I_A``.  Therefore ``N`` is an isometry
on the active domain if and only if the leakage block ``L`` is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import (
    flag_phases_are_removable,
)
from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    target_spectrum_forces_unit_weights,
    weighted_path_laplacian,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_LAPLACIAN_SPECTRUM,
)


def _sorted_eigenvalues(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact eigenvalues with multiplicity in deterministic order."""

    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def active_domain_basis() -> sp.Matrix:
    """Return embedding columns for ``A = span{|a>, |b>}`` in ``(u,a,b)``."""

    return sp.Matrix(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )


def repaired_range_projector() -> sp.Matrix:
    """Return ``P_R`` for ``R = span{|u>, |a>}``."""

    return sp.diag(1, 1, 0)


def active_projector() -> sp.Matrix:
    """Return ``P_A`` for ``A = span{|a>, |b>}``."""

    return sp.diag(0, 1, 1)


def active_repair_block(alpha: sp.Expr, beta: sp.Expr) -> sp.Matrix:
    """Return the full-space repair block ``alpha|u><a| + beta|a><b|``."""

    return sp.Matrix(
        [
            [0, alpha, 0],
            [0, 0, beta],
            [0, 0, 0],
        ]
    )


def active_repair_domain_matrix(alpha: sp.Expr, beta: sp.Expr) -> sp.Matrix:
    """Return ``N`` as a map from active-domain coordinates to full space."""

    return active_repair_block(alpha, beta) * active_domain_basis()


def leakage_domain_matrix(ell_a: sp.Expr, ell_b: sp.Expr) -> sp.Matrix:
    """Return leakage from active states to orthogonal unrepaired channels.

    The V10 identity is an abstract output-space decomposition, not a claim
    that both active states leak into the same residual port.  Orthogonal
    columns are the local normal form of the positive operator ``L.H L``.
    """

    return sp.Matrix(
        [
            [ell_a, 0],
            [0, ell_b],
        ]
    )


def active_identity() -> sp.Matrix:
    """Return the active-domain identity ``I_A``."""

    return sp.eye(2)


def active_repair_norm_matrix(alpha: sp.Expr, beta: sp.Expr) -> sp.Matrix:
    """Return ``N.H N`` on active-domain coordinates."""

    repair = active_repair_domain_matrix(alpha, beta)
    return sp.simplify(repair.H * repair)


def leakage_norm_matrix(ell_a: sp.Expr, ell_b: sp.Expr) -> sp.Matrix:
    """Return ``L.H L`` on active-domain coordinates."""

    leakage = leakage_domain_matrix(ell_a, ell_b)
    return sp.simplify(leakage.H * leakage)


def unitarity_balance_residual(
    alpha: sp.Expr,
    beta: sp.Expr,
    ell_a: sp.Expr,
    ell_b: sp.Expr,
) -> sp.Matrix:
    """Return ``N.H N + L.H L - I_A``."""

    return sp.simplify(
        active_repair_norm_matrix(alpha, beta)
        + leakage_norm_matrix(ell_a, ell_b)
        - active_identity()
    )


def unitarity_balance_formula_pass() -> bool:
    """Return whether V10's block identity has the expected symbolic form."""

    alpha, beta, ell_a, ell_b = sp.symbols("alpha beta ell_a ell_b", real=True)
    expected = sp.diag(
        alpha**2 + ell_a**2 - 1,
        beta**2 + ell_b**2 - 1,
    )
    return sp.simplify(unitarity_balance_residual(alpha, beta, ell_a, ell_b) - expected) == sp.zeros(
        2,
        2,
    )


def no_leakage_forces_unit_weights() -> bool:
    """Return whether ``L=0`` forces active repair weights to be unit."""

    alpha, beta = sp.symbols("alpha beta", real=True)
    equations = tuple(unitarity_balance_residual(alpha, beta, sp.Integer(0), sp.Integer(0)))
    solutions = set(
        tuple(solution[symbol] for symbol in (alpha, beta))
        for solution in sp.solve(equations, (alpha, beta), dict=True)
    )
    return solutions == {
        (-sp.Integer(1), -sp.Integer(1)),
        (-sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), -sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1)),
    }


def unit_weights_force_no_leakage() -> bool:
    """Return whether unit active weights force zero leakage by unitarity."""

    ell_a, ell_b = sp.symbols("ell_a ell_b", real=True)
    equations = tuple(unitarity_balance_residual(sp.Integer(1), sp.Integer(1), ell_a, ell_b))
    solutions = set(
        tuple(solution[symbol] for symbol in (ell_a, ell_b))
        for solution in sp.solve(equations, (ell_a, ell_b), dict=True)
    )
    return solutions == {
        (-sp.Integer(0), -sp.Integer(0)),
    }


def symmetric_leakage_preserves_ratio_but_rescales() -> bool:
    """Return whether equal leakage preserves the 1:3 ratio but changes scale."""

    leakage = sp.Rational(1, 2)
    weight = sp.simplify(1 - leakage**2)
    spectrum = _sorted_eigenvalues(weighted_path_laplacian(weight, weight))
    return (
        spectrum == (sp.Integer(0), weight, 3 * weight)
        and spectrum != EXPECTED_LAPLACIAN_SPECTRUM
    )


def unequal_leakage_breaks_ratio() -> bool:
    """Return whether unequal leakage destroys the path's 1:3 ratio."""

    w1 = sp.Rational(3, 4)
    w2 = sp.Rational(1, 2)
    spectrum = _sorted_eigenvalues(weighted_path_laplacian(w1, w2))
    return sp.simplify(spectrum[2] - 3 * spectrum[1]) != 0


def repair_isometry_saturation_pass() -> bool:
    """Return whether V10's algebraic isometry-saturation theorem passes."""

    return (
        unitarity_balance_formula_pass()
        and no_leakage_forces_unit_weights()
        and unit_weights_force_no_leakage()
        and flag_phases_are_removable()
        and target_spectrum_forces_unit_weights()
        and symmetric_leakage_preserves_ratio_but_rescales()
        and unequal_leakage_breaks_ratio()
    )


@dataclass(frozen=True)
class RepairIsometryPayload:
    """V10 payload for active repair isometry saturation."""

    final_verdict: str
    unitarity_balance_formula_pass: bool
    no_leakage_forces_unit_weights: bool
    unit_weights_force_no_leakage: bool
    tree_phases_removable: bool
    target_spectrum_forces_unit_weights: bool
    symmetric_leakage_preserves_ratio_but_rescales: bool
    unequal_leakage_breaks_ratio: bool
    no_leakage_microscopically_derived: bool
    interpretation: str


def repair_isometry_payload() -> RepairIsometryPayload:
    """Return the V10 repair-isometry saturation verdict."""

    balance = unitarity_balance_formula_pass()
    no_leakage_unit = no_leakage_forces_unit_weights()
    unit_no_leakage = unit_weights_force_no_leakage()
    phases = flag_phases_are_removable()
    target_weights = target_spectrum_forces_unit_weights()
    symmetric = symmetric_leakage_preserves_ratio_but_rescales()
    unequal = unequal_leakage_breaks_ratio()

    checks_pass = (
        balance
        and no_leakage_unit
        and unit_no_leakage
        and phases
        and target_weights
        and symmetric
        and unequal
    )

    if checks_pass:
        final_verdict = "V10_REPAIR_ISOMETRY_SATURATION_PASS"
        interpretation = (
            "For the active repair block N=P_R U P_A, unitarity gives "
            "N^dagger N + L^dagger L = I_A. Thus N is a unit isometry exactly "
            "when leakage L vanishes. Under the V9 path support this forces "
            "unit edge weights up to removable tree phases; leakage gives a "
            "weighted path instead. The theorem does not derive L=0."
        )
    elif not balance:
        final_verdict = "REPAIR_ISOMETRY_BALANCE_KILL"
        interpretation = "The projected unitarity balance identity failed."
    elif not no_leakage_unit or not unit_no_leakage:
        final_verdict = "REPAIR_ISOMETRY_EQUIVALENCE_KILL"
        interpretation = "No-leakage was not equivalent to unit active repair weights."
    else:
        final_verdict = "REPAIR_ISOMETRY_CONTROL_KILL"
        interpretation = "A phase, target-weight, symmetric-leakage, or unequal-leakage control failed."

    return RepairIsometryPayload(
        final_verdict=final_verdict,
        unitarity_balance_formula_pass=balance,
        no_leakage_forces_unit_weights=no_leakage_unit,
        unit_weights_force_no_leakage=unit_no_leakage,
        tree_phases_removable=phases,
        target_spectrum_forces_unit_weights=target_weights,
        symmetric_leakage_preserves_ratio_but_rescales=symmetric,
        unequal_leakage_breaks_ratio=unequal,
        no_leakage_microscopically_derived=False,
        interpretation=interpretation,
    )
