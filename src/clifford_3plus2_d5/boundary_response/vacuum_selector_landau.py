"""V33 tetrahedral invariant Landau minimality gate.

V32 proves that the continuous locking potential

    (|h|^2 - 1)^2 + (C(h) - 8/9)^2

selects the four accepted BCC tetrahedral exits.  V33 audits whether this lock
is the canonical lowest-order tetrahedral Landau construction.

The proper tetrahedral rotation group is realized by the 12 exact signed
permutation matrices that preserve the four V27 BCC selector directions.  Its
homogeneous invariant spaces in degrees 1, 2, and 3 are:

    degree 1: none,
    degree 2: span{|h|^2},
    degree 3: span{x*y*z}.

Thus the first selector-capable anisotropy appears at cubic order, and V32's
``C(h)=8*x*y*z/sqrt(3)`` is exactly the first non-radial tetrahedral invariant.

This still does not derive microscopic QCA condensation.  It narrows the
remaining physical input to the local vacuum entering the positive cubic
lowest-order tetrahedral Landau phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import permutations, product

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    generic_order_parameter,
    zero_order_parameter,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    coordinate_axis_controls,
    cubic_only_control_rejected,
    norm_squared,
    order_parameter_symbols,
    radial_only_control_rejected,
    selector_locking_potential,
    tetrahedral_cubic_polynomial,
    vacuum_selector_condensation_audit_payload,
    wrong_sign_control_rejected,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_potential import (
    tetrahedral_antipodal_controls,
    tetrahedral_cubic_invariant,
    tetrahedral_midpoint_controls,
    tetrahedral_selector_candidates,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3

REMAINING_DECLARED_INPUTS_AFTER_LANDAU_MINIMALITY = (
    "local_vacuum_enters_lowest_order_positive_tetrahedral_landau_phase",
)


def _same_expr(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.simplify(left - right) == 0


def _vector_from_matrix(vector: sp.Matrix) -> Vector3:
    return tuple(sp.simplify(vector[index, 0]) for index in range(3))  # type: ignore[return-value]


def _signed_permutation_matrix(
    permutation: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> sp.Matrix:
    matrix = sp.zeros(3, 3)
    for row, column in enumerate(permutation):
        matrix[row, column] = signs[row]
    return matrix


def _matrix_sort_key(matrix: sp.Matrix) -> tuple[str, ...]:
    return tuple(str(entry) for entry in matrix)


def _apply_rotation(matrix: sp.Matrix, vector: Vector3) -> Vector3:
    return _vector_from_matrix(matrix * sp.Matrix(vector))


@cache
def tetrahedral_rotation_group() -> tuple[sp.Matrix, ...]:
    """Return the 12 proper signed-permutation rotations of the BCC tetrahedron."""

    selectors = set(tetrahedral_selector_candidates())
    rotations: list[sp.Matrix] = []
    for permutation in permutations((0, 1, 2)):
        for signs in product((1, -1), repeat=3):
            matrix = _signed_permutation_matrix(permutation, signs)
            rotated = {_apply_rotation(matrix, selector) for selector in selectors}
            if matrix.det() == 1 and rotated == selectors:
                rotations.append(matrix)
    return tuple(sorted(rotations, key=_matrix_sort_key))


def rotations_are_proper() -> bool:
    """Return true when every tetrahedral rotation is orthogonal with determinant one."""

    identity = sp.eye(3)
    return all(
        matrix.T * matrix == identity and matrix.det() == 1
        for matrix in tetrahedral_rotation_group()
    )


def rotations_permute_selectors() -> bool:
    """Return true when every rotation permutes the four selector directions."""

    selectors = set(tetrahedral_selector_candidates())
    return all(
        {_apply_rotation(matrix, selector) for selector in selectors} == selectors
        for matrix in tetrahedral_rotation_group()
    )


@cache
def homogeneous_monomials(degree: int) -> tuple[sp.Expr, ...]:
    """Return the degree-``degree`` monomial basis in ``x,y,z``."""

    if degree < 0:
        raise ValueError("degree must be nonnegative")
    x, y, z = order_parameter_symbols()
    return tuple(
        x**power_x * y**power_y * z ** (degree - power_x - power_y)
        for power_x in range(degree + 1)
        for power_y in range(degree - power_x + 1)
    )


def transform_polynomial(polynomial: sp.Expr, rotation: sp.Matrix) -> sp.Expr:
    """Return ``polynomial(rotation * h)`` for ``h=(x,y,z)``."""

    x, y, z = order_parameter_symbols()
    transformed = rotation * sp.Matrix((x, y, z))
    return sp.expand(
        polynomial.subs(
            {
                x: transformed[0, 0],
                y: transformed[1, 0],
                z: transformed[2, 0],
            },
            simultaneous=True,
        )
    )


def polynomial_is_tetrahedral_invariant(polynomial: sp.Expr) -> bool:
    """Return true when ``polynomial`` is invariant under the tetrahedral group."""

    return all(
        _same_expr(transform_polynomial(polynomial, rotation), polynomial)
        for rotation in tetrahedral_rotation_group()
    )


@cache
def invariant_space_basis(degree: int) -> tuple[sp.Expr, ...]:
    """Return an exact basis for homogeneous tetrahedral invariants of a degree."""

    monomials = homogeneous_monomials(degree)
    coefficients = sp.symbols(f"c0:{len(monomials)}")
    x, y, z = order_parameter_symbols()
    polynomial = sum(
        coefficient * monomial
        for coefficient, monomial in zip(coefficients, monomials, strict=True)
    )

    rows: list[list[sp.Expr]] = []
    for rotation in tetrahedral_rotation_group():
        difference = sp.expand(transform_polynomial(polynomial, rotation) - polynomial)
        for coefficient in sp.Poly(difference, x, y, z).coeffs():
            row = [sp.expand(coefficient).coeff(symbol) for symbol in coefficients]
            if any(entry != 0 for entry in row):
                rows.append(row)

    if not rows:
        return tuple(monomials)

    nullspace = sp.Matrix(rows).nullspace()
    basis = []
    for vector in nullspace:
        basis.append(
            sp.expand(
                sum(
                    vector[index, 0] * monomials[index]
                    for index in range(len(monomials))
                )
            )
        )
    return tuple(basis)


def invariant_space_dimension(degree: int) -> int:
    """Return the dimension of homogeneous tetrahedral invariants of a degree."""

    return len(invariant_space_basis(degree))


def _polynomial_proportional(left: sp.Expr, right: sp.Expr) -> bool:
    x, y, z = order_parameter_symbols()
    left_poly = sp.Poly(sp.expand(left), x, y, z)
    right_poly = sp.Poly(sp.expand(right), x, y, z)
    monomials = set(left_poly.monoms()) | set(right_poly.monoms())

    ratio: sp.Expr | None = None
    for monomial in monomials:
        left_coeff = left_poly.coeff_monomial(monomial)
        right_coeff = right_poly.coeff_monomial(monomial)
        if right_coeff == 0:
            if left_coeff != 0:
                return False
            continue
        candidate_ratio = sp.simplify(left_coeff / right_coeff)
        if ratio is None:
            ratio = candidate_ratio
        elif not _same_expr(candidate_ratio, ratio):
            return False
    return ratio is not None and _same_expr(left - ratio * right, 0)


def degree_two_invariant_is_radial() -> bool:
    """Return true when degree-two invariants are exactly radial."""

    basis = invariant_space_basis(2)
    radial = norm_squared(order_parameter_symbols())
    return len(basis) == 1 and _polynomial_proportional(basis[0], radial)


def degree_three_invariant_matches_cubic() -> bool:
    """Return true when degree-three invariants are exactly the tetrahedral cubic."""

    basis = invariant_space_basis(3)
    x, y, z = order_parameter_symbols()
    return (
        len(basis) == 1
        and _polynomial_proportional(basis[0], x * y * z)
        and _polynomial_proportional(tetrahedral_cubic_polynomial(), x * y * z)
    )


def lowest_selector_anisotropy_degree() -> int:
    """Return the first homogeneous degree that can distinguish selector branches."""

    if (
        invariant_space_dimension(1) == 0
        and degree_two_invariant_is_radial()
        and degree_three_invariant_matches_cubic()
    ):
        return 3
    raise ValueError("tetrahedral invariant audit did not find the expected degree")


def tetrahedral_landau_locking_potential(order_parameter: Vector3) -> sp.Expr:
    """Return the canonical lowest-order positive tetrahedral Landau lock."""

    return sp.simplify(
        (norm_squared(order_parameter) - 1) ** 2
        + (tetrahedral_cubic_invariant(order_parameter) - sp.Rational(8, 9)) ** 2
    )


def landau_lock_matches_v32() -> bool:
    """Return true when the V33 Landau lock exactly recovers V32's lock."""

    checks = (
        tetrahedral_selector_candidates()
        + tetrahedral_antipodal_controls()
        + coordinate_axis_controls()
        + tetrahedral_midpoint_controls()
        + (zero_order_parameter(), generic_order_parameter())
    )
    return all(
        _same_expr(
            tetrahedral_landau_locking_potential(candidate),
            selector_locking_potential(candidate),
        )
        for candidate in checks
    )


def landau_lock_zero_only_at_selectors() -> bool:
    """Return true when the Landau lock zero set matches V32 on audited controls."""

    selectors = tetrahedral_selector_candidates()
    controls = (
        tetrahedral_antipodal_controls()
        + coordinate_axis_controls()
        + tetrahedral_midpoint_controls()
        + (zero_order_parameter(), generic_order_parameter())
    )
    return all(tetrahedral_landau_locking_potential(selector) == 0 for selector in selectors) and all(
        sp.simplify(tetrahedral_landau_locking_potential(control)) > 0
        for control in controls
    )


def non_tetrahedral_low_order_controls_rejected() -> bool:
    """Return true when ad hoc linear/quadratic anisotropies are not invariants."""

    x, y, _z = order_parameter_symbols()
    return (
        not polynomial_is_tetrahedral_invariant(x)
        and not polynomial_is_tetrahedral_invariant(x**2 - y**2)
    )


def v32_recovered() -> bool:
    """Return true when the V32 continuous selector-condensation gate passes."""

    return (
        vacuum_selector_condensation_audit_payload().final_verdict
        == "CONTINUOUS_TETRAHEDRAL_SELECTOR_CONDENSATION_PASS"
    )


@dataclass(frozen=True)
class TetrahedralLandauAuditPayload:
    """Verdict payload for the V33 tetrahedral Landau minimality gate."""

    final_verdict: str
    rotation_group_size: int
    rotations_are_proper: bool
    rotations_permute_selectors: bool
    degree_one_invariant_dimension: int
    degree_two_invariant_dimension: int
    degree_three_invariant_dimension: int
    degree_two_invariant_radial: bool
    degree_three_invariant_matches_cubic: bool
    lowest_selector_anisotropy_degree: int | None
    landau_lock_matches_v32: bool
    selector_zero_set_matches_v32: bool
    radial_only_control_rejected: bool
    cubic_only_control_rejected: bool
    wrong_sign_control_rejected: bool
    non_tetrahedral_low_order_controls_rejected: bool
    v32_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def tetrahedral_landau_audit_payload() -> TetrahedralLandauAuditPayload:
    """Return the V33 tetrahedral invariant Landau-minimality verdict."""

    group = tetrahedral_rotation_group()
    proper = rotations_are_proper()
    permutes = rotations_permute_selectors()
    degree_one = invariant_space_dimension(1)
    degree_two = invariant_space_dimension(2)
    degree_three = invariant_space_dimension(3)
    radial_degree_two = degree_two_invariant_is_radial()
    cubic_degree_three = degree_three_invariant_matches_cubic()
    try:
        lowest_degree: int | None = lowest_selector_anisotropy_degree()
    except ValueError:
        lowest_degree = None

    lock_matches = landau_lock_matches_v32()
    zero_set_matches = landau_lock_zero_only_at_selectors()
    radial_rejected = radial_only_control_rejected()
    cubic_rejected = cubic_only_control_rejected()
    wrong_sign_rejected = wrong_sign_control_rejected()
    low_order_controls_rejected = non_tetrahedral_low_order_controls_rejected()
    v32 = v32_recovered()

    checks_pass = (
        len(group) == 12
        and proper
        and permutes
        and degree_one == 0
        and degree_two == 1
        and degree_three == 1
        and radial_degree_two
        and cubic_degree_three
        and lowest_degree == 3
        and lock_matches
        and zero_set_matches
        and radial_rejected
        and cubic_rejected
        and wrong_sign_rejected
        and low_order_controls_rejected
        and v32
    )

    if checks_pass:
        final_verdict = "TETRAHEDRAL_INVARIANT_LANDAU_MINIMALITY_PASS"
        interpretation = (
            "The proper tetrahedral rotation group has 12 exact signed "
            "permutation rotations preserving the BCC selector directions. "
            "Its degree-one invariant space is zero, degree two is only "
            "radial, and the first selector-capable anisotropy is the "
            "degree-three tetrahedral cubic. The canonical lowest-order "
            "positive Landau lock exactly recovers V32's selector lock. "
            "The remaining physical input is that the local vacuum enters "
            "this positive cubic tetrahedral Landau phase."
        )
    else:
        final_verdict = "TETRAHEDRAL_INVARIANT_LANDAU_MINIMALITY_KILL"
        interpretation = (
            "The tetrahedral group, invariant-space dimensions, cubic "
            "minimality, Landau lock, controls, or V32 regression failed."
        )

    return TetrahedralLandauAuditPayload(
        final_verdict=final_verdict,
        rotation_group_size=len(group),
        rotations_are_proper=proper,
        rotations_permute_selectors=permutes,
        degree_one_invariant_dimension=degree_one,
        degree_two_invariant_dimension=degree_two,
        degree_three_invariant_dimension=degree_three,
        degree_two_invariant_radial=radial_degree_two,
        degree_three_invariant_matches_cubic=cubic_degree_three,
        lowest_selector_anisotropy_degree=lowest_degree,
        landau_lock_matches_v32=lock_matches,
        selector_zero_set_matches_v32=zero_set_matches,
        radial_only_control_rejected=radial_rejected,
        cubic_only_control_rejected=cubic_rejected,
        wrong_sign_control_rejected=wrong_sign_rejected,
        non_tetrahedral_low_order_controls_rejected=low_order_controls_rejected,
        v32_recovered=v32,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_LANDAU_MINIMALITY,
        interpretation=interpretation,
    )
