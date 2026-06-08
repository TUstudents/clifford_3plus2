"""Session 02 chiral D3 clock source certificate.

The goal of this module is deliberately narrow:

* prove the exact D3 clock identities that select the tangent current ``b``;
* record the up/down quark heads as conditional theorem targets;
* keep controls that prevent a source identity from becoming an unproved mass
  theorem.

No mass data are used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.threeclocks.clock_spine import matrix_equal


def standard_basis() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return the selected three-port basis ``(e1,e2,e3)``."""

    return (
        sp.Matrix([1, 0, 0]),
        sp.Matrix([0, 1, 0]),
        sp.Matrix([0, 0, 1]),
    )


def residual_basis() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return the trace/radial/tangent basis ``(u,a,b)``."""

    return (
        sp.Matrix([1, 1, 1]) / sp.sqrt(3),
        sp.Matrix([2, -1, -1]) / sp.sqrt(6),
        sp.Matrix([0, 1, -1]) / sp.sqrt(2),
    )


def residual_frame() -> sp.Matrix:
    """Return the orthonormal frame whose columns are ``(u,a,b)``."""

    return sp.Matrix.hstack(*residual_basis())


def cyclic_clock_matrix() -> sp.Matrix:
    """Return ``C`` with ``C e1=e2, C e2=e3, C e3=e1``."""

    return sp.Matrix(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )


def reflection_matrix() -> sp.Matrix:
    """Return the D3 reflection fixing ``e1`` and swapping ``e2,e3``."""

    return sp.Matrix(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )


def d3_relations_hold() -> bool:
    """Return whether ``C`` and ``rho`` satisfy the exact D3 relations."""

    c = cyclic_clock_matrix()
    rho = reflection_matrix()
    identity = sp.eye(3)
    return (
        matrix_equal(c**3, identity)
        and matrix_equal(rho**2, identity)
        and matrix_equal(rho * c * rho, c.inv())
    )


def selected_tooth_components() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return components of ``e1`` in the ``(u,a,b)`` frame."""

    e1, _, _ = standard_basis()
    frame = residual_frame()
    components = frame.T * e1
    return tuple(sp.simplify(value) for value in components)


def tangent_current_from_clock() -> sp.Matrix:
    """Return ``(C e1 - C^{-1} e1)/sqrt(2)``."""

    c = cyclic_clock_matrix()
    e1, _, _ = standard_basis()
    return sp.simplify((c * e1 - c.inv() * e1) / sp.sqrt(2))


def radial_second_difference_from_clock() -> sp.Matrix:
    """Return ``(2 e1 - C e1 - C^{-1} e1)/sqrt(6)``."""

    c = cyclic_clock_matrix()
    e1, _, _ = standard_basis()
    return sp.simplify((2 * e1 - c * e1 - c.inv() * e1) / sp.sqrt(6))


def tangent_identity_holds() -> bool:
    """Return whether the oriented first difference is exactly ``b``."""

    _, _, b = residual_basis()
    return matrix_equal(tangent_current_from_clock(), b)


def radial_identity_holds() -> bool:
    """Return whether the second difference is exactly ``a``."""

    _, a, _ = residual_basis()
    return matrix_equal(radial_second_difference_from_clock(), a)


def three_port_laplacian() -> sp.Matrix:
    """Return ``J^dagger J = 2I - C - C^{-1}`` on the three ports."""

    c = cyclic_clock_matrix()
    return 2 * sp.eye(3) - c - c.inv()


def three_port_laplacian_spectrum() -> tuple[sp.Expr, ...]:
    """Return the exact three-port Laplacian spectrum with multiplicity."""

    eigenvalues = three_port_laplacian().eigenvals()
    expanded: list[sp.Expr] = []
    for value, multiplicity in eigenvalues.items():
        expanded.extend([sp.simplify(value)] * multiplicity)
    return tuple(sorted(expanded, key=lambda value: float(value)))


def nilpotent_repair_flag_target() -> sp.Matrix:
    """Return the target flag ``N=|u><a|+|a><b|`` in ``(u,a,b)`` coordinates."""

    return sp.Matrix(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )


def literal_port_cut_matrix() -> sp.Matrix:
    """Return the literal cut ``e1->e2->e3->0`` in the port basis."""

    return sp.Matrix(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )


def literal_port_cut_in_residual_frame() -> sp.Matrix:
    """Return the literal port cut expressed in the ``(u,a,b)`` frame."""

    frame = residual_frame()
    return sp.simplify(frame.T * literal_port_cut_matrix() * frame)


def literal_port_cut_equals_repair_flag() -> bool:
    """Return whether the literal port cut already equals the repair flag."""

    return matrix_equal(literal_port_cut_in_residual_frame(), nilpotent_repair_flag_target())


def repair_flag_is_nilpotent() -> bool:
    """Return whether the target repair flag has ``N^3=0`` and ``N^2 != 0``."""

    n_flag = nilpotent_repair_flag_target()
    return matrix_equal(n_flag**3, sp.zeros(3, 3)) and not matrix_equal(
        n_flag**2, sp.zeros(3, 3)
    )


def first_return_orders_from_b_target() -> tuple[int, int, int]:
    """Return first-hit orders from ``b`` to ``(u,a,b)`` under target ``N``."""

    n_flag = nilpotent_repair_flag_target()
    source_b = sp.Matrix([0, 0, 1])
    orders: list[int] = []
    for coordinate in range(3):
        for order in range(3):
            vector = n_flag**order * source_b
            if sp.simplify(vector[coordinate]) != 0:
                orders.append(order)
                break
        else:
            raise ValueError("target flag did not reach one residual coordinate")
    return tuple(orders)


def up_profile_if_repair_flag_supplied() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the conditional up profile in light-to-heavy order ``(u,a,b)``."""

    x = 1 / sp.sqrt(2)
    n_flag = nilpotent_repair_flag_target()
    source_b = sp.Matrix([0, 0, 1])
    finite_head = sp.eye(3) + x * n_flag + x**2 * n_flag**2 / 2
    profile = finite_head * source_b
    return tuple(sp.simplify(value) for value in profile)


def down_profile_if_rank_five_shell_supplied() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the conditional down shell profile from ranks ``(6,2,5)``."""

    return (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(5, 6)))


def down_profile_if_contact_return_allowed() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the baseline down shell profile from ranks ``(6,2,4)``."""

    return (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)))


@dataclass(frozen=True)
class D3ClockSourcePayload:
    """Session 02 source-identity payload."""

    final_verdict: str
    d3_relations_pass: bool
    selected_tooth_components: tuple[sp.Expr, sp.Expr, sp.Expr]
    selected_tooth_has_no_tangent: bool
    tangent_current_is_b: bool
    radial_second_difference_is_a: bool
    quark_source_b_derived: bool
    three_port_laplacian_spectrum: tuple[sp.Expr, ...]
    three_port_laplacian_is_degenerate_control: bool
    literal_cut_equals_repair_flag: bool
    repair_flag_nilpotent_target_pass: bool
    first_return_orders_target: tuple[int, int, int]
    up_profile_conditional: tuple[sp.Expr, sp.Expr, sp.Expr]
    down_rank_five_profile_conditional: tuple[sp.Expr, sp.Expr, sp.Expr]
    down_contact_allowed_profile_control: tuple[sp.Expr, sp.Expr, sp.Expr]
    open_theorem_targets: tuple[str, ...]
    interpretation: str


def d3_clock_source_payload() -> D3ClockSourcePayload:
    """Return the Session 02 D3 clock source certificate."""

    components = selected_tooth_components()
    spectrum = three_port_laplacian_spectrum()
    literal_cut_equals_flag = literal_port_cut_equals_repair_flag()
    source_identity_pass = tangent_identity_holds() and radial_identity_holds()
    open_targets = (
        "derive_representation_basis_repair_flag_N_from_clock_defect",
        "embed_B_plus_tensor_C_plus_B_minus_tensor_C_inverse_in_unitary_dilation",
        "select_active_quark_shell_over_spectator_embedding",
        "derive_bottom_contact_veto_from_retarded_boundary_current",
        "derive_closure_exponents_k_u_3n_and_k_d_2n_plus_2",
        "compute_CKM_from_two_sided_clock_kernels",
    )

    return D3ClockSourcePayload(
        final_verdict="D3_CLOCK_SOURCE_IDENTITY_PASS",
        d3_relations_pass=d3_relations_hold(),
        selected_tooth_components=components,
        selected_tooth_has_no_tangent=sp.simplify(components[2]) == 0,
        tangent_current_is_b=tangent_identity_holds(),
        radial_second_difference_is_a=radial_identity_holds(),
        quark_source_b_derived=source_identity_pass,
        three_port_laplacian_spectrum=spectrum,
        three_port_laplacian_is_degenerate_control=spectrum == (0, 3, 3),
        literal_cut_equals_repair_flag=literal_cut_equals_flag,
        repair_flag_nilpotent_target_pass=repair_flag_is_nilpotent(),
        first_return_orders_target=first_return_orders_from_b_target(),
        up_profile_conditional=up_profile_if_repair_flag_supplied(),
        down_rank_five_profile_conditional=down_profile_if_rank_five_shell_supplied(),
        down_contact_allowed_profile_control=down_profile_if_contact_return_allowed(),
        open_theorem_targets=open_targets,
        interpretation=(
            "The D3 clock exactly selects b as the oriented tangent current and a as "
            "the radial second difference at the selected tooth.  It does not yet "
            "derive the representation-basis repair flag, the active down shell, "
            "the retarded contact veto, closure exponents, or CKM."
        ),
    )
