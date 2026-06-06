"""Session 03 neutrino product-bath gate.

This module rederives the neutrino product bath from the Session 02 frozen
sources and the universal period-one tail.  It does not import the upstream
``boundary_response`` product-sterile verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    projector,
    residual_basis_matrix,
    residual_projectors,
    residual_vectors,
)
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceAnchor,
    source_dictionary_payload,
)
from clifford_3plus2_d5.universal_bath.tail import (
    period_one_tail,
    silver_epsilon,
    silver_selected_z,
    tail_fixed_point_residual,
)


NEUTRINO_SOURCE_LABELS = ("neutrino_collective_u", "neutrino_edge_b")


@dataclass(frozen=True)
class NeutrinoProductBathPayload:
    """Session 03 neutrino product-bath verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    frozen_neutrino_sources: tuple[str, ...]
    checked_moment_powers: tuple[int, ...]
    diagonal_moments_equal: bool
    cross_moments_zero: bool
    fixed_point_residual: sp.Expr
    tail_value: sp.Expr
    tail_value_matches_epsilon: bool
    response_matches_target: bool
    response_diagonal: sp.Matrix
    mass_ratio: sp.Expr
    mass_squared_ratio: sp.Expr
    rank_one_control_has_cross_return: bool
    wrong_source_control_rejected: bool
    alternate_tail_control_rejected: bool
    pmns_ckm_parked: bool
    interpretation: str


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices agree after exact simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def frozen_neutrino_sources() -> tuple[SourceAnchor, SourceAnchor]:
    """Return the frozen Session 02 neutrino ``u`` and ``b`` source anchors."""

    payload = source_dictionary_payload()
    anchors = {anchor.label: anchor for anchor in payload.frozen_sources}
    return tuple(anchors[label] for label in NEUTRINO_SOURCE_LABELS)  # type: ignore[return-value]


def finite_path_adjacency(shells: int) -> sp.Matrix:
    """Return the finite nearest-neighbor half-line truncation."""

    if shells < 1:
        raise ValueError("shells must be positive")
    matrix = sp.zeros(shells, shells)
    for index in range(shells - 1):
        matrix[index, index + 1] = 1
        matrix[index + 1, index] = 1
    return matrix


def finite_product_hamiltonian(shells: int) -> sp.Matrix:
    """Return ``H_chain^(N) tensor I_family``."""

    return sp.kronecker_product(finite_path_adjacency(shells), sp.eye(3))


def shell_family_state(shells: int, shell: int, family_vector: sp.Matrix) -> sp.Matrix:
    """Return ``|shell> tensor |family_vector>`` in shell-major order."""

    if shells < 1:
        raise ValueError("shells must be positive")
    if shell < 0 or shell >= shells:
        raise ValueError("shell index out of range")
    if family_vector.shape != (3, 1):
        raise ValueError("family_vector must be a 3x1 column")

    state = sp.zeros(3 * shells, 1)
    start = 3 * shell
    state[start : start + 3, 0] = family_vector
    return state.applyfunc(sp.simplify)


def neutrino_product_states(shells: int) -> dict[str, sp.Matrix]:
    """Return product-bath states for the frozen neutrino sources."""

    collective, edge = frozen_neutrino_sources()
    if collective.port_vector is None or edge.port_vector is None:
        raise ValueError("neutrino sources must be frozen")
    return {
        collective.label: shell_family_state(shells, 0, collective.port_vector),
        edge.label: shell_family_state(shells, 0, edge.port_vector),
    }


def product_return_moment(
    shells: int,
    power: int,
    left_label: str,
    right_label: str,
) -> sp.Expr:
    """Return ``<left|H_Q^power|right>`` for product-bath states."""

    if power < 0:
        raise ValueError("power must be non-negative")
    states = neutrino_product_states(shells)
    h_q = finite_product_hamiltonian(shells)
    left = states[left_label]
    right = states[right_label]
    return sp.simplify((left.T * (h_q**power) * right)[0])


def product_return_matrix(shells: int, power: int) -> sp.Matrix:
    """Return the ``(u,b)`` return-moment matrix at a given power."""

    return sp.Matrix(
        [
            [
                product_return_moment(shells, power, left, right)
                for right in NEUTRINO_SOURCE_LABELS
            ]
            for left in NEUTRINO_SOURCE_LABELS
        ]
    )


def cross_moments(shells: int, powers: tuple[int, ...] = (0, 1, 2, 3, 4)) -> tuple[sp.Expr, ...]:
    """Return ``u``-``b`` cross-return moments for the requested powers."""

    return tuple(
        product_return_moment(shells, power, "neutrino_collective_u", "neutrino_edge_b")
        for power in powers
    )


def diagonal_moment_differences(
    shells: int,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[sp.Expr, ...]:
    """Return ``u`` diagonal minus ``b`` diagonal return moments."""

    return tuple(
        sp.simplify(
            product_return_moment(shells, power, "neutrino_collective_u", "neutrino_collective_u")
            - product_return_moment(shells, power, "neutrino_edge_b", "neutrino_edge_b")
        )
        for power in powers
    )


def neutrino_tail_response(z: sp.Expr) -> sp.Matrix:
    """Return the normalized product-bath neutrino response."""

    tail = period_one_tail(z)
    projectors = residual_projectors()
    return sp.simplify(tail**2 * projectors["u"] + projectors["b"])


def neutrino_target_response() -> sp.Matrix:
    """Return the selected-probe target ``epsilon^2 P_u + P_b``."""

    projectors = residual_projectors()
    return sp.simplify(silver_epsilon() ** 2 * projectors["u"] + projectors["b"])


def neutrino_response_diagonal(z: sp.Expr) -> sp.Matrix:
    """Return the response in the residual ``(a,u,b)`` basis."""

    basis = residual_basis_matrix(("a", "u", "b"))
    response = neutrino_tail_response(z)
    return (basis.T * response * basis).applyfunc(sp.simplify)


def rank_one_no_family_response(z: sp.Expr | None = None) -> sp.Matrix:
    """Return the rank-one control with the family factor removed."""

    probe = silver_selected_z() if z is None else z
    tail = period_one_tail(probe)
    vectors = residual_vectors()
    direction = tail * vectors["u"] + vectors["b"]
    return sp.simplify(projector(direction))


def rank_one_control_has_cross_return(z: sp.Expr | None = None) -> bool:
    """Return true when the rank-one control contaminates the ``u``/``b`` block."""

    basis = residual_basis_matrix(("a", "u", "b"))
    response = rank_one_no_family_response(z)
    in_basis = (basis.T * response * basis).applyfunc(sp.simplify)
    return sp.simplify(in_basis[1, 2]) != 0 or sp.simplify(in_basis[2, 1]) != 0


def wrong_source_control_response(z: sp.Expr | None = None) -> sp.Matrix:
    """Return a wrong-source control that uses ``a`` instead of ``b``."""

    probe = silver_selected_z() if z is None else z
    tail = period_one_tail(probe)
    projectors = residual_projectors()
    return sp.simplify(tail**2 * projectors["u"] + projectors["a"])


def alternate_tail_control_response() -> sp.Matrix:
    """Return the target-form response with a non-silver constant tail."""

    alternate = sp.Rational(1, 2)
    projectors = residual_projectors()
    return sp.simplify(alternate**2 * projectors["u"] + projectors["b"])


def neutrino_product_bath_payload(
    *,
    shells: int = 6,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> NeutrinoProductBathPayload:
    """Return the Session 03 neutrino product-bath verdict."""

    source_payload = source_dictionary_payload()
    source_pass = source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
    labels = tuple(anchor.label for anchor in frozen_neutrino_sources())
    cross_zero = all(sp.simplify(moment) == 0 for moment in cross_moments(shells, powers))
    diagonal_equal = all(
        sp.simplify(difference) == 0 for difference in diagonal_moment_differences(shells, powers)
    )

    z = silver_selected_z()
    tail_value = sp.simplify(period_one_tail(z))
    fixed_residual = tail_fixed_point_residual(sp.Symbol("z"))
    tail_matches = sp.simplify(tail_value - silver_epsilon()) == 0
    response = neutrino_tail_response(z)
    target = neutrino_target_response()
    response_matches = _matrix_equal(response, target)
    diagonal = neutrino_response_diagonal(z)
    rank_one_cross = rank_one_control_has_cross_return(z)
    wrong_source_rejected = not _matrix_equal(wrong_source_control_response(z), target)
    alternate_rejected = not _matrix_equal(alternate_tail_control_response(), target)

    checks_pass = (
        source_pass
        and labels == NEUTRINO_SOURCE_LABELS
        and cross_zero
        and diagonal_equal
        and fixed_residual == 0
        and tail_matches
        and response_matches
        and rank_one_cross
        and wrong_source_rejected
        and alternate_rejected
    )

    if checks_pass:
        final_verdict = "NEUTRINO_PRODUCT_BATH_CORE_PASS"
        interpretation = (
            "With the Session 02 frozen neutrino sources, the product bath "
            "H_chain tensor I_family makes the u and b diagonal return moments "
            "equal and all checked u/b cross-return moments vanish exactly. "
            "The normalized semi-infinite response is t(z)^2 P_u + P_b; at "
            "z=2 sqrt(2), t=epsilon, giving epsilon^2 P_u + P_b and the "
            "neutrino mass-squared ratio epsilon^4.  The result is C:9 inside "
            "the product half-line bath and C:7 as a physical source model "
            "until product factorization is derived microscopically."
        )
    else:
        final_verdict = "NEUTRINO_PRODUCT_BATH_KILL"
        interpretation = (
            "The neutrino product bath failed the source prerequisite, "
            "return-moment factorization, Weyl-tail target, or negative "
            "control gate."
        )

    return NeutrinoProductBathPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        frozen_neutrino_sources=labels,
        checked_moment_powers=powers,
        diagonal_moments_equal=diagonal_equal,
        cross_moments_zero=cross_zero,
        fixed_point_residual=fixed_residual,
        tail_value=tail_value,
        tail_value_matches_epsilon=tail_matches,
        response_matches_target=response_matches,
        response_diagonal=diagonal,
        mass_ratio=sp.simplify(silver_epsilon() ** 2),
        mass_squared_ratio=sp.simplify(silver_epsilon() ** 4),
        rank_one_control_has_cross_return=rank_one_cross,
        wrong_source_control_rejected=wrong_source_rejected,
        alternate_tail_control_rejected=alternate_rejected,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
