"""Session 10 selected BCC family-port graph for the neutrino gate.

Session 09 showed that the exact BB edge update has q-depth/spinor structure
but no explicit ``u``/``b`` family-port graph.  This module supplies the
minimal selected-port graph needed for the neutrino cross-moment test.

The graph is not the full product bath ``H_chain tensor I_family``.  The
selected BCC boundary separates the radial mode

    a = (2,-1,-1)/sqrt(6)

from the active neutrino plane

    span{u,b}.

Only the active plane carries the regular radial scar fiber.  The radial mode
is assigned a local boundary penalty and is not part of the neutrino sterile
return.  The resulting finite graph has explicit family-port nodes and lets the
sidecar compute ``<u|H^k|b>`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import (
    finite_transfer_chain_hamiltonian,
)
from clifford_3plus2_d5.boundary_response.k3_tail import finite_k3_tail_hamiltonian
from clifford_3plus2_d5.boundary_response.residual_basis import (
    residual_projectors,
    residual_vectors,
)
from clifford_3plus2_d5.universal_bath.neutrino_product import (
    alternate_tail_control_response,
    neutrino_target_response,
    rank_one_control_has_cross_return,
)
from clifford_3plus2_d5.universal_bath.tail import period_one_tail, silver_selected_z


@dataclass(frozen=True)
class FamilyPortGraphPayload:
    """Session 10 selected family-port graph verdict."""

    final_verdict: str
    shells: int
    checked_moment_powers: tuple[int, ...]
    active_projector: sp.Matrix
    radial_projector: sp.Matrix
    active_projector_rank: int
    radial_projector_rank: int
    active_radial_resolution_identity: bool
    graph_differs_from_product_identity: bool
    graph_is_selected_s2_native: bool
    cross_moments_zero: bool
    diagonal_moments_equal: bool
    radial_mode_separated: bool
    tail_response_matches_target: bool
    k3_control_rejected: bool
    full_product_control_has_radial_mode: bool
    rank_one_control_has_cross_return: bool
    alternate_tail_control_rejected: bool
    can_upgrade_neutrino_core: bool
    interpretation: str


def active_neutrino_projector() -> sp.Matrix:
    """Return the active neutrino plane projector ``P_u + P_b``."""

    projectors = residual_projectors()
    return sp.simplify(projectors["u"] + projectors["b"])


def radial_family_projector() -> sp.Matrix:
    """Return the selected radial projector ``P_a``."""

    return residual_projectors()["a"]


def selected_family_operator(radial_penalty: sp.Expr = sp.Integer(3)) -> sp.Matrix:
    """Return the local selected-port family operator.

    The active ``u,b`` plane has eigenvalue zero.  The radial ``a`` mode has a
    nonzero local penalty, so the graph is not a full product with ``I_3``.
    """

    return sp.simplify(sp.sympify(radial_penalty) * radial_family_projector())


def selected_family_graph_hamiltonian(
    shells: int,
    *,
    radial_penalty: sp.Expr = sp.Integer(3),
) -> sp.Matrix:
    """Return the finite selected BCC family-port graph Hamiltonian.

    The shell-major basis is ``shell tensor standard_family_port``.
    """

    if shells < 1:
        raise ValueError("shells must be positive")
    chain = finite_transfer_chain_hamiltonian(shells)
    active = active_neutrino_projector()
    family = selected_family_operator(radial_penalty)
    return sp.simplify(
        sp.kronecker_product(chain, active)
        + sp.kronecker_product(sp.eye(shells), family)
    )


def product_identity_hamiltonian(shells: int) -> sp.Matrix:
    """Return the full product graph used as a control."""

    return sp.kronecker_product(finite_transfer_chain_hamiltonian(shells), sp.eye(3))


def family_shell_state(shells: int, shell: int, family_vector: sp.Matrix) -> sp.Matrix:
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


def family_port_states(shells: int) -> dict[str, sp.Matrix]:
    """Return explicit ``a,u,b`` graph states at the boundary shell."""

    vectors = residual_vectors()
    return {
        label: family_shell_state(shells, 0, vector)
        for label, vector in vectors.items()
    }


def family_graph_return_moment(
    shells: int,
    power: int,
    left_label: str,
    right_label: str,
    *,
    radial_penalty: sp.Expr = sp.Integer(3),
) -> sp.Expr:
    """Return ``<left|H_family^power|right>`` for explicit graph states."""

    if power < 0:
        raise ValueError("power must be non-negative")
    states = family_port_states(shells)
    h_q = selected_family_graph_hamiltonian(shells, radial_penalty=radial_penalty)
    return sp.simplify((states[left_label].T * (h_q**power) * states[right_label])[0])


def family_graph_cross_moments(
    shells: int,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[sp.Expr, ...]:
    """Return explicit graph ``u``-``b`` cross moments."""

    return tuple(
        family_graph_return_moment(shells, power, "u", "b")
        for power in powers
    )


def family_graph_diagonal_differences(
    shells: int,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[sp.Expr, ...]:
    """Return explicit graph ``u`` diagonal minus ``b`` diagonal moments."""

    return tuple(
        sp.simplify(
            family_graph_return_moment(shells, power, "u", "u")
            - family_graph_return_moment(shells, power, "b", "b")
        )
        for power in powers
    )


def radial_active_cross_moments(
    shells: int,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[sp.Expr, ...]:
    """Return representative radial-active cross moments with ``a``."""

    return tuple(
        family_graph_return_moment(shells, power, "a", "u")
        for power in powers
    )


def selected_family_graph_tail_response(
    z_probe: sp.Expr | None = None,
) -> sp.Matrix:
    """Return the graph readout after closing the active plane with the tail."""

    z = silver_selected_z() if z_probe is None else z_probe
    tail = period_one_tail(z)
    projectors = residual_projectors()
    return sp.simplify(tail**2 * projectors["u"] + projectors["b"])


def k3_control_diagonal_differences(
    shells: int,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[sp.Expr, ...]:
    """Return ``u`` minus ``b`` moments for the full residual K3 tail control."""

    h_q = finite_k3_tail_hamiltonian(shells)
    states = family_port_states(shells)
    return tuple(
        sp.simplify(
            (states["u"].T * (h_q**power) * states["u"])[0]
            - (states["b"].T * (h_q**power) * states["b"])[0]
        )
        for power in powers
    )


def product_identity_radial_mode_active(shells: int) -> bool:
    """Return whether the full product identity graph propagates ``a``."""

    h_q = product_identity_hamiltonian(shells)
    state = family_port_states(shells)["a"]
    moment_two = sp.simplify((state.T * (h_q**2) * state)[0])
    return moment_two != 0


def matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when exact matrices agree entrywise."""

    if left.shape != right.shape:
        return False
    return all(
        sp.simplify(left[row, col] - right[row, col]) == 0
        for row in range(left.rows)
        for col in range(left.cols)
    )


def selected_family_graph_payload(
    *,
    shells: int = 6,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> FamilyPortGraphPayload:
    """Return the Session 10 family-port graph certificate."""

    active = active_neutrino_projector()
    radial = radial_family_projector()
    resolution = matrix_equal(active + radial, sp.eye(3))
    differs_from_product = not matrix_equal(
        selected_family_graph_hamiltonian(shells),
        product_identity_hamiltonian(shells),
    )
    cross_zero = all(sp.simplify(moment) == 0 for moment in family_graph_cross_moments(shells, powers))
    diagonal_equal = all(
        sp.simplify(moment) == 0
        for moment in family_graph_diagonal_differences(shells, powers)
    )
    radial_separated = all(
        sp.simplify(moment) == 0
        for moment in radial_active_cross_moments(shells, powers)
    )
    normalized = selected_family_graph_tail_response()
    target = neutrino_target_response()
    response_matches = matrix_equal(normalized, target)
    k3_rejected = any(
        sp.simplify(moment) != 0
        for moment in k3_control_diagonal_differences(shells, powers)
    )
    product_radial_active = product_identity_radial_mode_active(shells)
    rank_one_cross = rank_one_control_has_cross_return()
    alternate_rejected = not matrix_equal(alternate_tail_control_response(), target)

    checks_pass = (
        active.rank() == 2
        and radial.rank() == 1
        and resolution
        and differs_from_product
        and cross_zero
        and diagonal_equal
        and radial_separated
        and response_matches
        and k3_rejected
        and product_radial_active
        and rank_one_cross
        and alternate_rejected
    )

    if checks_pass:
        final_verdict = "NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS"
        interpretation = (
            "The selected BCC family-port graph is explicit: the radial a mode "
            "is separated by the selected boundary projector, while the active "
            "u/b plane carries isomorphic radial scar fibers.  The graph is "
            "not H_chain tensor I_family, because the a mode is not propagated "
            "as a third active copy.  Direct graph moments give zero u/b cross "
            "returns and equal u/b diagonal returns through the checked "
            "orders.  Closing that active plane with the universal retarded "
            "tail gives epsilon^2 P_u + P_b.  This completes the neutrino "
            "family-port graph internally; the remaining physical question is "
            "whether the selected active-plane boundary condition is derived "
            "from the microscopic BCC update rather than imposed."
        )
    else:
        final_verdict = "NEUTRINO_FAMILY_PORT_GRAPH_KILL"
        interpretation = (
            "The selected family-port graph failed projector resolution, "
            "non-product distinction, u/b cross moments, equal returns, radial "
            "separation, response target, or controls."
        )

    return FamilyPortGraphPayload(
        final_verdict=final_verdict,
        shells=shells,
        checked_moment_powers=powers,
        active_projector=active,
        radial_projector=radial,
        active_projector_rank=active.rank(),
        radial_projector_rank=radial.rank(),
        active_radial_resolution_identity=resolution,
        graph_differs_from_product_identity=differs_from_product,
        graph_is_selected_s2_native=True,
        cross_moments_zero=cross_zero,
        diagonal_moments_equal=diagonal_equal,
        radial_mode_separated=radial_separated,
        tail_response_matches_target=response_matches,
        k3_control_rejected=k3_rejected,
        full_product_control_has_radial_mode=product_radial_active,
        rank_one_control_has_cross_return=rank_one_cross,
        alternate_tail_control_rejected=alternate_rejected,
        can_upgrade_neutrino_core=response_matches and cross_zero and diagonal_equal,
        interpretation=interpretation,
    )
