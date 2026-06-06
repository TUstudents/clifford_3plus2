"""Session 22 quark current-parity selector.

Session 21 froze a useful quark-source candidate:

    quark source = active current line b.

This session removes one layer of ansatz from that statement.  Once a selected
family port is fixed, the residual symmetry preserving it is the ``S_2`` swap
of the two unselected ports.  In the residual basis

    u = (1,1,1)/sqrt(3),  a = (2,-1,-1)/sqrt(6),  b = (0,1,-1)/sqrt(2),

that selected ``S_2`` acts as

    u -> u,   a -> a,   b -> -b.

Thus ``b`` is not merely the active non-scalar line.  It is the unique
selected-``S_2`` odd current line.  The remaining physical premise is smaller:
a colored quark mass source must be represented by this odd boundary current,
not by an even scalar or radial incidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_zero_matrix,
    residual_basis_matrix,
    residual_projectors,
    residual_vectors,
    selected_port_s2_matrices,
    standard_basis,
)
from clifford_3plus2_d5.universal_bath.neutrino_active_plane import (
    active_plane_incidence_payload,
)

ODD_CURRENT_PHYSICAL_PREMISE = "colored_quark_mass_source_is_selected_S2_odd_boundary_current"


@dataclass(frozen=True)
class QuarkCurrentParitySelectorPayload:
    """Session 22 current-parity selector verdict."""

    final_verdict: str
    selected_s2_residual_action_uab: sp.Matrix
    parity_by_line: dict[str, int]
    even_projector_is_u_plus_a: bool
    odd_projector_is_b: bool
    active_plane_pass: bool
    active_plane_projector: sp.Matrix
    current_line_standard: sp.Matrix
    current_line_residual_uab: sp.Matrix
    current_line_is_b: bool
    current_line_is_selected_pair_current: bool
    current_line_is_odd: bool
    selected_scalar_has_no_current_component: bool
    active_plane_alone_insufficient: bool
    u_rejected_as_scalar_even: bool
    a_rejected_as_radial_even: bool
    current_parity_selects_b: bool
    session21_source_premise_reduced: bool
    remaining_physical_premise: str
    interpretation: str


def matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices are symbolically equal."""

    return is_zero_matrix((left - right).applyfunc(sp.simplify))


def selected_s2_swap_standard() -> sp.Matrix:
    """Return the nontrivial selected-port ``S_2`` swap on standard ports."""

    return selected_port_s2_matrices()[1]


def selected_s2_action_residual_uab() -> sp.Matrix:
    """Return the selected ``S_2`` action in residual coordinate order ``(u,a,b)``."""

    basis = residual_basis_matrix(("u", "a", "b"))
    return (basis.T * selected_s2_swap_standard() * basis).applyfunc(sp.simplify)


def selected_s2_parity_by_line() -> dict[str, int]:
    """Return the selected ``S_2`` parity eigenvalue of each residual line."""

    action = selected_s2_action_residual_uab()
    labels = ("u", "a", "b")
    return {label: int(action[index, index]) for index, label in enumerate(labels)}


def selected_s2_even_projector() -> sp.Matrix:
    """Return the selected-``S_2`` even projector."""

    swap = selected_s2_swap_standard()
    return ((sp.eye(3) + swap) / 2).applyfunc(sp.simplify)


def selected_s2_odd_projector() -> sp.Matrix:
    """Return the selected-``S_2`` odd projector."""

    swap = selected_s2_swap_standard()
    return ((sp.eye(3) - swap) / 2).applyfunc(sp.simplify)


def even_projector_is_u_plus_a() -> bool:
    """Return whether the selected-even plane is ``span(u,a)``."""

    projectors = residual_projectors()
    return matrix_equal(selected_s2_even_projector(), projectors["u"] + projectors["a"])


def odd_projector_is_b() -> bool:
    """Return whether the selected-odd line is exactly ``b``."""

    return matrix_equal(selected_s2_odd_projector(), residual_projectors()["b"])


def selected_pair_current_line_standard() -> sp.Matrix:
    """Return the oriented current across the two unselected ports."""

    _, e2, e3 = standard_basis()
    return (e2 - e3) / sp.sqrt(2)


def selected_pair_current_line_residual_uab() -> sp.Matrix:
    """Return the selected-pair current in residual order ``(u,a,b)``."""

    basis = residual_basis_matrix(("u", "a", "b"))
    return (basis.T * selected_pair_current_line_standard()).applyfunc(sp.simplify)


def selected_pair_current_is_b() -> bool:
    """Return whether the selected-pair current is the residual ``b`` line."""

    return selected_pair_current_line_residual_uab() == sp.Matrix([0, 0, 1])


def line_has_parity(vector: sp.Matrix, parity: int) -> bool:
    """Return whether ``vector`` is a selected-``S_2`` parity eigenline."""

    swap = selected_s2_swap_standard()
    return matrix_equal(swap * vector, parity * vector)


def selected_scalar_has_no_current_component() -> bool:
    """Return whether the selected scalar port has no odd-current component."""

    e1, _, _ = standard_basis()
    b = residual_vectors()["b"]
    return sp.simplify((b.T * e1)[0]) == 0


def active_plane_alone_insufficient() -> bool:
    """Return whether the active plane still contains an even scalar line."""

    active = active_plane_incidence_payload()
    projectors = residual_projectors()
    return (
        active.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
        and matrix_equal(active.active_projector, projectors["u"] + projectors["b"])
        and not matrix_equal(active.active_projector, projectors["b"])
        and matrix_equal(active.active_projector * projectors["u"], projectors["u"])
    )


def u_rejected_as_scalar_even() -> bool:
    """Return whether the collective line ``u`` is rejected as an odd current."""

    u = residual_vectors()["u"]
    return line_has_parity(u, 1) and not line_has_parity(u, -1)


def a_rejected_as_radial_even() -> bool:
    """Return whether the radial line ``a`` is rejected as an odd current."""

    a = residual_vectors()["a"]
    active = active_plane_incidence_payload()
    radial_projector = residual_projectors()["a"]
    return (
        line_has_parity(a, 1)
        and not line_has_parity(a, -1)
        and matrix_equal(active.radial_projector, radial_projector)
    )


def current_parity_selects_b() -> bool:
    """Return whether active incidence plus selected-odd current selects ``b``."""

    active = active_plane_incidence_payload()
    projectors = residual_projectors()
    odd = selected_s2_odd_projector()
    active_odd = (active.active_projector * odd).applyfunc(sp.simplify)
    return (
        active.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
        and odd_projector_is_b()
        and matrix_equal(active_odd, projectors["b"])
        and selected_pair_current_is_b()
        and line_has_parity(selected_pair_current_line_standard(), -1)
    )


def quark_current_parity_selector_payload() -> QuarkCurrentParitySelectorPayload:
    """Return the Session 22 current-parity selector payload."""

    active = active_plane_incidence_payload()
    projectors = residual_projectors()
    action = selected_s2_action_residual_uab()
    parity = selected_s2_parity_by_line()
    even_is_ua = even_projector_is_u_plus_a()
    odd_is_b = odd_projector_is_b()
    current = selected_pair_current_line_standard()
    current_uab = selected_pair_current_line_residual_uab()
    current_is_b = selected_pair_current_is_b()
    current_is_odd = line_has_parity(current, -1)
    scalar_no_current = selected_scalar_has_no_current_component()
    active_insufficient = active_plane_alone_insufficient()
    u_rejected = u_rejected_as_scalar_even()
    a_rejected = a_rejected_as_radial_even()
    parity_selects = current_parity_selects_b()

    checks_pass = (
        action == sp.diag(1, 1, -1)
        and parity == {"u": 1, "a": 1, "b": -1}
        and even_is_ua
        and odd_is_b
        and active.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
        and matrix_equal(active.active_projector, projectors["u"] + projectors["b"])
        and current_uab == sp.Matrix([0, 0, 1])
        and current_is_b
        and current_is_odd
        and scalar_no_current
        and active_insufficient
        and u_rejected
        and a_rejected
        and parity_selects
    )

    if checks_pass:
        final_verdict = "QUARK_CURRENT_PARITY_SELECTOR_PASS"
        interpretation = (
            "The selected-port S2 swap acts on residual lines as "
            "diag(+,+,-) in (u,a,b).  Its odd projector is exactly P_b, "
            "and the oriented current across the two unselected ports is "
            "(e2-e3)/sqrt(2)=b.  Intersecting this odd-current line with "
            "the Session 11 active plane selects b uniquely.  This reduces "
            "the Session 21 current-source input to the physical premise "
            "that a colored quark mass event is a selected-S2 odd boundary "
            "current."
        )
    else:
        final_verdict = "QUARK_CURRENT_PARITY_SELECTOR_KILL"
        interpretation = (
            "The selected-S2 action, parity projectors, active-plane "
            "intersection, current-line, or scalar/radial controls failed."
        )

    return QuarkCurrentParitySelectorPayload(
        final_verdict=final_verdict,
        selected_s2_residual_action_uab=action,
        parity_by_line=parity,
        even_projector_is_u_plus_a=even_is_ua,
        odd_projector_is_b=odd_is_b,
        active_plane_pass=active.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS",
        active_plane_projector=active.active_projector,
        current_line_standard=current,
        current_line_residual_uab=current_uab,
        current_line_is_b=current_is_b,
        current_line_is_selected_pair_current=current_is_b,
        current_line_is_odd=current_is_odd,
        selected_scalar_has_no_current_component=scalar_no_current,
        active_plane_alone_insufficient=active_insufficient,
        u_rejected_as_scalar_even=u_rejected,
        a_rejected_as_radial_even=a_rejected,
        current_parity_selects_b=parity_selects,
        session21_source_premise_reduced=checks_pass,
        remaining_physical_premise=ODD_CURRENT_PHYSICAL_PREMISE,
        interpretation=interpretation,
    )
