"""Session 21 quark active-current readout ansatz.

The lepton-side source is a selected scalar incidence port.  A quark mass event
is different: it is a colored boundary current that must return with the same
visible color.  Once selected incidence has split the residual family space into

    P_rad = P_a,     P_act = P_u + P_b,

the unique active non-scalar line is ``b``.  This session tests the resulting
quark-source ansatz:

    quark current source = b.

For the up channel, the readout is a coherent retarded first-passage amplitude
through the certified nilpotent flag.  For the down channel, the readout is not
a scalar ``b`` vector; it is a Hermitian current covariance / word-shell
measure over the primitive quark shell.  The session therefore freezes the
current source direction conditionally, while keeping the down measure choice
visible rather than forcing it from data.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    projector,
    residual_basis_matrix,
    residual_projectors,
)
from clifford_3plus2_d5.depth_scar.nilpotent_flag import nilpotent_flag_operator
from clifford_3plus2_d5.scalar_clebsch.down_subset_counts import (
    down_baseline_clebsch_vector,
    down_baseline_counts,
)
from clifford_3plus2_d5.universal_bath.neutrino_active_plane import (
    active_plane_incidence_payload,
    opposite_edge_line_from_incidence,
)
from clifford_3plus2_d5.universal_bath.quark_down_odd_shell import (
    primitive_counts,
    quark_down_odd_shell_payload,
)
from clifford_3plus2_d5.universal_bath.quark_height_orientation_bridge import (
    quark_height_orientation_bridge_payload,
)
from clifford_3plus2_d5.universal_bath.up_quark_nilpotent_cmv import (
    geometric_control_profile,
    tail_injection_amplitude,
    up_quark_nilpotent_cmv_payload,
)

CURRENT_SOURCE_PREMISE = "colored_quark_current_selects_active_non_scalar_b_line"
DOWN_CURRENT_MEASURE_PREMISE = "down_reads_hermitian_current_covariance_not_scalar_b_vector"
DOWN_IDENTITY_VETO_PREMISE = "one_tick_retarded_down_return_vetoes_identity_word"


@dataclass(frozen=True)
class QuarkActiveCurrentReadoutPayload:
    """Session 21 quark active-current readout verdict."""

    final_verdict: str
    active_plane_pass: bool
    height_bridge_pass: bool
    up_head_pass: bool
    down_odd_shell_pass: bool
    current_line_standard: sp.Matrix
    current_line_residual_uab: sp.Matrix
    active_current_is_b: bool
    active_current_unique_non_scalar_line: bool
    first_return_orders: dict[str, int]
    first_return_orders_light_to_heavy: tuple[int, int, int]
    up_radial_depths: dict[str, int]
    down_radial_depths: dict[str, int]
    up_coherent_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    up_profile_matches_conditional_head: bool
    up_geometric_control_rejected: bool
    down_readout_is_covariance: bool
    down_baseline_counts: dict[str, int]
    down_baseline_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    down_odd_shell_counts: dict[str, int]
    down_odd_shell_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    down_identity_veto_selects_odd_shell_if_assumed: bool
    down_identity_veto_microscopically_derived: bool
    source_freeze_candidate: bool
    source_freeze_ready: bool
    remaining_physical_inputs: tuple[str, ...]
    interpretation: str


def active_current_line_standard() -> sp.Matrix:
    """Return the active current line in standard family-port coordinates."""

    return opposite_edge_line_from_incidence()


def active_current_line_residual_uab() -> sp.Matrix:
    """Return the active current line in residual coordinate order ``(u,a,b)``."""

    basis = residual_basis_matrix(("u", "a", "b"))
    return (basis.T * active_current_line_standard()).applyfunc(sp.simplify)


def active_current_is_b() -> bool:
    """Return whether the active current line is exactly residual ``b``."""

    return active_current_line_residual_uab() == sp.Matrix([0, 0, 1])


def active_current_unique_non_scalar_line() -> bool:
    """Return whether selected incidence leaves ``b`` as the active non-scalar line."""

    incidence = active_plane_incidence_payload()
    projectors = residual_projectors()
    current_projector = projector(active_current_line_standard())
    return (
        incidence.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
        and incidence.active_projector == projectors["u"] + projectors["b"]
        and current_projector == projectors["b"]
        and incidence.active_projector - projectors["u"] == projectors["b"]
    )


def residual_unit(label: str) -> sp.Matrix:
    """Return residual coordinate basis vector in order ``(u,a,b)``."""

    labels = {"u": 0, "a": 1, "b": 2}
    vector = sp.zeros(3, 1)
    vector[labels[label], 0] = 1
    return vector


def first_return_orders_from_b(max_order: int = 3) -> dict[str, int]:
    """Return first nonzero retarded passage order from current endpoint ``b``."""

    flag = nilpotent_flag_operator()
    source = residual_unit("b")
    orders: dict[str, int] = {}
    for target_label in ("u", "a", "b"):
        target = residual_unit(target_label)
        for order in range(max_order + 1):
            amplitude = sp.simplify((target.T * (flag**order) * source)[0])
            if amplitude != 0:
                orders[target_label] = order
                break
        else:
            raise ValueError(f"no return found for {target_label} through order {max_order}")
    return orders


def up_depths_from_first_return() -> dict[str, int]:
    """Return up-sector silver-depth exponents from holomorphic closure."""

    orders = first_return_orders_from_b()
    return {label: 3 * order for label, order in orders.items()}


def down_depths_from_first_return() -> dict[str, int]:
    """Return down-sector silver-depth exponents from paired baseline closure."""

    orders = first_return_orders_from_b()
    return {label: 2 * (order + 1) for label, order in orders.items()}


def coherent_up_profile_from_b() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``exp(xN)|b>`` in residual order ``(u,a,b)``."""

    x = tail_injection_amplitude()
    flag = nilpotent_flag_operator()
    source = residual_unit("b")
    head = sp.eye(3) + x * flag + (x**2 / 2) * flag**2
    vector = (head * source).applyfunc(sp.simplify)
    return (vector[0, 0], vector[1, 0], vector[2, 0])


def down_odd_shell_profile() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the current-covariance odd-shell candidate profile."""

    return quark_down_odd_shell_payload().candidate_clebsch_vector


def down_identity_veto_selects_odd_shell_if_assumed() -> bool:
    """Return whether removing the even direct line selects the odd-shell candidate."""

    return primitive_counts() == {"d": 6, "s": 2, "b": 5}


def quark_active_current_readout_payload() -> QuarkActiveCurrentReadoutPayload:
    """Return the Session 21 active-current readout payload."""

    incidence = active_plane_incidence_payload()
    height = quark_height_orientation_bridge_payload()
    up = up_quark_nilpotent_cmv_payload()
    down = quark_down_odd_shell_payload()

    active_pass = incidence.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
    height_pass = height.final_verdict == "QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT"
    up_pass = up.final_verdict == "UP_NILPOTENT_HEAD_CONDITIONAL_PASS"
    down_pass = down.final_verdict == "QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS"
    current_is_b = active_current_is_b()
    current_unique = active_current_unique_non_scalar_line()
    orders = first_return_orders_from_b()
    up_depths = up_depths_from_first_return()
    down_depths = down_depths_from_first_return()
    up_profile = coherent_up_profile_from_b()
    up_matches = up_profile == up.taylor_profile
    geometric_rejected = geometric_control_profile() != up_profile
    baseline_profile = down_baseline_clebsch_vector()
    odd_profile = down_odd_shell_profile()
    identity_veto_selects = down_identity_veto_selects_odd_shell_if_assumed()

    checks_pass = (
        active_pass
        and height_pass
        and up_pass
        and down_pass
        and current_is_b
        and current_unique
        and orders == {"u": 2, "a": 1, "b": 0}
        and up_depths == {"u": 6, "a": 3, "b": 0}
        and down_depths == {"u": 6, "a": 4, "b": 2}
        and up_matches
        and geometric_rejected
        and down_baseline_counts() == {"d": 6, "s": 2, "b": 4}
        and primitive_counts() == {"d": 6, "s": 2, "b": 5}
        and baseline_profile == (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)))
        and odd_profile == (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(5, 6)))
        and identity_veto_selects
    )

    if checks_pass:
        final_verdict = "QUARK_ACTIVE_CURRENT_READOUT_CONDITIONAL_PASS"
        interpretation = (
            "Given selected incidence, the unique active non-scalar line is b. "
            "Taking the colored quark source to be this current endpoint turns "
            "normal depth into first-passage order: b,a,u have orders 0,1,2, "
            "giving up depths (6,3,0) and down depths (6,4,2).  The coherent "
            "up readout exp((1/sqrt(2))N)|b> reproduces "
            "(1/4,1/sqrt(2),1), while the geometric control fails.  The down "
            "readout is kept as a Hermitian current covariance over shell "
            "measures, with baseline (6,2,4) and odd-shell (6,2,5) visible. "
            "The session does not derive colored-current selection or the "
            "down identity-word veto microscopically."
        )
    else:
        final_verdict = "QUARK_ACTIVE_CURRENT_READOUT_KILL"
        interpretation = (
            "The active-current readout failed active-plane, height-bridge, "
            "up-head, down-shell, current-line, first-passage, depth, profile, "
            "or covariance controls."
        )

    return QuarkActiveCurrentReadoutPayload(
        final_verdict=final_verdict,
        active_plane_pass=active_pass,
        height_bridge_pass=height_pass,
        up_head_pass=up_pass,
        down_odd_shell_pass=down_pass,
        current_line_standard=active_current_line_standard(),
        current_line_residual_uab=active_current_line_residual_uab(),
        active_current_is_b=current_is_b,
        active_current_unique_non_scalar_line=current_unique,
        first_return_orders=orders,
        first_return_orders_light_to_heavy=(orders["u"], orders["a"], orders["b"]),
        up_radial_depths=up_depths,
        down_radial_depths=down_depths,
        up_coherent_profile=up_profile,
        up_profile_matches_conditional_head=up_matches,
        up_geometric_control_rejected=geometric_rejected,
        down_readout_is_covariance=True,
        down_baseline_counts=down_baseline_counts(),
        down_baseline_profile=baseline_profile,
        down_odd_shell_counts=primitive_counts(),
        down_odd_shell_profile=odd_profile,
        down_identity_veto_selects_odd_shell_if_assumed=identity_veto_selects,
        down_identity_veto_microscopically_derived=False,
        source_freeze_candidate=checks_pass,
        source_freeze_ready=False,
        remaining_physical_inputs=(
            CURRENT_SOURCE_PREMISE,
            DOWN_CURRENT_MEASURE_PREMISE,
            DOWN_IDENTITY_VETO_PREMISE,
        ),
        interpretation=interpretation,
    )
