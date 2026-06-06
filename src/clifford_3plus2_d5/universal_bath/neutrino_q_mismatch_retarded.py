"""Session 12 q-mismatch hard gap and retarded active-plane closure.

Session 11 derived the active family plane from selected-port incidence:

    detrace(e1) = a,     P_act = I - P_a = P_u + P_b.

This module connects that incidence result to the microscopic Bialynicki-Birula
BCC edge blocks.  The exact BB edge update splits hops by the relative-depth
charge ``q = r1 - r2``:

    same-normal:  sigma1 = sigma2  ->  Delta q = 0,
    mixed-normal: sigma1 = -sigma2 ->  Delta q = +-2.

The mixed-normal blocks carry the missing half of the BB norm.  A local
single-clock mismatch penalty ``g q^2`` gives the adjacent sectors a gap ``4g``.
Schur elimination then makes their feedback vanish in the hard-gap limit.

Finally, the retarded/outgoing boundary condition is encoded by a block
triangular update.  With no incoming clock-error data, visible powers are
exactly the q=0 survival powers.  A recurrent wedge control has nonzero
two-step return and is rejected.

The result is still a conditional microscopic boundary model: it proves the
finite algebra once the single-clock locking field and outgoing asymptotic
condition are accepted.  It does not derive those from a deeper boundary
material dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp

from clifford_3plus2_d5.universal_bath.neutrino_active_plane import (
    active_plane_incidence_payload,
    active_projector_from_incidence,
    radial_projector_from_incidence,
)
from clifford_3plus2_d5.universal_bath.neutrino_bcc_moments import (
    BBEdgeBlocks,
    bb_edge_blocks,
    mixed_normal_norm,
    same_normal_norm,
)
from clifford_3plus2_d5.universal_bath.neutrino_family_port_graph import (
    active_neutrino_projector,
    matrix_equal,
    radial_family_projector,
    selected_family_graph_payload,
)

REMAINING_DECLARED_INPUTS_AFTER_Q_MISMATCH = (
    "single_clock_locking_field_is_realized_by_boundary_material",
    "mixed_normal_clock_error_ports_are_outgoing_asymptotic_leads",
)


@dataclass(frozen=True)
class QMismatchRetardedPayload:
    """Session 12 q-mismatch/retarded-compression verdict."""

    final_verdict: str
    same_normal_direction_count: int
    mixed_normal_direction_count: int
    same_normal_norm: sp.Matrix
    mixed_normal_norm: sp.Matrix
    total_norm_is_identity: bool
    family_radial_projector_matches_incidence: bool
    family_active_projector_matches_incidence: bool
    leakage_gap: sp.Expr
    mixed_schur_feedback: sp.Matrix
    hard_gap_feedback_zero: bool
    finite_gap_feedback_scalar: bool
    retarded_weyl_equation_passes: bool
    retarded_weyl_normalization_passes: bool
    retarded_feedback_limit_zero: bool
    retarded_visible_powers_match_survival: bool
    recurrent_wedge_return_nonzero: bool
    recurrent_visible_powers_differ: bool
    session_10_family_graph_passes: bool
    session_11_active_plane_passes: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def delta_q(direction: tuple[int, int, int]) -> int:
    """Return the relative-depth change for a BCC body-diagonal hop."""

    sigma_1, sigma_2, _sigma_3 = direction
    return sigma_1 - sigma_2


def same_normal_directions() -> tuple[tuple[int, int, int], ...]:
    """Return body-diagonal directions preserving ``q``."""

    return tuple(direction for direction in product((1, -1), repeat=3) if delta_q(direction) == 0)


def mixed_normal_directions() -> tuple[tuple[int, int, int], ...]:
    """Return body-diagonal directions changing ``q`` by ``+-2``."""

    return tuple(direction for direction in product((1, -1), repeat=3) if delta_q(direction) != 0)


def mixed_coupling_vertical(blocks: BBEdgeBlocks | None = None) -> sp.Matrix:
    """Return the vertical mixed-normal coupling ``M=(M_+2,M_-2)^T``."""

    edge = bb_edge_blocks() if blocks is None else blocks
    return sp.Matrix.vstack(edge.m_plus2, edge.m_minus2).applyfunc(sp.simplify)


def q_mismatch_gap(stiffness: sp.Expr) -> sp.Expr:
    """Return the adjacent leakage gap from ``g q^2`` at ``q=+-2``."""

    return sp.simplify(4 * stiffness)


def mixed_normal_schur_feedback(
    z_probe: sp.Expr,
    stiffness: sp.Expr,
    blocks: BBEdgeBlocks | None = None,
) -> sp.Matrix:
    """Return ``M^H (z - 4g)^-1 M`` for the degenerate leakage gap."""

    coupling = mixed_coupling_vertical(blocks)
    gap = q_mismatch_gap(stiffness)
    return (coupling.H * coupling / (z_probe - gap)).applyfunc(sp.simplify)


def hard_gap_feedback_is_zero() -> bool:
    """Return whether the mixed-normal Schur feedback vanishes as ``g -> oo``."""

    z = sp.symbols("z")
    g = sp.symbols("g", positive=True)
    feedback = mixed_normal_schur_feedback(z, g)
    limit_matrix = feedback.applyfunc(lambda entry: sp.limit(entry, g, sp.oo))
    return matrix_equal(limit_matrix, sp.zeros(2))


def finite_gap_feedback_is_scalar() -> bool:
    """Return whether finite-gap feedback is scalar in the Weyl spinor."""

    z = sp.symbols("z")
    g = sp.symbols("g", positive=True)
    feedback = mixed_normal_schur_feedback(z, g)
    scalar = sp.simplify(1 / (2 * (z - 4 * g)))
    return matrix_equal(feedback, scalar * sp.eye(2))


def retarded_weyl_tail(spectral_parameter: sp.Expr) -> sp.Expr:
    """Return the normalized retarded/free half-line Weyl branch."""

    w = spectral_parameter
    return sp.simplify((w - sp.sqrt(w**2 - 4)) / 2)


def retarded_weyl_equation_passes() -> bool:
    """Return whether ``m = 1/(w-m)`` holds exactly."""

    w = sp.symbols("w", positive=True)
    m = retarded_weyl_tail(w)
    return sp.simplify(m * (w - m) - 1) == 0


def retarded_weyl_normalization_passes() -> bool:
    """Return whether the selected branch has ``m(w) ~ 1/w``."""

    w = sp.symbols("w", positive=True)
    m = retarded_weyl_tail(w)
    return sp.limit(w * m, w, sp.oo) == 1


def retarded_feedback_limit_is_zero() -> bool:
    """Return whether an outgoing high-threshold channel gives zero feedback."""

    w = sp.symbols("w", positive=True)
    feedback = (retarded_weyl_tail(w) * mixed_normal_norm()).applyfunc(sp.simplify)
    limit_matrix = feedback.applyfunc(lambda entry: sp.limit(entry, w, sp.oo))
    return matrix_equal(limit_matrix, sp.zeros(2))


def visible_scar_operator(shells: int, blocks: BBEdgeBlocks | None = None) -> sp.Matrix:
    """Return the finite half-line q=0 visible scar operator."""

    if shells < 1:
        raise ValueError("shells must be positive")
    edge = bb_edge_blocks() if blocks is None else blocks
    dim = 2 * shells
    operator = sp.zeros(dim, dim)
    for dest in range(shells):
        row = 2 * dest
        if dest - 1 >= 0:
            col = 2 * (dest - 1)
            operator[row : row + 2, col : col + 2] = edge.b_plus
        if dest + 1 < shells:
            col = 2 * (dest + 1)
            operator[row : row + 2, col : col + 2] = edge.b_minus
    return operator.applyfunc(sp.simplify)


def mixed_emission_operator(shells: int, blocks: BBEdgeBlocks | None = None) -> sp.Matrix:
    """Return mixed-normal emission from visible q=0 to clock-error leads."""

    if shells < 1:
        raise ValueError("shells must be positive")
    edge = bb_edge_blocks() if blocks is None else blocks
    emission = sp.zeros(4 * shells, 2 * shells)
    mixed = mixed_coupling_vertical(edge)
    for shell in range(shells):
        emission[4 * shell : 4 * shell + 4, 2 * shell : 2 * shell + 2] = mixed
    return emission.applyfunc(sp.simplify)


def recurrent_return_operator(shells: int, blocks: BBEdgeBlocks | None = None) -> sp.Matrix:
    """Return the recurrent hidden-to-visible wedge control block."""

    if shells < 1:
        raise ValueError("shells must be positive")
    edge = bb_edge_blocks() if blocks is None else blocks
    return_block = sp.Matrix.hstack(edge.m_minus2, edge.m_plus2)
    recurrent = sp.zeros(2 * shells, 4 * shells)
    for shell in range(shells):
        recurrent[2 * shell : 2 * shell + 2, 4 * shell : 4 * shell + 4] = return_block
    return recurrent.applyfunc(sp.simplify)


def hidden_outgoing_shift(shells: int) -> sp.Matrix:
    """Return a finite outgoing shift on the clock-error lead."""

    if shells < 1:
        raise ValueError("shells must be positive")
    shift = sp.zeros(4 * shells, 4 * shells)
    for shell in range(shells - 1):
        shift[4 * (shell + 1) : 4 * (shell + 2), 4 * shell : 4 * shell + 4] = sp.eye(4)
    return shift


def retarded_edge_update(shells: int) -> sp.Matrix:
    """Return the block-triangular retarded edge update."""

    visible = visible_scar_operator(shells)
    emission = mixed_emission_operator(shells)
    hidden = hidden_outgoing_shift(shells)
    top = sp.Matrix.hstack(visible, sp.zeros(visible.rows, hidden.cols))
    bottom = sp.Matrix.hstack(emission, hidden)
    return sp.Matrix.vstack(top, bottom).applyfunc(sp.simplify)


def recurrent_wedge_update(shells: int) -> sp.Matrix:
    """Return the recurrent control with hidden-to-visible feedback."""

    visible = visible_scar_operator(shells)
    emission = mixed_emission_operator(shells)
    hidden = hidden_outgoing_shift(shells)
    recurrent = recurrent_return_operator(shells)
    top = sp.Matrix.hstack(visible, recurrent)
    bottom = sp.Matrix.hstack(emission, hidden)
    return sp.Matrix.vstack(top, bottom).applyfunc(sp.simplify)


def visible_block(matrix: sp.Matrix, shells: int) -> sp.Matrix:
    """Return the visible-visible block of a full edge update."""

    dim = 2 * shells
    return matrix[:dim, :dim].applyfunc(sp.simplify)


def retarded_visible_powers_match_survival(
    shells: int = 4,
    powers: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> bool:
    """Return whether retarded visible powers equal q=0 survival powers."""

    update = retarded_edge_update(shells)
    visible = visible_scar_operator(shells)
    return all(
        matrix_equal(visible_block(update**power, shells), visible**power)
        for power in powers
    )


def recurrent_visible_powers_differ(shells: int = 4) -> bool:
    """Return whether recurrent hidden feedback changes visible powers."""

    update = recurrent_wedge_update(shells)
    visible = visible_scar_operator(shells)
    return not matrix_equal(visible_block(update**2, shells), visible**2)


def recurrent_two_step_return(blocks: BBEdgeBlocks | None = None) -> sp.Matrix:
    """Return the local two-step recurrent wedge correction ``G E``."""

    edge = bb_edge_blocks() if blocks is None else blocks
    return (edge.m_minus2 * edge.m_plus2 + edge.m_plus2 * edge.m_minus2).applyfunc(
        sp.simplify
    )


def q_mismatch_retarded_payload() -> QMismatchRetardedPayload:
    """Return the Session 12 q-mismatch/retarded-compression certificate."""

    blocks = bb_edge_blocks()
    same_norm = same_normal_norm(blocks)
    mixed_norm = mixed_normal_norm(blocks)
    total_norm = matrix_equal(same_norm + mixed_norm, sp.eye(2))
    family_radial = matrix_equal(radial_projector_from_incidence(), radial_family_projector())
    family_active = matrix_equal(active_projector_from_incidence(), active_neutrino_projector())
    z = sp.symbols("z")
    g = sp.symbols("g", positive=True)
    gap = q_mismatch_gap(g)
    feedback = mixed_normal_schur_feedback(z, g, blocks)
    hard_gap = hard_gap_feedback_is_zero()
    scalar_feedback = finite_gap_feedback_is_scalar()
    retarded_eq = retarded_weyl_equation_passes()
    retarded_norm = retarded_weyl_normalization_passes()
    retarded_limit = retarded_feedback_limit_is_zero()
    retarded_powers = retarded_visible_powers_match_survival()
    recurrent_return = recurrent_two_step_return(blocks)
    recurrent_return_nonzero = not matrix_equal(recurrent_return, sp.zeros(2))
    recurrent_differs = recurrent_visible_powers_differ()
    session_10 = selected_family_graph_payload()
    session_11 = active_plane_incidence_payload()
    session_10_pass = session_10.final_verdict == "NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS"
    session_11_pass = session_11.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"

    checks_pass = (
        len(same_normal_directions()) == 4
        and len(mixed_normal_directions()) == 4
        and matrix_equal(same_norm, sp.eye(2) / 2)
        and matrix_equal(mixed_norm, sp.eye(2) / 2)
        and total_norm
        and family_radial
        and family_active
        and sp.simplify(gap - 4 * g) == 0
        and hard_gap
        and scalar_feedback
        and retarded_eq
        and retarded_norm
        and retarded_limit
        and retarded_powers
        and recurrent_return_nonzero
        and recurrent_differs
        and session_10_pass
        and session_11_pass
    )

    if checks_pass:
        final_verdict = "NEUTRINO_Q_MISMATCH_RETARDED_COMPRESSION_PASS"
        interpretation = (
            "The BB edge directions split exactly into four same-normal "
            "q=0 hops and four mixed-normal q=+-2 hops.  The same-normal and "
            "mixed-normal blocks carry I/2 each.  The selected-port incidence "
            "radial projector is P_a, so a single-clock mismatch penalty gives "
            "a family penalty Lambda P_a and leaves the active P_u+P_b plane. "
            "A q^2 locking gap gives mixed-normal Schur feedback "
            "I/[2(z-4g)], which vanishes in the hard-gap limit.  With "
            "retarded/outgoing clock-error leads, the update is block "
            "triangular and visible powers are exactly the q=0 survival "
            "powers.  The recurrent wedge control has nonzero two-step return "
            "and changes the visible branch.  The remaining inputs are the "
            "physical realization of the single-clock locking field and the "
            "outgoing asymptotic condition by deeper boundary material "
            "dynamics."
        )
    else:
        final_verdict = "NEUTRINO_Q_MISMATCH_RETARDED_COMPRESSION_KILL"
        interpretation = (
            "The q-split, BB norm split, family incidence projector, hard-gap "
            "Schur limit, retarded Weyl branch, triangular compression, "
            "recurrent control, or Session 10/11 prerequisites failed."
        )

    return QMismatchRetardedPayload(
        final_verdict=final_verdict,
        same_normal_direction_count=len(same_normal_directions()),
        mixed_normal_direction_count=len(mixed_normal_directions()),
        same_normal_norm=same_norm,
        mixed_normal_norm=mixed_norm,
        total_norm_is_identity=total_norm,
        family_radial_projector_matches_incidence=family_radial,
        family_active_projector_matches_incidence=family_active,
        leakage_gap=gap,
        mixed_schur_feedback=feedback,
        hard_gap_feedback_zero=hard_gap,
        finite_gap_feedback_scalar=scalar_feedback,
        retarded_weyl_equation_passes=retarded_eq,
        retarded_weyl_normalization_passes=retarded_norm,
        retarded_feedback_limit_zero=retarded_limit,
        retarded_visible_powers_match_survival=retarded_powers,
        recurrent_wedge_return_nonzero=recurrent_return_nonzero,
        recurrent_visible_powers_differ=recurrent_differs,
        session_10_family_graph_passes=session_10_pass,
        session_11_active_plane_passes=session_11_pass,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_Q_MISMATCH,
        interpretation=interpretation,
    )
