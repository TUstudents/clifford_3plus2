"""Aggregate verdict for the path-defect depth-scar theorem."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    depth_scar_operator,
    defect_mode_depths,
    hand_written_diagonal_is_not_graph_native,
    path_spectrum_passes,
    transfer_operator_eigenvalues,
    transition_depth_differences,
    unbroken_k3_control_fails_hierarchy,
    weighted_scar_controls_pass,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon


@dataclass(frozen=True)
class DepthScarAuditPayload:
    """Verdict payload for the boundary repair-scar sidecar."""

    final_verdict: str
    depth_spectrum: tuple[sp.Expr, ...]
    mode_depths: tuple[sp.Expr, ...]
    transfer_eigenvalues: tuple[sp.Expr, ...]
    transition_depths: dict[tuple[int, int], int]
    path_spectrum_passes: bool
    k3_control_rejected: bool
    diagonal_control_rejected: bool
    weighted_controls_pass: bool
    scar_dynamically_derived: bool
    interpretation: str


def depth_scar_audit_payload() -> DepthScarAuditPayload:
    """Return the v1 prove-or-kill verdict for the path-defect depth operator."""

    path_pass = path_spectrum_passes()
    mode_depths = defect_mode_depths()
    transfer = transfer_operator_eigenvalues()
    k3_rejected = unbroken_k3_control_fails_hierarchy()
    diagonal_rejected = hand_written_diagonal_is_not_graph_native()
    weighted_pass = weighted_scar_controls_pass()

    transfer_pass = transfer == (
        sp.Integer(1),
        sp.simplify(epsilon() ** 2),
        sp.simplify(epsilon() ** 6),
    )
    mode_pass = mode_depths == EXPECTED_DEPTH_SPECTRUM
    transition_pass = transition_depth_differences() == {
        (1, 2): 2,
        (2, 3): 4,
        (1, 3): 6,
    }

    checks_pass = (
        path_pass
        and mode_pass
        and transfer_pass
        and transition_pass
        and k3_rejected
        and diagonal_rejected
        and weighted_pass
    )

    if checks_pass:
        final_verdict = "PATH_DEFECT_LAPLACIAN_DEPTH_PASS"
        interpretation = (
            "The S3 -> Z2 repair scar with graph P3 gives a positive graph-native "
            "operator D_scar = 2 Delta(P3) with exact spectrum {0,2,6}. Its normal "
            "modes are the uniform, endpoint-antisymmetric, and middle-compression "
            "family modes, and epsilon**D_scar gives transfer factors "
            "{1, epsilon^2, epsilon^6}. Unbroken K3 remains degenerate, and the "
            "hand-written diagonal control is rejected as non-graph-native. This "
            "passes only the operator-origin gate; the dynamical origin of the "
            "scar is still open."
        )
    elif not path_pass:
        final_verdict = "PATH_DEFECT_LAPLACIAN_SPECTRUM_KILL"
        interpretation = "The path-defect Laplacian does not have the target depth spectrum."
    elif not transfer_pass:
        final_verdict = "TRANSFER_OPERATOR_DEPTH_KILL"
        interpretation = "The transfer operator does not produce the target suppression factors."
    elif not k3_rejected:
        final_verdict = "UNBROKEN_K3_CONTROL_FAILED"
        interpretation = "The unbroken K3 control did not remain the expected degenerate kill."
    elif not diagonal_rejected:
        final_verdict = "SCAR_NOT_GRAPH_NATIVE_KILL"
        interpretation = "The diagonal control was not separated from the graph-native scar."
    else:
        final_verdict = "PATH_DEFECT_LAPLACIAN_CONTROL_KILL"
        interpretation = "The path-defect theorem or one of its controls failed."

    return DepthScarAuditPayload(
        final_verdict=final_verdict,
        depth_spectrum=tuple(EXPECTED_DEPTH_SPECTRUM),
        mode_depths=mode_depths,
        transfer_eigenvalues=transfer,
        transition_depths=transition_depth_differences(),
        path_spectrum_passes=path_pass,
        k3_control_rejected=k3_rejected,
        diagonal_control_rejected=diagonal_rejected,
        weighted_controls_pass=weighted_pass,
        scar_dynamically_derived=False,
        interpretation=interpretation,
    )


def depth_scar_operator_trace_spurion() -> sp.Matrix:
    """Return the traceless part of ``D_scar`` in the port basis."""

    operator = depth_scar_operator()
    return sp.simplify(operator - sp.Rational(operator.trace(), 3) * sp.eye(3))

