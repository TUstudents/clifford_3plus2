"""V30 local boundary-fiber isomorphism gate.

V24 showed that a regular unresolved boundary fiber forces equal degeneracy
per conserved primitive label.  V30 makes the regularity statement concrete:
the unresolved shell is

    H_Q = C^6_label tensor B_local,

so each conserved primitive label carries an isomorphic copy of the same local
boundary patch.  Tracing out that local patch gives ``I_6 / 6`` on the
conserved-label sector.  Sector-dependent or arbitrary label-preserving baths
remain controls, not theorem inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.conserved_label_partition import (
    conserved_label_partition_audit_payload,
    conserved_label_partition_is_complete,
)
from clifford_3plus2_d5.boundary_response.label_conserving_dynamics import (
    label_conserving_dynamics_audit_payload,
)
from clifford_3plus2_d5.boundary_response.microcanonical_reduction import (
    compressed_macro_degeneracy_control_phase,
    compressed_macro_degeneracy_control_ratio,
    microcanonical_label_weights,
    microcanonical_reduced_density,
    phase_from_degeneracies,
    ratio_from_degeneracies,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import SHELL_DIMENSION
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    BCC,
    COLOR,
    DIRECT,
    quark_boundary_phase_angle,
    quark_primitive_channels,
)
from clifford_3plus2_d5.boundary_response.regular_boundary_fiber import (
    regular_boundary_fiber_audit_payload,
)

REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER = (
    "physical_vacuum_order_parameter_exists",
)


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    if left.shape != right.shape:
        return False
    return all(
        sp.simplify(left[row, col] - right[row, col]) == 0
        for row in range(left.rows)
        for col in range(left.cols)
    )


def _positive_integer(value: sp.Expr) -> int:
    selected = sp.sympify(value)
    if not selected.is_integer or not selected.is_positive:
        raise ValueError("fiber_dim must be a positive integer for explicit matrices")
    return int(selected)


def local_fiber_degeneracies(fiber_dim: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return equal primitive degeneracies from ``C^6_label tensor B_local``."""

    selected = sp.sympify(fiber_dim)
    return tuple(selected for _ in range(SHELL_DIMENSION))


def local_fiber_total_dimension(fiber_dim: sp.Expr) -> sp.Expr:
    """Return ``dim(C^6_label tensor B_local) = 6D``."""

    return sp.simplify(SHELL_DIMENSION * sp.sympify(fiber_dim))


def local_fiber_label_projectors(fiber_dim: int | sp.Integer) -> tuple[sp.Matrix, ...]:
    """Return six equal-rank projectors for a concrete local fiber dimension."""

    dim = _positive_integer(sp.sympify(fiber_dim))
    total_dim = SHELL_DIMENSION * dim
    projectors: list[sp.Matrix] = []
    for label_index in range(SHELL_DIMENSION):
        projector = sp.zeros(total_dim, total_dim)
        start = label_index * dim
        for offset in range(dim):
            projector[start + offset, start + offset] = 1
        projectors.append(projector)
    return tuple(projectors)


def local_fiber_projector_ranks(fiber_dim: int | sp.Integer) -> tuple[int, ...]:
    """Return ranks of the six concrete label-fiber projectors."""

    return tuple(projector.rank() for projector in local_fiber_label_projectors(fiber_dim))


def fiber_isomorphism_witness(
    left: int,
    right: int,
    fiber_dim: int | sp.Integer,
) -> sp.Matrix:
    """Return a unitary permutation swapping two label fibers."""

    if left < 0 or left >= SHELL_DIMENSION or right < 0 or right >= SHELL_DIMENSION:
        raise ValueError(f"left and right must be in [0, {SHELL_DIMENSION})")
    dim = _positive_integer(sp.sympify(fiber_dim))
    total_dim = SHELL_DIMENSION * dim
    witness = sp.eye(total_dim)
    if left == right:
        return witness

    left_start = left * dim
    right_start = right * dim
    for offset in range(dim):
        left_index = left_start + offset
        right_index = right_start + offset
        witness[left_index, left_index] = 0
        witness[right_index, right_index] = 0
        witness[left_index, right_index] = 1
        witness[right_index, left_index] = 1
    return witness


def fiber_isomorphism_witness_maps_projectors(
    left: int,
    right: int,
    fiber_dim: int | sp.Integer,
) -> bool:
    """Return whether the witness maps the left label fiber to the right one."""

    projectors = local_fiber_label_projectors(fiber_dim)
    witness = fiber_isomorphism_witness(left, right, fiber_dim)
    return (
        _matrix_equal(witness.T * witness, sp.eye(witness.rows))
        and _matrix_equal(witness * projectors[left] * witness.T, projectors[right])
        and projectors[left].rank() == projectors[right].rank()
    )


def all_local_fibers_pairwise_isomorphic(fiber_dim: int | sp.Integer) -> bool:
    """Return true when every pair of label fibers has an explicit witness."""

    return all(
        fiber_isomorphism_witness_maps_projectors(left, right, fiber_dim)
        for left in range(SHELL_DIMENSION)
        for right in range(SHELL_DIMENSION)
    )


def local_fiber_reduced_density(fiber_dim: sp.Expr) -> sp.Matrix:
    """Return the reduced six-label density after tracing out ``B_local``."""

    return microcanonical_reduced_density(local_fiber_degeneracies(fiber_dim))


def local_fiber_ratio_and_phase(fiber_dim: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return the primitive ratio and phase induced by local-fiber isomorphism."""

    degeneracies = local_fiber_degeneracies(fiber_dim)
    return ratio_from_degeneracies(degeneracies), phase_from_degeneracies(degeneracies)


def sector_dependent_fiber_degeneracies(
    *,
    even_dim: sp.Expr,
    bcc_dim: sp.Expr,
    color_dim: sp.Expr,
) -> tuple[sp.Expr, ...]:
    """Return a sector-dependent fiber control."""

    degeneracies: list[sp.Expr] = []
    for channel in quark_primitive_channels():
        if channel.sector == DIRECT:
            degeneracies.append(sp.sympify(even_dim))
        elif channel.sector == BCC:
            degeneracies.append(sp.sympify(bcc_dim))
        elif channel.sector == COLOR:
            degeneracies.append(sp.sympify(color_dim))
        else:
            raise ValueError(f"unknown primitive channel sector: {channel.sector}")
    return tuple(degeneracies)


def sector_dependent_control_rejected() -> bool:
    """Return true when sector-dependent fibers fail full local isomorphism."""

    degeneracies = sector_dependent_fiber_degeneracies(
        even_dim=2,
        bcc_dim=1,
        color_dim=3,
    )
    density = microcanonical_reduced_density(degeneracies)
    return (
        len(set(degeneracies)) > 1
        and not _matrix_equal(density, sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION)
        and sp.simplify(phase_from_degeneracies(degeneracies) - quark_boundary_phase_angle())
        != 0
    )


def arbitrary_label_degeneracy_control_rejected() -> bool:
    """Return true when conservation without local isomorphism leaves weights free."""

    degeneracies = sp.symbols(f"d0:{SHELL_DIMENSION}", positive=True)
    weights = microcanonical_label_weights(degeneracies)
    return sp.simplify(weights[0] - weights[1]) != 0


def compressed_macro_fiber_control_rejected() -> bool:
    """Return true when compressed macro-fiber counting gives the wrong branch."""

    compressed_phase = compressed_macro_degeneracy_control_phase()
    return (
        sp.simplify(compressed_macro_degeneracy_control_ratio() - 1 / sp.sqrt(5)) == 0
        and sp.simplify(compressed_phase - sp.pi / 4) == 0
        and sp.simplify(compressed_phase - quark_boundary_phase_angle()) != 0
    )


@dataclass(frozen=True)
class LocalBoundaryFiberAuditPayload:
    """Verdict payload for the V30 local boundary-fiber isomorphism gate."""

    final_verdict: str
    local_degeneracies_equal: bool
    total_dimension: sp.Expr
    concrete_projector_ranks: tuple[int, ...]
    pairwise_isomorphism_witnesses_exist: bool
    reduced_density_uniform: bool
    local_ratio: sp.Expr
    local_phase: sp.Expr
    sector_dependent_control_rejected: bool
    arbitrary_degeneracy_control_rejected: bool
    compressed_macro_control_rejected: bool
    v21_labels_complete: bool
    v22_no_go_respected: bool
    v24_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def local_boundary_fiber_audit_payload() -> LocalBoundaryFiberAuditPayload:
    """Return the V30 local boundary-fiber isomorphism verdict."""

    fiber_dim = sp.symbols("D", positive=True)
    concrete_dim = 2
    degeneracies = local_fiber_degeneracies(fiber_dim)
    local_equal = all(sp.simplify(value - degeneracies[0]) == 0 for value in degeneracies)
    total_dimension = local_fiber_total_dimension(fiber_dim)
    ranks = local_fiber_projector_ranks(concrete_dim)
    witnesses_exist = all_local_fibers_pairwise_isomorphic(concrete_dim)
    density_uniform = _matrix_equal(
        local_fiber_reduced_density(fiber_dim),
        sp.eye(SHELL_DIMENSION) / SHELL_DIMENSION,
    )
    ratio, phase = local_fiber_ratio_and_phase(fiber_dim)
    sector_rejected = sector_dependent_control_rejected()
    arbitrary_rejected = arbitrary_label_degeneracy_control_rejected()
    compressed_rejected = compressed_macro_fiber_control_rejected()

    v21 = conserved_label_partition_audit_payload()
    v22 = label_conserving_dynamics_audit_payload()
    v24 = regular_boundary_fiber_audit_payload()
    v21_complete = (
        v21.final_verdict == "CONSERVED_LABEL_PARTITION_THEOREM_PASS"
        and conserved_label_partition_is_complete()
    )
    v22_respected = (
        v22.final_verdict == "LABEL_CONSERVING_DYNAMICS_MAX_ENTROPY_NO_GO_PASS"
        and v22.max_entropy_prior_remains_declared
    )
    v24_recovered = (
        v24.final_verdict == "REGULAR_BOUNDARY_FIBER_EQUAL_DEGENERACY_PASS"
        and v24.regular_density_uniform
    )

    checks_pass = (
        local_equal
        and sp.simplify(total_dimension - SHELL_DIMENSION * fiber_dim) == 0
        and ranks == tuple(concrete_dim for _ in range(SHELL_DIMENSION))
        and witnesses_exist
        and density_uniform
        and sp.simplify(ratio - 1) == 0
        and sp.simplify(phase - quark_boundary_phase_angle()) == 0
        and sector_rejected
        and arbitrary_rejected
        and compressed_rejected
        and v21_complete
        and v22_respected
        and v24_recovered
    )

    if checks_pass:
        final_verdict = "LOCAL_BOUNDARY_FIBER_ISOMORPHISM_PASS"
        interpretation = (
            "The unresolved shell factors as C^6_label tensor B_local. Each "
            "conserved primitive label carries an isomorphic copy of the same "
            "local boundary patch, so every label fiber has equal rank. "
            "Tracing out B_local gives I6/6 and recovers r=1 with "
            "atan(sqrt(5)). Sector-dependent, arbitrary-degeneracy, and "
            "compressed macro-fiber controls fail. This is a structural "
            "isomorphism theorem, not label-conserving thermalization."
        )
    else:
        final_verdict = "LOCAL_BOUNDARY_FIBER_ISOMORPHISM_KILL"
        interpretation = (
            "The local tensor factorization, equal ranks, pairwise "
            "isomorphism witnesses, uniform reduction, CKM phase recovery, "
            "controls, V21 compatibility, V22 no-go, or V24 recovery failed."
        )

    return LocalBoundaryFiberAuditPayload(
        final_verdict=final_verdict,
        local_degeneracies_equal=local_equal,
        total_dimension=total_dimension,
        concrete_projector_ranks=ranks,
        pairwise_isomorphism_witnesses_exist=witnesses_exist,
        reduced_density_uniform=density_uniform,
        local_ratio=ratio,
        local_phase=phase,
        sector_dependent_control_rejected=sector_rejected,
        arbitrary_degeneracy_control_rejected=arbitrary_rejected,
        compressed_macro_control_rejected=compressed_rejected,
        v21_labels_complete=v21_complete,
        v22_no_go_respected=v22_respected,
        v24_recovered=v24_recovered,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER,
        interpretation=interpretation,
    )
