"""Session 07 down-quark indefinite Jacobi head.

The down sector is real/symmetric rather than CMV.  This module compares the
3-port residual shell with the 6-element regular S3 shell and records exactly
which Clebsch heads are available.  It deliberately does not select the
``4 -> 5`` bottom line: S3 makes the rank-5 line available but not unique.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.scalar_clebsch.down_subset_counts import (
    DOWN_FAMILIES,
    down_baseline_clebsch_vector,
    down_baseline_counts,
    down_candidate_clebsch_vector,
    down_candidate_counts,
    down_subset_audit_payload,
)
from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    rank_five_projector_ranks,
    s3_projector_audit_payload,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceAnchor,
    SourceStatus,
    source_dictionary_payload,
)

DOWN_QUARK_SOURCE_LABEL = "down_quark_boundary_source"


@dataclass(frozen=True)
class DownJacobiHeadCandidate:
    """One count-level real symmetric down-head candidate."""

    label: str
    graph: str
    denominator: int
    ranks: dict[str, int]
    clebsch_vector: tuple[sp.Expr, sp.Expr, sp.Expr]
    residue_matrix: sp.Matrix
    matches_baseline: bool
    matches_candidate: bool
    forced_by_symmetry: bool
    interpretation: str


@dataclass(frozen=True)
class DownQuarkIndefiniteJacobiPayload:
    """Session 07 down-quark indefinite Jacobi verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    quark_source_unresolved: bool
    source_label: str
    source_reduction: ReductionKind
    subset_prerequisite_pass: bool
    s3_projector_prerequisite_pass: bool
    three_port_head: DownJacobiHeadCandidate
    regular_baseline_head: DownJacobiHeadCandidate
    regular_candidate_head: DownJacobiHeadCandidate
    three_port_cannot_host_candidate: bool
    regular_candidate_available: bool
    regular_candidate_forced_by_s3_alone: bool
    rank_two_requires_defect_polarization: bool
    rank_five_not_unique: bool
    signature_breakdown_control_detected: bool
    selected_physical_head: str
    interpretation: str


def unresolved_down_quark_source() -> SourceAnchor:
    """Return the unresolved Session 02 down-quark source anchor."""

    payload = source_dictionary_payload()
    anchors = {anchor.label: anchor for anchor in payload.unresolved_sources}
    return anchors[DOWN_QUARK_SOURCE_LABEL]


def clebsch_from_ranks(
    ranks: dict[str, int],
    denominator: int,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``sqrt(rank / denominator)`` in ``(d,s,b)`` order."""

    return tuple(
        sp.sqrt(sp.Rational(ranks[family], denominator))
        for family in DOWN_FAMILIES
    )


def diagonal_residue_matrix(vector: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Matrix:
    """Return a real symmetric residue matrix for the count head."""

    return sp.diag(*vector)


def _candidate(
    *,
    label: str,
    graph: str,
    denominator: int,
    ranks: dict[str, int],
    forced_by_symmetry: bool,
    interpretation: str,
) -> DownJacobiHeadCandidate:
    """Build a count-level head candidate."""

    vector = clebsch_from_ranks(ranks, denominator)
    return DownJacobiHeadCandidate(
        label=label,
        graph=graph,
        denominator=denominator,
        ranks=ranks,
        clebsch_vector=vector,
        residue_matrix=diagonal_residue_matrix(vector),
        matches_baseline=vector == down_baseline_clebsch_vector(),
        matches_candidate=vector == down_candidate_clebsch_vector(),
        forced_by_symmetry=forced_by_symmetry,
        interpretation=interpretation,
    )


def three_port_permutation_head() -> DownJacobiHeadCandidate:
    """Return the residual 3-port head ``(3,1,2)/3``."""

    return _candidate(
        label="three_port_permutation_baseline",
        graph="residual_3_port_permutation_shell",
        denominator=3,
        ranks={"d": 3, "s": 1, "b": 2},
        forced_by_symmetry=True,
        interpretation=(
            "The 3-port residual shell can reproduce the clean baseline ratios "
            "(1,1/sqrt(3),sqrt(2/3)) but has no integer rank corresponding to "
            "sqrt(5/6)."
        ),
    )


def regular_s3_baseline_head() -> DownJacobiHeadCandidate:
    """Return the regular S3 baseline head ``(6,2,4)/6``."""

    return _candidate(
        label="regular_s3_projector_baseline",
        graph="regular_s3_cayley_shell",
        denominator=6,
        ranks=down_baseline_counts(),
        forced_by_symmetry=True,
        interpretation=(
            "The 6-element regular shell reproduces the same clean baseline "
            "(1,1/sqrt(3),sqrt(2/3)) using ranks (6,2,4)."
        ),
    )


def regular_s3_candidate_head() -> DownJacobiHeadCandidate:
    """Return the available, not forced, odd-shell candidate ``(6,2,5)/6``."""

    return _candidate(
        label="regular_s3_rank_five_candidate",
        graph="regular_s3_cayley_shell",
        denominator=6,
        ranks=down_candidate_counts(),
        forced_by_symmetry=False,
        interpretation=(
            "The 6-element regular shell can host the data-improved "
            "(1,1/sqrt(3),sqrt(5/6)) vector, but choosing the rank-5 line is "
            "not forced by S3 alone."
        ),
    )


def three_port_cannot_host_rank_five_candidate() -> bool:
    """Return whether a 3-port denominator cannot realize the 5/6 line."""

    required_rank = sp.Rational(5, 6) * 3
    return required_rank != int(required_rank)


def signed_norm_squared(vector: tuple[sp.Expr, ...], signature: tuple[int, ...]) -> sp.Expr:
    """Return a toy signed norm used to detect indefinite breakdown controls."""

    if len(vector) != len(signature):
        raise ValueError("vector and signature must have the same length")
    return sp.simplify(
        sum(
            sign * component**2
            for component, sign in zip(vector, signature, strict=True)
        )
    )


def signature_breakdown_control_detected() -> bool:
    """Return whether a negative signed norm is detected rather than clamped."""

    return sp.N(signed_norm_squared((sp.Integer(0), sp.Integer(1)), (1, -1))) < 0


def down_quark_indefinite_jacobi_payload() -> DownQuarkIndefiniteJacobiPayload:
    """Return the Session 07 down-quark indefinite Jacobi verdict."""

    source_payload = source_dictionary_payload()
    source = unresolved_down_quark_source()
    subset = down_subset_audit_payload()
    projector = s3_projector_audit_payload()
    three_port = three_port_permutation_head()
    baseline = regular_s3_baseline_head()
    candidate = regular_s3_candidate_head()
    rank_five_ranks = rank_five_projector_ranks()

    source_pass = source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
    source_unresolved = (
        source.label == DOWN_QUARK_SOURCE_LABEL
        and source.status == SourceStatus.UNRESOLVED
        and source.reduction == ReductionKind.INDEFINITE_LOOKAHEAD_JACOBI
        and source.port_vector is None
        and source.normal_depth is None
    )
    subset_pass = subset.final_verdict == "DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_PASS"
    projector_pass = projector.final_verdict == "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
    three_port_no_candidate = three_port_cannot_host_rank_five_candidate()
    candidate_available = projector.count_vector_available and candidate.matches_candidate
    candidate_forced = projector.count_vector_forced_by_s3_alone
    rank_five_ambiguous = projector.rank_five_not_unique and set(rank_five_ranks.values()) == {5}
    signature_control = signature_breakdown_control_detected()

    checks_pass = (
        source_pass
        and source_unresolved
        and subset_pass
        and projector_pass
        and three_port.matches_baseline
        and baseline.matches_baseline
        and candidate.matches_candidate
        and three_port_no_candidate
        and candidate_available
        and not candidate_forced
        and projector.rank_two_requires_defect_polarization
        and rank_five_ambiguous
        and signature_control
    )

    if checks_pass:
        final_verdict = "DOWN_INDEFINITE_JACOBI_HEAD_CONDITIONAL_PASS"
        interpretation = (
            "The down real-symmetric head comparison is implemented.  The "
            "3-port shell and the regular S3 shell both reproduce the clean "
            "(6,2,4) baseline, while only the regular shell can host the "
            "(6,2,5) candidate.  S3 makes the rank-5 line available but not "
            "unique, so the bottom 4->5 shift remains an open selection "
            "theorem.  The down BCC source vector is still unresolved."
        )
        selected = "unselected_open_rank_five_gate"
    else:
        final_verdict = "DOWN_INDEFINITE_JACOBI_HEAD_KILL"
        interpretation = (
            "The down Jacobi head failed the source, subset, projector, "
            "candidate-availability, ambiguity, or signature-control checks."
        )
        selected = "none"

    return DownQuarkIndefiniteJacobiPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        quark_source_unresolved=source_unresolved,
        source_label=source.label,
        source_reduction=source.reduction,
        subset_prerequisite_pass=subset_pass,
        s3_projector_prerequisite_pass=projector_pass,
        three_port_head=three_port,
        regular_baseline_head=baseline,
        regular_candidate_head=candidate,
        three_port_cannot_host_candidate=three_port_no_candidate,
        regular_candidate_available=candidate_available,
        regular_candidate_forced_by_s3_alone=candidate_forced,
        rank_two_requires_defect_polarization=projector.rank_two_requires_defect_polarization,
        rank_five_not_unique=rank_five_ambiguous,
        signature_breakdown_control_detected=signature_control,
        selected_physical_head=selected,
        interpretation=interpretation,
    )
