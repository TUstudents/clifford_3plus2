"""Session 18 down-sector odd-shell rank-five audit.

S3 alone makes two rank-five complements available.  The primitive quark shell
has extra structure: one even direct return and five odd returns.  Given the
active six-label shell from Session 17, the data-improved bottom coefficient is
the full odd-shell projector.  This removes the S3 rank-five ambiguity inside
the primitive-shell model, while keeping the physical readout premise explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    BCC,
    DIRECT,
    EVEN,
    ODD,
    QuarkBoundaryChannel,
    quark_primitive_channels,
)
from clifford_3plus2_d5.scalar_clebsch.down_subset_counts import (
    color_only_middle_control,
    compressed_partition_cannot_derive_bcc_count,
    down_candidate_clebsch_vector,
    down_candidate_counts,
    down_subset_audit_payload,
)
from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    rank_five_is_not_unique,
    s3_projector_audit_payload,
)
from clifford_3plus2_d5.universal_bath.quark_active_color_microcanonical import (
    quark_active_color_microcanonical_payload,
)


BOTTOM_ODD_SHELL_READOUT_PREMISE = "down_bottom_reads_full_primitive_odd_shell"


@dataclass(frozen=True)
class PrimitiveSubset:
    """One primitive-shell subset used by the down readout."""

    label: str
    channels: tuple[QuarkBoundaryChannel, ...]
    count: int
    predicate: str


@dataclass(frozen=True)
class QuarkDownOddShellPayload:
    """Session 18 down odd-shell rank-five verdict."""

    final_verdict: str
    active_color_microcanonical_pass: bool
    down_subset_pass: bool
    s3_projector_pass: bool
    s3_rank_five_ambiguous: bool
    full_shell_subset: PrimitiveSubset
    bcc_odd_subset: PrimitiveSubset
    odd_shell_subset: PrimitiveSubset
    primitive_counts: dict[str, int]
    candidate_counts: dict[str, int]
    candidate_clebsch_vector: tuple[sp.Expr, sp.Expr, sp.Expr]
    odd_shell_rank_five: bool
    odd_shell_is_complement_of_even_direct: bool
    bcc_middle_rank_two: bool
    color_only_middle_control_rejected: bool
    compressed_parity_control_rejected_for_middle: bool
    primitive_parity_selects_rank_five_line: bool
    remaining_readout_premise: str
    source_freeze_ready: bool
    interpretation: str


def _subset(label: str, channels: tuple[QuarkBoundaryChannel, ...], predicate: str) -> PrimitiveSubset:
    """Return a primitive subset record."""

    return PrimitiveSubset(label=label, channels=channels, count=len(channels), predicate=predicate)


def full_shell_subset() -> PrimitiveSubset:
    """Return the full six-label primitive shell."""

    channels = quark_primitive_channels()
    return _subset("full_primitive_shell", channels, "all primitive labels")


def bcc_odd_subset() -> PrimitiveSubset:
    """Return the BCC odd doublet subset."""

    channels = tuple(
        channel
        for channel in quark_primitive_channels()
        if channel.parity == ODD and channel.sector == BCC
    )
    return _subset("bcc_odd_doublet", channels, "parity=odd and sector=BCC")


def odd_shell_subset() -> PrimitiveSubset:
    """Return the full primitive odd shell."""

    channels = tuple(channel for channel in quark_primitive_channels() if channel.parity == ODD)
    return _subset("full_odd_shell", channels, "parity=odd")


def even_direct_subset() -> PrimitiveSubset:
    """Return the even direct one-dimensional subset."""

    channels = tuple(
        channel
        for channel in quark_primitive_channels()
        if channel.parity == EVEN and channel.sector == DIRECT
    )
    return _subset("even_direct_line", channels, "parity=even and sector=direct")


def primitive_counts() -> dict[str, int]:
    """Return primitive-predicate counts in down family order."""

    return {
        "d": full_shell_subset().count,
        "s": bcc_odd_subset().count,
        "b": odd_shell_subset().count,
    }


def odd_shell_is_complement_of_even_direct() -> bool:
    """Return whether the odd shell is the complement of the even direct line."""

    full_names = {channel.name for channel in full_shell_subset().channels}
    odd_names = {channel.name for channel in odd_shell_subset().channels}
    even_names = {channel.name for channel in even_direct_subset().channels}
    return (
        even_direct_subset().count == 1
        and odd_shell_subset().count == 5
        and odd_names.isdisjoint(even_names)
        and odd_names | even_names == full_names
    )


def bcc_middle_rank_two() -> bool:
    """Return whether the middle subset is exactly the BCC odd doublet."""

    subset = bcc_odd_subset()
    return (
        subset.count == 2
        and all(channel.parity == ODD and channel.sector == BCC for channel in subset.channels)
    )


def primitive_parity_selects_rank_five_line() -> bool:
    """Return whether primitive parity selects the rank-five complement."""

    return odd_shell_is_complement_of_even_direct() and odd_shell_subset().count == 5


def quark_down_odd_shell_payload() -> QuarkDownOddShellPayload:
    """Return the Session 18 down odd-shell rank-five audit payload."""

    active = quark_active_color_microcanonical_payload()
    down = down_subset_audit_payload()
    s3 = s3_projector_audit_payload()
    full = full_shell_subset()
    bcc = bcc_odd_subset()
    odd = odd_shell_subset()
    counts = primitive_counts()
    candidate_counts = down_candidate_counts()
    candidate_vector = down_candidate_clebsch_vector()
    color_control = color_only_middle_control()
    compressed_rejected = compressed_partition_cannot_derive_bcc_count()

    active_pass = active.final_verdict == "QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS"
    down_pass = down.final_verdict == "DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_PASS"
    s3_pass = s3.final_verdict == "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
    s3_ambiguous = rank_five_is_not_unique()
    odd_rank_five = odd.count == 5
    odd_complement = odd_shell_is_complement_of_even_direct()
    middle_rank_two = bcc_middle_rank_two()
    parity_selects = primitive_parity_selects_rank_five_line()
    source_freeze_ready = False

    checks_pass = (
        active_pass
        and down_pass
        and s3_pass
        and s3_ambiguous
        and full.count == 6
        and bcc.count == 2
        and odd_rank_five
        and counts == {"d": 6, "s": 2, "b": 5}
        and counts == candidate_counts
        and candidate_vector == (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(5, 6)))
        and odd_complement
        and middle_rank_two
        and color_control != candidate_vector
        and compressed_rejected
        and parity_selects
        and not source_freeze_ready
    )

    if checks_pass:
        final_verdict = "QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS"
        interpretation = (
            "The active primitive shell removes the S3 rank-five ambiguity: "
            "the bottom candidate is the full primitive odd shell, the "
            "complement of the even direct line, while the strange middle "
            "subset is the BCC odd doublet.  This gives counts (6,2,5) and "
            "Clebsches (1,1/sqrt(3),sqrt(5/6)).  The remaining premise is the "
            "physical down-head readout that assigns bottom to the full odd "
            "shell; this is not derived from S3 alone."
        )
    else:
        final_verdict = "QUARK_DOWN_ODD_SHELL_RANK_FIVE_AUDIT_KILL"
        interpretation = (
            "The active-shell prerequisite, down-subset prerequisite, S3 "
            "ambiguity check, primitive subset counts, or controls failed."
        )

    return QuarkDownOddShellPayload(
        final_verdict=final_verdict,
        active_color_microcanonical_pass=active_pass,
        down_subset_pass=down_pass,
        s3_projector_pass=s3_pass,
        s3_rank_five_ambiguous=s3_ambiguous,
        full_shell_subset=full,
        bcc_odd_subset=bcc,
        odd_shell_subset=odd,
        primitive_counts=counts,
        candidate_counts=candidate_counts,
        candidate_clebsch_vector=candidate_vector,
        odd_shell_rank_five=odd_rank_five,
        odd_shell_is_complement_of_even_direct=odd_complement,
        bcc_middle_rank_two=middle_rank_two,
        color_only_middle_control_rejected=color_control != candidate_vector,
        compressed_parity_control_rejected_for_middle=compressed_rejected,
        primitive_parity_selects_rank_five_line=parity_selects,
        remaining_readout_premise=BOTTOM_ODD_SHELL_READOUT_PREMISE,
        source_freeze_ready=source_freeze_ready,
        interpretation=interpretation,
    )
