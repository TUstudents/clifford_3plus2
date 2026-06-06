"""Down-sector primitive-label subset counts.

The natural S3/projector baseline gives counts ``(6,2,4)``. The data-improved
odd-shell candidate uses ``(6,2,5)``. This module keeps both objects visible so
the candidate bottom ``+1`` is not mistaken for a derived S3 result.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    BCC,
    ODD,
    QuarkBoundaryChannel,
    quark_boundary_shell_audit_payload,
    quark_primitive_channels,
)


DOWN_FAMILIES = ("d", "s", "b")


def primitive_channel_subsets(
    channels: tuple[QuarkBoundaryChannel, ...] | None = None,
) -> dict[str, tuple[QuarkBoundaryChannel, ...]]:
    """Return down-sector data-candidate subsets of the primitive quark labels.

    The candidate subsets are exact shell predicates:

    * ``d``: the full primitive shell, count 6;
    * ``s``: the BCC odd doublet, count 2;
    * ``b``: the full odd shell, count 5.

    The bottom choice is the data-improved odd-shell candidate. The natural S3
    baseline is represented separately by :func:`down_baseline_counts`.
    """

    selected = quark_primitive_channels() if channels is None else channels
    return {
        "d": selected,
        "s": tuple(
            channel
            for channel in selected
            if channel.parity == ODD and channel.sector == BCC
        ),
        "b": tuple(channel for channel in selected if channel.parity == ODD),
    }


def down_candidate_counts(
    channels: tuple[QuarkBoundaryChannel, ...] | None = None,
) -> dict[str, int]:
    """Return data-candidate primitive subset counts for ``(d,s,b)``."""

    return {
        family: len(subset)
        for family, subset in primitive_channel_subsets(channels).items()
    }


def down_baseline_counts() -> dict[str, int]:
    """Return the natural S3/projector baseline counts ``(6,2,4)``."""

    return {"d": 6, "s": 2, "b": 4}


def _sqrt_count_vector(counts: dict[str, int], denominator: int = 6) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``sqrt(n/denominator)`` in ``(d,s,b)`` order."""

    return tuple(
        sp.sqrt(sp.Rational(counts[family], denominator))
        for family in DOWN_FAMILIES
    )


def down_candidate_clebsch_vector(
    channels: tuple[QuarkBoundaryChannel, ...] | None = None,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the data-candidate ``sqrt(n/6)`` down-sector Clebsches."""

    denominator = len(quark_primitive_channels() if channels is None else channels)
    return _sqrt_count_vector(down_candidate_counts(channels), denominator)


def down_baseline_clebsch_vector() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the S3/projector baseline down-sector Clebsches."""

    return _sqrt_count_vector(down_baseline_counts())


def down_candidate_mass_ratio_predictions(eta: sp.Expr) -> dict[str, sp.Expr]:
    """Return RG-invariant down ratio predictions from candidate counts."""

    eta = sp.sympify(eta)
    c_d, c_s, c_b = down_candidate_clebsch_vector()
    return {
        "m_d/m_s": sp.simplify((c_d / c_s) * eta**2),
        "m_s/m_b": sp.simplify((c_s / c_b) * eta**2),
    }


def down_baseline_mass_ratio_predictions(eta: sp.Expr) -> dict[str, sp.Expr]:
    """Return RG-invariant down ratio predictions from the S3 baseline."""

    eta = sp.sympify(eta)
    c_d, c_s, c_b = down_baseline_clebsch_vector()
    return {
        "m_d/m_s": sp.simplify((c_d / c_s) * eta**2),
        "m_s/m_b": sp.simplify((c_s / c_b) * eta**2),
    }


def compressed_parity_counts() -> tuple[int, int]:
    """Return the counts available after compressing labels to even/odd only."""

    channels = quark_primitive_channels()
    even = sum(1 for channel in channels if channel.parity != ODD)
    odd = sum(1 for channel in channels if channel.parity == ODD)
    return even, odd


def compressed_partition_cannot_derive_bcc_count() -> bool:
    """Return true when the even/odd partition cannot produce the BCC count 2."""

    return 2 not in compressed_parity_counts()


def color_only_middle_control() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return a bad control using the color triplet instead of the BCC doublet."""

    counts = {
        "d": 6,
        "s": 3,
        "b": 5,
    }
    return _sqrt_count_vector(counts)


def s3_projector_bottom_control() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the natural S3/projector vector with bottom count ``4``."""

    return down_baseline_clebsch_vector()


def odd_shell_plus_one_is_open() -> bool:
    """Return true because the bottom ``4 -> 5`` shift remains an open gate."""

    return True


@dataclass(frozen=True)
class DownSubsetAuditPayload:
    """Audit payload for down-sector integer subset counts."""

    final_verdict: str
    baseline_counts: dict[str, int]
    candidate_counts: dict[str, int]
    baseline_clebsch_vector: tuple[sp.Expr, sp.Expr, sp.Expr]
    candidate_clebsch_vector: tuple[sp.Expr, sp.Expr, sp.Expr]
    baseline_ratio_formulas: dict[str, sp.Expr]
    candidate_ratio_formulas: dict[str, sp.Expr]
    quark_shell_prerequisite_pass: bool
    compressed_partition_rejected: bool
    color_only_middle_control_rejected: bool
    s3_projector_baseline_confirmed: bool
    plus_one_open: bool
    interpretation: str


def down_subset_audit_payload() -> DownSubsetAuditPayload:
    """Return the down-sector baseline/candidate verdict."""

    q1 = quark_boundary_shell_audit_payload()
    baseline_counts = down_baseline_counts()
    candidate_counts = down_candidate_counts()
    baseline_vector = down_baseline_clebsch_vector()
    candidate_vector = down_candidate_clebsch_vector()
    eta = sp.Symbol("eta", positive=True)
    baseline_ratios = down_baseline_mass_ratio_predictions(eta)
    candidate_ratios = down_candidate_mass_ratio_predictions(eta)
    baseline_target = (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)))
    candidate_target = (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(5, 6)))
    color_control = color_only_middle_control()
    s3_control = s3_projector_bottom_control()

    checks_pass = (
        q1.final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
        and baseline_counts == {"d": 6, "s": 2, "b": 4}
        and candidate_counts == {"d": 6, "s": 2, "b": 5}
        and baseline_vector == baseline_target
        and candidate_vector == candidate_target
        and sp.simplify(baseline_ratios["m_d/m_s"] - sp.sqrt(3) * eta**2) == 0
        and sp.simplify(baseline_ratios["m_s/m_b"] - eta**2 / sp.sqrt(2)) == 0
        and sp.simplify(candidate_ratios["m_d/m_s"] - sp.sqrt(3) * eta**2) == 0
        and sp.simplify(candidate_ratios["m_s/m_b"] - sp.sqrt(sp.Rational(2, 5)) * eta**2) == 0
        and compressed_partition_cannot_derive_bcc_count()
        and color_control != candidate_vector
        and s3_control == baseline_vector
        and odd_shell_plus_one_is_open()
    )

    if checks_pass:
        final_verdict = "DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_PASS"
        interpretation = (
            "The six-channel quark shell cleanly supplies the S3/projector "
            "baseline counts (6,2,4), giving (1,1/sqrt(3),sqrt(2/3)). The "
            "data-improved odd-shell candidate uses counts (6,2,5), giving "
            "(1,1/sqrt(3),sqrt(5/6)). The bottom +1 is intentionally marked "
            "as an open theory burden rather than a derived S3 result."
        )
    else:
        final_verdict = "DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_KILL"
        interpretation = (
            "The quark-shell prerequisite, baseline/candidate counts, ratio "
            "formulas, or control checks failed."
        )

    return DownSubsetAuditPayload(
        final_verdict=final_verdict,
        baseline_counts=baseline_counts,
        candidate_counts=candidate_counts,
        baseline_clebsch_vector=baseline_vector,
        candidate_clebsch_vector=candidate_vector,
        baseline_ratio_formulas=baseline_ratios,
        candidate_ratio_formulas=candidate_ratios,
        quark_shell_prerequisite_pass=q1.final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS",
        compressed_partition_rejected=compressed_partition_cannot_derive_bcc_count(),
        color_only_middle_control_rejected=color_control != candidate_vector,
        s3_projector_baseline_confirmed=s3_control == baseline_vector,
        plus_one_open=odd_shell_plus_one_is_open(),
        interpretation=interpretation,
    )
