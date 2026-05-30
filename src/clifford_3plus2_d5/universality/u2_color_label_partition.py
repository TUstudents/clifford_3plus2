"""U2 — the sector difference is the color quantum number.

Universality requires that the quark and lepton boundary shells are not
independent operators but differ by *exactly* the color label.  The quark
primitive shell is

    S_q = 1_even + 5_odd = 1_direct + (2_BCC + 3_color)   (6 channels),

and the lepton/neutrino residual is the 3-channel ``K_3`` residual ``(u, a, b)``
with ``u`` the S_3-trivial collective mode and ``(a, b)`` the S_3 doublet — i.e.
``1 + 2`` with no color.

The gate checks that the quark shell's non-color core ``1_direct + 2_BCC`` (a
``1 + 2`` structure) matches the lepton residual ``1 + 2``, and that the quark
shell is exactly that core plus the three color ports.  The conserved-label
partition ``(parity, bcc_index, color_index)`` separates color cleanly: three
channels carry a color index, three do not.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.universality.reuse import (
    conserved_label_partition_is_complete,
    primitive_conserved_labels,
    quark_shell_dimension_breakdown,
    residual_vectors,
)

# Lepton K_3 residual irrep split under S_3: u is the trivial (1), {a, b} the
# doublet (2).  Stated by residual_basis: 3 = 1 + 2, no color.
LEPTON_RESIDUAL_SPLIT = {"trivial": 1, "doublet": 2}


def quark_noncolor_core_split() -> dict[str, int]:
    """Return the ``1 + 2`` split of the quark shell's non-color core."""

    breakdown = quark_shell_dimension_breakdown()
    return {"trivial": breakdown["even_direct"], "doublet": breakdown["bcc_odd"]}


def quark_color_dimension() -> int:
    """Return the number of color ports in the quark shell (expected 3)."""

    return quark_shell_dimension_breakdown()["color_odd"]


def lepton_residual_dimension() -> int:
    """Return the lepton residual channel count (expected 3, no color)."""

    return len(residual_vectors())


def noncolor_cores_match() -> bool:
    """Return true when the quark non-color core equals the lepton ``1 + 2`` residual."""

    return quark_noncolor_core_split() == LEPTON_RESIDUAL_SPLIT


def conserved_label_color_split() -> tuple[int, int]:
    """Return ``(n_color, n_noncolor)`` from the conserved-label partition."""

    labels = primitive_conserved_labels()
    n_color = sum(1 for label in labels if label.color_index != "none")
    n_noncolor = sum(1 for label in labels if label.color_index == "none")
    return n_color, n_noncolor


def quark_shell_is_lepton_core_plus_color() -> bool:
    """Return true when the quark shell = lepton non-color core + 3 color ports."""

    breakdown = quark_shell_dimension_breakdown()
    core = breakdown["even_direct"] + breakdown["bcc_odd"]
    return (
        core == lepton_residual_dimension()
        and quark_color_dimension() == 3
        and breakdown["total"] == core + quark_color_dimension()
    )


@dataclass(frozen=True)
class ColorLabelPartitionAuditPayload:
    """Verdict payload for the U2 color-label gate."""

    final_verdict: str
    lepton_residual_split: dict[str, int]
    quark_noncolor_core_split: dict[str, int]
    quark_color_dimension: int
    noncolor_cores_match: bool
    conserved_label_color_split: tuple[int, int]
    label_partition_complete: bool
    quark_is_core_plus_color: bool
    interpretation: str


def color_label_partition_audit_payload() -> ColorLabelPartitionAuditPayload:
    """Return the U2 color-label verdict."""

    cores_match = noncolor_cores_match()
    color_split = conserved_label_color_split()
    partition_complete = conserved_label_partition_is_complete()
    core_plus_color = quark_shell_is_lepton_core_plus_color()

    checks_pass = (
        cores_match
        and color_split == (3, 3)
        and partition_complete
        and core_plus_color
    )

    if checks_pass:
        final_verdict = "SECTOR_DIFFERENCE_IS_COLOR_LABEL"
        interpretation = (
            "The quark shell's non-color core (1_direct + 2_BCC) is the same "
            "1 + 2 residual structure as the lepton K3 residual (trivial u + "
            "doublet a,b). The quark shell is exactly that core plus three "
            "color ports, and the conserved-label partition separates three "
            "color-labeled channels from three non-color channels. The quark "
            "and lepton boundaries differ by exactly the color quantum number, "
            "consistent with both being projections of one colored boundary."
        )
    else:
        final_verdict = "SHELLS_INDEPENDENT_KILL"
        interpretation = (
            "The quark non-color core does not match the lepton 1 + 2 residual, "
            "or the color-label partition is not a clean 3 + 3 split. The "
            "boundaries differ by more than the color quantum number; "
            "universality is not supported."
        )

    return ColorLabelPartitionAuditPayload(
        final_verdict=final_verdict,
        lepton_residual_split=LEPTON_RESIDUAL_SPLIT,
        quark_noncolor_core_split=quark_noncolor_core_split(),
        quark_color_dimension=quark_color_dimension(),
        noncolor_cores_match=cores_match,
        conserved_label_color_split=color_split,
        label_partition_complete=partition_complete,
        quark_is_core_plus_color=core_plus_color,
        interpretation=interpretation,
    )
