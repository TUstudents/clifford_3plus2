"""B2 — free-input enumeration.

Lists the texture program's free inputs (the things assigned rather than derived),
each with its source. These set the generation HIERARCHY; the derived factors
(B1) set the structure.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.texture_provenance.reuse import (
    REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER,
    quark_family_depths,
    quark_transition_depths,
)


@dataclass(frozen=True)
class FreeInput:
    """A free (assigned) texture input."""

    name: str
    source: str
    assigned: str


def quark_depths_are_even_and_additive() -> bool:
    """Return true when the assigned depths are even and additive (the checked structure)."""

    depths = quark_transition_depths()
    even = all(d % 2 == 0 for d in depths.values())
    additive = depths[(1, 3)] == depths[(1, 2)] + depths[(2, 3)]
    return even and additive


def free_inputs() -> tuple[FreeInput, ...]:
    """Return the enumerated free inputs of the texture program."""

    depths = quark_family_depths()
    return (
        FreeInput(
            name="quark_depth_embedding",
            source="quark_transfer_hierarchy.quark_family_depths",
            assigned=(
                f"{depths} — fit to the CKM hierarchy; even+additive is checked, "
                "so given that structure two depths are free"
            ),
        ),
        FreeInput(
            name="charged_lepton_leakage_depth",
            source="charged_lepton_leakage (depth=2)",
            assigned="two-step leakage depth (sets sin theta_e = sqrt(3/2) eps^2)",
        ),
        FreeInput(
            name="ergodicity_prior_r_equals_1",
            source="local_boundary_fiber / regular_boundary_fiber (Jaynes max-entropy)",
            assigned=(
                "flat primitive ratio r=1 (gives atan(sqrt(5))), reduced to "
                f"{REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER[0]}"
            ),
        ),
        FreeInput(
            name="cp_phase_branch",
            source="pmns_conditional.conjugate_branch",
            assigned="discrete sign branch of the leptonic CP phase (+/- 5 pi / 12)",
        ),
    )


@dataclass(frozen=True)
class FreeInputsAuditPayload:
    """Verdict payload for the B2 free-input enumeration."""

    final_verdict: str
    free_inputs: tuple[FreeInput, ...]
    n_free: int
    quark_depths_even_additive: bool
    interpretation: str


def free_inputs_audit_payload() -> FreeInputsAuditPayload:
    """Return the B2 verdict."""

    inputs = free_inputs()
    even_additive = quark_depths_are_even_and_additive()

    final_verdict = "FREE_INPUTS_ENUMERATED"
    interpretation = (
        f"The texture program has {len(inputs)} free inputs: the quark depth "
        "embedding {0,2,6} (fit to the CKM hierarchy; even+additive checked), the "
        "charged-lepton two-step depth, the flat-coin r=1 ergodicity prior "
        "(ultimately physical_vacuum_order_parameter_exists), and the CP-phase "
        "branch. These set the hierarchy; the derived factors (B1) set the "
        "structure."
    )

    return FreeInputsAuditPayload(
        final_verdict=final_verdict,
        free_inputs=inputs,
        n_free=len(inputs),
        quark_depths_even_additive=even_additive,
        interpretation=interpretation,
    )
