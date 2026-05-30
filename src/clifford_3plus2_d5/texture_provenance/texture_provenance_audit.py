"""B4 — combined texture-provenance verdict (roadmap gate A3b).

Aggregates B1 (derived-factor ledger), B2 (free-input enumeration), and B3
(parameter count). Produces the honest "derive or count" verdict: the texture
structure (CP phases, mixing structure, group-theory factors) is derived, while
the generation hierarchy rides on the free depth embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.texture_provenance.b1_derived_factors import (
    DerivedFactorsAuditPayload,
    derived_factors_audit_payload,
)
from clifford_3plus2_d5.texture_provenance.b2_free_inputs import (
    FreeInputsAuditPayload,
    free_inputs_audit_payload,
)
from clifford_3plus2_d5.texture_provenance.b3_parameter_count import (
    ParameterCountAuditPayload,
    parameter_count_audit_payload,
)

REMAINING_DECLARED_INPUTS = ("generation_depth_embedding_derived",)


@dataclass(frozen=True)
class TextureProvenanceAuditPayload:
    """Combined verdict payload for the A3b texture-provenance gate."""

    final_verdict: str
    derived_factors: DerivedFactorsAuditPayload
    free_inputs: FreeInputsAuditPayload
    parameter_count: ParameterCountAuditPayload
    texture_is_predictive: bool
    full_hierarchy_derived: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def texture_provenance_audit_payload() -> TextureProvenanceAuditPayload:
    """Return the combined A3b verdict with the generation-mechanism deferral."""

    b1 = derived_factors_audit_payload()
    b2 = free_inputs_audit_payload()
    b3 = parameter_count_audit_payload()

    structure_derived = b1.final_verdict == "DERIVED_FACTORS_CATALOGUED"
    predictive = b3.final_verdict == "TEXTURE_PREDICTIVE"

    if structure_derived and predictive:
        final_verdict = "TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT"
        interpretation = (
            "The CKM/PMNS texture's group-theoretic and geometric factors are "
            "derived (B1): C_F = 4/3, coin base sqrt(5) = (2_BCC + 3_color), BCC "
            "sqrt(2)/(1/sqrt(2)), charged-lepton sqrt(3/2), and the V10 leptonic "
            "phase word — giving the CP phases atan(sqrt(5)) and 5 pi/12 and the "
            f"PMNS angle structure. The {b2.n_free} free inputs (B2) — the quark "
            "depth embedding, the charged-lepton depth, the r=1 ergodicity prior, "
            "and the CP-phase branch — set the hierarchy, and are fewer than the "
            f"{b3.n_observables} CKM/PMNS observables (B3, surplus {b3.surplus}), "
            "so the textures are predictive for structure, NOT numerology. The "
            "framework does not derive the hierarchy: deriving the depth "
            "embedding {0,2,6} is a generation mechanism (N=3 is empirical per "
            "the closed kill-sidecars), recorded as the remaining input."
        )
    elif not predictive:
        final_verdict = "TEXTURE_NUMEROLOGY_KILL"
        interpretation = (
            "The free-input count is not below the observable count: the textures "
            f"have {b2.n_free} free knobs vs {b3.n_observables} observables. They "
            "are numerology, not predictions."
        )
    else:
        final_verdict = "TEXTURE_PROVENANCE_KILL"
        interpretation = (
            f"A derived-factor check failed (B1={b1.final_verdict}); a "
            "claimed-derived factor is not actually derived."
        )

    return TextureProvenanceAuditPayload(
        final_verdict=final_verdict,
        derived_factors=b1,
        free_inputs=b2,
        parameter_count=b3,
        texture_is_predictive=predictive,
        full_hierarchy_derived=False,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS,
        interpretation=interpretation,
    )
