"""Phase A-5 combined verdict: framework + mapping + ε constraint.

Aggregates the three sub-phase payloads:

- ``sme_framework_identification.framework_identification_payload``
  (Phase A-1)
- ``sme_tensor_mapping.mapping_audit_payload`` (Phase A-2)
- ``epsilon_constraint.epsilon_constraint_payload`` (Phase A-4)

and produces a single ``SMEAuditPayload`` summarising the Bold A
verdict.

The Phase A-3 literature note (``SME_LITERATURE_NOTE.md``) feeds into
this audit indirectly via the bound constant in
``epsilon_constraint.DIM5_FERMION_BOUND_GEV_INVERSE``.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.sme.epsilon_constraint import (
    EpsilonConstraintPayload,
    epsilon_constraint_payload,
)
from clifford_3plus2_d5.sme.sme_framework_identification import (
    FrameworkIdentificationPayload,
    framework_identification_payload,
)
from clifford_3plus2_d5.sme.sme_tensor_mapping import (
    SMETensorMappingPayload,
    mapping_audit_payload,
)


@dataclass(frozen=True)
class SMEAuditPayload:
    framework: FrameworkIdentificationPayload
    mapping: SMETensorMappingPayload
    epsilon_constraint: EpsilonConstraintPayload
    all_phases_consistent: bool
    provisional_caveats: tuple[str, ...]
    is_provisional: bool
    final_scale_verdict: str
    verdict: str
    interpretation: str


def _collect_provisional_caveats(
    mapping: SMETensorMappingPayload,
    constraint: EpsilonConstraintPayload,
) -> tuple[str, ...]:
    caveats: list[str] = []
    if not mapping.field_redefinition_checked:
        caveats.append(
            "F-sme-5 field-redefinition triviality unchecked "
            "(d^{(5)} direction may be vacuous under SME field "
            "redefinitions)"
        )
    if not mapping.km_hamiltonian_normalization_derived:
        caveats.append(
            "Kostelecky-Mewes Hamiltonian-form normalization of "
            "d^{(5)}_{αβγ} not derived (mapping is structural "
            "identification; ε bound carries O(1) normalization "
            "uncertainty)"
        )
    if not constraint.kr_entry_ids_verified:
        caveats.append(
            "Kostelecky-Russell entry IDs (arXiv:0801.0287 v19) not "
            "verified — d^{(5)} bound is the representative "
            "order-of-magnitude estimate 10⁻¹⁷ GeV⁻¹"
        )
    return tuple(caveats)


def sme_audit_payload() -> SMEAuditPayload:
    framework = framework_identification_payload()
    mapping = mapping_audit_payload()
    constraint = epsilon_constraint_payload()

    all_consistent = (
        framework.all_classes_consistent
        and mapping.nonzero_component_count == 3
        and mapping.identity_coefficient_vanishes
        and mapping.lower_block_equals_upper
    )

    final = constraint.scale_verdict
    caveats = _collect_provisional_caveats(mapping, constraint)
    is_provisional = len(caveats) > 0

    if not all_consistent:
        verdict = "SME AUDIT INCONSISTENCY"
        interpretation = (
            "One or more sub-phase consistency checks failed.  "
            f"framework: {framework.verdict}; "
            f"mapping: {mapping.verdict}; "
            f"epsilon_constraint: {constraint.scale_verdict}.  "
            "Investigate before drawing conclusions."
        )
    else:
        qualifier = "PROVISIONAL " if is_provisional else ""
        verdict = f"{qualifier}SME AUDIT — {final}"
        if is_provisional:
            caveat_block = (
                "  Provisional caveats: " + "; ".join(caveats) + "."
            )
        else:
            caveat_block = ""
        interpretation = (
            f"Phase A-1 identified the SME sector as {framework.sme_sector_label}.  "
            f"Phase A-2 mapped H^(1) to {mapping.nonzero_component_count} non-zero "
            f"components of {mapping.sme_target_label} ({mapping.cpt_class}).  "
            f"Phase A-3 (literature note) supplied the tightest representative "
            f"bound on d^(5).  Phase A-4 produced an ε bound of "
            f"{float(constraint.epsilon_bound_metres):.3e} m, which is "
            f"{float(constraint.epsilon_bound_orders_above_planck):.2f} orders "
            f"of magnitude above the Planck length.  Final scale verdict: "
            f"{final}.  {constraint.verdict_class_explanation}"
            f"{caveat_block}"
        )

    return SMEAuditPayload(
        framework=framework,
        mapping=mapping,
        epsilon_constraint=constraint,
        all_phases_consistent=all_consistent,
        provisional_caveats=caveats,
        is_provisional=is_provisional,
        final_scale_verdict=final,
        verdict=verdict,
        interpretation=interpretation,
    )
