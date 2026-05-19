"""Phase D-5 combined verdict: continuous + global + discrete anomalies.

Aggregates the three sub-phase payloads:

- ``bcc_anomaly_polynomial.anomaly_polynomial_payload`` (FD-11/FD-12)
- ``global_anomaly_check.global_anomaly_check_payload`` (FD-13)
- ``discrete_anomaly_constraint.discrete_anomaly_constraint_payload`` (FD-14)

and produces a single ``AnomalyAuditPayload`` summarising the verdict
for the topology sidecar.

Expected outcome: ANOMALY KILL — anomalies cancel per generation and
impose no constraint on N.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.topology.anomaly_cancellation.bcc_anomaly_polynomial import (
    AnomalyPolynomialPayload,
    anomaly_polynomial_payload,
)
from clifford_3plus2_d5.topology.anomaly_cancellation.discrete_anomaly_constraint import (
    DiscreteAnomalyConstraintPayload,
    discrete_anomaly_constraint_payload,
)
from clifford_3plus2_d5.topology.anomaly_cancellation.global_anomaly_check import (
    GlobalAnomalyCheckPayload,
    global_anomaly_check_payload,
)


@dataclass(frozen=True)
class AnomalyAuditPayload:
    generations: int
    polynomial: AnomalyPolynomialPayload
    global_check: GlobalAnomalyCheckPayload
    discrete_constraint: DiscreteAnomalyConstraintPayload
    all_cancellations_satisfied: bool
    forces_three_generations: bool
    verdict: str
    interpretation: str


def anomaly_audit_payload(generations: int = 3) -> AnomalyAuditPayload:
    poly = anomaly_polynomial_payload(generations)
    glob = global_anomaly_check_payload(generations)
    constraint = discrete_anomaly_constraint_payload()

    all_ok = (
        poly.all_cancel
        and glob.witten_anomaly_free
        and constraint.per_generation_continuous_cancels
        and constraint.per_generation_global_anomaly_free
    )
    forces_three = constraint.forces_N_equals_three

    if all_ok and not forces_three:
        verdict = "ANOMALY KILL — anomalies cancel for any N, do not force N = 3"
        interpretation = (
            "Phase D-5 verdict: all four continuous SM anomalies "
            "(gravitational, U(1)_Y³, SU(2)²·Y, SU(3)²·Y) cancel exactly "
            "per generation.  Witten's global SU(2) anomaly is satisfied "
            "because each generation contributes 4 SU(2)_L doublets (even).  "
            "The combined discrete anomaly constraint on N reduces to "
            "0 = 0, satisfied for every N ≥ 0.  Three generations is "
            "admissible but not unique — the anomaly route does NOT force "
            "N = 3 on the BCC × chiral-16 carrier."
        )
    elif all_ok and forces_three:
        verdict = "ANOMALY POSITIVE — N = 3 forced"
        interpretation = (
            "Anomalies cancel AND the constraint pins N to 3.  Investigate "
            "the constraint source — this would be a major positive."
        )
    else:
        verdict = "ANOMALY ISSUE — cancellation fails"
        interpretation = (
            f"Some anomaly does not cancel.  Polynomial cancel: "
            f"{poly.all_cancel}; Witten free: {glob.witten_anomaly_free}; "
            f"per-generation continuous: "
            f"{constraint.per_generation_continuous_cancels}; per-generation "
            f"global: {constraint.per_generation_global_anomaly_free}.  "
            "Investigate before drawing conclusions."
        )

    return AnomalyAuditPayload(
        generations=generations,
        polynomial=poly,
        global_check=glob,
        discrete_constraint=constraint,
        all_cancellations_satisfied=all_ok,
        forces_three_generations=forces_three,
        verdict=verdict,
        interpretation=interpretation,
    )
