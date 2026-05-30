"""B3 — parameter count vs observables.

Counts the texture program's free inputs against the flavor observables it
targets, and separates the genuine derived predictions (the CP phases, the PMNS
angle structure, the CKM additivity relation) from the hierarchy (the depth
embedding). Kills if the textures are pure numerology (as many free knobs as
data).
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.flavor_a_track.b2_free_inputs import free_inputs

# CKM: 3 mixing angles + 1 CP phase; PMNS: 3 mixing angles + 1 Dirac CP phase.
N_CKM_OBSERVABLES = 4
N_PMNS_OBSERVABLES = 4
N_TEXTURE_OBSERVABLES = N_CKM_OBSERVABLES + N_PMNS_OBSERVABLES

# Genuine derived predictions (not just the free inputs re-expressed).
GENUINE_DERIVED_PREDICTIONS = (
    "CKM CP phase delta_q = atan(sqrt(5))",
    "PMNS Dirac CP phase = 5 pi / 12 (V10 word)",
    "PMNS angle structure (TBM + theta_e = arcsin(sqrt(3/2) eps^2))",
    "CKM additivity relation s_13 ~ depth_12 + depth_23",
)


def parameter_count_verdict(n_free: int, n_observables: int) -> str:
    """Return the verdict for a given (n_free, n_observables)."""

    if n_free >= n_observables:
        return "TEXTURE_NUMEROLOGY_KILL"
    return "TEXTURE_PREDICTIVE"


@dataclass(frozen=True)
class ParameterCountAuditPayload:
    """Verdict payload for the B3 parameter-count gate."""

    final_verdict: str
    n_free: int
    n_observables: int
    surplus: int
    genuine_derived_predictions: tuple[str, ...]
    interpretation: str


def parameter_count_audit_payload() -> ParameterCountAuditPayload:
    """Return the B3 verdict."""

    n_free = len(free_inputs())
    n_observables = N_TEXTURE_OBSERVABLES
    surplus = n_observables - n_free
    verdict = parameter_count_verdict(n_free, n_observables)

    if verdict == "TEXTURE_PREDICTIVE":
        interpretation = (
            f"The texture program has {n_free} free inputs against "
            f"{n_observables} CKM/PMNS observables (surplus {surplus} > 0), so it "
            "is not pure numerology. The surplus is realized as genuine derived "
            "predictions: the two CP phases, the TBM-based PMNS angle structure, "
            "and the CKM additivity relation. The magnitude hierarchy still rides "
            "on the free depth embedding."
        )
    else:
        interpretation = (
            f"The texture program has {n_free} free inputs against "
            f"{n_observables} observables (surplus {surplus} <= 0): as many free "
            "knobs as data points. The textures are numerology, not predictions."
        )

    return ParameterCountAuditPayload(
        final_verdict=verdict,
        n_free=n_free,
        n_observables=n_observables,
        surplus=surplus,
        genuine_derived_predictions=GENUINE_DERIVED_PREDICTIONS,
        interpretation=interpretation,
    )
