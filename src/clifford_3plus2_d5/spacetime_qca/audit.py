"""Small result payloads for spacetime QCA audit sessions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpacetimeQCAInfrastructureAudit:
    bcc_direction_count: int
    dirac_dimension: int
    hypercube_corner_count: int
    notes: tuple[str, ...] = ()


def infrastructure_audit_payload() -> SpacetimeQCAInfrastructureAudit:
    return SpacetimeQCAInfrastructureAudit(
        bcc_direction_count=8,
        dirac_dimension=4,
        hypercube_corner_count=8,
        notes=(
            "Bialynicki-Birula hop matrices are intentionally not guessed.",
            "Hypercube control is available as the doubling baseline.",
        ),
    )
