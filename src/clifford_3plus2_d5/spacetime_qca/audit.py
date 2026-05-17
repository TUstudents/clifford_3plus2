"""Small result payloads for spacetime QCA audit sessions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpacetimeQCAInfrastructureAudit:
    bcc_direction_count: int
    dirac_dimension: int
    hypercube_corner_count: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BCCDiracAudit:
    source: str
    right_weyl_hamiltonian: str
    left_weyl_hamiltonian: str
    dirac_hamiltonian: str
    hypercube_corner_gapless_count: int
    bcc_cubic_corner_gapless_count: int
    bcc_gapless_corner_note: str
    gauge_lift: str


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


def bcc_dirac_audit_payload() -> BCCDiracAudit:
    return BCCDiracAudit(
        source="Bialynicki-Birula Phys. Rev. D 49, 6920 (1994), Section II",
        right_weyl_hamiltonian="H_R(k) = sigma . k",
        left_weyl_hamiltonian="H_L(k) = -sigma . k",
        dirac_hamiltonian="H_D(k) = alpha . k",
        hypercube_corner_gapless_count=8,
        bcc_cubic_corner_gapless_count=4,
        bcc_gapless_corner_note=(
            "The four gapless cubic-corner representatives are reciprocal-lattice "
            "origin equivalents for the body-diagonal lattice."
        ),
        gauge_lift="constant background link gives H = alpha.k x I + I x iA",
    )
