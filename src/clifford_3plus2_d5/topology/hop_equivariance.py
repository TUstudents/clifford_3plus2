"""Phase D-1: Bialynicki-Birula hops Z_3-equivariance check.

For each BCC direction h with sigma(h) = h', check whether the BB hop
matrices satisfy

    W_{h'} = U_3 . W_h . U_3^{-1}

where U_3 is the 2-spinor lift of the body-diagonal Z_3 rotation.

If yes: the bare massless Weyl walk on BCC has body-diagonal Z_3
symmetry on its (spatial × Dirac) sector.

If no: the BB construction picks a direction-asymmetric convention
that breaks the Z_3 (would be unexpected and worth investigating).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.topology.bcc_z3_rotation import (
    bcc_direction_permutation,
    dirac_spinor_lift,
)
from clifford_3plus2_d5.topology.reuse import (
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    same_matrix,
)


def equivariance_residual_for_index(index: int) -> sp.Matrix:
    """Return ``W_{sigma(h)} - U_3 . W_h . U_3^{-1}`` for the i-th BCC direction."""

    hops = bialynicki_birula_hops()
    perm = bcc_direction_permutation()
    u3 = dirac_spinor_lift()
    u3_inv = u3.inv()

    transformed = (u3 * hops[index] * u3_inv).applyfunc(sp.simplify)
    target = hops[perm[index]]
    return (target - transformed).applyfunc(sp.simplify)


def all_hops_equivariant() -> bool:
    """Return whether all 8 BB hops are Z_3-equivariant under U_3."""

    for index in range(8):
        residual = equivariance_residual_for_index(index)
        if not same_matrix(residual, sp.zeros(2, 2)):
            return False
    return True


def equivariance_failures() -> tuple[tuple[int, sp.Matrix], ...]:
    """Return ``(index, residual)`` pairs for non-equivariant hops."""

    failures: list[tuple[int, sp.Matrix]] = []
    for index in range(8):
        residual = equivariance_residual_for_index(index)
        if not same_matrix(residual, sp.zeros(2, 2)):
            failures.append((index, residual))
    return tuple(failures)


@dataclass(frozen=True)
class HopEquivarianceAuditPayload:
    direction_count: int
    permutation: tuple[int, ...]
    cycle_lengths: tuple[int, ...]
    all_equivariant: bool
    failure_count: int
    verdict: str
    interpretation: str


def hop_equivariance_audit_payload() -> HopEquivarianceAuditPayload:
    """Run the BB hops Z_3-equivariance test."""

    from clifford_3plus2_d5.topology.bcc_z3_rotation import cycle_lengths

    perm = bcc_direction_permutation()
    cycles = cycle_lengths()
    equivariant = all_hops_equivariant()
    failures = equivariance_failures()

    if equivariant:
        verdict = "HOPS Z_3-EQUIVARIANT"
        interpretation = (
            "All 8 Bialynicki-Birula BCC hops satisfy "
            "W_{sigma(h)} = U_3 . W_h . U_3^{-1} under the body-diagonal "
            "Z_3 rotation.  The BB construction respects the Z_3 symmetry "
            "of the BCC lattice.  However, this Z_3 acts on the SPATIAL × "
            "DIRAC sector only — the chiral-16 INTERNAL carrier is not "
            "touched.  See internal_triviality.py for the chiral-16 audit."
        )
    else:
        verdict = "HOPS NOT EQUIVARIANT"
        interpretation = (
            f"{len(failures)} of 8 BB hops fail the Z_3 equivariance "
            f"condition.  The Bialynicki-Birula construction may pick a "
            f"direction-asymmetric convention.  Investigate the failure "
            f"residuals before drawing conclusions."
        )

    return HopEquivarianceAuditPayload(
        direction_count=len(bialynicki_birula_directions()),
        permutation=perm,
        cycle_lengths=cycles,
        all_equivariant=equivariant,
        failure_count=len(failures),
        verdict=verdict,
        interpretation=interpretation,
    )
