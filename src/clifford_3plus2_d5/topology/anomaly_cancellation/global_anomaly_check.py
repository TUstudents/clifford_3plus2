"""FD-13: Witten global SU(2) anomaly + mod-2 lattice anomaly check.

Witten's global SU(2) anomaly (Phys. Lett. B 117, 324, 1982) arises
because

    pi_4(SU(2)) = Z/2

i.e. there exist gauge transformations on S^4 that cannot be continuously
deformed to the identity.  An SU(2) gauge theory with Weyl fermions is
anomaly-free under such transformations only if the number of SU(2)_L
doublets is EVEN.

Per Standard Model generation, the SU(2)_L-doublet content is:

    Q_L  (color triplet × SU(2)_L doublet)  = 3 doublets (one per color)
    L_L  (color singlet × SU(2)_L doublet)  = 1 doublet
    Total per generation:                    = 4 doublets  (even ✓)

For N generations: 4N doublets, always even.  Witten's anomaly does NOT
constrain N.

The mod-2 lattice anomaly counts the SU(2)_L doublet number modulo 2 on
the BCC walk.  Since the chiral-16 carries 4 doublets per generation,
4N mod 2 = 0 for any N.  No lattice-level constraint on N either.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.topology.anomaly_cancellation.bcc_anomaly_polynomial import (
    SM_LEFT_HANDED_CONTENT_PER_GENERATION,
)


def su2_doublet_count_per_generation() -> int:
    """Return the number of SU(2)_L doublets per SM generation.

    Each doublet is a single Weyl fermion living in the SU(2)_L 2.
    A color-triplet doublet (Q_L) counts as 3 doublets (one per color)
    because Witten's anomaly counts the SU(2)_L Weyl-doublet number
    weighted by ``dim(other gauge reps)``.
    """

    total = 0
    for (_, _, n_color, _, n_su2, _) in SM_LEFT_HANDED_CONTENT_PER_GENERATION:
        if n_su2 == 2:
            total += n_color  # color triplet contributes 3, singlet contributes 1
    return total


def su2_doublet_count(generations: int = 1) -> int:
    """Return total SU(2)_L doublets for ``N`` generations."""

    return generations * su2_doublet_count_per_generation()


def witten_anomaly_free(generations: int = 1) -> bool:
    """Return whether the SU(2) global anomaly cancels for ``N`` generations.

    Cancellation condition: doublet count is even (≡ 0 mod 2).
    """

    return su2_doublet_count(generations) % 2 == 0


def mod2_lattice_anomaly(generations: int = 1) -> int:
    """Return the mod-2 anomaly count for the BCC walk's SU(2)_L sector.

    Returns 0 (anomaly-free) or 1 (anomalous).  Since the BCC walk's
    continuum limit is free Dirac and the SU(2)_L gauging is the same
    as the continuum SM, the mod-2 count coincides with Witten's.
    """

    return su2_doublet_count(generations) % 2


def witten_constrains_generation_count() -> bool:
    """Return whether Witten's anomaly forces a specific ``N``.

    The doublet count per generation is 4 (even), so 4·N ≡ 0 (mod 2)
    for every N ≥ 0.  Witten's anomaly imposes NO constraint on N.
    """

    return su2_doublet_count_per_generation() % 2 != 0


@dataclass(frozen=True)
class GlobalAnomalyCheckPayload:
    generations: int
    doublets_per_generation: int
    total_doublets: int
    witten_anomaly_free: bool
    mod2_lattice_anomaly: int
    witten_constrains_N: bool
    verdict: str
    interpretation: str


def global_anomaly_check_payload(generations: int = 1) -> GlobalAnomalyCheckPayload:
    per_gen = su2_doublet_count_per_generation()
    total = su2_doublet_count(generations)
    anomaly_free = witten_anomaly_free(generations)
    mod2 = mod2_lattice_anomaly(generations)
    constrains = witten_constrains_generation_count()

    if anomaly_free and not constrains:
        verdict = "GLOBAL ANOMALY OK — Witten satisfied for any N"
        interpretation = (
            f"Each SM generation contains {per_gen} SU(2)_L Weyl doublets "
            f"(3 from Q_L colors + 1 from L_L).  For N = {generations}, the "
            f"total doublet count is {total} (even).  Witten's global SU(2) "
            f"anomaly (π_4(SU(2)) = Z/2) is satisfied because 4N is even for "
            f"every N ≥ 0.  Therefore Witten's anomaly imposes NO constraint "
            f"on the number of generations.  Mod-2 lattice anomaly count = "
            f"{mod2}."
        )
    elif not anomaly_free:
        verdict = "WITTEN ANOMALY VIOLATION"
        interpretation = (
            f"Total SU(2)_L doublets for N = {generations}: {total} (odd).  "
            f"Witten's global anomaly is NOT cancelled — this would indicate "
            f"a structural problem with the chiral-16 content."
        )
    else:
        verdict = "WITTEN CONSTRAINS N"
        interpretation = (
            f"Per-generation doublet count is {per_gen} (odd).  Witten's "
            f"anomaly forces N to be even.  Unexpected for the standard "
            f"SM content — investigate."
        )

    return GlobalAnomalyCheckPayload(
        generations=generations,
        doublets_per_generation=per_gen,
        total_doublets=total,
        witten_anomaly_free=anomaly_free,
        mod2_lattice_anomaly=mod2,
        witten_constrains_N=constrains,
        verdict=verdict,
        interpretation=interpretation,
    )
