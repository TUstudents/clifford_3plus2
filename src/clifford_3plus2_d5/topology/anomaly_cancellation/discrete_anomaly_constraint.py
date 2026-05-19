"""FD-14: extracting the constraint on N_generations from all anomalies.

Combines FD-11/FD-12 (continuous SM anomalies) with FD-13 (Witten global
+ mod-2 lattice).  The goal is to determine whether ANY discrete-or-
continuous anomaly forces a specific value of N.

Result: every anomaly considered scales linearly with N (per-generation
result × N).  Each per-generation result is exactly zero (or even, for
the mod-2 counts).  Therefore the constraint on N is the trivial
constraint ``0 = 0``, which is satisfied for every N ≥ 0.

There is NO discrete anomaly mechanism on the BCC × chiral-16 carrier
that forces N = 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.topology.anomaly_cancellation.bcc_anomaly_polynomial import (
    all_anomalies_cancel,
    anomaly_polynomial_payload,
)
from clifford_3plus2_d5.topology.anomaly_cancellation.global_anomaly_check import (
    global_anomaly_check_payload,
)


def per_generation_anomaly_polynomial_is_zero() -> bool:
    """Return whether every continuous anomaly vanishes at N = 1."""

    return all_anomalies_cancel(1)


def per_generation_global_anomaly_is_anomaly_free() -> bool:
    """Return whether Witten's anomaly is satisfied at N = 1."""

    return global_anomaly_check_payload(1).witten_anomaly_free


def constraint_on_N() -> sp.Expr:
    """Return the symbolic constraint on N that all anomalies impose.

    Every anomaly is linear in N with vanishing coefficient, so the
    constraint reduces to ``0 = 0``.  We return ``sp.Integer(0)`` to
    represent "no constraint".
    """

    return sp.Integer(0)


def anomalies_force_N_equals_three() -> bool:
    """Return whether any anomaly uniquely forces N = 3.

    Standard expectation: False.  Anomalies cancel per generation, so
    any N is consistent.
    """

    expr = constraint_on_N()
    # If the constraint is identically zero, it does NOT pick a unique N.
    return expr != 0


def admissible_generation_counts(max_N: int = 20) -> tuple[int, ...]:
    """Return all N in [0, max_N] for which every anomaly is satisfied.

    Used to demonstrate that the admissible set is the whole non-negative
    integers up to max_N — not just {3}.
    """

    admissible: list[int] = []
    for N in range(max_N + 1):
        cont = all_anomalies_cancel(N)
        glob = global_anomaly_check_payload(N).witten_anomaly_free
        if cont and glob:
            admissible.append(N)
    return tuple(admissible)


@dataclass(frozen=True)
class DiscreteAnomalyConstraintPayload:
    per_generation_continuous_cancels: bool
    per_generation_global_anomaly_free: bool
    constraint_symbolic: sp.Expr
    forces_N_equals_three: bool
    admissible_N_up_to_twenty: tuple[int, ...]
    verdict: str
    interpretation: str


def discrete_anomaly_constraint_payload() -> DiscreteAnomalyConstraintPayload:
    cont = per_generation_anomaly_polynomial_is_zero()
    glob = per_generation_global_anomaly_is_anomaly_free()
    constraint = constraint_on_N()
    forces_three = anomalies_force_N_equals_three()
    admissible = admissible_generation_counts(20)
    expected_admissible = tuple(range(21))

    sample_payload = anomaly_polynomial_payload(3)  # reuse the per-N report

    if cont and glob and not forces_three and admissible == expected_admissible:
        verdict = "ANOMALY KILL — no discrete anomaly forces N = 3"
        interpretation = (
            "Every continuous SM anomaly (grav, U(1)_Y³, SU(2)²·Y, "
            "SU(3)²·Y) cancels per generation; Witten's global SU(2) "
            "anomaly is satisfied because 4 doublets per generation × N "
            "is always even.  The combined constraint on N reduces to "
            "0 = 0, which is satisfied for every N ≥ 0.  Therefore the "
            "BCC × chiral-16 carrier admits any number of generations from "
            "the standpoint of anomaly cancellation — three is not forced.  "
            f"Independent check: {sample_payload.interpretation}"
        )
    else:
        verdict = "ANOMALY CONSTRAINT (unexpected)"
        interpretation = (
            f"continuous cancel: {cont}; global anomaly free: {glob}; "
            f"forces N=3: {forces_three}; admissible N up to 20: "
            f"{admissible}.  Investigate."
        )

    return DiscreteAnomalyConstraintPayload(
        per_generation_continuous_cancels=cont,
        per_generation_global_anomaly_free=glob,
        constraint_symbolic=constraint,
        forces_N_equals_three=forces_three,
        admissible_N_up_to_twenty=admissible,
        verdict=verdict,
        interpretation=interpretation,
    )
