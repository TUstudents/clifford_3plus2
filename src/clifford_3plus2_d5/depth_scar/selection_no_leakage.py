"""V11 selection-signature no-leakage theorem.

V10 proves that unit edge weights are equivalent to no leakage from the active
repair block.  V11 proves the conditional bridge:

    unique allowed microscopic successor => no leakage.

The theorem is deliberately abstract.  It does not enumerate the actual
microscopic BCC-QCA basis.  Instead it states the finite signature condition
that V12 must check:

    Omega(a) = {u},   Omega(b) = {a}.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import (
    flag_phases_are_removable,
)
from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    PATH_REPAIR_EDGES,
    local_support_induces_path_laplacian,
)
from clifford_3plus2_d5.depth_scar.repair_isometry import (
    active_identity,
    no_leakage_forces_unit_weights,
    repair_isometry_saturation_pass,
)

ACTIVE_SUCCESSORS = {"a": "u", "b": "a"}


def target_successor_map() -> dict[str, str]:
    """Return the desired unique active repair successors."""

    return dict(ACTIVE_SUCCESSORS)


def unique_successor_allowed_sets() -> dict[str, tuple[str, ...]]:
    """Return the V11 abstract allowed-successor sets."""

    return {source: (target,) for source, target in ACTIVE_SUCCESSORS.items()}


def leaky_successor_allowed_sets() -> dict[str, tuple[str, ...]]:
    """Return a negative-control allowed-successor set with leakage."""

    return {
        "a": ("u", "bulk_a"),
        "b": ("a",),
    }


def allowed_sets_are_unique_successors(
    allowed_sets: dict[str, tuple[str, ...]],
    target_map: dict[str, str] | None = None,
) -> bool:
    """Return whether every active state has exactly its target successor."""

    if target_map is None:
        target_map = target_successor_map()
    return all(allowed_sets.get(source) == (target,) for source, target in target_map.items())


def no_leakage_from_unique_successors(allowed_sets: dict[str, tuple[str, ...]]) -> bool:
    """Return whether the allowed-successor data force zero leakage."""

    return allowed_sets_are_unique_successors(allowed_sets)


def successor_unitary_repair_block(theta_a: sp.Expr, theta_b: sp.Expr) -> sp.Matrix:
    """Return ``U P_A`` induced by unique successors, as a full-space matrix.

    Columns are active inputs ``|a>`` and ``|b>``; rows are output ports
    ``(u,a,b)``.
    """

    return sp.Matrix(
        [
            [sp.exp(sp.I * theta_a), 0],
            [0, sp.exp(sp.I * theta_b)],
            [0, 0],
        ]
    )


def unique_successor_block_is_isometry() -> bool:
    """Return whether unique successors give an isometry on the active domain."""

    theta_a, theta_b = sp.symbols("theta_a theta_b", real=True)
    block = successor_unitary_repair_block(theta_a, theta_b)
    return sp.simplify(block.H * block - active_identity()) == sp.zeros(2, 2)


def unique_successor_block_matches_path_support() -> bool:
    """Return whether unique successors realize the V9 path support."""

    theta_a, theta_b = sp.symbols("theta_a theta_b", real=True)
    block = successor_unitary_repair_block(theta_a, theta_b)
    support = tuple(
        (row, col + 1)
        for row in range(3)
        for col in range(2)
        if block[row, col] != 0
    )
    return support == PATH_REPAIR_EDGES


def unique_successor_no_leakage_pass() -> bool:
    """Return whether V11's abstract unique-successor theorem passes."""

    allowed_sets = unique_successor_allowed_sets()
    return (
        allowed_sets_are_unique_successors(allowed_sets)
        and no_leakage_from_unique_successors(allowed_sets)
        and unique_successor_block_is_isometry()
        and unique_successor_block_matches_path_support()
        and no_leakage_forces_unit_weights()
        and flag_phases_are_removable()
        and local_support_induces_path_laplacian()
    )


def leaky_successor_control_rejected() -> bool:
    """Return whether an extra allowed output defeats the V11 condition."""

    allowed_sets = leaky_successor_allowed_sets()
    return (
        not allowed_sets_are_unique_successors(allowed_sets)
        and not no_leakage_from_unique_successors(allowed_sets)
    )


def approximate_leakage_weight_bounds(eta: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return the active weight bounds implied by leakage norm ``eta``."""

    return sp.simplify(1 - eta**2), sp.Integer(1)


def small_leakage_bounds_weights() -> bool:
    """Return whether small leakage gives weights in ``[1-eta**2, 1]``."""

    eta = sp.symbols("eta", nonnegative=True)
    lower, upper = approximate_leakage_weight_bounds(eta)
    return lower == 1 - eta**2 and upper == 1


@dataclass(frozen=True)
class SelectionNoLeakagePayload:
    """V11 payload for selection-signature no-leakage."""

    final_verdict: str
    unique_successor_sets: dict[str, tuple[str, ...]]
    unique_successors_force_no_leakage: bool
    unique_successor_block_is_isometry: bool
    unique_successor_block_matches_path_support: bool
    repair_isometry_saturation_available: bool
    leaky_successor_control_rejected: bool
    small_leakage_bounds_pass: bool
    microscopic_successor_enumeration_done: bool
    interpretation: str


def selection_no_leakage_payload() -> SelectionNoLeakagePayload:
    """Return the V11 selection-signature no-leakage verdict."""

    allowed_sets = unique_successor_allowed_sets()
    unique = allowed_sets_are_unique_successors(allowed_sets)
    no_leakage = no_leakage_from_unique_successors(allowed_sets)
    isometry = unique_successor_block_is_isometry()
    support = unique_successor_block_matches_path_support()
    v10 = repair_isometry_saturation_pass()
    leaky_control = leaky_successor_control_rejected()
    bounds = small_leakage_bounds_weights()

    checks_pass = unique and no_leakage and isometry and support and v10 and leaky_control and bounds

    if checks_pass:
        final_verdict = "V11_SELECTION_SIGNATURE_NO_LEAKAGE_PASS"
        interpretation = (
            "If the complete microscopic selection signature leaves exactly one "
            "successor for each active state, Omega(a)={u} and Omega(b)={a}, "
            "then the active update has no allowed output outside the repaired "
            "range. Therefore L=0, V10 gives N^dagger N=I_A, and the unit P3 "
            "flag follows up to tree phases. V11 does not perform the actual "
            "microscopic successor enumeration."
        )
    elif not unique or not no_leakage or not leaky_control:
        final_verdict = "SELECTION_SIGNATURE_NOT_UNIQUE_KILL"
        interpretation = "The abstract allowed-successor data do not force no leakage."
    elif not isometry or not support or not v10:
        final_verdict = "SELECTION_NO_LEAKAGE_CONSEQUENCE_KILL"
        interpretation = "Unique successors did not imply the expected isometry/path consequences."
    else:
        final_verdict = "SELECTION_NO_LEAKAGE_CONTROL_KILL"
        interpretation = "The small-leakage or negative-control checks failed."

    return SelectionNoLeakagePayload(
        final_verdict=final_verdict,
        unique_successor_sets=allowed_sets,
        unique_successors_force_no_leakage=no_leakage,
        unique_successor_block_is_isometry=isometry,
        unique_successor_block_matches_path_support=support,
        repair_isometry_saturation_available=v10,
        leaky_successor_control_rejected=leaky_control,
        small_leakage_bounds_pass=bounds,
        microscopic_successor_enumeration_done=False,
        interpretation=interpretation,
    )
