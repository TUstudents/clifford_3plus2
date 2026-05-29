"""V28 vacuum-selector order-parameter gate.

V27 proves the BCC orbit quotient once one primitive tetrahedral exit is
selected.  V28 supplies a minimal order-parameter realization of that
selection:

    E_i = - h . v_i

where ``v_i`` are the four normalized tetrahedral BCC exits.  If
``h = v_selected``, the selected exit is the unique ground state, the gap is
``4/3``, and the energy stabilizer is the selected-exit ``S_3``.

This still does not derive why the physical vacuum develops such an order
parameter.  It converts the previous "select one exit" declaration into a
concrete rank-one boundary Hamiltonian with explicit controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_framing import (
    VacuumFramingAuditPayload,
    bcc_unoriented_exit_representatives,
    induced_residual_permutation,
    selected_exit_stabilizer_permutations,
    vacuum_framing_audit_payload,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3, vector_dot

REMAINING_DECLARED_INPUTS_AFTER_SELECTOR = (
    "physical_vacuum_order_parameter_exists",
    "unit_outward_causal_continuation_or_chain_normalization",
    "regular_boundary_fiber_or_max_entropy_prior",
)


def _validate_exit_index(selected: int, *, size: int = 4) -> None:
    if selected < 0 or selected >= size:
        raise ValueError(f"selected must be in [0, {size})")


def _zero_vector() -> Vector3:
    return sp.Integer(0), sp.Integer(0), sp.Integer(0)


def _energy_sort_key(value: sp.Expr) -> tuple[float, str]:
    simplified = sp.simplify(value)
    return float(sp.N(simplified)), str(simplified)


def _same_expr(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.simplify(left - right) == 0


def selector_order_parameter(selected: int = 0) -> Vector3:
    """Return the rank-one vacuum selector ``h = v_selected``."""

    exits = bcc_unoriented_exit_representatives()
    _validate_exit_index(selected, size=len(exits))
    return exits[selected]


def zero_order_parameter() -> Vector3:
    """Return the zero-field control order parameter."""

    return _zero_vector()


def midpoint_order_parameter(left: int = 0, right: int = 1) -> Vector3:
    """Return a two-exit midpoint control field.

    For ``left=0`` and ``right=1``, the two selected exits are degenerate
    ground states, so the field is rejected by the unique-selector gate.
    """

    exits = bcc_unoriented_exit_representatives()
    _validate_exit_index(left, size=len(exits))
    _validate_exit_index(right, size=len(exits))
    return tuple(
        sp.simplify(exits[left][index] + exits[right][index])
        for index in range(3)
    )


def generic_order_parameter() -> Vector3:
    """Return a generic field with a unique ground state but trivial stabilizer."""

    return sp.Integer(2), sp.Integer(3), sp.Integer(5)


def selector_energies(order_parameter: Vector3) -> tuple[sp.Expr, ...]:
    """Return exact selector energies ``E_i = -h . v_i``."""

    exits = bcc_unoriented_exit_representatives()
    return tuple(
        sp.simplify(-vector_dot(order_parameter, exit_vector))
        for exit_vector in exits
    )


def selector_hamiltonian(order_parameter: Vector3) -> sp.Matrix:
    """Return the diagonal selector Hamiltonian on four exit labels."""

    return sp.diag(*selector_energies(order_parameter))


def ground_exit_indices(order_parameter: Vector3) -> tuple[int, ...]:
    """Return all exit indices attaining the exact ground energy."""

    energies = selector_energies(order_parameter)
    min_energy = min(energies, key=_energy_sort_key)
    return tuple(
        index for index, energy in enumerate(energies) if _same_expr(energy, min_energy)
    )


def selector_gap(order_parameter: Vector3) -> sp.Expr:
    """Return the first excitation gap, or zero for a degenerate ground state."""

    energies = selector_energies(order_parameter)
    ground = ground_exit_indices(order_parameter)
    if len(ground) != 1:
        return sp.Integer(0)
    distinct = sorted(
        {sp.simplify(energy) for energy in energies},
        key=_energy_sort_key,
    )
    if len(distinct) < 2:
        return sp.Integer(0)
    return sp.simplify(distinct[1] - distinct[0])


def energy_stabilizer_permutations(order_parameter: Vector3) -> tuple[tuple[int, ...], ...]:
    """Return permutations of exit labels preserving the selector energies."""

    energies = selector_energies(order_parameter)
    stabilizer = []
    for perm in permutations(range(len(energies))):
        if all(_same_expr(energies[perm[index]], energies[index]) for index in range(4)):
            stabilizer.append(tuple(perm))
    return tuple(stabilizer)


def selector_stabilizer(selected: int = 0) -> tuple[tuple[int, ...], ...]:
    """Return the expected selected-exit stabilizer from the V27 orbit theorem."""

    return selected_exit_stabilizer_permutations(selected)


def selector_induces_residual_s3(selected: int = 0) -> bool:
    """Return whether the selected-exit stabilizer induces all residual ``S_3``."""

    induced = {
        induced_residual_permutation(perm, selected)
        for perm in selected_exit_stabilizer_permutations(selected)
    }
    all_s3 = {tuple(perm) for perm in permutations(range(3))}
    return induced == all_s3


def selector_energy_multiset(order_parameter: Vector3) -> tuple[sp.Expr, ...]:
    """Return the exact energy multiset in deterministic order."""

    return tuple(sorted(selector_energies(order_parameter), key=_energy_sort_key))


def all_selector_energy_multisets() -> tuple[tuple[sp.Expr, ...], ...]:
    """Return the energy multisets for the four tetrahedral selector fields."""

    return tuple(
        selector_energy_multiset(selector_order_parameter(selected))
        for selected in range(4)
    )


@dataclass(frozen=True)
class VacuumSelectorAuditPayload:
    """Verdict payload for the V28 vacuum-selector order-parameter gate."""

    final_verdict: str
    selected_ground_indices: tuple[int, ...]
    selected_gap: sp.Expr
    selector_stabilizer_size: int
    selector_induces_s3: bool
    all_four_selectors_conjugate: bool
    zero_control_rejected: bool
    midpoint_control_rejected: bool
    generic_control_rejected: bool
    v27_applies_after_selection: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def _v27_applies_after_selection(v27: VacuumFramingAuditPayload) -> bool:
    return (
        v27.final_verdict == "BCC_VACUUM_FRAMING_ORBIT_PASS"
        and v27.residual_exit_count == 3
        and v27.residual_adjacency_matches_k3
        and v27.stabilizer_induces_s3
    )


def vacuum_selector_audit_payload() -> VacuumSelectorAuditPayload:
    """Return the V28 vacuum-selector verdict."""

    selected = selector_order_parameter(0)
    selected_ground = ground_exit_indices(selected)
    selected_gap = selector_gap(selected)
    energy_stabilizer = energy_stabilizer_permutations(selected)
    expected_stabilizer = selector_stabilizer(0)
    selector_induces_s3 = selector_induces_residual_s3(0)

    multisets = all_selector_energy_multisets()
    all_four_selectors_conjugate = all(
        multiset == multisets[0] for multiset in multisets
    )

    zero_control_rejected = (
        ground_exit_indices(zero_order_parameter()) == (0, 1, 2, 3)
        and len(energy_stabilizer_permutations(zero_order_parameter())) == 24
    )
    midpoint_control_rejected = (
        ground_exit_indices(midpoint_order_parameter(0, 1)) == (0, 1)
        and selector_gap(midpoint_order_parameter(0, 1)) == 0
    )
    generic_control_rejected = (
        len(ground_exit_indices(generic_order_parameter())) == 1
        and len(energy_stabilizer_permutations(generic_order_parameter())) == 1
    )

    v27 = vacuum_framing_audit_payload()
    v27_applies = _v27_applies_after_selection(v27)
    checks_pass = (
        selector_energies(selected)
        == (-sp.Integer(1), sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 3))
        and selected_ground == (0,)
        and sp.simplify(selected_gap - sp.Rational(4, 3)) == 0
        and set(energy_stabilizer) == set(expected_stabilizer)
        and len(energy_stabilizer) == 6
        and selector_induces_s3
        and all_four_selectors_conjugate
        and zero_control_rejected
        and midpoint_control_rejected
        and generic_control_rejected
        and v27_applies
    )

    if checks_pass:
        final_verdict = "VACUUM_SELECTOR_ORDER_PARAMETER_PASS"
        interpretation = (
            "A rank-one selector order parameter h=v_i gives the selected "
            "tetrahedral BCC exit a unique ground energy, with gap 4/3 and "
            "selected-exit stabilizer S3. Zero, midpoint, and generic fields "
            "are rejected because they do not produce the combined unique "
            "selection plus residual-S3 structure. The existence of the "
            "physical vacuum order parameter remains a named input."
        )
    else:
        final_verdict = "VACUUM_SELECTOR_ORDER_PARAMETER_KILL"
        interpretation = (
            "The selector energy spectrum, stabilizer, conjugacy, controls, "
            "or V27 compatibility failed."
        )

    return VacuumSelectorAuditPayload(
        final_verdict=final_verdict,
        selected_ground_indices=selected_ground,
        selected_gap=selected_gap,
        selector_stabilizer_size=len(energy_stabilizer),
        selector_induces_s3=selector_induces_s3,
        all_four_selectors_conjugate=all_four_selectors_conjugate,
        zero_control_rejected=zero_control_rejected,
        midpoint_control_rejected=midpoint_control_rejected,
        generic_control_rejected=generic_control_rejected,
        v27_applies_after_selection=v27_applies,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_SELECTOR,
        interpretation=interpretation,
    )
