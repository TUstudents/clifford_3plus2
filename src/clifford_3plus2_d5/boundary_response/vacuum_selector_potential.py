"""V31 tetrahedral vacuum-selector potential gate.

V28 realizes a selected BCC exit as a rank-one order parameter ``h = v_i``.
V31 adds a minimal tetrahedral selector potential whose symmetry-broken
candidate minima are exactly the four tetrahedral BCC exit directions.

The audit uses the tetrahedral cubic invariant

    C(h) = sum_i (h . v_i)^3

and the finite-candidate energy ``E(h) = -a C(h)``.  For positive anisotropy
``a``, the four ``v_i`` have energy ``-8a/9``.  Zero and midpoint controls have
energy zero, and antipodal controls have energy ``+8a/9``.  Reversing the sign
selects the antipodal branch, which is rejected for the V27/V28 selector
convention.

This still does not derive microscopic condensation.  It proves the narrower
gate: if the local vacuum sector admits this tetrahedral selector anisotropy
in its broken phase, the selected exit is one of four degenerate
symmetry-related BCC selectors, not an arbitrary label.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector import (
    energy_stabilizer_permutations,
    generic_order_parameter,
    midpoint_order_parameter,
    selector_energy_multiset,
    selector_induces_residual_s3,
    vacuum_selector_audit_payload,
    zero_order_parameter,
)
from clifford_3plus2_d5.boundary_response.vacuum_framing import (
    bcc_unoriented_exit_representatives,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3, vector_dot

REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_POTENTIAL = (
    "tetrahedral_selector_order_parameter_condenses",
)


def _negate_vector(vector: Vector3) -> Vector3:
    return tuple(sp.simplify(-component) for component in vector)  # type: ignore[return-value]


def _same_expr(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.simplify(left - right) == 0


def _energy_sort_key(value: sp.Expr) -> tuple[float, str]:
    simplified = sp.simplify(value)
    return float(sp.N(simplified)), str(simplified)


def tetrahedral_selector_candidates() -> tuple[Vector3, ...]:
    """Return the four BCC tetrahedral selector directions."""

    return bcc_unoriented_exit_representatives()


def tetrahedral_antipodal_controls() -> tuple[Vector3, ...]:
    """Return the four antipodal controls ``-v_i``."""

    return tuple(_negate_vector(candidate) for candidate in tetrahedral_selector_candidates())


def tetrahedral_midpoint_controls() -> tuple[Vector3, ...]:
    """Return representative two-selector midpoint controls."""

    return tuple(
        midpoint_order_parameter(left, right)
        for left, right in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    )


def tetrahedral_cubic_invariant(order_parameter: Vector3) -> sp.Expr:
    """Return ``sum_i (h . v_i)^3`` for the tetrahedral exits."""

    return sp.simplify(
        sum(
            vector_dot(order_parameter, candidate) ** 3
            for candidate in tetrahedral_selector_candidates()
        )
    )


def selector_potential_energy(
    order_parameter: Vector3,
    *,
    anisotropy: sp.Expr = sp.Integer(1),
) -> sp.Expr:
    """Return the finite-candidate selector potential energy ``-a C(h)``."""

    return sp.simplify(-sp.sympify(anisotropy) * tetrahedral_cubic_invariant(order_parameter))


def selector_candidate_energies(
    *,
    anisotropy: sp.Expr = sp.Integer(1),
) -> tuple[sp.Expr, ...]:
    """Return energies for the four accepted selector candidates."""

    return tuple(
        selector_potential_energy(candidate, anisotropy=anisotropy)
        for candidate in tetrahedral_selector_candidates()
    )


def antipodal_control_energies(
    *,
    anisotropy: sp.Expr = sp.Integer(1),
) -> tuple[sp.Expr, ...]:
    """Return energies for the four antipodal controls."""

    return tuple(
        selector_potential_energy(control, anisotropy=anisotropy)
        for control in tetrahedral_antipodal_controls()
    )


def selector_potential_minimizers(
    *,
    anisotropy: sp.Expr = sp.Integer(1),
) -> tuple[int, ...]:
    """Return selector indices minimizing the accepted selector candidates."""

    energies = selector_candidate_energies(anisotropy=anisotropy)
    minimum = min(energies, key=_energy_sort_key)
    return tuple(index for index, energy in enumerate(energies) if _same_expr(energy, minimum))


def wrong_sign_antipodal_minimizers() -> tuple[int, ...]:
    """Return antipodal indices selected when the anisotropy sign is reversed."""

    energies = antipodal_control_energies(anisotropy=-1)
    minimum = min(energies, key=_energy_sort_key)
    return tuple(index for index, energy in enumerate(energies) if _same_expr(energy, minimum))


def selector_potential_gap(
    *,
    anisotropy: sp.Expr = sp.Integer(1),
) -> sp.Expr:
    """Return the exact finite-candidate gap above selector minima.

    The comparison set contains zero, all pair midpoint controls, and all
    antipodal controls.  For positive anisotropy the gap is ``8/9``.
    """

    accepted_minimum = min(selector_candidate_energies(anisotropy=anisotropy), key=_energy_sort_key)
    controls = (
        (zero_order_parameter(),)
        + tetrahedral_midpoint_controls()
        + tetrahedral_antipodal_controls()
    )
    control_minimum = min(
        (selector_potential_energy(control, anisotropy=anisotropy) for control in controls),
        key=_energy_sort_key,
    )
    return sp.simplify(control_minimum - accepted_minimum)


def selector_candidates_reproduce_v28_spectra() -> bool:
    """Return true when every selector candidate reproduces V28 up to permutation."""

    expected = (
        -sp.Integer(1),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
        sp.Rational(1, 3),
    )
    return all(
        selector_energy_multiset(candidate) == expected
        for candidate in tetrahedral_selector_candidates()
    )


def selector_candidates_have_residual_s3_stabilizers() -> bool:
    """Return true when each selector candidate has selected-exit residual ``S_3``."""

    return all(
        len(energy_stabilizer_permutations(candidate)) == 6
        and selector_induces_residual_s3(index)
        for index, candidate in enumerate(tetrahedral_selector_candidates())
    )


def zero_anisotropy_control_rejected() -> bool:
    """Return true when zero anisotropy has no selector gap."""

    energies = selector_candidate_energies(anisotropy=0) + antipodal_control_energies(
        anisotropy=0
    )
    return all(energy == 0 for energy in energies) and selector_potential_gap(anisotropy=0) == 0


def wrong_sign_control_rejected() -> bool:
    """Return true when wrong-sign anisotropy selects the antipodal branch."""

    selector_min = min(selector_candidate_energies(anisotropy=-1), key=_energy_sort_key)
    antipodal_min = min(antipodal_control_energies(anisotropy=-1), key=_energy_sort_key)
    return (
        wrong_sign_antipodal_minimizers() == (0, 1, 2, 3)
        and sp.simplify(antipodal_min - selector_min) < 0
    )


def midpoint_control_rejected() -> bool:
    """Return true when midpoint controls are above selector minima."""

    selector_min = min(selector_candidate_energies(), key=_energy_sort_key)
    midpoint_min = min(
        (selector_potential_energy(control) for control in tetrahedral_midpoint_controls()),
        key=_energy_sort_key,
    )
    return sp.simplify(midpoint_min - selector_min) > 0


def generic_control_rejected() -> bool:
    """Return true when the generic V28 field has trivial selector stabilizer."""

    return len(energy_stabilizer_permutations(generic_order_parameter())) == 1


@dataclass(frozen=True)
class VacuumSelectorPotentialAuditPayload:
    """Verdict payload for the V31 tetrahedral selector potential gate."""

    final_verdict: str
    selector_candidate_count: int
    selector_energy: sp.Expr
    selector_minimizers: tuple[int, ...]
    selector_gap: sp.Expr
    selector_candidates_degenerate: bool
    selector_candidates_reproduce_v28: bool
    selector_stabilizers_induce_s3: bool
    zero_anisotropy_control_rejected: bool
    wrong_sign_control_rejected: bool
    midpoint_control_rejected: bool
    generic_control_rejected: bool
    v28_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def vacuum_selector_potential_audit_payload() -> VacuumSelectorPotentialAuditPayload:
    """Return the V31 tetrahedral selector-potential verdict."""

    candidates = tetrahedral_selector_candidates()
    energies = selector_candidate_energies()
    selector_energy = energies[0]
    degenerate = all(_same_expr(energy, selector_energy) for energy in energies)
    minimizers = selector_potential_minimizers()
    gap = selector_potential_gap()
    reproduces_v28 = selector_candidates_reproduce_v28_spectra()
    stabilizers_s3 = selector_candidates_have_residual_s3_stabilizers()
    zero_rejected = zero_anisotropy_control_rejected()
    wrong_sign_rejected = wrong_sign_control_rejected()
    midpoint_rejected = midpoint_control_rejected()
    generic_rejected = generic_control_rejected()
    v28 = vacuum_selector_audit_payload()
    v28_recovered = v28.final_verdict == "VACUUM_SELECTOR_ORDER_PARAMETER_PASS"

    checks_pass = (
        len(candidates) == 4
        and degenerate
        and selector_energy == -sp.Rational(8, 9)
        and minimizers == (0, 1, 2, 3)
        and sp.simplify(gap - sp.Rational(8, 9)) == 0
        and reproduces_v28
        and stabilizers_s3
        and zero_rejected
        and wrong_sign_rejected
        and midpoint_rejected
        and generic_rejected
        and v28_recovered
    )

    if checks_pass:
        final_verdict = "TETRAHEDRAL_SELECTOR_POTENTIAL_PASS"
        interpretation = (
            "The tetrahedral cubic selector anisotropy has four degenerate "
            "accepted finite-candidate minima at the BCC tetrahedral exits. "
            "Each minimum reproduces the V28 selector spectrum and residual "
            "S3 stabilizer. Zero-anisotropy, wrong-sign, midpoint, and generic "
            "controls fail. The remaining physical input is condensation of "
            "this tetrahedral selector order parameter."
        )
    else:
        final_verdict = "TETRAHEDRAL_SELECTOR_POTENTIAL_KILL"
        interpretation = (
            "The selector energies, finite-candidate minimizers, V28 recovery, "
            "stabilizers, gap, or controls failed."
        )

    return VacuumSelectorPotentialAuditPayload(
        final_verdict=final_verdict,
        selector_candidate_count=len(candidates),
        selector_energy=selector_energy,
        selector_minimizers=minimizers,
        selector_gap=gap,
        selector_candidates_degenerate=degenerate,
        selector_candidates_reproduce_v28=reproduces_v28,
        selector_stabilizers_induce_s3=stabilizers_s3,
        zero_anisotropy_control_rejected=zero_rejected,
        wrong_sign_control_rejected=wrong_sign_rejected,
        midpoint_control_rejected=midpoint_rejected,
        generic_control_rejected=generic_rejected,
        v28_recovered=v28_recovered,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_SELECTOR_POTENTIAL,
        interpretation=interpretation,
    )
