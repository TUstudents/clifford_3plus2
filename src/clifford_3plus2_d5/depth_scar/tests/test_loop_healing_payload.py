"""Aggregate payload tests for V4 loop healing and CP deformation."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.loop_healing import loop_healing_cp_payload


def test_loop_healing_cp_payload_passes() -> None:
    delta = sp.symbols("delta")
    payload = loop_healing_cp_payload()

    assert payload.final_verdict == "LOOP_HEALING_CP_DEFORMATION_PASS"
    assert payload.real_spectrum_formula == (0, 1 + 2 * delta, 3)
    assert payload.doubled_spectrum_formula == (0, 2 + 4 * delta, 6)
    assert payload.real_spectrum_formula_pass
    assert payload.doubled_spectrum_formula_pass
    assert payload.magnetic_laplacian_hermitian
    assert payload.path_phases_removable
    assert payload.path_cycle_rank == 0
    assert payload.healed_triangle_cycle_rank == 1
    assert payload.loop_phase_gauge_invariant
    assert payload.phi_zero_control
    assert payload.path_limit_control
    assert payload.conjugation_flips_phase
    assert payload.microscopic_delta_phi_derived is False
