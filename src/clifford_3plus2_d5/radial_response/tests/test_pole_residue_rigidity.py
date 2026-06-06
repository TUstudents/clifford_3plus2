"""Tests for the R12 radial pole/residue rigidity no-go."""

import sympy as sp

from clifford_3plus2_d5.radial_response.pole_residue_rigidity import (
    bath_coupling_norm,
    bath_has_same_triality_head_data,
    bath_self_energy,
    one_level_triality_bath,
    pole_residue_rigidity_no_go_pass,
    pole_residue_rigidity_payload,
    self_energy_poles,
    self_energy_residues,
    two_level_tail_triality_bath,
    two_triality_head_amplitudes,
)
from clifford_3plus2_d5.radial_response.silver_transfer_inheritance import (
    radial_silver_transfer_inheritance_pass,
)
from clifford_3plus2_d5.radial_response.two_channel_isometry import total_probability


def test_triality_baths_have_same_normalized_head_data() -> None:
    one = one_level_triality_bath()
    tail = two_level_tail_triality_bath()
    assert bath_has_same_triality_head_data(one)
    assert bath_has_same_triality_head_data(tail)
    assert sp.simplify(bath_coupling_norm(one) - 1) == 0
    assert sp.simplify(bath_coupling_norm(tail) - 1) == 0
    assert sp.simplify(total_probability(two_triality_head_amplitudes()) - 1) == 0


def test_admissible_baths_have_different_self_energies() -> None:
    z = sp.Symbol("z")
    one_sigma = bath_self_energy(one_level_triality_bath(), z)
    tail_sigma = bath_self_energy(two_level_tail_triality_bath(), z)
    assert one_sigma == 1 / z
    assert tail_sigma == (z - 1) / (z**2 - z - 1)
    assert sp.simplify(one_sigma - tail_sigma) != 0


def test_admissible_baths_have_different_poles_and_residues() -> None:
    z = sp.Symbol("z")
    one_sigma = bath_self_energy(one_level_triality_bath(), z)
    tail_sigma = bath_self_energy(two_level_tail_triality_bath(), z)
    assert self_energy_poles(one_sigma, z) == (sp.Integer(0),)
    assert self_energy_residues(one_sigma, z) == (sp.Integer(1),)
    assert self_energy_poles(tail_sigma, z) == (
        sp.Rational(1, 2) - sp.sqrt(5) / 2,
        sp.Rational(1, 2) + sp.sqrt(5) / 2,
    )
    assert self_energy_residues(tail_sigma, z) == (
        (5 + sp.sqrt(5)) / 10,
        (5 - sp.sqrt(5)) / 10,
    )
    assert self_energy_poles(one_sigma, z) != self_energy_poles(tail_sigma, z)
    assert self_energy_residues(one_sigma, z) != self_energy_residues(tail_sigma, z)


def test_pole_residue_rigidity_payload_reports_no_go() -> None:
    payload = pole_residue_rigidity_payload()
    assert radial_silver_transfer_inheritance_pass()
    assert pole_residue_rigidity_no_go_pass()
    assert payload.final_verdict == "RADIAL_POLE_RESIDUE_RIGIDITY_NO_GO_PASS"
    assert payload.allowed_successors == ("triality_plus", "triality_minus")
    assert payload.head_amplitudes == (1 / sp.sqrt(2), 1 / sp.sqrt(2))
    assert payload.coupling_norms_match
    assert payload.inherited_transfer_unchanged
    assert payload.self_energies_differ
    assert payload.poles_differ
    assert payload.residues_differ
    assert payload.rigidity_forced is False
