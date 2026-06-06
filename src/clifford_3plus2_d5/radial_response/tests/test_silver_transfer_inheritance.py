"""Tests for the R11 silver-transfer inheritance gate."""

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    residual_graph_decaying_factor,
)
from clifford_3plus2_d5.radial_response.silver_transfer_inheritance import (
    INDEPENDENT_ETA_CONTROL,
    candidate_eta_is_inherited,
    candidate_transfer_root_is_inherited,
    independent_eta_control_rejected,
    inherited_eta,
    inherited_intensity_ratio,
    inherited_transfer_root,
    minimal_unitary_value_forcing_rejected,
    radial_silver_transfer_inheritance_pass,
    residual_graph_source_matches,
    second_transfer_root_controls_rejected,
    silver_transfer_inheritance_payload,
    weyl_chain_source_matches,
)


def test_inherited_silver_root_and_powers_are_exact() -> None:
    root = inherited_transfer_root()
    eta = inherited_eta()
    intensity = inherited_intensity_ratio()
    assert sp.simplify(root - (sp.sqrt(2) - 1)) == 0
    assert sp.simplify(eta - (3 - 2 * sp.sqrt(2))) == 0
    assert sp.simplify(intensity - (17 - 12 * sp.sqrt(2))) == 0
    assert sp.simplify(eta - root**2) == 0
    assert sp.simplify(intensity - root**4) == 0


def test_boundary_and_flavor_transfer_sources_match() -> None:
    assert residual_graph_source_matches()
    assert weyl_chain_source_matches()
    assert candidate_transfer_root_is_inherited(residual_graph_decaying_factor(3))


def test_independent_eta_and_alternate_roots_are_rejected() -> None:
    assert not candidate_eta_is_inherited(INDEPENDENT_ETA_CONTROL)
    assert independent_eta_control_rejected()
    assert not candidate_transfer_root_is_inherited(residual_graph_decaying_factor(2))
    assert not candidate_transfer_root_is_inherited(residual_graph_decaying_factor(4))
    assert second_transfer_root_controls_rejected()


def test_minimal_unitary_toy_is_not_value_forcing() -> None:
    assert minimal_unitary_value_forcing_rejected()


def test_silver_transfer_inheritance_payload_passes() -> None:
    payload = silver_transfer_inheritance_payload()
    assert radial_silver_transfer_inheritance_pass()
    assert payload.final_verdict == "RADIAL_SILVER_TRANSFER_INHERITANCE_PASS"
    assert payload.eta_matches_root_square
    assert payload.intensity_matches_root_fourth
    assert payload.residual_graph_source_matches
    assert payload.weyl_chain_source_matches
    assert payload.flavor_shared_transfer_passes
    assert payload.quark_transfer_schur_passes
    assert payload.independent_eta_control_rejected
    assert payload.k2_root_control_rejected
    assert payload.k4_root_control_rejected
    assert payload.duplicate_local_derivation_used is False
    assert payload.minimal_unitary_value_forcing_rejected
    assert "boundary_response.transfer" in payload.source_modules
    assert "flavor_a_track.u1_shared_transfer" in payload.source_modules
