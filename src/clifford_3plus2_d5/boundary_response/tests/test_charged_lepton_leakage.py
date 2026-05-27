"""Tests for the V7 charged-lepton leakage gate."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    charged_lepton_leakage_audit_payload,
    charged_lepton_leakage_depth_amplitude,
    charged_lepton_rotation_sine,
    charged_lepton_rotation_sine_for_depth,
    charged_lepton_rotation_sine_squared,
    selected_charged_lepton_port,
    selected_port_residual_components,
    synthetic_b_leakage_response,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    residual_basis_matrix,
    standard_basis,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon, epsilon_fourth
from clifford_3plus2_d5.boundary_response.weyl_sterile import weyl_sterile_audit_payload


def test_selected_charged_lepton_port_is_first_residual_port() -> None:
    assert selected_charged_lepton_port() == standard_basis()[0]


def test_selected_port_decomposes_in_residual_basis() -> None:
    components = selected_port_residual_components()
    assert sp.simplify(components["a"] - sp.sqrt(sp.Rational(2, 3))) == 0
    assert sp.simplify(components["u"] - 1 / sp.sqrt(3)) == 0
    assert components["b"] == 0


def test_two_step_leakage_amplitude_is_epsilon_squared() -> None:
    assert sp.simplify(charged_lepton_leakage_depth_amplitude(2) - epsilon() ** 2) == 0


def test_charged_lepton_rotation_sine_matches_l1_prediction() -> None:
    expected = sp.sqrt(sp.Rational(3, 2)) * epsilon() ** 2
    assert sp.simplify(charged_lepton_rotation_sine() - expected) == 0


def test_charged_lepton_rotation_sine_squared_matches_l1_prediction() -> None:
    expected = sp.Rational(3, 2) * epsilon_fourth()
    assert sp.simplify(charged_lepton_rotation_sine_squared() - expected) == 0


def test_wrong_depth_controls_do_not_match_l1_prediction() -> None:
    expected = charged_lepton_rotation_sine()
    assert sp.simplify(charged_lepton_rotation_sine_for_depth(1) - expected) != 0
    assert sp.simplify(charged_lepton_rotation_sine_for_depth(3) - expected) != 0


def test_negative_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        charged_lepton_leakage_depth_amplitude(-1)


def test_synthetic_b_leakage_control_is_detectable() -> None:
    synthetic = synthetic_b_leakage_response()
    basis = residual_basis_matrix(("a", "u", "b"))
    components = (basis.T * synthetic).applyfunc(sp.simplify)
    assert sp.simplify(components[2, 0]) != 0


def test_charged_lepton_leakage_audit_reports_pass_but_parks_textures() -> None:
    payload = charged_lepton_leakage_audit_payload()
    assert payload.final_verdict == "CHARGED_LEPTON_LEAKAGE_PASS"
    assert payload.component_b == 0
    assert sp.simplify(payload.rotation_sine - sp.sqrt(sp.Rational(3, 2)) * epsilon() ** 2) == 0
    assert not payload.depth_one_control_matches
    assert not payload.depth_three_control_matches
    assert payload.b_leakage_control_detected
    assert payload.pmns_ckm_parked


def test_v6_limit_pass_regression() -> None:
    assert weyl_sterile_audit_payload().final_verdict == "PRODUCT_STERILE_LIMIT_PASS"
