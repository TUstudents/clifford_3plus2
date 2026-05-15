from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.spatial_1d import (
    alpha_eta_projectors,
    local_hopping_reconstructs_transfer,
    local_hopping_terms,
    local_qca_laurent_orthogonal,
    local_qca_symbol_at_root,
    local_qca_symbol_reconstructs_transfer,
    local_qca_symbol_unitary_on_samples,
    mode_windings_from_hopping,
    root_of_unity,
    spatial_1d_alpha_certificate,
    spatial_1d_combined_local_qca_layer,
    spatial_1d_combined_route_certificate,
    spatial_1d_local_hopping_certificate,
    spatial_1d_local_qca_certificate,
    spatial_1d_unseeded_search_summary,
    spatial_alpha_local_qca_layer,
    spatial_alpha_prototype,
    spatial_orientation_orbits,
    transfer_matrix_at_root,
    transfer_matrix_from_hopping_at_root,
)


ROOT = Path(__file__).resolve().parents[1]


def test_spatial_transfer_rule_builds_exact_root_of_unity_matrix() -> None:
    rule = spatial_alpha_prototype()
    transfer = transfer_matrix_at_root(rule, 1)
    zeta = root_of_unity(12, 1)

    assert transfer.shape == (10, 10)
    assert transfer[0, 0] == sp.simplify(zeta**4)
    assert transfer[3, 3] == sp.simplify(zeta**3)
    assert transfer[5, 5] == sp.simplify(zeta**4)
    assert transfer[8, 8] == sp.simplify(zeta**3)


def test_spatial_root_samples_are_unitary() -> None:
    rule = spatial_alpha_prototype()

    for sample in range(rule.period):
        transfer = transfer_matrix_at_root(rule, sample)
        assert sp.simplify(transfer.conjugate().T * transfer - identity(10)) == sp.zeros(10)


def test_spatial_local_hopping_terms_reconstruct_transfer() -> None:
    rule = spatial_alpha_prototype()
    terms = local_hopping_terms(rule)

    assert tuple(term.shift for term in terms) == (3, 4)
    assert mode_windings_from_hopping(rule) == (4, 4, 4, 3, 3)
    assert local_hopping_reconstructs_transfer(rule)
    for sample in range(rule.period):
        assert (
            transfer_matrix_from_hopping_at_root(
                terms,
                period=rule.period,
                sample=sample,
            )
            == transfer_matrix_at_root(rule, sample)
        )


def test_spatial_local_qca_layer_is_exact_laurent_unitary() -> None:
    rule = spatial_alpha_prototype()
    layer = spatial_alpha_local_qca_layer(rule)

    assert layer.name == "spatial_1d_alpha_projector_shift_qca"
    assert tuple(term.shift for term in layer.terms) == (3, 4)
    assert layer.locality_radius == 4
    assert local_qca_laurent_orthogonal(layer)
    assert local_qca_symbol_reconstructs_transfer(layer, rule)
    assert local_qca_symbol_unitary_on_samples(layer)
    for sample in range(rule.period):
        assert local_qca_symbol_at_root(layer, sample) == transfer_matrix_at_root(rule, sample)


def test_spatial_alpha_eta_rank_profile() -> None:
    rule = spatial_alpha_prototype()
    alpha_projector, eta_projector = alpha_eta_projectors(rule)

    assert alpha_projector.rank() == 6
    assert eta_projector.rank() == 4
    assert alpha_projector + eta_projector == identity(10)
    assert alpha_projector * eta_projector == sp.zeros(10)


def test_spatial_orientation_orbits_couple_block_signs() -> None:
    rule = spatial_alpha_prototype()
    orbits = spatial_orientation_orbits(rule)
    allowed = tuple(orbit for orbit in orbits if orbit.transport_allowed)

    assert len(orbits) == 4
    assert {(orbit.alpha_sign, orbit.eta_sign) for orbit in allowed} == {
        (1, 1),
        (-1, -1),
    }


def test_spatial_1d_alpha_certificate_reports_sign_coupling() -> None:
    certificate = spatial_1d_alpha_certificate()

    assert certificate.period == 12
    assert certificate.alpha_winding == 4
    assert certificate.eta_winding == 3
    assert certificate.winding_gcd == 1
    assert certificate.winding_lcm == 12
    assert certificate.locality_radius == 4
    assert certificate.transfer_unitary_on_samples
    assert certificate.coarse_6_4_band_split
    assert certificate.orientation_choices_before_transport == 4
    assert certificate.orientation_choices_after_transport == 2
    assert certificate.sign_coupled_to_global_pm
    assert certificate.strict_bridge_candidates == 0
    assert certificate.route_label == "spatial_signs_coupled_to_global_pm"
    assert not certificate.load_bearing_qca_bridge


def test_spatial_1d_local_hopping_certificate_reports_computed_sign_coupling() -> None:
    certificate = spatial_1d_local_hopping_certificate()

    assert certificate.hopping_term_count == 2
    assert certificate.hopping_shifts == (3, 4)
    assert certificate.hopping_locality_radius == 4
    assert certificate.mode_windings == (4, 4, 4, 3, 3)
    assert certificate.computed_alpha_winding == 4
    assert certificate.computed_eta_winding == 3
    assert certificate.computed_winding_gcd == 1
    assert certificate.computed_winding_lcm == 12
    assert certificate.reconstructs_transfer_on_samples
    assert certificate.transfer_unitary_on_samples
    assert certificate.coarse_6_4_band_split
    assert certificate.orientation_choices_before_transport == 4
    assert certificate.orientation_choices_after_transport == 2
    assert certificate.sign_coupled_to_global_pm
    assert certificate.route_label == "spatial_local_hopping_signs_coupled"
    assert not certificate.load_bearing_qca_bridge


def test_spatial_1d_local_qca_certificate_reports_exact_local_rule() -> None:
    certificate = spatial_1d_local_qca_certificate()

    assert certificate.layer_name == "spatial_1d_alpha_projector_shift_qca"
    assert certificate.qca_term_count == 2
    assert certificate.qca_shifts == (3, 4)
    assert certificate.qca_locality_radius == 4
    assert certificate.finite_radius
    assert certificate.coefficient_matrices_real
    assert certificate.laurent_orthogonal
    assert certificate.symbol_reconstructs_transfer_on_samples
    assert certificate.symbol_unitary_on_samples
    assert certificate.coefficient_algebra_dimension == 2
    assert certificate.coefficient_center_dimension == 2
    assert certificate.central_idempotent_ranks == (0, 4, 6, 10)
    assert certificate.lower_rank_central_idempotents == 0
    assert certificate.coarse_6_4_center_pair
    assert certificate.mode_windings == (4, 4, 4, 3, 3)
    assert certificate.computed_alpha_winding == 4
    assert certificate.computed_eta_winding == 3
    assert certificate.computed_winding_gcd == 1
    assert certificate.computed_winding_lcm == 12
    assert certificate.orientation_choices_after_transport == 2
    assert certificate.sign_coupled_to_global_pm
    assert certificate.strict_bridge_candidates == 0
    assert certificate.route_label == "spatial_local_qca_signs_coupled_not_load_bearing"
    assert not certificate.load_bearing_qca_bridge


def test_spatial_1d_combined_route_couples_signs_without_generating_j() -> None:
    layer = spatial_1d_combined_local_qca_layer()
    certificate = spatial_1d_combined_route_certificate()

    assert layer.name == "spatial_1d_route1_route2_combined_qca"
    assert tuple(term.shift for term in layer.terms) == (3, 4)
    assert local_qca_laurent_orthogonal(layer)
    assert local_qca_symbol_unitary_on_samples(layer)
    assert certificate.qca_shifts == (3, 4)
    assert certificate.finite_radius
    assert certificate.coefficient_matrices_real
    assert certificate.laurent_orthogonal
    assert certificate.symbol_unitary_on_samples
    assert certificate.onsite_generated_algebra_dimension == 22
    assert certificate.onsite_center_dimension == 3
    assert certificate.onsite_central_idempotent_ranks == (0, 4, 6, 10)
    assert certificate.onsite_lower_rank_central_idempotents == 0
    assert certificate.onsite_compatible_j_count == 4
    assert certificate.transported_compatible_j_count == 2
    assert certificate.transported_j_commute_on_samples
    assert certificate.sign_coupled_to_global_pm
    assert certificate.coefficient_algebra_dimension == 8
    assert certificate.coefficient_algebra_generates_alpha_eta_projectors
    assert certificate.joint_rule_algebra_dimension == 22
    assert certificate.joint_rule_generated_transported_j_count == 0
    assert certificate.topological_pm_shape_candidate
    assert not certificate.strict_bridge_candidate
    assert certificate.route_label == "combined_route_signs_coupled_but_j_not_rule_generated"
    assert not certificate.load_bearing_qca_bridge


def test_spatial_1d_unseeded_search_rejects_seeded_projector_shift() -> None:
    summary = spatial_1d_unseeded_search_summary()

    assert summary.candidate_count == 4
    assert summary.unseeded_candidate_count == 3
    assert summary.seeded_guardrail_rejections == 1
    assert summary.laurent_orthogonal_candidates == 4
    assert summary.unseeded_coarse_6_4_center_candidates == 0
    assert summary.unseeded_sign_coupled_candidates == 0
    assert summary.unseeded_strict_bridge_candidates == 0
    assert summary.lower_rank_center_rejections == 1
    assert summary.route_label == "unseeded_spatial_no_bridge_candidates"
    assert not summary.load_bearing_qca_bridge

    by_name = {candidate.candidate_name: candidate for candidate in summary.candidates}
    assert by_name["unseeded_uniform_identity_shift"].central_idempotent_ranks == (0, 10)
    assert by_name["unseeded_uniform_clock_shift"].central_idempotent_ranks == (0, 10)
    assert by_name["unseeded_mode_5_cycle_shift"].lower_rank_central_idempotents == 2
    seeded = by_name["spatial_1d_alpha_projector_shift_qca"]
    assert not seeded.seeded_coefficient_guardrail_passed
    assert seeded.seeded_coefficient_witnesses == ("shift_3:P_eta", "shift_4:P_alpha")
    assert seeded.coarse_6_4_center_pair
    assert seeded.sign_coupled_to_global_pm
    assert seeded.route_label == "unseeded_spatial_seeded_coefficient_rejected"


def test_spatial_1d_alpha_cli_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/spatial_1d_alpha_search.py",
            "--json",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["family"] == "spatial_1d_alpha"
    assert payload["candidate_count"] == 1
    assert payload["unitary_candidates"] == 1
    assert payload["coarse_6_4_band_candidates"] == 1
    assert payload["orientation_choices_before_transport"] == 4
    assert payload["orientation_choices_after_transport"] == 2
    assert payload["sign_coupled_candidates"] == 1
    assert payload["strict_bridge_candidates"] == 0
    assert payload["route_label"] == "spatial_signs_coupled_to_global_pm"
    assert payload["load_bearing_qca_bridge"] is False
    assert payload["local_hopping"]["hopping_shifts"] == [3, 4]
    assert payload["local_hopping"]["mode_windings"] == [4, 4, 4, 3, 3]
    assert payload["local_hopping"]["computed_alpha_winding"] == 4
    assert payload["local_hopping"]["computed_eta_winding"] == 3
    assert payload["local_hopping"]["reconstructs_transfer_on_samples"] is True
    assert payload["local_hopping"]["orientation_choices_after_transport"] == 2
    assert payload["local_hopping"]["route_label"] == "spatial_local_hopping_signs_coupled"
    assert payload["local_qca"]["qca_shifts"] == [3, 4]
    assert payload["local_qca"]["finite_radius"] is True
    assert payload["local_qca"]["coefficient_matrices_real"] is True
    assert payload["local_qca"]["laurent_orthogonal"] is True
    assert payload["local_qca"]["symbol_reconstructs_transfer_on_samples"] is True
    assert payload["local_qca"]["symbol_unitary_on_samples"] is True
    assert payload["local_qca"]["coefficient_algebra_dimension"] == 2
    assert payload["local_qca"]["coefficient_center_dimension"] == 2
    assert payload["local_qca"]["central_idempotent_ranks"] == [0, 4, 6, 10]
    assert payload["local_qca"]["lower_rank_central_idempotents"] == 0
    assert payload["local_qca"]["coarse_6_4_center_pair"] is True
    assert payload["local_qca"]["mode_windings"] == [4, 4, 4, 3, 3]
    assert payload["local_qca"]["orientation_choices_after_transport"] == 2
    assert payload["local_qca"]["sign_coupled_to_global_pm"] is True
    assert payload["local_qca"]["strict_bridge_candidates"] == 0
    assert (
        payload["local_qca"]["route_label"]
        == "spatial_local_qca_signs_coupled_not_load_bearing"
    )
    assert payload["local_qca"]["load_bearing_qca_bridge"] is False


def test_spatial_1d_combined_cli_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/spatial_1d_alpha_search.py",
            "--variant",
            "combined",
            "--json",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["family"] == "spatial_1d_route1_route2_combined"
    assert payload["qca_shifts"] == [3, 4]
    assert payload["laurent_orthogonal"] is True
    assert payload["symbol_unitary_on_samples"] is True
    assert payload["onsite_central_idempotent_ranks"] == [0, 4, 6, 10]
    assert payload["onsite_compatible_j_count"] == 4
    assert payload["transported_compatible_j_count"] == 2
    assert payload["transported_j_commute_on_samples"] is True
    assert payload["coefficient_algebra_generates_alpha_eta_projectors"] is True
    assert payload["joint_rule_generated_transported_j_count"] == 0
    assert payload["topological_pm_shape_candidate"] is True
    assert payload["strict_bridge_candidate"] is False
    assert payload["route_label"] == "combined_route_signs_coupled_but_j_not_rule_generated"
    assert payload["load_bearing_qca_bridge"] is False


def test_spatial_1d_unseeded_cli_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/spatial_1d_unseeded_search.py",
            "--json",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["family"] == "spatial_1d_unseeded"
    assert payload["candidate_count"] == 4
    assert payload["unseeded_candidate_count"] == 3
    assert payload["seeded_guardrail_rejections"] == 1
    assert payload["unseeded_coarse_6_4_center_candidates"] == 0
    assert payload["unseeded_sign_coupled_candidates"] == 0
    assert payload["unseeded_strict_bridge_candidates"] == 0
    assert payload["lower_rank_center_rejections"] == 1
    assert payload["route_label"] == "unseeded_spatial_no_bridge_candidates"
    assert payload["load_bearing_qca_bridge"] is False

    by_name = {candidate["candidate_name"]: candidate for candidate in payload["candidates"]}
    seeded = by_name["spatial_1d_alpha_projector_shift_qca"]
    assert seeded["seeded_coefficient_guardrail_passed"] is False
    assert seeded["seeded_coefficient_witnesses"] == ["shift_3:P_eta", "shift_4:P_alpha"]
    assert seeded["central_idempotent_ranks"] == [0, 4, 6, 10]
    assert seeded["sign_coupled_to_global_pm"] is True
