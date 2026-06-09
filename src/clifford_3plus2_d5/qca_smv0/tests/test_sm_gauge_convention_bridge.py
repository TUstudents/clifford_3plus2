"""Tests for QCA_SMv0 gauge-convention bridge audit."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import sm_yukawa_site_gauge_from_higgs_site_theta
from clifford_3plus2_d5.qca_smv0.sm_gauge_convention_bridge import (
    sm_gauge_convention_bridge_diagnostics,
    sm_gauge_convention_generator_residuals,
    sm_hypercharge_spectral_mismatch,
    sm_left_doublet_projector,
    sm_right_singlet_projector,
    sm_transport_electroweak_generators,
    sm_yukawa_door_electroweak_generators,
    sm_yukawa_door_site_gauge_from_generators,
    sm_yukawa_gauge_convention_energy_residuals,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import deterministic_higgs_site_theta


def test_yukawa_door_generators_reproduce_stage10_helper() -> None:
    theta = deterministic_higgs_site_theta((1, 1, 1), scale=0.17)

    helper = sm_yukawa_site_gauge_from_higgs_site_theta(theta)
    explicit = sm_yukawa_door_site_gauge_from_generators(theta)

    assert jnp.max(jnp.abs(helper - explicit)) < 1e-7


def test_left_doublets_match_and_right_singlets_are_charge_conjugate() -> None:
    left_residual, right_su2_residual, right_hypercharge_residual, full_difference = (
        sm_gauge_convention_generator_residuals()
    )

    assert left_residual < 1e-7
    assert right_su2_residual < 1e-7
    assert right_hypercharge_residual < 1e-7
    assert full_difference > 1.0


def test_hypercharge_spectrum_obstructs_unitary_similarity() -> None:
    assert sm_hypercharge_spectral_mismatch() > 0.1


def test_projectors_are_orthogonal_and_complete_on_internal_carrier_split() -> None:
    left = sm_left_doublet_projector()
    right = sm_right_singlet_projector()

    assert jnp.max(jnp.abs(left @ left - left)) < 1e-7
    assert jnp.max(jnp.abs(right @ right - right)) < 1e-7
    assert jnp.max(jnp.abs(left @ right)) < 1e-7
    assert jnp.max(jnp.abs(left + right - jnp.eye(left.shape[0], dtype=left.dtype))) < 1e-7


def test_transport_and_yukawa_generators_are_distinct_on_singlets() -> None:
    transport = sm_transport_electroweak_generators()
    yukawa = sm_yukawa_door_electroweak_generators()
    right = sm_right_singlet_projector()

    assert jnp.max(jnp.abs(right @ (transport[3] - yukawa[3]) @ right)) > 1.0
    assert jnp.max(jnp.abs(right @ (transport[3] + yukawa[3]) @ right)) < 1e-7


def test_yukawa_energy_is_covariant_only_in_yukawa_door_convention() -> None:
    physical_residual, transport_residual = sm_yukawa_gauge_convention_energy_residuals()

    assert physical_residual < 1e-7
    assert transport_residual > 1e-4


def test_gauge_convention_bridge_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_gauge_convention_bridge_diagnostics()

    assert diagnostics.generator_helper_residual < 1e-7
    assert diagnostics.left_doublet_generator_residual < 1e-7
    assert diagnostics.right_singlet_su2_residual < 1e-7
    assert diagnostics.right_hypercharge_conjugation_residual < 1e-7
    assert diagnostics.full_generator_difference_norm > 1.0
    assert diagnostics.hypercharge_spectral_mismatch > 0.1
    assert diagnostics.physical_yukawa_energy_covariance_residual < 1e-7
    assert diagnostics.transport_yukawa_energy_noninvariance_residual > 1e-4
    assert diagnostics.jit_delta_physical_covariance < 1e-8
    assert diagnostics.jit_delta_transport_noninvariance < 1e-8


def test_energy_residuals_are_jittable() -> None:
    expected = sm_yukawa_gauge_convention_energy_residuals()
    actual = jax.jit(sm_yukawa_gauge_convention_energy_residuals)()

    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert jnp.abs(actual_part - expected_part) < 1e-8
