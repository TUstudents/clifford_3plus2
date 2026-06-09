"""Tests for QCA_SMv0 antiunitary singlet bridge."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import (
    sm_antiunitary_bridge_diagnostics,
    sm_antiunitary_bridge_gauge_from_transport,
    sm_antiunitary_bridge_generator_residuals,
    sm_full_bridge_yukawa_energy_residuals,
    sm_physical_right_generators,
    sm_physical_right_gauge_unitarity_residual,
    sm_physical_right_site_gauge_from_algebra,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import deterministic_sm_site_theta, sm_generators, sm_site_gauge_from_algebra
from clifford_3plus2_d5.qca_smv0.sm_gauge_convention_bridge import (
    sm_left_doublet_projector,
    sm_right_singlet_projector,
    sm_yukawa_door_electroweak_generators,
)


def test_physical_right_generators_are_left_linear_and_right_antilinear() -> None:
    left_residual, right_residual, ew_residual, difference_norm = sm_antiunitary_bridge_generator_residuals()

    assert left_residual < 1e-7
    assert right_residual < 1e-7
    assert ew_residual < 1e-7
    assert difference_norm > 1.0


def test_physical_right_electroweak_slice_equals_yukawa_door_generators() -> None:
    assert jnp.max(jnp.abs(sm_physical_right_generators()[8:12] - sm_yukawa_door_electroweak_generators())) < 1e-7


def test_bridge_projector_formula_matches_generators() -> None:
    transport = sm_generators()
    physical = sm_physical_right_generators()
    left = sm_left_doublet_projector()
    right = sm_right_singlet_projector()
    expected = jnp.stack([left @ generator @ left + right @ jnp.conj(generator) @ right for generator in transport])

    assert jnp.max(jnp.abs(physical - expected)) < 1e-7


def test_finite_bridge_matches_exponentiated_physical_right_gauge() -> None:
    site_theta = deterministic_sm_site_theta((1, 1, 1), scale=0.17)
    transport_gauge = sm_site_gauge_from_algebra(site_theta)
    physical_gauge = sm_physical_right_site_gauge_from_algebra(site_theta)
    bridge_gauge = sm_antiunitary_bridge_gauge_from_transport(transport_gauge)

    assert jnp.max(jnp.abs(physical_gauge - bridge_gauge)) < 2e-7
    assert sm_physical_right_gauge_unitarity_residual(physical_gauge) < 5e-7


def test_full_physical_right_gauge_restores_yukawa_energy_covariance() -> None:
    physical_residual, transport_residual = sm_full_bridge_yukawa_energy_residuals()

    assert physical_residual < 1e-7
    assert transport_residual > 1e-4


def test_antiunitary_bridge_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_antiunitary_bridge_diagnostics()

    assert diagnostics.left_linear_generator_residual < 1e-7
    assert diagnostics.right_antilinear_generator_residual < 1e-7
    assert diagnostics.electroweak_yukawa_slice_residual < 1e-7
    assert diagnostics.finite_bridge_residual < 2e-7
    assert diagnostics.finite_bridge_unitarity_residual < 5e-7
    assert diagnostics.transport_physical_generator_difference_norm > 1.0
    assert diagnostics.full_physical_yukawa_energy_covariance_residual < 1e-7
    assert diagnostics.full_transport_yukawa_energy_noninvariance_residual > 1e-4
    assert diagnostics.jit_delta_physical_covariance < 1e-8
    assert diagnostics.jit_delta_transport_noninvariance < 1e-8


def test_full_bridge_energy_residuals_are_jittable() -> None:
    expected = sm_full_bridge_yukawa_energy_residuals()
    actual = jax.jit(sm_full_bridge_yukawa_energy_residuals)()

    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert jnp.abs(actual_part - expected_part) < 1e-8
