"""Tests for the QCA_SMv0 physical-right production variational audit."""

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_variational import (
    sm_physical_right_production_higgs_directional_derivative_residuals,
    sm_physical_right_production_higgs_force_decomposition_residual,
    sm_physical_right_production_link_directional_derivative_residual,
    sm_physical_right_production_link_force_decomposition_residual,
    sm_physical_right_production_variational_diagnostics,
)


def test_physical_right_production_force_decompositions_are_exact() -> None:
    state = sm_physical_right_production_initial_state()

    assert sm_physical_right_production_link_force_decomposition_residual(state) < 1e-8
    assert sm_physical_right_production_higgs_force_decomposition_residual(state) < 1e-8


def test_physical_right_production_directional_derivatives_match_forces() -> None:
    state = sm_physical_right_production_initial_state()
    higgs_real, higgs_imag = sm_physical_right_production_higgs_directional_derivative_residuals(state)

    assert sm_physical_right_production_link_directional_derivative_residual(state) < 1e-4
    assert higgs_real < 1e-4
    assert higgs_imag < 1e-4


def test_physical_right_production_variational_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_variational_diagnostics()

    assert diagnostics.link_force_decomposition_residual < 1e-8
    assert diagnostics.higgs_force_decomposition_residual < 1e-8
    assert diagnostics.link_directional_derivative_residual < 1e-4
    assert diagnostics.higgs_real_directional_derivative_residual < 1e-4
    assert diagnostics.higgs_imag_directional_derivative_residual < 1e-4
    assert diagnostics.vacuum_link_force_norm < 1e-6
    assert diagnostics.vacuum_higgs_force_norm < 1e-6
    assert diagnostics.deterministic_link_force_norm > 1e-2
    assert diagnostics.deterministic_higgs_force_norm > 1e-2
    assert bool(diagnostics.all_residuals_finite)
    assert jnp.isfinite(diagnostics.deterministic_link_force_norm)
