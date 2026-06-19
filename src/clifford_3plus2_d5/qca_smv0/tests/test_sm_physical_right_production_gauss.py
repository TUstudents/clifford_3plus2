"""Tests for QCA_SMv0 physical-right production Gauss monitor."""

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_production_tick import (
    sm_zero_family_lepton_yukawas,
    sm_zero_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_gauss,
    sm_physical_right_production_gauss_diagnostics,
    sm_physical_right_production_gauss_history,
    sm_physical_right_production_gauss_observables,
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)


def test_physical_right_production_vacuum_gauss_is_zero_and_stays_zero() -> None:
    vacuum = sm_physical_right_production_vacuum_state()
    final, history = sm_physical_right_production_gauss_history(vacuum, steps=2, step_size=0.001)

    assert jnp.linalg.norm(sm_physical_right_production_gauss(vacuum)) < 1e-8
    assert jnp.linalg.norm(sm_physical_right_production_gauss(final)) < 1e-8
    assert jnp.max(history["gauss_norm"]) < 1e-8
    assert history["family_norm"][-1] < 1e-8


def test_physical_right_production_gauss_history_records_finite_nonzero_deterministic_signal() -> None:
    initial = sm_physical_right_production_initial_state()
    final, history = sm_physical_right_production_gauss_history(initial, steps=2, step_size=0.001)
    observations = sm_physical_right_production_gauss_observables(final)

    assert history["gauss_norm"].shape == (3,)
    assert jnp.all(jnp.isfinite(history["gauss_norm"]))
    assert history["gauss_norm"][0] > 1e-1
    assert history["gauss_norm"][-1] > 1e-1
    assert jnp.linalg.norm(sm_physical_right_production_gauss(final) - sm_physical_right_production_gauss(initial)) > 1e-3
    assert observations["sm_link_unitarity_residual"] < 8e-7
    assert observations["higgs_link_unitarity_residual"] < 8e-7


def test_physical_right_production_gauss_resolves_default_vs_zero_yukawa_rollout() -> None:
    initial = sm_physical_right_production_initial_state()
    default_final, _ = sm_physical_right_production_gauss_history(initial, steps=3, step_size=0.001)
    zero_final, _ = sm_physical_right_production_gauss_history(
        initial,
        steps=3,
        step_size=0.001,
        quark_yukawas=sm_zero_quark_yukawas(),
        lepton_yukawas=sm_zero_family_lepton_yukawas(),
    )

    assert (
        jnp.linalg.norm(sm_physical_right_production_gauss(default_final) - sm_physical_right_production_gauss(zero_final))
        > 5e-7
    )


def test_physical_right_production_gauss_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_gauss_diagnostics()

    assert diagnostics.vacuum_initial_gauss_norm < 1e-8
    assert diagnostics.vacuum_final_gauss_norm < 1e-8
    assert diagnostics.vacuum_family_norm < 1e-8
    assert diagnostics.deterministic_initial_gauss_norm > 1e-1
    assert diagnostics.deterministic_final_gauss_norm > 1e-1
    assert diagnostics.deterministic_max_gauss_norm > 1e-1
    assert diagnostics.deterministic_gauss_delta_norm > 1e-3
    assert diagnostics.zero_yukawa_final_gauss_difference_norm > 5e-7
    assert diagnostics.rollout_family_norm_drift < 1e-5
    assert diagnostics.max_sm_link_unitarity_residual < 8e-7
    assert diagnostics.max_higgs_link_unitarity_residual < 8e-7
    assert diagnostics.history_count == 3
    assert bool(diagnostics.history_all_finite)
