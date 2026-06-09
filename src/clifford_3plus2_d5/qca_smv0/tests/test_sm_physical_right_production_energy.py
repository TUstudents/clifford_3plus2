"""Tests for QCA_SMv0 physical-right production energy monitor."""

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_production_tick import (
    sm_zero_family_lepton_yukawas,
    sm_zero_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_energy import (
    sm_physical_right_production_energy_diagnostics,
    sm_physical_right_production_energy_history,
    sm_physical_right_production_energy_observables,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)


def test_physical_right_production_vacuum_energy_stays_zero() -> None:
    vacuum = sm_physical_right_production_vacuum_state()
    _, history = sm_physical_right_production_energy_history(vacuum, steps=2, step_size=0.001)

    assert jnp.max(jnp.abs(history["total_energy"])) < 1e-7
    assert jnp.max(jnp.abs(history["family_norm"])) < 1e-8


def test_physical_right_production_energy_history_records_components() -> None:
    initial = sm_physical_right_production_initial_state()
    final, history = sm_physical_right_production_energy_history(initial, steps=2, step_size=0.001)
    final_observables = sm_physical_right_production_energy_observables(final)

    assert tuple(history.keys()) == (
        "family_norm",
        "gauge_energy",
        "higgs_energy",
        "streaming_energy",
        "yukawa_energy",
        "total_energy",
    )
    assert history["total_energy"].shape == (3,)
    assert jnp.all(jnp.isfinite(history["total_energy"]))
    assert jnp.min(history["gauge_energy"]) > 1e-8
    assert jnp.min(history["higgs_energy"]) > 1e-3
    assert jnp.max(jnp.abs(history["streaming_energy"])) > 1.0
    assert jnp.max(jnp.abs(history["yukawa_energy"])) > 1e-3
    assert jnp.abs(final_observables["total_energy"] - history["total_energy"][-1]) < 1e-7


def test_physical_right_production_energy_resolves_zero_yukawa_control() -> None:
    initial = sm_physical_right_production_initial_state()
    _, default_history = sm_physical_right_production_energy_history(initial, steps=2, step_size=0.001)
    _, zero_history = sm_physical_right_production_energy_history(
        initial,
        steps=2,
        step_size=0.001,
        quark_yukawas=sm_zero_quark_yukawas(),
        lepton_yukawas=sm_zero_family_lepton_yukawas(),
    )

    assert jnp.abs(default_history["total_energy"][-1] - zero_history["total_energy"][-1]) > 1e-3
    assert jnp.abs(zero_history["yukawa_energy"][-1]) < 1e-7


def test_physical_right_production_energy_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_energy_diagnostics()

    assert diagnostics.vacuum_initial_total_energy_abs < 1e-7
    assert diagnostics.vacuum_final_total_energy_abs < 1e-7
    assert diagnostics.deterministic_initial_total_energy > 1.0
    assert diagnostics.deterministic_final_total_energy > 1.0
    assert diagnostics.deterministic_total_energy_delta_abs < 5e-6
    assert diagnostics.deterministic_max_total_energy_abs > 1.0
    assert diagnostics.deterministic_gauge_energy_positive > 1e-8
    assert diagnostics.deterministic_higgs_energy_positive > 1e-3
    assert diagnostics.deterministic_streaming_energy_abs > 1.0
    assert diagnostics.deterministic_yukawa_energy_abs > 1e-3
    assert diagnostics.zero_yukawa_final_total_energy_difference_abs > 1e-3
    assert diagnostics.rollout_family_norm_drift < 1e-5
    assert diagnostics.history_count == 3
    assert bool(diagnostics.history_all_finite)
