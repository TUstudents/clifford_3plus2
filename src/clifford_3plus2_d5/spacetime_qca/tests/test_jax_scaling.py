"""Session 43 scaling-diagnostics tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca.jax_scaling import (
    ScalingRunConfig,
    jax_coupled_scaling_trial,
    jax_default_scaling_initial_state,
    jax_neutral_vacuum_density_probe,
    jax_scaling_snapshot,
    jax_step_size_scaling_sweep,
    session43_scaling_audit_payload,
)


def _assert_finite_scalar(value: jnp.ndarray) -> None:
    assert np.asarray(value).shape == ()
    assert bool(jnp.all(jnp.isfinite(value)))


def test_scaling_run_config_defaults_are_memory_safe() -> None:
    config = ScalingRunConfig()

    assert config.lattice_shape == (1, 1, 1)
    assert config.sector == "u1_y"
    assert config.step_size == 0.005
    assert config.shapes is None


def test_default_scaling_initial_state_shapes() -> None:
    config = ScalingRunConfig(lattice_shape=(2, 1, 1), sector="u1_y")
    fields = jax_default_scaling_initial_state(config)

    assert fields.state.shape == (2, 1, 1, 4, 32)
    assert fields.links.shape == (2, 1, 1, 8, 32, 32)
    assert fields.momenta.shape == (2, 1, 1, 8, 1)
    assert fields.phi.shape == (2, 1, 1, 2)
    assert fields.higgs_momentum.shape == (2, 1, 1, 2)
    assert fields.higgs_links.shape == (2, 1, 1, 8, 2, 2)


def test_scaling_snapshot_returns_expected_finite_scalars() -> None:
    config = ScalingRunConfig()
    fields = jax_default_scaling_initial_state(config)

    snapshot = jax_scaling_snapshot(
        fields.state,
        fields.links,
        fields.momenta,
        fields.phi,
        fields.higgs_momentum,
        fields.higgs_links,
        config=config,
        reference_state=fields.state,
    )

    for value in (
        snapshot.fermion_norm,
        snapshot.gauge_hamiltonian_density,
        snapshot.higgs_total_energy,
        snapshot.higgs_energy_density_mean,
        snapshot.gauss_residual_norm,
        snapshot.yukawa_norm_drift,
        snapshot.total_energy_proxy,
    ):
        _assert_finite_scalar(value)
    np.testing.assert_allclose(
        np.asarray(snapshot.total_energy_proxy),
        np.asarray(snapshot.gauge_hamiltonian_density + snapshot.higgs_energy_density_mean),
        atol=1e-6,
    )


def test_neutral_vacuum_density_probe_is_volume_normalized() -> None:
    probes = jax_neutral_vacuum_density_probe(((1, 1, 1), (2, 1, 1)))

    assert tuple(probe.lattice_shape for probe in probes) == ((1, 1, 1), (2, 1, 1))
    for probe in probes:
        np.testing.assert_allclose(np.asarray(probe.gauge_hamiltonian_density), 0.0, atol=2e-6)
        np.testing.assert_allclose(np.asarray(probe.higgs_energy_density_mean), 0.0, atol=2e-6)
        np.testing.assert_allclose(np.asarray(probe.total_energy_proxy), 0.0, atol=2e-6)
        assert bool(probe.all_zero)


def test_session43_scaling_audit_payload_is_stable() -> None:
    payload = session43_scaling_audit_payload()

    assert payload["default_sector"] == "u1_y"
    assert payload["default_lattice_shape"] == (1, 1, 1)
    assert payload["neutral_probe_shapes"] == ((1, 1, 1), (2, 1, 1))
    assert payload["step_sizes"] == (0.0, 0.0025, 0.005)
    assert any("not a continuum renormalization proof" in note for note in payload["notes"])


@pytest.mark.slow
def test_zero_step_coupled_scaling_trial_has_zero_energy_drifts() -> None:
    trial = jax_coupled_scaling_trial(ScalingRunConfig(step_size=0.0))

    assert bool(trial.all_finite)
    np.testing.assert_allclose(np.asarray(trial.fermion_norm_drift), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(trial.gauge_energy_drift), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(trial.higgs_energy_drift), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(trial.total_energy_proxy_drift), 0.0, atol=3e-6)


@pytest.mark.slow
def test_nonzero_coupled_scaling_trial_has_finite_drift_diagnostics() -> None:
    trial = jax_coupled_scaling_trial(ScalingRunConfig(step_size=0.0025))

    assert bool(trial.all_finite)
    for value in (
        trial.fermion_norm_drift,
        trial.gauge_energy_drift,
        trial.higgs_energy_drift,
        trial.gauss_residual_drift,
        trial.total_energy_proxy_drift,
    ):
        _assert_finite_scalar(value)
        assert float(value) >= 0.0


@pytest.mark.slow
def test_step_size_scaling_sweep_returns_ordered_finite_trials() -> None:
    step_sizes = (0.0, 0.0025, 0.005)
    trials = jax_step_size_scaling_sweep(ScalingRunConfig(), step_sizes)

    assert len(trials) == len(step_sizes)
    assert all(bool(trial.all_finite) for trial in trials)
    assert float(trials[0].total_energy_proxy_drift) <= float(trials[-1].total_energy_proxy_drift) + 1e-5
