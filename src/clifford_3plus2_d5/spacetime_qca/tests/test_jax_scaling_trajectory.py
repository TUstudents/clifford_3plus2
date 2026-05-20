"""Session 44 multi-step scaling trajectory tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import clifford_3plus2_d5.spacetime_qca as spacetime_qca
from clifford_3plus2_d5.spacetime_qca.jax_scaling import (
    ScalingRunConfig,
    jax_advance_scaling_fields,
    jax_compare_yukawa_modes,
    jax_coupled_scaling_trajectory,
    jax_scaling_timing_probe,
)


def _assert_finite_scalar(value: jnp.ndarray) -> None:
    assert np.asarray(value).shape == ()
    assert bool(jnp.all(jnp.isfinite(value)))


def test_zero_step_trajectory_has_zero_drifts() -> None:
    trajectory = jax_coupled_scaling_trajectory(ScalingRunConfig(step_size=0.0), steps=4)

    assert tuple(sample.step_index for sample in trajectory.samples) == (0, 1, 2, 3, 4)
    assert bool(trajectory.all_finite)
    np.testing.assert_allclose(np.asarray(trajectory.max_fermion_norm_drift), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(trajectory.max_gauge_energy_drift), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(trajectory.max_higgs_energy_drift), 0.0, atol=3e-6)
    np.testing.assert_allclose(np.asarray(trajectory.max_total_energy_proxy_drift), 0.0, atol=3e-6)


def test_record_every_keeps_requested_samples_and_final() -> None:
    trajectory = jax_coupled_scaling_trajectory(
        ScalingRunConfig(step_size=0.0),
        steps=4,
        record_every=2,
    )

    assert tuple(sample.step_index for sample in trajectory.samples) == (0, 2, 4)
    assert trajectory.initial.step_index == 0
    assert trajectory.final.step_index == 4


def test_record_every_keeps_final_when_not_divisible() -> None:
    trajectory = jax_coupled_scaling_trajectory(
        ScalingRunConfig(step_size=0.0),
        steps=5,
        record_every=2,
    )

    assert tuple(sample.step_index for sample in trajectory.samples) == (0, 2, 4, 5)
    assert trajectory.final.step_index == 5


def test_trajectory_rejects_invalid_step_controls() -> None:
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        jax_coupled_scaling_trajectory(ScalingRunConfig(), steps=-1)

    with pytest.raises(ValueError, match="record_every must be positive"):
        jax_coupled_scaling_trajectory(ScalingRunConfig(), record_every=0)


def test_timing_probe_zero_step_returns_nonnegative_timing() -> None:
    timing = jax_scaling_timing_probe(ScalingRunConfig(step_size=0.0), steps=1)

    assert timing.compile_seconds >= 0
    assert timing.run_seconds >= 0


def test_timing_probe_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        jax_scaling_timing_probe(ScalingRunConfig(), steps=-1)


def test_session44_public_exports_are_available() -> None:
    assert spacetime_qca.ScalingTrajectory is not None
    assert spacetime_qca.ScalingTrajectorySample is not None
    assert spacetime_qca.YukawaModeComparison is not None
    assert spacetime_qca.jax_advance_scaling_fields is jax_advance_scaling_fields
    assert spacetime_qca.jax_coupled_scaling_trajectory is jax_coupled_scaling_trajectory
    assert spacetime_qca.jax_compare_yukawa_modes is jax_compare_yukawa_modes
    assert spacetime_qca.jax_scaling_timing_probe is jax_scaling_timing_probe


@pytest.mark.slow
def test_nonzero_unitary_trajectory_has_finite_drift_diagnostics() -> None:
    trajectory = jax_coupled_scaling_trajectory(
        ScalingRunConfig(step_size=0.0025, matter_coupling=0.0, yukawa_mode="unitary"),
        steps=2,
    )

    assert tuple(sample.step_index for sample in trajectory.samples) == (0, 1, 2)
    assert bool(trajectory.all_finite)
    for value in (
        trajectory.max_fermion_norm_drift,
        trajectory.max_gauge_energy_drift,
        trajectory.max_higgs_energy_drift,
        trajectory.max_gauss_residual_drift,
        trajectory.max_total_energy_proxy_drift,
    ):
        _assert_finite_scalar(value)
        assert float(value) >= 0.0


@pytest.mark.slow
def test_unitary_yukawa_mode_does_not_increase_fermion_norm_drift() -> None:
    comparison = jax_compare_yukawa_modes(
        ScalingRunConfig(step_size=0.005, matter_coupling=0.0),
        steps=1,
    )

    assert bool(comparison.all_finite)
    assert float(comparison.fermion_norm_drift_ratio) <= 1.0 + 1e-4
    _assert_finite_scalar(comparison.total_energy_proxy_drift_ratio)
