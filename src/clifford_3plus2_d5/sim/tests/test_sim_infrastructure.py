"""Shared simulator infrastructure tests."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5 import sim
from clifford_3plus2_d5.spacetime_qca import (
    jax_constant_link_field as spacetime_constant_link_field,
    jax_identity_link_field as spacetime_identity_link_field,
)


def test_identity_and_constant_link_fields_have_expected_layout() -> None:
    identity = sim.jax_identity_link_field((2, 3, 4), 2, edge_count=6)
    link = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex64)
    constant = sim.jax_constant_link_field((2, 3, 4), link, edge_count=6)

    assert identity.shape == (2, 3, 4, 6, 2, 2)
    assert constant.shape == (2, 3, 4, 6, 2, 2)
    np.testing.assert_allclose(np.asarray(identity[1, 2, 3, 5]), np.eye(2))
    np.testing.assert_allclose(np.asarray(constant[1, 2, 3, 5]), np.asarray(link))


def test_spacetime_public_link_helpers_match_shared_sim_defaults() -> None:
    link = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)

    np.testing.assert_allclose(
        np.asarray(spacetime_identity_link_field((2, 2, 2), 2)),
        np.asarray(sim.jax_identity_link_field((2, 2, 2), 2)),
    )
    np.testing.assert_allclose(
        np.asarray(spacetime_constant_link_field((2, 2, 2), link)),
        np.asarray(sim.jax_constant_link_field((2, 2, 2), link)),
    )


def test_source_roll_matches_manual_jnp_roll() -> None:
    state = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    displacement = (1, -1, 1)

    rolled = sim.source_roll(state, displacement)
    expected = jnp.roll(state, shift=(-1, 1, -1), axis=(0, 1, 2))

    np.testing.assert_array_equal(np.asarray(rolled), np.asarray(expected))


def test_generic_gauge_transform_uses_pull_link_convention() -> None:
    links = sim.jax_identity_link_field((2, 2, 2), 2, edge_count=2)
    site_gauge = jnp.zeros((2, 2, 2, 2, 2), dtype=jnp.complex64)
    site_gauge = site_gauge.at[..., 0, 0].set(1)
    site_gauge = site_gauge.at[..., 1, 1].set(1)
    site_gauge = site_gauge.at[0, 0, 0].set(jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex64))
    displacements = ((1, 0, 0), (-1, 0, 0))

    transformed = sim.jax_transform_link_field(links, site_gauge, displacements)
    source = sim.source_roll(site_gauge, displacements[0])
    expected = site_gauge @ jnp.swapaxes(jnp.conj(source), -1, -2)

    np.testing.assert_allclose(np.asarray(transformed[..., 0, :, :]), np.asarray(expected))


def test_dirac_internal_flatten_roundtrip_and_norm_metrics() -> None:
    state = jnp.arange(2 * 2 * 2 * 4 * 3, dtype=jnp.float32).reshape((2, 2, 2, 4, 3))

    flat = sim.flatten_dirac_internal_state(state)
    roundtrip = sim.unflatten_dirac_internal_state(flat, internal_dim=3)
    metrics = sim.state_transition_metrics(state, roundtrip)

    assert flat.shape == (2, 2, 2, 12)
    np.testing.assert_array_equal(np.asarray(roundtrip), np.asarray(state))
    assert metrics.all_finite
    assert metrics.norm_drift == 0


def test_benchmark_helper_returns_nonnegative_timing() -> None:
    def add_one(value: jnp.ndarray) -> jnp.ndarray:
        return value + 1

    timing = sim.benchmark_jitted_kernel(add_one, jnp.ones((4,), dtype=jnp.float32))

    assert timing.compile_seconds >= 0
    assert timing.run_seconds >= 0


def test_profile_callable_returns_json_safe_payload() -> None:
    def add_one(value: jnp.ndarray) -> jnp.ndarray:
        return value + 1

    profile = sim.profile_callable(
        "add_one",
        add_one,
        args=(jnp.ones((4,), dtype=jnp.float32),),
        include_jit=True,
    )
    payload = profile.as_payload()

    assert payload["label"] == "add_one"
    assert payload["python_seconds"] >= 0
    assert payload["jit_compile_seconds"] >= 0
    assert payload["jit_run_seconds"] >= 0
    assert payload["all_finite"] is True
    json.dumps(payload)


def test_profile_callable_repeated_returns_warm_timing_stats() -> None:
    def add_one(value: jnp.ndarray) -> jnp.ndarray:
        return value + 1

    profile = sim.profile_callable_repeated(
        "add_one_repeat",
        add_one,
        args=(jnp.ones((4,), dtype=jnp.float32),),
        warmup_runs=1,
        timed_runs=2,
    )
    payload = profile.as_payload()

    assert payload["label"] == "add_one_repeat"
    assert payload["warmup_runs"] == 1
    assert payload["timed_runs"] == 2
    assert len(payload["run_seconds"]) == 2
    assert payload["min_seconds"] <= payload["mean_seconds"] <= payload["max_seconds"]
    assert payload["all_finite"] is True
    json.dumps(payload)


def test_recorded_loop_records_initial_requested_and_final_steps() -> None:
    def step(value: jnp.ndarray) -> jnp.ndarray:
        return value + 1

    def observe(value: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {"value": value, "double": 2 * value}

    result = sim.run_recorded_loop(
        jnp.asarray(0, dtype=jnp.int32),
        step,
        observe,
        sim.GenericRunConfig(steps=5, record_every=2),
    )

    np.testing.assert_array_equal(np.asarray(result.step_indices), np.asarray((0, 2, 4, 5)))
    np.testing.assert_array_equal(np.asarray(result.observations["value"]), np.asarray((0, 2, 4, 5)))
    np.testing.assert_array_equal(np.asarray(result.observations["double"]), np.asarray((0, 4, 8, 10)))
    assert bool(result.all_finite)


def test_recorded_scan_matches_loop_for_toy_state() -> None:
    def step(value: jnp.ndarray) -> jnp.ndarray:
        return value + 1

    def observe(value: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return {"value": value}

    config = sim.GenericRunConfig(steps=4, record_every=3, use_jit=False)
    loop = sim.run_recorded_loop(jnp.asarray(0, dtype=jnp.int32), step, observe, config)
    scan = sim.run_recorded_scan(jnp.asarray(0, dtype=jnp.int32), step, observe, config)

    np.testing.assert_array_equal(np.asarray(scan.step_indices), np.asarray(loop.step_indices))
    np.testing.assert_array_equal(np.asarray(scan.observations["value"]), np.asarray(loop.observations["value"]))
    np.testing.assert_array_equal(np.asarray(scan.final_state), np.asarray(loop.final_state))


def test_generic_runner_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        sim.recorded_step_indices(sim.GenericRunConfig(steps=-1))

    with pytest.raises(ValueError, match="record_every must be positive"):
        sim.recorded_step_indices(sim.GenericRunConfig(record_every=0))


def test_save_npz_json_writes_arrays_and_metadata(tmp_path) -> None:
    npz_path, json_path = sim.save_npz_json(
        {"values": jnp.asarray((1, 2, 3))},
        {"runner": "test"},
        tmp_path / "run.npz",
    )

    assert npz_path.exists()
    assert json_path.exists()
    with np.load(npz_path) as payload:
        np.testing.assert_array_equal(payload["values"], np.asarray((1, 2, 3)))
    assert sim.load_json_metadata(json_path)["runner"] == "test"


def test_save_npz_json_requires_npz_suffix(tmp_path) -> None:
    with pytest.raises(ValueError, match="must end with .npz"):
        sim.save_npz_json({"values": jnp.asarray((1,))}, {}, tmp_path / "run.data")
