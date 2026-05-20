"""Prototype tiny simulation lab runner tests."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

import clifford_3plus2_d5.spacetime_qca.lab as spacetime_lab
from clifford_3plus2_d5.spacetime_qca.lab.tiny_runner import (
    SimulationRunConfig,
    run_simulation,
    save_simulation_result,
    simulation_summary,
)
from clifford_3plus2_d5.spacetime_qca.lab.scripts.run_tiny_sim import main as tiny_sim_main


def test_default_runner_config_is_memory_safe() -> None:
    config = SimulationRunConfig()

    assert config.lattice_shape == (1, 1, 1)
    assert config.sector == "u1_y"
    assert config.steps == 4
    assert config.record_every == 1
    assert config.step_size == 0.0025
    assert config.yukawa_mode == "unitary"


def test_zero_step_run_records_finite_history_and_zero_drift() -> None:
    result = run_simulation(SimulationRunConfig(step_size=0.0, steps=4))

    np.testing.assert_array_equal(np.asarray(result.history.step_indices), np.asarray((0, 1, 2, 3, 4)))
    assert bool(result.all_finite)
    np.testing.assert_allclose(
        np.asarray(result.history.fermion_norm),
        np.asarray(result.history.fermion_norm[0]),
        atol=3e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.history.total_energy_proxy),
        np.asarray(result.history.total_energy_proxy[0]),
        atol=3e-6,
    )


def test_runner_record_every_keeps_final_step() -> None:
    result = run_simulation(SimulationRunConfig(step_size=0.0, steps=5, record_every=2))

    np.testing.assert_array_equal(np.asarray(result.history.step_indices), np.asarray((0, 2, 4, 5)))


def test_runner_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        run_simulation(SimulationRunConfig(steps=-1))

    with pytest.raises(ValueError, match="record_every must be positive"):
        run_simulation(SimulationRunConfig(record_every=0))

    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        run_simulation(SimulationRunConfig(lattice_shape=(1, 0, 1)))


def test_save_simulation_result_writes_npz_and_json(tmp_path) -> None:
    result = run_simulation(SimulationRunConfig(step_size=0.0, steps=2))
    npz_path, json_path = save_simulation_result(result, tmp_path / "run.npz")

    assert npz_path.exists()
    assert json_path.exists()
    with np.load(npz_path) as payload:
        assert set(payload.files) >= {
            "step_indices",
            "fermion_norm",
            "gauge_hamiltonian_density",
            "higgs_total_energy",
            "gauss_residual_norm",
            "final_state",
            "final_phi",
        }
        np.testing.assert_array_equal(payload["step_indices"], np.asarray((0, 1, 2)))

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["metadata"]["runner"] == "spacetime_qca.lab.tiny_runner"
    assert metadata["summary"]["steps"] == 2
    assert metadata["summary"]["all_finite"] is True


def test_save_simulation_result_requires_npz_suffix(tmp_path) -> None:
    result = run_simulation(SimulationRunConfig(step_size=0.0, steps=0))

    with pytest.raises(ValueError, match="must end with .npz"):
        save_simulation_result(result, tmp_path / "run.data")


def test_runner_summary_mode_cli_prints_json(capsys) -> None:
    exit_code = tiny_sim_main(("--steps", "0", "--step-size", "0.0"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == 0
    assert payload["all_finite"] is True
    assert payload["yukawa_mode"] == "unitary"


def test_runner_output_cli_writes_files(tmp_path, capsys) -> None:
    output = tmp_path / "tiny.npz"
    exit_code = tiny_sim_main(("--steps", "0", "--step-size", "0.0", "--output", str(output)))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output)
    assert output.exists()
    assert output.with_suffix(".json").exists()


def test_session46_public_exports_are_available() -> None:
    assert spacetime_lab.SimulationRunConfig is SimulationRunConfig
    assert spacetime_lab.run_simulation is run_simulation
    assert spacetime_lab.save_simulation_result is save_simulation_result
    assert spacetime_lab.simulation_summary is simulation_summary


@pytest.mark.slow
def test_nonzero_unitary_runner_produces_finite_history() -> None:
    result = run_simulation(
        SimulationRunConfig(steps=1, step_size=0.0025, matter_coupling=0.0, yukawa_mode="unitary"),
    )

    assert bool(result.all_finite)
    assert result.final_fields.state.shape == (1, 1, 1, 4, 32)
    assert result.final_fields.links.shape == (1, 1, 1, 8, 32, 32)
    assert result.history.step_indices.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(result.history.fermion_norm)))
