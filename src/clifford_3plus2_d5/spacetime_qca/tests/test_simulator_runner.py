"""Scan-backed main spacetime-QCA simulator tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca.lab.tiny_runner import (
    SimulationRunConfig,
    run_simulation as run_lab_simulation,
)
from clifford_3plus2_d5.spacetime_qca.simulator import (
    SpacetimeSimulationConfig,
    run_spacetime_simulation,
    save_spacetime_simulation_result,
    sm_smoke,
    spacetime_simulation_summary,
    su2_l_tiny,
    u1_y_tiny,
)
from clifford_3plus2_d5.spacetime_qca.simulator.scripts.run_sim import main as main_sim_cli


def test_main_simulator_zero_step_matches_lab_runner() -> None:
    main = run_spacetime_simulation(SpacetimeSimulationConfig(step_size=0.0, steps=0, use_jit=False))
    lab = run_lab_simulation(SimulationRunConfig(step_size=0.0, steps=0))

    np.testing.assert_array_equal(np.asarray(main.step_indices), np.asarray((0,)))
    np.testing.assert_allclose(np.asarray(main.observations["fermion_norm"]), np.asarray(lab.history.fermion_norm))
    np.testing.assert_allclose(
        np.asarray(main.observations["total_energy_proxy"]),
        np.asarray(lab.history.total_energy_proxy),
    )
    assert bool(main.all_finite)


def test_main_simulator_record_every_keeps_final_step() -> None:
    result = run_spacetime_simulation(SpacetimeSimulationConfig(step_size=0.0, steps=5, record_every=2, use_jit=False))

    np.testing.assert_array_equal(np.asarray(result.step_indices), np.asarray((0, 2, 4, 5)))
    assert result.final_fields.state.shape == (1, 1, 1, 4, 32)


def test_main_simulator_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        run_spacetime_simulation(SpacetimeSimulationConfig(steps=-1))

    with pytest.raises(ValueError, match="record_every must be positive"):
        run_spacetime_simulation(SpacetimeSimulationConfig(record_every=0))

    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        run_spacetime_simulation(SpacetimeSimulationConfig(lattice_shape=(1, 0, 1)))

    with pytest.raises(ValueError, match="force_chunk_size must be positive"):
        run_spacetime_simulation(SpacetimeSimulationConfig(force_chunk_size=0))

    with pytest.raises(ValueError, match="gauss_projection_steps must be nonnegative"):
        run_spacetime_simulation(SpacetimeSimulationConfig(gauss_projection_steps=-1))

    with pytest.raises(ValueError, match="gauss_projection_step_size must be nonnegative"):
        run_spacetime_simulation(SpacetimeSimulationConfig(gauss_projection_step_size=-0.1))


def test_main_simulator_presets_are_memory_safe() -> None:
    assert u1_y_tiny().sector == "u1_y"
    assert su2_l_tiny().sector == "su2_l"
    assert sm_smoke().sector == "sm"
    assert sm_smoke().steps == 1


def test_main_simulator_save_writes_npz_and_json(tmp_path) -> None:
    result = run_spacetime_simulation(SpacetimeSimulationConfig(step_size=0.0, steps=1, use_jit=False))
    npz_path, json_path = save_spacetime_simulation_result(result, tmp_path / "run.npz")

    assert npz_path.exists()
    assert json_path.exists()
    with np.load(npz_path) as payload:
        assert set(payload.files) >= {"step_indices", "fermion_norm", "final_state", "final_phi"}
        np.testing.assert_array_equal(payload["step_indices"], np.asarray((0, 1)))

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["metadata"]["runner"] == "spacetime_qca.simulator"
    assert metadata["summary"]["all_finite"] is True


def test_main_simulator_cli_prints_json(capsys) -> None:
    exit_code = main_sim_cli(("--steps", "0", "--step-size", "0.0"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == 0
    assert payload["all_finite"] is True
    assert payload["force_method"] == "finite_difference"
    assert payload["use_jit"] is False


def test_main_simulator_cli_accepts_batched_force_method(capsys) -> None:
    exit_code = main_sim_cli(
        (
            "--steps",
            "0",
            "--step-size",
            "0.0",
            "--force-method",
            "finite_difference_batched",
            "--force-chunk-size",
            "16",
        ),
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["force_method"] == "finite_difference_batched"
    assert payload["force_chunk_size"] == 16


def test_main_simulator_cli_accepts_analytic_force_method(capsys) -> None:
    exit_code = main_sim_cli(("--steps", "0", "--step-size", "0.0", "--force-method", "analytic_staple"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["force_method"] == "analytic_staple"


def test_main_simulator_cli_accepts_eigh_yukawa_oracle_mode(capsys) -> None:
    exit_code = main_sim_cli(("--steps", "0", "--step-size", "0.0", "--yukawa-mode", "unitary_eigh"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["yukawa_mode"] == "unitary_eigh"


def test_main_simulator_cli_accepts_gauss_projection_controls(capsys) -> None:
    exit_code = main_sim_cli(
        (
            "--steps",
            "0",
            "--step-size",
            "0.0",
            "--gauss-projection-steps",
            "2",
            "--gauss-projection-step-size",
            "0.01",
        ),
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gauss_projection_steps"] == 2
    assert payload["gauss_projection_step_size"] == 0.01


def test_main_simulator_cli_output_writes_metadata(tmp_path, capsys) -> None:
    output = tmp_path / "sim.npz"
    exit_code = main_sim_cli(("--steps", "0", "--step-size", "0.0", "--output", str(output)))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output)
    assert output.exists()
    json_path = output.with_suffix(".json")
    assert json_path.exists()
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["metadata"]["runner"] == "spacetime_qca.simulator"


@pytest.mark.slow
def test_main_simulator_one_step_matches_lab_runner() -> None:
    main = run_spacetime_simulation(
        SpacetimeSimulationConfig(steps=1, step_size=0.0025, matter_coupling=0.0, use_jit=False),
    )
    lab = run_lab_simulation(
        SimulationRunConfig(steps=1, step_size=0.0025, matter_coupling=0.0, yukawa_mode="unitary"),
    )

    np.testing.assert_allclose(np.asarray(main.final_fields.state), np.asarray(lab.final_fields.state), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(main.observations["fermion_norm"]),
        np.asarray(lab.history.fermion_norm),
        atol=1e-6,
    )
    assert spacetime_simulation_summary(main)["all_finite"] is True
