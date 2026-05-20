"""Import-boundary tests for the lab/main simulator split."""

from __future__ import annotations

import importlib
import sys

import clifford_3plus2_d5.spacetime_qca as spacetime_qca
import clifford_3plus2_d5.spacetime_qca.lab as spacetime_lab
import clifford_3plus2_d5.spacetime_qca.lab.tiny_runner as tiny_runner
import clifford_3plus2_d5.spacetime_qca.simulator as spacetime_simulator
import clifford_3plus2_d5.spacetime_qca.simulator.runner as simulator_runner


def test_old_runner_import_paths_are_not_supported() -> None:
    for module_name in (
        "clifford_3plus2_d5.spacetime_qca.jax_runner",
        "clifford_3plus2_d5.spacetime_qca.scripts.run_tiny_sim",
    ):
        sys.modules.pop(module_name, None)
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module_name} should not be importable after the simulator split")


def test_spacetime_qca_top_level_does_not_export_runner_symbols() -> None:
    for name in (
        "SimulationRunConfig",
        "SimulationHistory",
        "SimulationResult",
        "run_simulation",
        "save_simulation_result",
        "simulation_summary",
    ):
        assert not hasattr(spacetime_qca, name)


def test_lab_runner_exports_are_supported() -> None:
    assert spacetime_lab.SimulationRunConfig is tiny_runner.SimulationRunConfig
    assert spacetime_lab.run_simulation is tiny_runner.run_simulation
    assert spacetime_lab.save_simulation_result is tiny_runner.save_simulation_result
    assert spacetime_lab.simulation_summary is tiny_runner.simulation_summary


def test_main_simulator_exports_are_supported() -> None:
    assert spacetime_simulator.SpacetimeSimulationConfig is not None
    assert spacetime_simulator.SpacetimeFields is not None
    assert spacetime_simulator.run_spacetime_simulation is simulator_runner.run_spacetime_simulation
    assert spacetime_simulator.save_spacetime_simulation_result is simulator_runner.save_spacetime_simulation_result
    assert spacetime_simulator.spacetime_simulation_summary is simulator_runner.spacetime_simulation_summary
