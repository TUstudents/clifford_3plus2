"""Named small-run presets for the spacetime-QCA simulator."""

from __future__ import annotations

from clifford_3plus2_d5.spacetime_qca.simulator.config import SpacetimeSimulationConfig


def u1_y_tiny() -> SpacetimeSimulationConfig:
    """Return the default memory-safe physical-hypercharge smoke preset."""

    return SpacetimeSimulationConfig(sector="u1_y", label="u1_y_tiny")


def su2_l_tiny() -> SpacetimeSimulationConfig:
    """Return a memory-safe electroweak SU(2)_L smoke preset."""

    return SpacetimeSimulationConfig(sector="su2_l", label="su2_l_tiny")


def sm_smoke() -> SpacetimeSimulationConfig:
    """Return a memory-safe full SM-sector smoke preset."""

    return SpacetimeSimulationConfig(sector="sm", steps=1, label="sm_smoke")
