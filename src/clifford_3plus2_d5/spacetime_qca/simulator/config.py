"""Production-facing spacetime-QCA simulator configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from clifford_3plus2_d5.spacetime_qca.jax_coupled_higgs import (
    HiggsCoupledSector,
    YukawaUpdateMode,
)
from clifford_3plus2_d5.spacetime_qca.jax_scaling import ScalingRunConfig


@dataclass(frozen=True)
class SpacetimeSimulationConfig:
    """Stable controls for the scan-backed spacetime-QCA simulator."""

    lattice_shape: tuple[int, int, int] = (1, 1, 1)
    sector: HiggsCoupledSector = "u1_y"
    steps: int = 4
    record_every: int = 1
    step_size: float = 0.0025
    matter_coupling: float = 1.0
    yukawa_coupling: float = 1.0
    beta: float = 1.0
    vev_squared: float = 1.0
    quartic: float = 1.0
    force_epsilon: float = 1e-3
    current_epsilon: float = 1e-3
    yukawa_mode: YukawaUpdateMode = "unitary"
    use_jit: bool = False
    label: str = "spacetime_qca"
    description: str = "Scan-backed spacetime-QCA simulator run"


def validate_spacetime_simulation_config(config: SpacetimeSimulationConfig) -> None:
    """Validate spacetime-QCA simulator controls."""

    if len(config.lattice_shape) != 3 or any(size <= 0 for size in config.lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {config.lattice_shape}")
    if config.steps < 0:
        raise ValueError(f"steps must be nonnegative, got {config.steps}")
    if config.record_every <= 0:
        raise ValueError(f"record_every must be positive, got {config.record_every}")


def scaling_config_from_spacetime_config(config: SpacetimeSimulationConfig) -> ScalingRunConfig:
    """Convert a main simulator config to the existing coupled-step config."""

    return ScalingRunConfig(
        lattice_shape=config.lattice_shape,
        sector=config.sector,
        step_size=config.step_size,
        matter_coupling=config.matter_coupling,
        yukawa_coupling=config.yukawa_coupling,
        beta=config.beta,
        vev_squared=config.vev_squared,
        quartic=config.quartic,
        force_epsilon=config.force_epsilon,
        current_epsilon=config.current_epsilon,
        yukawa_mode=config.yukawa_mode,
    )


def spacetime_simulation_metadata(config: SpacetimeSimulationConfig) -> dict[str, Any]:
    """Return JSON-safe metadata for a spacetime simulator config."""

    metadata = asdict(config)
    metadata["lattice_shape"] = tuple(config.lattice_shape)
    metadata["runner"] = "spacetime_qca.simulator"
    metadata["format"] = "npz-plus-json-v1"
    return metadata
