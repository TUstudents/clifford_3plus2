"""Generic periodic-lattice helpers for JAX simulation kernels."""

from __future__ import annotations

import jax.numpy as jnp


Displacement3D = tuple[int, int, int]
LatticeShape3D = tuple[int, int, int]


def validate_lattice_shape(shape: tuple[int, ...]) -> LatticeShape3D:
    """Validate and return a three-dimensional periodic lattice shape."""

    if len(shape) != 3:
        raise ValueError("lattice shape must have exactly three dimensions")
    if any(size <= 0 for size in shape):
        raise ValueError("lattice dimensions must be positive")
    return int(shape[0]), int(shape[1]), int(shape[2])


def validate_displacements(displacements: tuple[Displacement3D, ...], *, expected_count: int | None = None) -> None:
    """Validate a tuple of integer 3D displacements."""

    if expected_count is not None and len(displacements) != expected_count:
        raise ValueError(f"expected {expected_count} displacements")
    for displacement in displacements:
        if len(displacement) != 3:
            raise ValueError("each displacement must have exactly three components")


def source_roll(state: jnp.ndarray, displacement: Displacement3D) -> jnp.ndarray:
    """Return ``state[x+h]`` at target index ``x`` for pull-form displacement ``h``."""

    return jnp.roll(
        state,
        shift=tuple(-component for component in displacement),
        axis=(0, 1, 2),
    )


def source_rolls(state: jnp.ndarray, displacements: tuple[Displacement3D, ...]) -> jnp.ndarray:
    """Return stacked pull-form rolls with shape ``(len(displacements), *state.shape)``."""

    validate_displacements(displacements)
    return jnp.stack(tuple(source_roll(state, displacement) for displacement in displacements), axis=0)
