"""Scale-stability audit for the production echo Gram.

Stage 33 checks that the Stage 32 echo Gram is measured inside a local
finite-difference regime.  The same echo-Gram diagnostic is evaluated at
``epsilon`` and ``2 epsilon``.  Echo norms should scale linearly, while Gram
eigenvalues should scale quadratically.

This is a diagnostic of the current discrete production map.  It does not add
new dynamics or claim a continuum stability theorem.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_gram import (
    sm_physical_right_production_echo_gram_diagnostics,
)


class PhysicalRightProductionEchoScaleDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 33 echo-Gram scale stability."""

    small_epsilon: jnp.ndarray
    large_epsilon: jnp.ndarray
    epsilon_ratio: jnp.ndarray
    small_base_roundtrip_residual: jnp.ndarray
    large_base_roundtrip_residual: jnp.ndarray
    min_norm_scale_ratio: jnp.ndarray
    max_norm_scale_ratio: jnp.ndarray
    min_eigenvalue_scale_ratio: jnp.ndarray
    max_eigenvalue_scale_ratio: jnp.ndarray
    condition_number_delta: jnp.ndarray
    offdiag_correlation_delta: jnp.ndarray
    max_link_unitarity_residual: jnp.ndarray


def sm_physical_right_production_echo_scale_diagnostics(
    *,
    steps: int = 2,
    step_size: float = 1e-3,
    epsilon: float = 1e-4,
    scale: float = 2.0,
) -> PhysicalRightProductionEchoScaleDiagnostics:
    """Return Stage 33 scale-stability diagnostics for the echo Gram."""

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if scale <= 1:
        raise ValueError(f"scale must be greater than 1, got {scale}")

    small = sm_physical_right_production_echo_gram_diagnostics(
        steps=steps,
        step_size=step_size,
        epsilon=epsilon,
    )
    large = sm_physical_right_production_echo_gram_diagnostics(
        steps=steps,
        step_size=step_size,
        epsilon=scale * epsilon,
    )
    scale_value = jnp.asarray(scale, dtype=jnp.float32)
    quadratic_scale = scale_value * scale_value

    return PhysicalRightProductionEchoScaleDiagnostics(
        small_epsilon=small.perturbation_size,
        large_epsilon=large.perturbation_size,
        epsilon_ratio=large.perturbation_size / small.perturbation_size,
        small_base_roundtrip_residual=small.base_roundtrip_residual,
        large_base_roundtrip_residual=large.base_roundtrip_residual,
        min_norm_scale_ratio=large.min_echo_norm / jnp.maximum(small.min_echo_norm * scale_value, jnp.asarray(1e-30, dtype=jnp.float32)),
        max_norm_scale_ratio=large.max_echo_norm / jnp.maximum(small.max_echo_norm * scale_value, jnp.asarray(1e-30, dtype=jnp.float32)),
        min_eigenvalue_scale_ratio=large.gram_min_eigenvalue
        / jnp.maximum(small.gram_min_eigenvalue * quadratic_scale, jnp.asarray(1e-30, dtype=jnp.float32)),
        max_eigenvalue_scale_ratio=large.gram_max_eigenvalue
        / jnp.maximum(small.gram_max_eigenvalue * quadratic_scale, jnp.asarray(1e-30, dtype=jnp.float32)),
        condition_number_delta=jnp.abs(large.gram_condition_number - small.gram_condition_number),
        offdiag_correlation_delta=jnp.abs(large.max_offdiag_correlation - small.max_offdiag_correlation),
        max_link_unitarity_residual=jnp.max(
            jnp.asarray(
                [
                    small.max_inverse_sm_link_unitarity_residual,
                    small.max_inverse_higgs_link_unitarity_residual,
                    large.max_inverse_sm_link_unitarity_residual,
                    large.max_inverse_higgs_link_unitarity_residual,
                ],
            ),
        ),
    )
