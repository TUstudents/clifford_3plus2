"""Tests for the bare bulk BCC Weyl walk."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import (
    BCC_DISPLACEMENTS,
    bcc_anisotropy_scaling,
    bcc_axis_projectors,
    bcc_dirac_hop_completeness_residual,
    bcc_dirac_hop_matrices,
    bcc_dirac_split_symbol,
    bcc_dirac_split_axis_step,
    bcc_dirac_spin_axis_step,
    bcc_dirac_step,
    bcc_dirac_symbol_unitarity_residual,
    bcc_hop_completeness_residual,
    bcc_hop_matrices,
    bcc_split_symbol,
    bcc_symbol_from_hops,
    bcc_symbol_unitarity_residual,
    bcc_weyl_step,
    pauli_matrices,
    small_k_dirac_speed_error,
    small_k_weyl_speed_error,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def test_pauli_matrices_square_to_identity() -> None:
    tau = pauli_matrices()
    identity = jnp.eye(2, dtype=tau.dtype)

    for axis in range(3):
        assert jnp.max(jnp.abs(tau[axis] @ tau[axis] - identity)) < 1e-6


def test_bcc_projectors_are_orthogonal_resolutions() -> None:
    projectors = bcc_axis_projectors()
    identity = jnp.eye(2, dtype=projectors.dtype)

    for axis in range(3):
        p_minus = projectors[axis, 0]
        p_plus = projectors[axis, 1]
        assert jnp.max(jnp.abs(p_minus @ p_minus - p_minus)) < 1e-6
        assert jnp.max(jnp.abs(p_plus @ p_plus - p_plus)) < 1e-6
        assert jnp.max(jnp.abs(p_minus @ p_plus)) < 1e-6
        assert jnp.max(jnp.abs(p_minus + p_plus - identity)) < 1e-6


def test_bcc_hops_have_eight_displacements_and_complete_norm() -> None:
    hops = bcc_hop_matrices()

    assert len(BCC_DISPLACEMENTS) == 8
    assert hops.shape == (8, 2, 2)
    assert bcc_hop_completeness_residual() < 1e-6


def test_opposite_chirality_symbol_is_momentum_reverse() -> None:
    k = (0.11, -0.07, 0.05)
    right = bcc_split_symbol(k, chirality=-1)
    reversed_left = bcc_split_symbol(tuple(-value for value in k), chirality=1)

    assert jnp.max(jnp.abs(right - reversed_left)) < 1e-6


def test_bloch_symbol_matches_hop_sum_and_is_unitary() -> None:
    k = (0.19, -0.13, 0.07)
    split_symbol = bcc_split_symbol(k)
    hop_symbol = bcc_symbol_from_hops(k)

    assert jnp.max(jnp.abs(split_symbol - hop_symbol)) < 1e-6
    assert bcc_symbol_unitarity_residual(k) < 1e-6


def test_bcc_weyl_step_preserves_norm_on_periodic_lattice() -> None:
    key = jax.random.PRNGKey(101)
    shape = (3, 4, 2, 2)
    real = jax.random.normal(key, shape, dtype=jnp.float32)
    imag = jax.random.normal(jax.random.fold_in(key, 1), shape, dtype=jnp.float32)
    state = (real + 1j * imag).astype(jnp.complex64)

    updated = bcc_weyl_step(state)
    drift = state_norm_squared(updated) - state_norm_squared(state)

    assert updated.shape == state.shape
    assert jnp.abs(drift) < 2e-5


def test_bcc_weyl_step_is_jittable() -> None:
    state = jnp.zeros((2, 2, 2, 2), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0].set(1.0 + 0.0j)

    expected = bcc_weyl_step(state)
    actual = jax.jit(bcc_weyl_step)(state)

    assert jnp.max(jnp.abs(actual - expected)) < 1e-6


def test_small_k_symbol_has_weyl_phase_speed() -> None:
    assert small_k_weyl_speed_error(k=(1e-3, 0.0, 0.0)) < 1e-6
    assert small_k_weyl_speed_error() < 5e-6


def test_weyl_anisotropy_spread_decays_under_continuum_scaling() -> None:
    spreads, ratios = bcc_anisotropy_scaling(magnitudes=(0.08, 0.04, 0.02))

    assert spreads.shape == (3,)
    assert ratios.shape == (2,)
    assert spreads[0] > spreads[1] > spreads[2] > 0
    assert jnp.max(ratios) < 0.60
    assert jnp.min(ratios) > 0.35


def test_dirac_hops_are_block_assembled_from_opposite_chiralities() -> None:
    hops = bcc_dirac_hop_matrices()
    left_hops = bcc_hop_matrices(chirality=1)
    right_hops = bcc_hop_matrices(chirality=-1)

    assert hops.shape == (8, 4, 4)
    assert bcc_dirac_hop_completeness_residual() < 1e-6
    assert jnp.max(jnp.abs(hops[:, :2, :2] - left_hops)) < 1e-6
    assert jnp.max(jnp.abs(hops[:, 2:, 2:] - right_hops)) < 1e-6
    assert jnp.max(jnp.abs(hops[:, :2, 2:])) < 1e-6
    assert jnp.max(jnp.abs(hops[:, 2:, :2])) < 1e-6


def test_dirac_symbol_is_unitary_and_has_weyl_continuum_speed() -> None:
    k = (0.19, -0.13, 0.07)
    symbol = bcc_dirac_split_symbol(k)

    assert symbol.shape == (4, 4)
    assert bcc_dirac_symbol_unitarity_residual(k) < 1e-6
    assert small_k_dirac_speed_error(k=(1e-3, 0.0, 0.0)) < 1e-6
    assert small_k_dirac_speed_error() < 5e-6


def test_bcc_dirac_step_preserves_norm_on_periodic_lattice() -> None:
    key = jax.random.PRNGKey(202)
    shape = (3, 2, 2, 4)
    real = jax.random.normal(key, shape, dtype=jnp.float32)
    imag = jax.random.normal(jax.random.fold_in(key, 1), shape, dtype=jnp.float32)
    state = (real + 1j * imag).astype(jnp.complex64)

    updated = bcc_dirac_step(state)
    drift = state_norm_squared(updated) - state_norm_squared(state)

    assert updated.shape == state.shape
    assert jnp.abs(drift) < 2e-5


def test_bcc_dirac_spin_axis_step_matches_independent_spectator_steps() -> None:
    key = jax.random.PRNGKey(404)
    shape = (2, 2, 2, 4, 3)
    real = jax.random.normal(key, shape, dtype=jnp.float32)
    imag = jax.random.normal(jax.random.fold_in(key, 1), shape, dtype=jnp.float32)
    state = (real + 1j * imag).astype(jnp.complex64)

    expected = jnp.stack([bcc_dirac_step(state[..., spectator]) for spectator in range(shape[-1])], axis=-1)

    assert jnp.max(jnp.abs(bcc_dirac_spin_axis_step(state) - expected)) < 1e-6


def test_bcc_dirac_split_axis_step_matches_hop_sum_with_spectators() -> None:
    key = jax.random.PRNGKey(405)
    shape = (2, 2, 2, 4, 3)
    real = jax.random.normal(key, shape, dtype=jnp.float32)
    imag = jax.random.normal(jax.random.fold_in(key, 1), shape, dtype=jnp.float32)
    state = (real + 1j * imag).astype(jnp.complex64)

    assert jnp.max(jnp.abs(bcc_dirac_split_axis_step(state) - bcc_dirac_spin_axis_step(state))) < 2e-6


def test_bcc_dirac_step_is_jittable() -> None:
    state = jnp.zeros((2, 2, 2, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0].set(1.0 + 0.0j)
    state = state.at[0, 0, 0, 3].set(0.5 + 0.25j)

    expected = bcc_dirac_step(state)
    actual = jax.jit(bcc_dirac_step)(state)

    assert jnp.max(jnp.abs(actual - expected)) < 1e-6
