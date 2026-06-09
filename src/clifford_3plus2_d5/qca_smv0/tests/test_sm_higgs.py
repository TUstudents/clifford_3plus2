"""Tests for QCA_SMv0 local Higgs/Yukawa collision."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import deterministic_sm_state
from clifford_3plus2_d5.qca_smv0.sm_higgs import (
    SMYukawaCouplings,
    sm_apply_yukawa_collision,
    sm_chirality_norms,
    sm_constant_higgs,
    sm_higgs_tilde,
    sm_massive_dispersion_residual,
    sm_massive_positive_phase,
    sm_yukawa_hermitian_residual,
    sm_yukawa_internal_matrix,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def _q_index(color: int, weak: int) -> int:
    return 2 * color + weak


def _u_c_index(color: int) -> int:
    return 6 + color


def _d_c_index(color: int) -> int:
    return 9 + color


def _l_index(weak: int) -> int:
    return 12 + weak


def _e_c_index() -> int:
    return 14


def _nu_c_index() -> int:
    return 15


def test_constant_higgs_and_tilde_use_unitary_gauge_doors() -> None:
    higgs = sm_constant_higgs((1, 1, 1), vev=2.0)
    h_tilde = sm_higgs_tilde(higgs)

    assert higgs.shape == (1, 1, 1, 2)
    assert jnp.max(jnp.abs(higgs[..., 0])) < 1e-7
    assert jnp.max(jnp.abs(higgs[..., 1] - jnp.sqrt(2.0))) < 1e-7
    assert jnp.max(jnp.abs(h_tilde[..., 0] - jnp.sqrt(2.0))) < 1e-7
    assert jnp.max(jnp.abs(h_tilde[..., 1])) < 1e-7


def test_sm_yukawa_matrix_is_hermitian_and_has_sm_door_structure() -> None:
    higgs = sm_constant_higgs((1, 1, 1), vev=1.0)
    couplings = SMYukawaCouplings(up=1.1, down=0.7, electron=0.5, neutrino=0.3)
    y = sm_yukawa_internal_matrix(higgs, couplings=couplings)[0, 0, 0]
    block = y[:16, :16]

    assert sm_yukawa_hermitian_residual(y) < 1e-7
    for color in range(3):
        assert jnp.abs(block[_q_index(color, 0), _u_c_index(color)]) > 0
        assert jnp.abs(block[_q_index(color, 1), _d_c_index(color)]) > 0
        assert jnp.abs(block[_q_index(color, 1), _u_c_index(color)]) < 1e-7
        assert jnp.abs(block[_q_index(color, 0), _d_c_index(color)]) < 1e-7

    assert jnp.abs(block[_l_index(0), _nu_c_index()]) > 0
    assert jnp.abs(block[_l_index(1), _e_c_index()]) > 0
    assert jnp.abs(block[_l_index(1), _nu_c_index()]) < 1e-7
    assert jnp.abs(block[_l_index(0), _e_c_index()]) < 1e-7


def test_yukawa_collision_has_exact_identity_controls_and_preserves_norm() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    zero_higgs = jnp.zeros_like(higgs)

    assert jnp.max(jnp.abs(sm_apply_yukawa_collision(state, higgs, step_size=0.0) - state)) < 1e-7
    assert jnp.max(jnp.abs(sm_apply_yukawa_collision(state, zero_higgs, step_size=0.07) - state)) < 1e-7

    updated = sm_apply_yukawa_collision(state, higgs, step_size=0.07)
    assert jnp.abs(state_norm_squared(updated) - state_norm_squared(state)) < 5e-7


def test_yukawa_collision_flips_dirac_chirality_locally() -> None:
    lattice_shape = (1, 1, 1)
    state = jnp.zeros((*lattice_shape, 4, 32), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, _q_index(0, 0)].set(1.0 + 0.0j)
    higgs = sm_constant_higgs(lattice_shape)

    updated = sm_apply_yukawa_collision(state, higgs, step_size=0.07)
    _, right_norm = sm_chirality_norms(updated)

    assert right_norm > 1e-5


def test_constant_higgs_collision_has_massive_dispersion() -> None:
    assert sm_massive_dispersion_residual() < 1e-5
    assert jnp.abs(sm_massive_positive_phase((0.0, 0.0, 0.0), mass_angle=0.03) - 0.03) < 1e-6


def test_yukawa_collision_is_jittable() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    expected = sm_apply_yukawa_collision(state, higgs, step_size=0.07)
    jitted = jax.jit(sm_apply_yukawa_collision, static_argnames=("step_size",))
    actual = jitted(state, higgs, step_size=0.07)

    assert jnp.max(jnp.abs(actual - expected)) < 1e-7
