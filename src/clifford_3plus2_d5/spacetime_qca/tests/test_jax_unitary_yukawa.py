"""Session 45 exact unitary Yukawa insertion tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.sim.links import jax_identity_link_field
from clifford_3plus2_d5.sim.state import state_norm_squared
from clifford_3plus2_d5.spacetime_qca.jax_coupled_higgs import (
    _cos_sin_from_cubic_yukawa,
    _cos_sin_from_eigh,
    _selected_yukawa_lambda_squared,
    jax_apply_site_local_yukawa_kick,
    jax_apply_site_local_yukawa_unitary,
    jax_apply_site_local_yukawa_unitary_eigh,
    jax_apply_site_local_yukawa_update,
    jax_patisalam_fermion_gauge_higgs_step,
)
from clifford_3plus2_d5.spacetime_qca.jax_higgs import (
    jax_higgs_link_field_from_algebra,
    jax_higgs_yukawa_internal_control_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_scaling import ScalingRunConfig, jax_coupled_scaling_trial
from clifford_3plus2_d5.spacetime_qca.plaquette import canonical_bcc_plaquette_shapes


def _state() -> jnp.ndarray:
    state = jnp.zeros((1, 1, 1, 4, 32), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, 0].set(1.0 + 0.25j)
    state = state.at[0, 0, 0, 1, 7].set(-0.5 + 0.125j)
    state = state.at[0, 0, 0, 2, 13].set(0.25 - 0.2j)
    return state


def _neutral_phi(value: complex = 1.0 + 0.0j) -> jnp.ndarray:
    return jnp.zeros((1, 1, 1, 2), dtype=jnp.complex64).at[..., 1].set(value)


def _mixed_phi() -> jnp.ndarray:
    phi = jnp.zeros((1, 1, 1, 2), dtype=jnp.complex64)
    phi = phi.at[..., 0].set(0.25 + 0.125j)
    return phi.at[..., 1].set(0.75 - 0.2j)


def _shape():
    return (tuple(canonical_bcc_plaquette_shapes())[0],)


def test_selected_yukawa_map_satisfies_cubic_identity() -> None:
    phi = jnp.concatenate(
        (
            jnp.zeros((1, 1, 1, 2), dtype=jnp.complex64),
            _neutral_phi(0.5 + 0.25j),
            _mixed_phi(),
        ),
        axis=0,
    )
    hermitian = jax_higgs_yukawa_internal_control_field(phi, dtype=jnp.complex64)
    lambda_squared = _selected_yukawa_lambda_squared(phi)

    cubic_residual = hermitian @ hermitian @ hermitian - lambda_squared[..., None, None] * hermitian

    np.testing.assert_allclose(np.asarray(cubic_residual), np.zeros_like(np.asarray(cubic_residual)), atol=3e-3)


@pytest.mark.slow
def test_cubic_polynomial_cos_sin_matches_eigh_oracle() -> None:
    phi = _mixed_phi()
    hermitian = jax_higgs_yukawa_internal_control_field(phi, dtype=jnp.complex64)
    scale = jnp.asarray(0.0025, dtype=jnp.float32)

    polynomial = _cos_sin_from_cubic_yukawa(hermitian, phi, scale)
    oracle = _cos_sin_from_eigh(hermitian, scale)

    for actual, expected in zip(polynomial, oracle, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5)


@pytest.mark.slow
def test_unitary_yukawa_polynomial_matches_eigh_oracle() -> None:
    state = _state()
    phi = _mixed_phi()

    polynomial = jax_apply_site_local_yukawa_unitary(state, phi, step_size=0.0025)
    oracle = jax_apply_site_local_yukawa_unitary_eigh(state, phi, step_size=0.0025)

    np.testing.assert_allclose(np.asarray(polynomial), np.asarray(oracle), atol=3e-5)


def test_yukawa_update_zero_step_and_zero_coupling_are_noops() -> None:
    state = _state()
    phi = _neutral_phi()

    np.testing.assert_allclose(
        np.asarray(jax_apply_site_local_yukawa_unitary(state, phi, step_size=0.0)),
        np.asarray(state),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(jax_apply_site_local_yukawa_unitary(state, phi, step_size=0.1, yukawa_coupling=0.0)),
        np.asarray(state),
        atol=1e-7,
    )


@pytest.mark.slow
def test_unitary_yukawa_zero_higgs_field_is_noop() -> None:
    state = _state()
    zero_phi = jnp.zeros((1, 1, 1, 2), dtype=jnp.complex64)

    np.testing.assert_allclose(
        np.asarray(jax_apply_site_local_yukawa_unitary(state, zero_phi, step_size=0.1)),
        np.asarray(state),
        atol=2e-5,
    )


@pytest.mark.slow
def test_unitary_yukawa_preserves_norm_while_first_order_drifts() -> None:
    state = _state()
    phi = _neutral_phi()

    unitary = jax_apply_site_local_yukawa_unitary(state, phi, step_size=0.05)
    first_order = jax_apply_site_local_yukawa_kick(state, phi, step_size=0.05)

    np.testing.assert_allclose(
        np.asarray(state_norm_squared(unitary)),
        np.asarray(state_norm_squared(state)),
        atol=3e-5,
    )
    assert float(state_norm_squared(first_order) - state_norm_squared(state)) > 1e-5


@pytest.mark.slow
def test_unitary_yukawa_matches_first_order_to_first_order_in_step_size() -> None:
    state = _state()
    phi = _neutral_phi(0.75 + 0.0j)

    unitary = jax_apply_site_local_yukawa_unitary(state, phi, step_size=1e-4)
    first_order = jax_apply_site_local_yukawa_kick(state, phi, step_size=1e-4)

    np.testing.assert_allclose(np.asarray(unitary), np.asarray(first_order), atol=2e-6)


def test_yukawa_update_selector_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown Yukawa update mode"):
        jax_apply_site_local_yukawa_update(
            _state(),
            _neutral_phi(),
            step_size=0.1,
            mode="bad_mode",  # type: ignore[arg-type]
        )


@pytest.mark.slow
def test_yukawa_update_selector_accepts_eigh_oracle_mode() -> None:
    state = _state()
    phi = _mixed_phi()

    selected = jax_apply_site_local_yukawa_update(
        state,
        phi,
        step_size=0.0025,
        mode="unitary_eigh",
    )
    direct = jax_apply_site_local_yukawa_unitary_eigh(state, phi, step_size=0.0025)

    np.testing.assert_allclose(np.asarray(selected), np.asarray(direct), atol=1e-7)


@pytest.mark.slow
def test_coupled_step_accepts_unitary_yukawa_mode() -> None:
    lattice_shape = (1, 1, 1)
    state = _state()
    links = jax_identity_link_field(lattice_shape, 32, dtype=jnp.complex64)
    momenta = jnp.zeros((*lattice_shape, 8, 1), dtype=jnp.float32)
    phi = _neutral_phi()
    higgs_momentum = jnp.zeros_like(phi)
    higgs_links = jax_higgs_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, 4), dtype=jnp.float32))

    actual = jax_patisalam_fermion_gauge_higgs_step(
        state,
        links,
        momenta,
        phi,
        higgs_momentum,
        higgs_links,
        sector="u1_y",
        step_size=0.0025,
        matter_coupling=0.0,
        yukawa_mode="unitary",
        shapes=_shape(),
    )

    assert actual[0].shape == state.shape
    assert actual[1].shape == links.shape
    assert actual[2].shape == momenta.shape
    assert actual[3].shape == phi.shape
    assert actual[4].shape == higgs_momentum.shape
    assert actual[5].shape == higgs_links.shape
    for item in actual:
        assert bool(jnp.all(jnp.isfinite(item)))


@pytest.mark.slow
def test_unitary_yukawa_mode_reduces_scaling_norm_drift() -> None:
    first_order = jax_coupled_scaling_trial(
        ScalingRunConfig(step_size=0.005, matter_coupling=0.0, yukawa_mode="first_order"),
    )
    unitary = jax_coupled_scaling_trial(
        ScalingRunConfig(step_size=0.005, matter_coupling=0.0, yukawa_mode="unitary"),
    )

    assert bool(first_order.all_finite)
    assert bool(unitary.all_finite)
    assert float(jnp.abs(unitary.after.yukawa_norm_drift)) <= float(
        jnp.abs(first_order.after.yukawa_norm_drift),
    ) + 1e-5
