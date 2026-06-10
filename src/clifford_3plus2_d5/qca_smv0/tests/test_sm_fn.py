"""Tests for QCA_SMv0 FN recirculation paths."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FN_LAMBDA_SHEAR,
    FN_LAMBDA_WOLFENSTEIN,
    SM_FAMILY_DIM,
    fn_beam_splitter,
    fn_beam_splitter_unitarity_residual,
    fn_charge_exponents,
    fn_ckm_from_yukawas,
    fn_default_coefficients,
    fn_effective_yukawa,
    fn_apply_recirculation_collision,
    fn_path_transfer_amplitude,
    fn_path_transfer_residual,
    fn_path_unitary,
    fn_quark_yukawa_matrices,
    fn_recirculation_network,
    fn_recirculation_network_unitarity_residual,
    fn_recirculation_power_matrix,
    fn_recirculation_transfer_matrix,
    fn_singular_masses,
    fn_unitarity_residual,
    fn_unitary_dilation,
    fn_unitary_dilation_residual,
    fn_prepare_visible_collision_state,
    fn_read_visible_collision_output,
    fn_recirculation_collision_dilation,
    fn_visible_recirculation_readout,
    fn_visible_recirculation_transfer,
    fn_wolfenstein_scaling,
)


def test_fn_beam_splitter_and_path_chain_are_unitary() -> None:
    lam = FN_LAMBDA_WOLFENSTEIN
    splitter = fn_beam_splitter(lam)
    chain = fn_path_unitary(lam, 5)

    assert splitter.shape == (2, 2)
    assert fn_beam_splitter_unitarity_residual(lam) < 1e-7
    assert jnp.max(jnp.abs(chain.T @ chain - jnp.eye(6, dtype=chain.dtype))) < 1e-6


def test_fn_hidden_path_transfer_is_lambda_to_path_length() -> None:
    lam = FN_LAMBDA_WOLFENSTEIN

    for path_length in range(9):
        assert jnp.abs(fn_path_transfer_amplitude(lam, path_length) - lam**path_length) < 1e-7
    assert fn_path_transfer_residual(lam, max_path_length=8) < 1e-7


def test_fn_all_entry_recirculation_network_is_unitary_and_reads_powers() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    lam = FN_LAMBDA_WOLFENSTEIN
    network = fn_recirculation_network(lam, charges.q, charges.u)
    transfers = fn_recirculation_transfer_matrix(network)
    expected_exponents = fn_charge_exponents(charges.q, charges.u)
    expected = jnp.asarray(lam, dtype=jnp.float32) ** expected_exponents

    assert network.path_lengths.shape == (3, 3)
    assert network.source_indices.shape == (3, 3)
    assert network.sink_indices.shape == (3, 3)
    assert network.unitary.shape == (45, 45)
    assert fn_recirculation_network_unitarity_residual(network) < 1e-6
    assert jnp.max(jnp.abs(transfers - expected)) < 1e-7
    assert jnp.max(jnp.abs(fn_recirculation_power_matrix(lam, charges.q, charges.u) - expected)) < 1e-7


def test_fn_visible_hidden_visible_readout_generates_yukawa_transfer() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    lam = FN_LAMBDA_WOLFENSTEIN
    coeffs = fn_default_coefficients("down")
    readout = fn_visible_recirculation_readout(lam, charges.q, charges.d, coefficients=coeffs)
    powers = fn_recirculation_power_matrix(lam, charges.q, charges.d)
    expected = coeffs * powers.astype(jnp.complex64)

    assert readout.entry.shape[1] == 3
    assert readout.exit.shape[0] == 3
    assert readout.transfer.shape == (3, 3)
    assert jnp.max(jnp.abs(readout.transfer - expected)) < 1e-10
    assert jnp.max(jnp.abs(fn_visible_recirculation_transfer(lam, charges.q, charges.d, coefficients=coeffs) - expected)) < 1e-10


def test_fn_visible_transfer_has_exact_unitary_dilation() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    transfer = fn_visible_recirculation_transfer(
        FN_LAMBDA_WOLFENSTEIN,
        charges.q,
        charges.u,
        coefficients=fn_default_coefficients("up"),
    )
    dilation = fn_unitary_dilation(transfer)

    assert dilation.unitary.shape == (6, 6)
    assert dilation.normalization >= 1.0
    assert fn_unitary_dilation_residual(dilation) < 2e-6
    assert jnp.max(jnp.abs(dilation.normalization * dilation.transfer - transfer)) < 1e-8


def test_fn_recirculation_collision_is_unitary_and_exposes_visible_transfer() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    coeffs = fn_default_coefficients("up")
    operator = jnp.swapaxes(
        fn_visible_recirculation_transfer(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.u, coefficients=coeffs),
        -1,
        -2,
    )
    collision = fn_recirculation_collision_dilation(
        FN_LAMBDA_WOLFENSTEIN,
        charges.q,
        charges.u,
        coefficients=coeffs,
    )
    left_state = jnp.asarray([1.0 + 0.0j, -0.5 + 0.25j, 0.2 - 0.4j], dtype=jnp.complex64)
    prepared = fn_prepare_visible_collision_state(left_state)
    evolved = fn_apply_recirculation_collision(collision, prepared)
    visible = fn_read_visible_collision_output(evolved)

    assert prepared.shape == (6,)
    assert evolved.shape == (6,)
    assert fn_unitary_dilation_residual(collision) < 2e-6
    assert jnp.abs(jnp.linalg.norm(evolved) - jnp.linalg.norm(prepared)) < 1e-6
    assert jnp.max(jnp.abs(collision.normalization * visible - operator @ left_state)) < 1e-8


def test_fn_quark_charge_exponents_match_standard_assignment() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    expected_up = jnp.asarray([[8, 5, 3], [7, 4, 2], [5, 2, 0]], dtype=jnp.int32)
    expected_down = jnp.asarray([[4, 3, 3], [3, 2, 2], [1, 0, 0]], dtype=jnp.int32)

    assert jnp.array_equal(fn_charge_exponents(charges.q, charges.u), expected_up)
    assert jnp.array_equal(fn_charge_exponents(charges.q, charges.d), expected_down)


def test_fn_effective_yukawa_is_coefficients_times_recirculation_powers() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    coeffs = fn_default_coefficients("up")
    powers = fn_recirculation_power_matrix(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.u)
    y = fn_effective_yukawa(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.u, coefficients=coeffs)
    expected = coeffs * powers.astype(jnp.complex64)

    assert y.shape == (SM_FAMILY_DIM, SM_FAMILY_DIM)
    assert jnp.max(jnp.abs(y - expected)) < 1e-10


def test_fn_diagonal_scalings_and_wolfenstein_left_frame_scaling() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    identity = jnp.eye(SM_FAMILY_DIM, dtype=jnp.complex64)
    up = fn_effective_yukawa(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.u, coefficients=identity)
    down = fn_effective_yukawa(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.d, coefficients=identity)
    wolfenstein = fn_wolfenstein_scaling(charges.q, FN_LAMBDA_WOLFENSTEIN)

    assert jnp.max(jnp.abs(jnp.diag(up).real - FN_LAMBDA_WOLFENSTEIN ** jnp.array([8, 4, 0]))) < 1e-10
    assert jnp.max(jnp.abs(jnp.diag(down).real - FN_LAMBDA_WOLFENSTEIN ** jnp.array([4, 2, 0]))) < 1e-10
    assert jnp.max(jnp.abs(wolfenstein - FN_LAMBDA_WOLFENSTEIN ** jnp.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]]))) < 1e-10


def test_fn_quark_yukawas_generate_masses_and_ckm_from_same_matrices() -> None:
    yukawas = fn_quark_yukawa_matrices(lambda_rec=FN_LAMBDA_WOLFENSTEIN)
    up_masses = fn_singular_masses(yukawas.up)
    down_masses = fn_singular_masses(yukawas.down)
    ckm = fn_ckm_from_yukawas(yukawas.up, yukawas.down)

    assert yukawas.up.shape == (3, 3)
    assert yukawas.down.shape == (3, 3)
    assert jnp.all(up_masses > 0)
    assert jnp.all(down_masses > 0)
    assert up_masses[0] > up_masses[-1]
    assert down_masses[0] > down_masses[-1]
    assert fn_unitarity_residual(ckm) < 2e-6


def test_fn_module_supports_shear_candidate_lambda_without_changing_rule() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    y_shear = fn_effective_yukawa(FN_LAMBDA_SHEAR, charges.q, charges.u)
    y_wolf = fn_effective_yukawa(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.u)

    assert jnp.max(jnp.abs(y_shear - y_wolf)) > 1e-5
    assert fn_path_transfer_residual(FN_LAMBDA_SHEAR, max_path_length=8) < 1e-7


def test_fn_effective_yukawa_is_jittable() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    coeffs = fn_default_coefficients("up")
    expected = fn_effective_yukawa(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.u, coefficients=coeffs)
    jitted = jax.jit(fn_effective_yukawa, static_argnames=("left_charges", "right_charges"))
    actual = jitted(FN_LAMBDA_WOLFENSTEIN, charges.q, charges.u, coefficients=coeffs)

    assert jnp.max(jnp.abs(actual - expected)) < 1e-10
