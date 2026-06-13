"""Tests for QCA_SMv0 three-family Higgs/Yukawa collision."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_cp import (
    DEFAULT_CENTER_HOLONOMY_POWERS,
    sm_center_coefficients,
    sm_center_cp_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    SM_FAMILY_INTERNAL_DIM,
    deterministic_sm_family_state,
    sm_apply_family_fn_quark_aux_state,
    sm_apply_family_fn_quark_door_state,
    sm_apply_family_fn_quark_path_source_kick,
    sm_apply_family_fn_quark_source_kick,
    sm_apply_family_recirculated_quark_door,
    sm_apply_family_yukawa_collision,
    sm_default_family_lepton_yukawas,
    sm_family_fn_quark_path_state_source,
    sm_family_fn_quark_state_source,
    sm_family_chirality_norms,
    sm_family_embedding_residuals,
    sm_family_higgs_yukawa_diagnostics,
    sm_family_fn_quark_path_energy_local_density,
    sm_family_fn_quark_path_higgs_force,
    sm_family_recirculated_quark_path_readouts,
    sm_family_recirculated_quark_dilations,
    sm_family_recirculated_quark_yukawas,
    sm_family_yukawa_internal_matrix,
    sm_zero_family_fn_quark_aux_state,
    sm_zero_family_fn_quark_path_state_aux,
    sm_zero_family_fn_quark_state_aux,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FN_LAMBDA_WOLFENSTEIN,
    fn_unitary_dilation_residual,
    fn_visible_recirculation_transfer,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_INTERNAL_DIM
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs, sm_yukawa_hermitian_residual
from clifford_3plus2_d5.sim.state import state_norm_squared


def test_family_state_and_yukawa_matrix_layout() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    matrix = sm_family_yukawa_internal_matrix(higgs)

    assert state.shape == (*lattice_shape, 4, SM_INTERNAL_DIM, 3)
    assert matrix.shape == (*lattice_shape, SM_FAMILY_INTERNAL_DIM, SM_FAMILY_INTERNAL_DIM)
    assert sm_yukawa_hermitian_residual(matrix) < 1e-7


def test_family_yukawa_embeds_stage6_quark_matrices_in_correct_doors() -> None:
    higgs = sm_constant_higgs((1, 1, 1))
    quark_yukawas = sm_center_cp_quark_yukawas()
    lepton_yukawas = sm_default_family_lepton_yukawas()
    quark_embedding, wrong_door, ckm_embedding = sm_family_embedding_residuals(
        higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )

    assert quark_embedding < 2e-7
    assert wrong_door < 1e-7
    assert ckm_embedding < 3e-6


def test_family_quark_source_is_fn_recirculation_readout() -> None:
    charges = DEFAULT_FN_QUARK_CHARGES
    powers = DEFAULT_CENTER_HOLONOMY_POWERS
    recirculated = sm_family_recirculated_quark_yukawas()
    center_reference = sm_center_cp_quark_yukawas()
    expected_up = fn_visible_recirculation_transfer(
        FN_LAMBDA_WOLFENSTEIN,
        charges.q,
        charges.u,
        coefficients=sm_center_coefficients("up", powers=powers.up),
    )
    expected_down = fn_visible_recirculation_transfer(
        FN_LAMBDA_WOLFENSTEIN,
        charges.q,
        charges.d,
        coefficients=sm_center_coefficients("down", powers=powers.down),
    )
    higgs = sm_constant_higgs((1, 1, 1))
    lepton_yukawas = sm_default_family_lepton_yukawas()
    quark_embedding, wrong_door, ckm_embedding = sm_family_embedding_residuals(
        higgs,
        quark_yukawas=recirculated,
        lepton_yukawas=lepton_yukawas,
    )

    assert jnp.max(jnp.abs(recirculated.up - expected_up)) < 1e-10
    assert jnp.max(jnp.abs(recirculated.down - expected_down)) < 1e-10
    assert jnp.max(jnp.abs(recirculated.up - center_reference.up)) < 1e-10
    assert jnp.max(jnp.abs(recirculated.down - center_reference.down)) < 1e-10
    assert quark_embedding < 2e-7
    assert wrong_door < 1e-7
    assert ckm_embedding < 3e-6


def test_family_quark_door_uses_finite_fn_unitary_dilation() -> None:
    higgs = sm_constant_higgs((1, 1, 1))
    h_tilde = jnp.asarray([jnp.conj(higgs[0, 0, 0, 1]), -jnp.conj(higgs[0, 0, 0, 0])], dtype=jnp.complex64)
    left_family = jnp.asarray([1.0 + 0.0j, -0.35 + 0.2j, 0.1 - 0.45j], dtype=jnp.complex64)
    batched_left = jnp.stack([left_family, 0.5j * left_family], axis=0)
    batched_h = jnp.asarray([higgs[0, 0, 0, 1], 0.3 * higgs[0, 0, 0, 1]], dtype=jnp.complex64)
    dilations = sm_family_recirculated_quark_dilations()
    quark_yukawas = sm_family_recirculated_quark_yukawas()

    up_source = sm_apply_family_recirculated_quark_door(left_family, h_tilde[0], dilations.up)
    down_source = sm_apply_family_recirculated_quark_door(left_family, higgs[0, 0, 0, 1], dilations.down)
    batched_down_source = sm_apply_family_recirculated_quark_door(batched_left, batched_h, dilations.down)
    expected_up = h_tilde[0] * (jnp.swapaxes(quark_yukawas.up, -1, -2) @ left_family)
    expected_down = higgs[0, 0, 0, 1] * (jnp.swapaxes(quark_yukawas.down, -1, -2) @ left_family)
    expected_batched_down = batched_h[:, None] * jnp.einsum(
        "ij,bj->bi",
        jnp.swapaxes(quark_yukawas.down, -1, -2),
        batched_left,
    )

    assert fn_unitary_dilation_residual(dilations.up) < 2e-6
    assert fn_unitary_dilation_residual(dilations.down) < 2e-6
    assert jnp.max(jnp.abs(up_source - expected_up)) < 2e-7
    assert jnp.max(jnp.abs(down_source - expected_down)) < 2e-7
    assert jnp.max(jnp.abs(batched_down_source - expected_batched_down)) < 2e-7


def test_family_quark_door_has_persistent_unitary_fn_aux_state() -> None:
    higgs = sm_constant_higgs((1, 1, 1))
    h_tilde = jnp.asarray([jnp.conj(higgs[0, 0, 0, 1]), -jnp.conj(higgs[0, 0, 0, 0])], dtype=jnp.complex64)
    left = jnp.asarray([1.0 + 0.0j, -0.35 + 0.2j, 0.1 - 0.45j], dtype=jnp.complex64)
    hidden = jnp.asarray([0.2 - 0.1j, -0.05 + 0.15j, 0.1 + 0.05j], dtype=jnp.complex64)
    dilations = sm_family_recirculated_quark_dilations()

    zero_aux = sm_zero_family_fn_quark_aux_state()
    zero_aux_output = sm_apply_family_fn_quark_door_state(left, zero_aux.up, h_tilde[0], dilations.up)
    stateless_output = sm_apply_family_recirculated_quark_door(left, h_tilde[0], dilations.up)
    memory_output = sm_apply_family_fn_quark_door_state(left, hidden, h_tilde[0], dilations.up)
    raw_input_norm = jnp.sum(jnp.abs(left) ** 2) + jnp.sum(jnp.abs(hidden) ** 2)
    raw_output_norm = jnp.sum(jnp.abs(memory_output.raw_visible) ** 2) + jnp.sum(jnp.abs(memory_output.hidden) ** 2)

    assert jnp.max(jnp.abs(zero_aux_output.physical_visible - stateless_output)) < 2e-7
    assert jnp.abs(raw_output_norm - raw_input_norm) < 2e-6
    assert jnp.linalg.norm(memory_output.physical_visible - stateless_output) > 1e-5


def test_family_quark_aux_state_updates_both_higgs_doors() -> None:
    higgs = sm_constant_higgs((1, 1, 1))
    h_tilde = jnp.asarray([jnp.conj(higgs[0, 0, 0, 1]), -jnp.conj(higgs[0, 0, 0, 0])], dtype=jnp.complex64)
    up_left = jnp.asarray([1.0 + 0.0j, -0.35 + 0.2j, 0.1 - 0.45j], dtype=jnp.complex64)
    down_left = jnp.asarray([-0.2 + 0.3j, 0.4 - 0.1j, 0.7 + 0.0j], dtype=jnp.complex64)
    aux = sm_zero_family_fn_quark_aux_state()
    dilations = sm_family_recirculated_quark_dilations()
    quark_yukawas = sm_family_recirculated_quark_yukawas()

    up_source, down_source, updated_aux = sm_apply_family_fn_quark_aux_state(
        up_left,
        down_left,
        h_tilde[0],
        higgs[0, 0, 0, 1],
        aux,
        dilations,
    )
    expected_up = h_tilde[0] * (jnp.swapaxes(quark_yukawas.up, -1, -2) @ up_left)
    expected_down = higgs[0, 0, 0, 1] * (jnp.swapaxes(quark_yukawas.down, -1, -2) @ down_left)

    assert updated_aux.up.shape == (3,)
    assert updated_aux.down.shape == (3,)
    assert jnp.max(jnp.abs(up_source - expected_up)) < 2e-7
    assert jnp.max(jnp.abs(down_source - expected_down)) < 2e-7
    assert jnp.linalg.norm(updated_aux.up) > 1e-5
    assert jnp.linalg.norm(updated_aux.down) > 1e-5


def test_family_fn_quark_state_source_matches_matrix_door_with_zero_aux() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    aux = sm_zero_family_fn_quark_state_aux(lattice_shape)
    source = sm_family_fn_quark_state_source(state, higgs, aux)
    quark_yukawas = sm_family_recirculated_quark_yukawas()
    h_tilde = jnp.asarray([jnp.conj(higgs[..., 1]), -jnp.conj(higgs[..., 0])], dtype=jnp.complex64)
    expected_up = jnp.zeros_like(source.up)
    expected_down = jnp.zeros_like(source.down)
    expected_state_source = jnp.zeros_like(state)

    for spin in range(4):
        for color in range(3):
            up_total = jnp.zeros((*lattice_shape, 3), dtype=state.dtype)
            down_total = jnp.zeros_like(up_total)
            for weak in range(2):
                left = state[..., spin, 2 * color + weak, :]
                up_total = up_total + h_tilde[weak][..., None] * jnp.einsum(
                    "ij,...j->...i",
                    jnp.swapaxes(quark_yukawas.up, -1, -2),
                    left,
                )
                down_total = down_total + higgs[..., weak, None] * jnp.einsum(
                    "ij,...j->...i",
                    jnp.swapaxes(quark_yukawas.down, -1, -2),
                    left,
                )
            expected_up = expected_up.at[..., spin, color, :].set(up_total)
            expected_down = expected_down.at[..., spin, color, :].set(down_total)
            expected_state_source = expected_state_source.at[..., spin, 6 + color, :].set(up_total)
            expected_state_source = expected_state_source.at[..., spin, 9 + color, :].set(down_total)

    assert source.aux_state.up.shape == (*lattice_shape, 4, 3, 2, 3)
    assert source.aux_state.down.shape == (*lattice_shape, 4, 3, 2, 3)
    assert jnp.max(jnp.abs(source.up - expected_up)) < 2e-7
    assert jnp.max(jnp.abs(source.down - expected_down)) < 2e-7
    assert jnp.max(jnp.abs(source.state_source - expected_state_source)) < 2e-7
    assert jnp.linalg.norm(source.aux_state.up) > 1e-5
    assert jnp.linalg.norm(source.aux_state.down) > 1e-5


def test_family_fn_quark_state_source_uses_hidden_memory() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    zero_aux = sm_zero_family_fn_quark_state_aux(lattice_shape)
    memory = jnp.full((*lattice_shape, 4, 3, 2, 3), 0.03 - 0.02j, dtype=jnp.complex64)
    memory_aux = type(zero_aux)(up=memory, down=-0.5j * memory)

    zero_source = sm_family_fn_quark_state_source(state, higgs, zero_aux)
    memory_source = sm_family_fn_quark_state_source(state, higgs, memory_aux)

    assert jnp.linalg.norm(memory_source.state_source - zero_source.state_source) > 1e-5
    assert jnp.linalg.norm(memory_source.aux_state.up - zero_source.aux_state.up) > 1e-5
    assert jnp.linalg.norm(memory_source.aux_state.down - zero_source.aux_state.down) > 1e-5


def test_family_fn_quark_path_source_matches_matrix_door_with_zero_aux() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    readouts = sm_family_recirculated_quark_path_readouts()
    aux = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    source = sm_family_fn_quark_path_state_source(state, higgs, aux, readouts)
    matrix_source = sm_family_fn_quark_state_source(state, higgs, sm_zero_family_fn_quark_state_aux(lattice_shape))

    assert readouts.up.network.unitary.shape == (45, 45)
    assert readouts.down.network.unitary.shape == (27, 27)
    assert source.aux_state.up.shape == (*lattice_shape, 4, 3, 2, 45)
    assert source.aux_state.down.shape == (*lattice_shape, 4, 3, 2, 27)
    assert source.state_remainder.shape == state.shape
    assert jnp.max(jnp.abs(source.up - matrix_source.up)) < 2e-7
    assert jnp.max(jnp.abs(source.down - matrix_source.down)) < 2e-7
    assert jnp.max(jnp.abs(source.state_source - matrix_source.state_source)) < 2e-7
    assert jnp.linalg.norm(source.state_remainder - state) > 1e-5
    assert jnp.linalg.norm(source.aux_state.up) > 1e-5
    assert jnp.linalg.norm(source.aux_state.down) > 1e-5


def test_family_fn_quark_path_source_uses_hidden_path_memory() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    zero_aux = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    memory_aux = type(zero_aux)(
        up=jnp.full_like(zero_aux.up, 0.005 - 0.003j),
        down=jnp.full_like(zero_aux.down, -0.004 + 0.002j),
    )

    zero_source = sm_family_fn_quark_path_state_source(state, higgs, zero_aux)
    memory_source = sm_family_fn_quark_path_state_source(state, higgs, memory_aux)

    assert jnp.linalg.norm(memory_source.state_source - zero_source.state_source) > 1e-5
    assert jnp.linalg.norm(memory_source.aux_state.up - zero_source.aux_state.up) > 1e-5
    assert jnp.linalg.norm(memory_source.aux_state.down - zero_source.aux_state.down) > 1e-5


def test_family_fn_quark_path_higgs_force_uses_recirculated_source() -> None:
    lattice_shape = (1, 1, 1)
    state = jnp.zeros((*lattice_shape, 4, SM_INTERNAL_DIM, 3), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, 0, 0].set(0.8 + 0.1j)
    state = state.at[0, 0, 0, 2, 6, 1].set(-0.35 + 0.45j)
    state = state.at[0, 0, 0, 1, 3, 1].set(0.55 - 0.2j)
    state = state.at[0, 0, 0, 3, 10, 2].set(0.4 + 0.3j)
    higgs = sm_constant_higgs(lattice_shape)
    aux = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    zero_state = jnp.zeros_like(state)

    density = sm_family_fn_quark_path_energy_local_density(state, higgs, aux)
    force = sm_family_fn_quark_path_higgs_force(state, higgs, aux)
    zero_force = sm_family_fn_quark_path_higgs_force(zero_state, higgs, aux)

    assert density.shape == lattice_shape
    assert force.shape == higgs.shape
    assert jnp.linalg.norm(density) > 1e-7
    assert jnp.linalg.norm(force) > 1e-7
    assert jnp.linalg.norm(zero_force) < 1e-8


def test_family_fn_quark_source_kick_advances_state_from_persistent_source() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    aux = sm_zero_family_fn_quark_state_aux(lattice_shape)
    step_size = 0.025

    source = sm_family_fn_quark_state_source(state, higgs, aux)
    kicked = sm_apply_family_fn_quark_source_kick(state, higgs, aux, step_size=step_size)
    beta_source = jnp.einsum(
        "sr,...rif->...sif",
        jnp.asarray(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=state.dtype,
        ),
        source.state_source,
    )
    expected = state - 1j * step_size * beta_source

    assert jnp.max(jnp.abs(kicked.source - source.state_source)) < 2e-7
    assert jnp.max(jnp.abs(kicked.state - expected)) < 2e-7
    assert jnp.max(jnp.abs(kicked.aux_state.up - source.aux_state.up)) < 2e-7
    assert jnp.max(jnp.abs(kicked.aux_state.down - source.aux_state.down)) < 2e-7


def test_family_fn_quark_source_kick_zero_step_is_identity() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    aux = sm_zero_family_fn_quark_state_aux(lattice_shape)

    kicked = sm_apply_family_fn_quark_source_kick(state, higgs, aux, step_size=0.0)

    assert jnp.max(jnp.abs(kicked.state - state)) < 1e-8
    assert jnp.linalg.norm(kicked.source) < 1e-8
    assert jnp.max(jnp.abs(kicked.aux_state.up - aux.up)) < 1e-8
    assert jnp.max(jnp.abs(kicked.aux_state.down - aux.down)) < 1e-8


def test_family_fn_quark_path_source_kick_advances_state_from_path_source() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    aux = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    step_size = 0.025

    source = sm_family_fn_quark_path_state_source(state, higgs, aux)
    kicked = sm_apply_family_fn_quark_path_source_kick(state, higgs, aux, step_size=step_size)
    beta_source = jnp.einsum(
        "sr,...rif->...sif",
        jnp.asarray(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=state.dtype,
        ),
        source.state_source,
    )
    expected = state + step_size * (source.state_remainder - state) - 1j * step_size * beta_source
    expected_up_aux = aux.up + step_size * (source.aux_state.up - aux.up)
    expected_down_aux = aux.down + step_size * (source.aux_state.down - aux.down)
    tiny = sm_apply_family_fn_quark_path_source_kick(state, higgs, aux, step_size=1e-6)

    assert jnp.max(jnp.abs(kicked.source - source.state_source)) < 2e-7
    assert jnp.max(jnp.abs(kicked.state - expected)) < 2e-7
    assert jnp.max(jnp.abs(kicked.aux_state.up - expected_up_aux)) < 2e-7
    assert jnp.max(jnp.abs(kicked.aux_state.down - expected_down_aux)) < 2e-7
    assert jnp.linalg.norm(tiny.state - state) < 5e-6
    assert jnp.linalg.norm(tiny.aux_state.up - aux.up) < 5e-6
    assert jnp.linalg.norm(tiny.aux_state.down - aux.down) < 5e-6


def test_family_yukawa_collision_identity_controls_norm_and_chirality_flip() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    zero_higgs = jnp.zeros_like(higgs)
    quark_yukawas = sm_center_cp_quark_yukawas()
    lepton_yukawas = sm_default_family_lepton_yukawas()

    zero_step = sm_apply_family_yukawa_collision(
        state,
        higgs,
        step_size=0.0,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    zero_higgs_updated = sm_apply_family_yukawa_collision(
        state,
        zero_higgs,
        step_size=0.04,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    updated = sm_apply_family_yukawa_collision(
        state,
        higgs,
        step_size=0.04,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )

    left_seed = jnp.zeros_like(state)
    left_seed = left_seed.at[0, 0, 0, 0, 0, 0].set(1.0 + 0.0j)
    flipped = sm_apply_family_yukawa_collision(
        left_seed,
        higgs,
        step_size=0.04,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    _, right_norm = sm_family_chirality_norms(flipped)

    assert jnp.max(jnp.abs(zero_step - state)) < 1e-7
    assert jnp.max(jnp.abs(zero_higgs_updated - state)) < 5e-7
    assert jnp.abs(state_norm_squared(updated) - state_norm_squared(state)) < 1e-6
    assert right_norm > 1e-9


def test_family_higgs_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_family_higgs_yukawa_diagnostics()

    assert diagnostics.family_yukawa_hermitian_residual < 1e-7
    assert diagnostics.fn_recirculated_quark_yukawa_residual < 1e-10
    assert diagnostics.fn_recirculated_embedding_residual < 2e-7
    assert diagnostics.fn_quark_dilation_unitarity_residual < 2e-6
    assert diagnostics.fn_quark_door_transfer_residual < 2e-7
    assert diagnostics.quark_embedding_residual < 2e-7
    assert diagnostics.wrong_door_residual < 1e-7
    assert diagnostics.ckm_embedding_residual < 3e-6
    assert diagnostics.zero_step_residual < 1e-7
    assert diagnostics.zero_higgs_residual < 5e-7
    assert jnp.abs(diagnostics.norm_drift) < 1e-6
    assert diagnostics.chirality_flip_right_norm > 1e-9
    assert diagnostics.jit_delta < 2e-7


def test_family_yukawa_collision_is_jittable() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    quark_yukawas = sm_center_cp_quark_yukawas()
    lepton_yukawas = sm_default_family_lepton_yukawas()
    expected = sm_apply_family_yukawa_collision(
        state,
        higgs,
        step_size=0.04,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    jitted = jax.jit(sm_apply_family_yukawa_collision, static_argnames=("step_size",))
    actual = jitted(
        state,
        higgs,
        step_size=0.04,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )

    assert jnp.max(jnp.abs(actual - expected)) < 2e-7
