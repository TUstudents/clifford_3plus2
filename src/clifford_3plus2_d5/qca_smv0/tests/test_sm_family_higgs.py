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
    sm_apply_family_yukawa_collision,
    sm_default_family_lepton_yukawas,
    sm_family_chirality_norms,
    sm_family_embedding_residuals,
    sm_family_higgs_yukawa_diagnostics,
    sm_family_recirculated_quark_yukawas,
    sm_family_yukawa_internal_matrix,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FN_LAMBDA_WOLFENSTEIN,
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
    assert jnp.abs(state_norm_squared(updated) - state_norm_squared(state)) < 5e-7
    assert right_norm > 1e-9


def test_family_higgs_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_family_higgs_yukawa_diagnostics()

    assert diagnostics.family_yukawa_hermitian_residual < 1e-7
    assert diagnostics.fn_recirculated_quark_yukawa_residual < 1e-10
    assert diagnostics.fn_recirculated_embedding_residual < 2e-7
    assert diagnostics.quark_embedding_residual < 2e-7
    assert diagnostics.wrong_door_residual < 1e-7
    assert diagnostics.ckm_embedding_residual < 3e-6
    assert diagnostics.zero_step_residual < 1e-7
    assert diagnostics.zero_higgs_residual < 5e-7
    assert jnp.abs(diagnostics.norm_drift) < 5e-7
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
