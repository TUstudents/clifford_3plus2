"""Tests for the compact QCA_SMv0 rollout runner."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_step
from clifford_3plus2_d5.qca_smv0.scripts.benchmark_fn_dilation_rollout import (
    PRODUCTION_SCALING_PRESET_SHAPES,
    _benchmark_payload,
    _benchmark_sweep_payload,
    _apply_benchmark_preset,
    build_arg_parser as build_benchmark_arg_parser,
    _max_cubic_edge,
    _memory_budget_summary,
    _mode_comparisons,
    _memory_payload,
    _strategy_comparisons,
)
from clifford_3plus2_d5.qca_smv0.scripts.charged_lepton_probe import (
    main as charged_lepton_probe_main,
    run_charged_lepton_probe,
)
from clifford_3plus2_d5.qca_smv0.scripts.lepton_pmns_probe import (
    main as lepton_pmns_probe_main,
    run_lepton_pmns_probe,
)
from clifford_3plus2_d5.qca_smv0.scripts.neutrino_probe import (
    main as neutrino_probe_main,
    run_neutrino_probe,
)
from clifford_3plus2_d5.qca_smv0.scripts.phenomenology_rollout import (
    PhenomenologyRunConfig,
    _prepare_phenomenology_rollout,
    _mode_memory_estimate,
    config_from_args,
    main as phenomenology_main,
    run_phenomenology_rollout_modes,
    run_phenomenology_rollout,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    FamilyLeptonYukawas,
    sm_apply_family_yukawa_collision,
    sm_apply_family_yukawa_collision_from_cache,
    sm_family_quark_path_hidden_dims,
    sm_family_recirculated_quark_path_readouts,
    sm_family_yukawa_collision_cache,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import fn_ckm_from_yukawas, fn_singular_masses
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    sm_free_dirac_family_step,
    sm_free_dirac_family_split_axis_step,
    sm_free_dirac_internal_step,
    sm_free_dirac_internal_split_axis_step,
    sm_gauged_family_dirac_step,
    sm_gauged_dirac_step,
    sm_identity_links,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_lepton import (
    sm_default_pmns_matrix,
    sm_lepton_direct_pmns_neutrino_yukawa,
    sm_lepton_pmns_expected_transfer_weights,
    sm_lepton_seesaw_schur_from_pmns,
    sm_pmns_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    QCASMRolloutConfig,
    deterministic_qca_family_state,
    deterministic_qca_sm_state,
    sm_qca_center_cp_rollout_config,
    sm_qca_config_memory_footprint,
    sm_jit_qca_state_rollout,
    sm_jit_qca_production_rollout,
    sm_qca_family_carrier_basis_state,
    sm_qca_calibrated_quark_family_response,
    sm_qca_prepare_calibrated_production_rollout,
    sm_qca_prepared_quark_family_response,
    sm_qca_prepare_production_rollout,
    sm_qca_prepare_production_rollout_from_config,
    sm_qca_production_contract,
    sm_qca_rollout_memory_footprint,
    sm_qca_prepare_state,
    sm_qca_production_rollout_config,
    sm_qca_rollout_config_from_masses_ckm,
    sm_qca_state_memory_footprint,
    sm_qca_family_carrier_populations,
    sm_qca_total_norm,
    sm_qca_state_step,
    sm_qca_calibrated_coefficient_diagnostics,
    sm_jit_qca_production_rollout_observable_arrays,
    sm_run_qca_calibrated_production_rollout,
    sm_run_jitted_qca_calibrated_production_rollout_from_masses_ckm,
    sm_run_qca_calibrated_production_rollout_with_observables,
    sm_run_jitted_qca_calibrated_carrier_basis_probe,
    sm_run_jitted_qca_calibrated_production_rollout_with_observables,
    sm_run_jitted_qca_production_rollout_with_observables,
    sm_run_qca_production_rollout,
    sm_run_qca_production_rollout_with_observables,
    sm_run_qca_state_rollout,
    sm_run_qca_rollout,
)


def _benchmark_ckm() -> jnp.ndarray:
    lambda_rec = jnp.asarray(0.22501, dtype=jnp.float32)
    s23 = jnp.asarray(0.04183, dtype=jnp.float32)
    s13 = jnp.asarray(0.003732, dtype=jnp.float32)
    delta = jnp.asarray(1.147, dtype=jnp.float32)
    c12 = jnp.sqrt(1.0 - lambda_rec * lambda_rec)
    c23 = jnp.sqrt(1.0 - s23 * s23)
    c13 = jnp.sqrt(1.0 - s13 * s13)
    exp_pos = jnp.exp(1j * delta)
    exp_neg = jnp.exp(-1j * delta)
    return jnp.asarray(
        [
            [c12 * c13, lambda_rec * c13, s13 * exp_neg],
            [
                -lambda_rec * c23 - c12 * s23 * s13 * exp_pos,
                c12 * c23 - lambda_rec * s23 * s13 * exp_pos,
                s23 * c13,
            ],
            [
                lambda_rec * s23 - c12 * c23 * s13 * exp_pos,
                -c12 * s23 - lambda_rec * c23 * s13 * exp_pos,
                c23 * c13,
            ],
        ],
        dtype=jnp.complex64,
    )


def test_free_bcc_rollout_matches_dirac_step_and_records_observables() -> None:
    state = jnp.zeros((2, 1, 1, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0].set(1.0 + 0.25j)
    state = state.at[1, 0, 0, 3].set(-0.5 + 0.125j)

    result = sm_run_qca_rollout(QCASMRolloutConfig(record_density=True), state, steps=1)

    assert result.steps_completed == 1
    assert not result.used_gauge_links
    assert not result.used_higgs_fn_collision
    assert result.density_history.shape == (2, 2, 1, 1)
    assert jnp.max(jnp.abs(result.final_state - bcc_dirac_step(state))) < 2e-7
    assert jnp.max(jnp.abs(result.norm_history - result.norm_history[0])) < 2e-7


def test_identity_gauge_rollout_matches_free_internal_step() -> None:
    state = deterministic_qca_sm_state((2, 1, 1))
    links = sm_identity_links((2, 1, 1), dtype=state.dtype)

    result = sm_run_qca_rollout(QCASMRolloutConfig(links=links), state, steps=1)

    assert result.used_gauge_links
    assert jnp.max(jnp.abs(result.final_state - sm_free_dirac_internal_step(state))) < 2e-7
    assert jnp.max(jnp.abs(result.norm_history - result.norm_history[0])) < 2e-7


def test_direct_family_stream_matches_independent_family_streams() -> None:
    state = deterministic_qca_family_state((2, 1, 1))
    by_family = jnp.moveaxis(state, -1, 0)
    links = sm_identity_links((2, 1, 1), dtype=state.dtype)

    expected_free = jnp.moveaxis(jax.vmap(sm_free_dirac_internal_step)(by_family), 0, -1)
    expected_gauged = jnp.moveaxis(
        jax.vmap(lambda family_state: sm_gauged_dirac_step(family_state, links))(by_family),
        0,
        -1,
    )

    assert jnp.max(jnp.abs(sm_free_dirac_family_step(state) - expected_free)) < 2e-7
    assert jnp.max(jnp.abs(sm_free_dirac_family_split_axis_step(state) - expected_free)) < 2e-6
    assert jnp.max(jnp.abs(sm_gauged_family_dirac_step(state, links) - expected_gauged)) < 2e-7


def test_split_axis_stream_mode_matches_hop_sum_for_free_rollouts() -> None:
    lattice_shape = (2, 2, 2)
    sm_state = deterministic_qca_sm_state(lattice_shape)
    family_state = deterministic_qca_family_state(lattice_shape)

    assert (
        jnp.max(
            jnp.abs(sm_free_dirac_internal_split_axis_step(sm_state) - sm_free_dirac_internal_step(sm_state)),
        )
        < 2e-6
    )

    hop_config = QCASMRolloutConfig(stream_mode="hop_sum", record_observables=False)
    split_config = QCASMRolloutConfig(stream_mode="split_axis", record_observables=False)
    hop = sm_run_qca_rollout(hop_config, family_state, steps=2)
    split = sm_run_qca_rollout(split_config, family_state, steps=2)

    assert hop.norm_history.shape == (2,)
    assert split.norm_history.shape == (2,)
    assert jnp.max(jnp.abs(split.final_state - hop.final_state)) < 5e-6
    assert jnp.max(jnp.abs(split.norm_history - hop.norm_history)) < 5e-6


def test_center_cp_higgs_fn_rollout_defaults_to_compressed_collision() -> None:
    state = deterministic_qca_family_state((1, 1, 1))
    config = sm_qca_center_cp_rollout_config((1, 1, 1), yukawa_step_size=0.02, record_density=True)

    result = sm_run_qca_rollout(config, state, steps=1)
    stream_only = sm_run_qca_rollout(QCASMRolloutConfig(record_density=True), state, steps=1)

    assert config.collision_mode == "effective_yukawa"
    assert result.used_higgs_fn_collision
    assert not result.used_fn_dilation_collision
    assert result.final_fn_path_aux_state is None
    assert result.density_history.shape == (2, 1, 1, 1)
    assert jnp.abs(result.extended_norm_history[-1] - result.extended_norm_history[0]) < 1e-4
    assert jnp.abs(sm_qca_total_norm(result.final_state) - sm_qca_total_norm(state)) < 1e-4
    assert jnp.max(jnp.abs(result.final_state - stream_only.final_state)) > 1e-8


def test_calibrated_center_cp_rollout_config_from_masses_ckm_runs_family_rollout() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)

    calibrated = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
        center_fit_steps=0,
        yukawa_step_size=0.01,
    )
    result = sm_run_qca_rollout(calibrated.config, state, steps=1)

    assert calibrated.verdict.passed
    assert calibrated.verdict.selected_label == "wilson_flux_rule"
    assert calibrated.config.collision_mode == "effective_yukawa"
    assert result.used_higgs_fn_collision
    assert not result.used_fn_dilation_collision
    assert result.final_fn_path_aux_state is None
    assert jnp.abs(result.extended_norm_history[-1] - result.extended_norm_history[0]) < 1e-4


def test_scanned_rollout_matches_repeated_persistent_state_steps() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        record_density=True,
        collision_mode="fn_dilation",
    )

    manual = sm_qca_prepare_state(state, config)
    manual = sm_qca_state_step(manual, config)
    manual = sm_qca_state_step(manual, config)
    scanned = sm_run_qca_rollout(config, state, steps=2)

    assert scanned.final_qca_state.fn_path_aux_state is not None
    assert manual.fn_path_aux_state is not None
    assert scanned.density_history.shape == (3, 1, 1, 1)
    assert jnp.max(jnp.abs(scanned.final_qca_state.visible_state - manual.visible_state)) < 2e-7
    assert jnp.max(jnp.abs(scanned.final_qca_state.fn_path_aux_state.up - manual.fn_path_aux_state.up)) < 2e-7
    assert jnp.max(jnp.abs(scanned.final_qca_state.fn_path_aux_state.down - manual.fn_path_aux_state.down)) < 2e-7
    assert jnp.abs(scanned.extended_norm_history[-1] - scanned.extended_norm_history[0]) < 1e-4


def test_state_only_rollout_matches_diagnostic_final_state() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        collision_mode="effective_yukawa",
    )

    prepared = sm_qca_prepare_state(state, config)
    state_only = sm_run_qca_state_rollout(prepared, config, steps=3)
    diagnostic = sm_run_qca_rollout(config, state, steps=3)

    assert jnp.max(jnp.abs(state_only.visible_state - diagnostic.final_state)) < 2e-7
    assert state_only.fn_path_aux_state is None


def test_jitted_state_rollout_matches_eager_state_rollout() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        collision_mode="effective_yukawa",
    )

    prepared = sm_qca_prepare_state(state, config)
    eager = sm_run_qca_state_rollout(prepared, config, steps=2)
    jitted = sm_jit_qca_state_rollout(config, steps=2, donate_state=False)(prepared)

    assert jnp.max(jnp.abs(jitted.visible_state - eager.visible_state)) < 2e-7
    assert jitted.fn_path_aux_state is None


def test_production_rollout_api_uses_compressed_state_only_hot_path() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_production_rollout_config(lattice_shape, yukawa_step_size=0.02)
    setup = sm_qca_prepare_production_rollout(
        lattice_shape,
        initial_state=state,
        yukawa_step_size=0.02,
    )

    assert config.collision_mode == "effective_yukawa"
    assert config.stream_mode == "split_axis"
    assert config.yukawa_collision_strategy == "fast"
    assert not config.record_observables
    assert not config.record_density
    assert config.higgs is None
    assert config.quark_yukawas is None
    assert config.lepton_yukawas is None
    assert config.quark_path_readouts is None
    assert config.family_yukawa_collision_cache is not None
    assert setup.config.quark_yukawas is None
    assert setup.config.lepton_yukawas is None
    assert setup.config.quark_path_readouts is None
    assert setup.initial_qca_state.fn_path_aux_state is None
    assert setup.rollout_memory_footprint.state.fn_path_aux_complex_elements == 0
    assert setup.rollout_memory_footprint.config_array_bytes == 1152

    production = sm_run_qca_production_rollout(setup, steps=2)
    reference = sm_run_qca_state_rollout(setup.initial_qca_state, setup.config, steps=2)
    jitted = sm_jit_qca_production_rollout(setup.config, steps=2, donate_state=False)(setup.initial_qca_state)

    assert production.final_qca_state.fn_path_aux_state is None
    assert production.final_rollout_memory_footprint.state.fn_path_aux_complex_elements == 0
    assert jnp.max(jnp.abs(production.final_qca_state.visible_state - reference.visible_state)) < 2e-7
    assert jnp.max(jnp.abs(jitted.visible_state - reference.visible_state)) < 2e-7


def test_production_contract_distinguishes_lean_hot_path_from_diagnostic_config() -> None:
    lattice_shape = (1, 1, 1)
    production_config = sm_qca_production_rollout_config(lattice_shape, yukawa_step_size=0.02)
    diagnostic_config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        record_observables=False,
        record_density=False,
    )
    exact_config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        collision_mode="fn_dilation",
        record_observables=False,
        record_density=False,
    )

    production_contract = sm_qca_production_contract(production_config, uses_production_api=True)
    diagnostic_contract = sm_qca_production_contract(diagnostic_config, uses_production_api=False)

    assert production_contract.uses_production_api
    assert production_contract.state_only
    assert production_contract.structured_collision_cache_present
    assert not production_contract.raw_yukawa_arrays_present
    assert not production_contract.raw_readout_arrays_present
    assert not production_contract.higgs_field_present
    assert production_contract.lean_effective_yukawa
    assert diagnostic_contract.state_only
    assert diagnostic_contract.structured_collision_cache_present
    assert diagnostic_contract.raw_yukawa_arrays_present
    assert not diagnostic_contract.lean_effective_yukawa
    assert exact_config.lepton_yukawas is not None
    assert jnp.max(jnp.abs(exact_config.lepton_yukawas.neutrino)) < 2e-7
    assert jnp.max(jnp.abs(exact_config.lepton_yukawas.electron)) < 2e-7


def test_production_rollout_accepts_static_gauge_links_in_hop_sum_mode() -> None:
    lattice_shape = (2, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    links = sm_identity_links(lattice_shape, dtype=state.dtype)

    setup = sm_qca_prepare_production_rollout(
        lattice_shape,
        initial_state=state,
        links=links,
        yukawa_step_size=0.0,
    )
    production = sm_run_qca_production_rollout(setup, steps=1)
    contract = sm_qca_production_contract(setup.config, uses_production_api=True)

    assert setup.config.links is not None
    assert setup.config.stream_mode == "hop_sum"
    assert setup.config.family_yukawa_collision_cache is None
    assert setup.rollout_memory_footprint.config_array_bytes == links.size * links.dtype.itemsize
    assert contract.uses_production_api
    assert contract.gauge_links_present
    assert contract.lean_effective_yukawa
    assert jnp.max(jnp.abs(production.final_qca_state.visible_state - sm_free_dirac_family_step(state))) < 2e-7


def test_production_rollout_accepts_custom_lepton_yukawas_in_compressed_cache() -> None:
    lattice_shape = (1, 1, 1)
    zeros = jnp.zeros((3, 3), dtype=jnp.complex64)
    lepton_yukawas = FamilyLeptonYukawas(neutrino=zeros, electron=zeros)
    setup = sm_qca_prepare_production_rollout(
        lattice_shape,
        lepton_yukawas=lepton_yukawas,
        yukawa_step_size=0.02,
    )
    cache = setup.config.family_yukawa_collision_cache
    contract = sm_qca_production_contract(setup.config, uses_production_api=True)

    assert cache is not None
    assert setup.config.lepton_yukawas is None
    assert contract.lean_effective_yukawa
    assert not contract.raw_yukawa_arrays_present
    identity3 = jnp.eye(3, dtype=jnp.complex64)
    assert jnp.max(jnp.abs(cache.cos_left_blocks[2] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.cos_right_blocks[2] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_left_right_blocks[2])) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_right_left_blocks[2])) < 2e-7
    assert jnp.max(jnp.abs(cache.cos_left_blocks[3] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.cos_right_blocks[3] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_left_right_blocks[3])) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_right_left_blocks[3])) < 2e-7


def test_production_rollout_defaults_lepton_yukawas_to_zero_not_placeholders() -> None:
    setup = sm_qca_prepare_production_rollout((1, 1, 1), yukawa_step_size=0.02)
    cache = setup.config.family_yukawa_collision_cache

    assert cache is not None
    assert setup.config.lepton_yukawas is None
    identity3 = jnp.eye(3, dtype=jnp.complex64)
    for block_index in (2, 3):
        assert jnp.max(jnp.abs(cache.cos_left_blocks[block_index] - identity3)) < 2e-7
        assert jnp.max(jnp.abs(cache.cos_right_blocks[block_index] - identity3)) < 2e-7
        assert jnp.max(jnp.abs(cache.sin_left_right_blocks[block_index])) < 2e-7
        assert jnp.max(jnp.abs(cache.sin_right_left_blocks[block_index])) < 2e-7


def test_production_rollout_rejects_static_links_in_split_axis_mode() -> None:
    lattice_shape = (1, 1, 1)
    links = sm_identity_links(lattice_shape)

    with pytest.raises(ValueError, match="static gauge links require stream_mode='hop_sum'"):
        sm_qca_prepare_production_rollout(
            lattice_shape,
            links=links,
            stream_mode="split_axis",
        )


def test_production_rollout_fast_yukawa_strategy_matches_memory_strategy() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    memory_setup = sm_qca_prepare_production_rollout(
        lattice_shape,
        initial_state=state,
        yukawa_step_size=0.02,
        yukawa_collision_strategy="memory",
    )
    fast_setup = sm_qca_prepare_production_rollout(
        lattice_shape,
        initial_state=state,
        yukawa_step_size=0.02,
        yukawa_collision_strategy="fast",
    )

    memory = sm_run_qca_production_rollout(memory_setup, steps=2)
    fast = sm_run_qca_production_rollout(fast_setup, steps=2)

    assert fast_setup.config.yukawa_collision_strategy == "fast"
    assert fast_setup.config.family_yukawa_collision_cache is not None
    assert fast_setup.initial_qca_state.fn_path_aux_state is None
    assert fast.final_qca_state.fn_path_aux_state is None
    assert jnp.max(jnp.abs(fast.final_qca_state.visible_state - memory.final_qca_state.visible_state)) < 2e-7


def test_production_hot_path_rejects_diagnostic_recording_configs() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_production_rollout_config(lattice_shape, yukawa_step_size=0.02)

    for diagnostic_config in (
        config._replace(record_observables=True),
        config._replace(record_density=True),
    ):
        try:
            sm_qca_prepare_production_rollout_from_config(state, diagnostic_config)
        except ValueError as exc:
            assert "state-only output" in str(exc)
        else:  # pragma: no cover - defensive assertion shape for old pytest helpers.
            raise AssertionError("expected production setup to reject diagnostic recording")

        try:
            sm_jit_qca_production_rollout(diagnostic_config, steps=1, donate_state=False)
        except ValueError as exc:
            assert "state-only output" in str(exc)
        else:  # pragma: no cover - defensive assertion shape for old pytest helpers.
            raise AssertionError("expected production JIT helper to reject diagnostic recording")


def test_production_rollout_observables_helper_matches_state_rollout() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_production_rollout_config(lattice_shape, yukawa_step_size=0.02)
    setup = sm_qca_prepare_production_rollout_from_config(state, config)

    observed = sm_run_qca_production_rollout_with_observables(setup, steps=2)
    reference = sm_run_qca_production_rollout(setup, steps=2)

    assert observed.collision_mode == "effective_yukawa"
    assert observed.steps_completed == 2
    assert observed.used_higgs_fn_collision
    assert not observed.used_fn_dilation_collision
    assert observed.production.final_rollout_memory_footprint == reference.final_rollout_memory_footprint
    assert jnp.max(jnp.abs(observed.production.final_qca_state.visible_state - reference.final_qca_state.visible_state)) < 2e-7
    assert jnp.abs(observed.norm_initial - sm_qca_total_norm(state)) < 2e-7
    assert jnp.abs(observed.norm_final - sm_qca_total_norm(reference.final_qca_state.visible_state)) < 2e-7
    assert jnp.abs(jnp.sum(observed.carrier_populations_initial.sector) - observed.norm_initial) < 2e-7
    assert jnp.abs(jnp.sum(observed.carrier_populations_final.sector) - observed.norm_final) < 2e-7
    assert jnp.abs(jnp.sum(observed.carrier_populations_initial.dirac_chirality) - observed.norm_initial) < 5e-7
    assert jnp.abs(jnp.sum(observed.carrier_populations_final.family) - observed.norm_final) < 5e-7
    assert observed.carrier_populations_initial.sector.shape == (6,)
    assert observed.carrier_populations_initial.dirac_chirality.shape == (2,)
    assert observed.carrier_populations_initial.family.shape == (3,)
    assert observed.carrier_populations_initial.internal_copy.shape == (2,)
    assert observed.production.final_qca_state.fn_path_aux_state is None


def test_jitted_production_observables_match_eager_observables() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_production_rollout_config(lattice_shape, yukawa_step_size=0.02)
    setup = sm_qca_prepare_production_rollout_from_config(state, config)

    eager = sm_run_qca_production_rollout_with_observables(setup, steps=2)
    jitted = sm_run_jitted_qca_production_rollout_with_observables(setup, steps=2, donate_state=False)
    arrays = sm_jit_qca_production_rollout_observable_arrays(
        setup.config,
        steps=2,
        donate_state=False,
    )(setup.initial_qca_state)

    assert jitted.collision_mode == "effective_yukawa"
    assert jitted.used_higgs_fn_collision
    assert not jitted.used_fn_dilation_collision
    assert jnp.max(jnp.abs(jitted.production.final_qca_state.visible_state - eager.production.final_qca_state.visible_state)) < 2e-7
    assert jnp.max(jnp.abs(arrays.final_qca_state.visible_state - eager.production.final_qca_state.visible_state)) < 2e-7
    assert jnp.abs(jitted.norm_initial - eager.norm_initial) < 2e-7
    assert jnp.abs(jitted.norm_final - eager.norm_final) < 2e-7
    assert jnp.abs(arrays.extended_norm_final - eager.extended_norm_final) < 2e-7
    assert jnp.max(jnp.abs(jitted.carrier_populations_final.sector - eager.carrier_populations_final.sector)) < 2e-7
    assert jnp.max(jnp.abs(arrays.carrier_populations_final.family - eager.carrier_populations_final.family)) < 2e-7


def test_family_carrier_populations_partition_the_visible_fibre() -> None:
    state = deterministic_qca_family_state((1, 1, 1))
    populations = sm_qca_family_carrier_populations(state)
    total = sm_qca_total_norm(state)

    assert jnp.abs(jnp.sum(populations.sector) - total) < 2e-7
    assert jnp.abs(jnp.sum(populations.dirac_chirality) - total) < 5e-7
    assert jnp.abs(jnp.sum(populations.family) - total) < 5e-7
    assert jnp.abs(jnp.sum(populations.internal_copy) - total) < 5e-7


def test_family_carrier_basis_state_selects_physical_sector() -> None:
    state = sm_qca_family_carrier_basis_state(
        (2, 1, 1),
        site=(1, 0, 0),
        dirac=2,
        sector="u_c",
        color=1,
        family=2,
        internal_copy=1,
        amplitude=1.0,
    )
    populations = sm_qca_family_carrier_populations(state)

    assert state.shape == (2, 1, 1, 4, 32, 3)
    assert jnp.abs(sm_qca_total_norm(state) - 1.0) < 2e-7
    assert jnp.allclose(populations.sector, jnp.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert jnp.allclose(populations.dirac_chirality, jnp.asarray([0.0, 1.0]))
    assert jnp.allclose(populations.family, jnp.asarray([0.0, 0.0, 1.0]))
    assert jnp.allclose(populations.internal_copy, jnp.asarray([0.0, 1.0]))


def test_family_carrier_basis_state_runs_through_production_hot_path() -> None:
    state = sm_qca_family_carrier_basis_state(
        (1, 1, 1),
        dirac=0,
        sector="Q",
        color=0,
        weak=0,
        family=0,
    )
    setup = sm_qca_prepare_production_rollout(
        (1, 1, 1),
        initial_state=state,
        yukawa_step_size=0.0,
        stream_mode="split_axis",
    )
    observed = sm_run_qca_production_rollout_with_observables(setup, steps=1)

    assert observed.production.final_qca_state.visible_state.shape == state.shape
    assert jnp.abs(observed.norm_initial - 1.0) < 2e-7
    assert jnp.abs(observed.norm_final - 1.0) < 2e-7
    assert jnp.abs(jnp.sum(observed.carrier_populations_final.sector) - observed.norm_final) < 2e-7


def test_family_carrier_basis_state_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="sector must be one of"):
        sm_qca_family_carrier_basis_state((1, 1, 1), sector="top")
    with pytest.raises(ValueError, match="family must be in"):
        sm_qca_family_carrier_basis_state((1, 1, 1), family=3)


def test_calibrated_production_rollout_api_goes_from_masses_ckm_to_hot_path() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)
    ckm = _benchmark_ckm()

    calibrated = sm_qca_prepare_calibrated_production_rollout(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        initial_state=state,
        center_fit_steps=0,
        yukawa_step_size=0.01,
    )
    lower_level = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        center_fit_steps=0,
        yukawa_step_size=0.01,
        record_observables=False,
        record_density=False,
        collision_mode="effective_yukawa",
    )

    assert calibrated.verdict.passed
    assert calibrated.verdict.selected_label == "wilson_flux_rule"
    assert calibrated.setup.config.collision_mode == "effective_yukawa"
    assert calibrated.setup.config.stream_mode == "split_axis"
    assert calibrated.setup.config.yukawa_collision_strategy == "fast"
    assert not calibrated.setup.config.record_observables
    assert not calibrated.setup.config.record_density
    assert calibrated.setup.config.higgs is None
    assert calibrated.setup.config.quark_yukawas is None
    assert calibrated.setup.config.lepton_yukawas is None
    assert calibrated.setup.config.quark_path_readouts is None
    assert calibrated.setup.config.family_yukawa_collision_cache is not None
    assert calibrated.setup.initial_qca_state.fn_path_aux_state is None
    assert calibrated.setup.rollout_memory_footprint.state.fn_path_aux_complex_elements == 0
    assert lower_level.config.quark_yukawas is not None
    assert lower_level.config.family_yukawa_collision_cache is not None
    assert (
        jnp.max(
            jnp.abs(
                calibrated.setup.config.family_yukawa_collision_cache.cos_left_blocks
                - lower_level.config.family_yukawa_collision_cache.cos_left_blocks,
            ),
        )
        < 2e-7
    )
    assert (
        jnp.max(
            jnp.abs(
                calibrated.setup.config.family_yukawa_collision_cache.sin_left_right_blocks
                - lower_level.config.family_yukawa_collision_cache.sin_left_right_blocks,
            ),
        )
        < 2e-7
    )

    production = sm_run_qca_calibrated_production_rollout(calibrated, steps=2)
    observed = sm_run_qca_calibrated_production_rollout_with_observables(calibrated, steps=2)
    jitted_observed = sm_run_jitted_qca_calibrated_production_rollout_with_observables(
        calibrated,
        steps=2,
        donate_state=False,
    )
    direct_jitted_observed = sm_run_jitted_qca_calibrated_production_rollout_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        steps=2,
        initial_state=state,
        center_fit_steps=0,
        yukawa_step_size=0.01,
        donate_state=False,
    )
    reference = sm_run_qca_state_rollout(calibrated.setup.initial_qca_state, calibrated.setup.config, steps=2)

    assert production.verdict.passed
    assert production.production.final_qca_state.fn_path_aux_state is None
    assert jnp.max(jnp.abs(production.production.final_qca_state.visible_state - reference.visible_state)) < 2e-7
    assert observed.verdict.passed
    assert jitted_observed.verdict.passed
    assert direct_jitted_observed.verdict.passed
    assert observed.coefficient_diagnostics.passed
    assert jitted_observed.coefficient_diagnostics.passed
    assert direct_jitted_observed.coefficient_diagnostics.passed
    assert direct_jitted_observed.coefficient_diagnostics.selected_label == "wilson_flux_rule"
    assert direct_jitted_observed.coefficient_diagnostics.up_magnitudes.shape == (3, 3)
    assert direct_jitted_observed.coefficient_diagnostics.down_magnitudes.shape == (3, 3)
    assert direct_jitted_observed.coefficient_diagnostics.up_center_powers.shape == (3, 3)
    assert direct_jitted_observed.coefficient_diagnostics.down_center_powers.shape == (3, 3)
    assert direct_jitted_observed.coefficient_diagnostics.magnitude_min > 0.0
    assert direct_jitted_observed.coefficient_diagnostics.magnitude_max < 10.0
    assert direct_jitted_observed.coefficient_diagnostics.steps_completed == 0
    assert (
        jnp.max(
            jnp.abs(
                direct_jitted_observed.coefficient_diagnostics.up_magnitudes
                - sm_qca_calibrated_coefficient_diagnostics(direct_jitted_observed.verdict).up_magnitudes,
            ),
        )
        < 2e-7
    )
    assert direct_jitted_observed.production_contract.uses_production_api
    assert direct_jitted_observed.production_contract.state_only
    assert direct_jitted_observed.production_contract.structured_collision_cache_present
    assert not direct_jitted_observed.production_contract.raw_yukawa_arrays_present
    assert not direct_jitted_observed.production_contract.raw_readout_arrays_present
    assert not direct_jitted_observed.production_contract.higgs_field_present
    assert direct_jitted_observed.production_contract.lean_effective_yukawa
    assert observed.production_observables.collision_mode == "effective_yukawa"
    assert observed.production_observables.steps_completed == 2
    assert observed.production_observables.production.final_qca_state.fn_path_aux_state is None
    assert jitted_observed.production_observables.production.final_qca_state.fn_path_aux_state is None
    assert direct_jitted_observed.production_observables.production.final_qca_state.fn_path_aux_state is None
    assert (
        jnp.max(
            jnp.abs(
                observed.production_observables.production.final_qca_state.visible_state
                - production.production.final_qca_state.visible_state,
            ),
        )
        < 2e-7
    )
    assert (
        jnp.max(
            jnp.abs(
                jitted_observed.production_observables.production.final_qca_state.visible_state
                - production.production.final_qca_state.visible_state,
            ),
        )
        < 2e-7
    )
    assert jnp.abs(observed.production_observables.norm_final - sm_qca_total_norm(reference.visible_state)) < 2e-7
    assert (
        jnp.abs(
            jitted_observed.production_observables.extended_norm_final
            - observed.production_observables.extended_norm_final,
        )
        < 2e-7
    )
    assert (
        jnp.max(
            jnp.abs(
                direct_jitted_observed.production_observables.production.final_qca_state.visible_state
                - jitted_observed.production_observables.production.final_qca_state.visible_state,
            ),
        )
        < 2e-7
    )
    assert (
        jnp.abs(
            direct_jitted_observed.production_observables.norm_final
            - jitted_observed.production_observables.norm_final,
        )
        < 2e-7
    )


def test_calibrated_carrier_basis_probe_matches_manual_hot_path_and_shows_up_response() -> None:
    lattice_shape = (1, 1, 1)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)
    ckm = _benchmark_ckm()
    initial_state = sm_qca_family_carrier_basis_state(
        lattice_shape,
        dirac=0,
        sector="Q",
        color=0,
        weak=0,
        family=0,
    )

    probe = sm_run_jitted_qca_calibrated_carrier_basis_probe(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        steps=1,
        dirac=0,
        sector="Q",
        color=0,
        weak=0,
        family=0,
        center_fit_steps=0,
        yukawa_step_size=0.05,
        donate_state=False,
    )
    manual = sm_run_jitted_qca_calibrated_production_rollout_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        steps=1,
        initial_state=initial_state,
        center_fit_steps=0,
        yukawa_step_size=0.05,
        donate_state=False,
    )

    probe_observed = probe.production_observables
    manual_observed = manual.production_observables
    assert probe.verdict.passed
    assert jnp.max(
        jnp.abs(
            probe_observed.production.final_qca_state.visible_state
            - manual_observed.production.final_qca_state.visible_state,
        ),
    ) < 2e-7
    assert jnp.allclose(
        probe_observed.carrier_populations_initial.sector,
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert probe_observed.carrier_populations_final.sector[1] > 1e-8
    assert jnp.abs(jnp.sum(probe_observed.carrier_populations_final.sector) - probe_observed.norm_final) < 2e-7


def test_lepton_carrier_probe_routes_charged_lepton_door_to_e_c() -> None:
    lattice_shape = (1, 1, 1)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)
    zeros = jnp.zeros((3, 3), dtype=jnp.complex64)
    electron = jnp.diag(jnp.asarray([0.10, 0.20, 0.50], dtype=jnp.complex64))

    probe = sm_run_jitted_qca_calibrated_carrier_basis_probe(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
        steps=1,
        dirac=0,
        sector="L",
        weak=1,
        family=2,
        lepton_yukawas=FamilyLeptonYukawas(neutrino=zeros, electron=electron),
        center_fit_steps=0,
        yukawa_step_size=0.10,
        donate_state=False,
    )
    observed = probe.production_observables

    assert probe.verdict.passed
    assert observed.production.final_qca_state.fn_path_aux_state is None
    assert probe.production_contract.uses_production_api
    assert probe.production_contract.lean_effective_yukawa
    assert probe.production_contract.structured_collision_cache_present
    assert not probe.production_contract.raw_yukawa_arrays_present
    assert jnp.allclose(
        observed.carrier_populations_initial.sector,
        jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    assert observed.carrier_populations_final.sector[4] > 1e-4
    assert observed.carrier_populations_final.sector[5] < 1e-9
    assert observed.carrier_populations_final.sector[0] < 1e-9
    assert jnp.abs(jnp.sum(observed.carrier_populations_final.sector) - observed.norm_final) < 3e-7
    assert jnp.abs(observed.extended_norm_final - observed.extended_norm_initial) < 3e-7


def test_lepton_carrier_probe_routes_neutrino_door_to_nu_c() -> None:
    lattice_shape = (1, 1, 1)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)
    zeros = jnp.zeros((3, 3), dtype=jnp.complex64)
    neutrino = jnp.diag(jnp.asarray([0.10, 0.20, 0.50], dtype=jnp.complex64))

    probe = sm_run_jitted_qca_calibrated_carrier_basis_probe(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
        steps=1,
        dirac=0,
        sector="L",
        weak=0,
        family=2,
        lepton_yukawas=FamilyLeptonYukawas(neutrino=neutrino, electron=zeros),
        center_fit_steps=0,
        yukawa_step_size=0.10,
        donate_state=False,
    )
    observed = probe.production_observables

    assert probe.verdict.passed
    assert observed.production.final_qca_state.fn_path_aux_state is None
    assert probe.production_contract.uses_production_api
    assert probe.production_contract.lean_effective_yukawa
    assert probe.production_contract.structured_collision_cache_present
    assert not probe.production_contract.raw_yukawa_arrays_present
    assert jnp.allclose(
        observed.carrier_populations_initial.sector,
        jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    assert observed.carrier_populations_final.sector[5] > 1e-4
    assert observed.carrier_populations_final.sector[4] < 1e-9
    assert observed.carrier_populations_final.sector[0] < 1e-9
    assert jnp.abs(jnp.sum(observed.carrier_populations_final.sector) - observed.norm_final) < 3e-7
    assert jnp.abs(observed.extended_norm_final - observed.extended_norm_initial) < 3e-7


def test_charged_lepton_probe_payload_passes_physics_tests() -> None:
    payload = run_charged_lepton_probe()
    e_populations = [probe["e_c_population"] for probe in payload["probes"]]

    assert payload["physics_tests"]["passed"]
    assert all(payload["physics_tests"]["checks"].values())
    assert payload["probe_source"]["sector"] == "L"
    assert payload["probe_source"]["weak"] == 1
    assert payload["probe_source"]["target"] == "e_c"
    assert payload["probe_source"]["wrong_target"] == "nu_c"
    assert e_populations[2] > e_populations[1] > e_populations[0]
    assert all(probe["nu_c_population"] < payload["physics_tests"]["thresholds"]["wrong_sector_max"] for probe in payload["probes"])
    assert payload["production_contract"]["uses_production_api"]
    assert payload["production_contract"]["lean_effective_yukawa"]
    assert payload["production_contract"]["structured_collision_cache_present"]
    assert not payload["production_contract"]["raw_yukawa_arrays_present"]


def test_charged_lepton_probe_cli_writes_json_and_report(tmp_path, capsys) -> None:
    json_path = tmp_path / "charged_lepton_probe.json"
    report_path = tmp_path / "charged_lepton_probe.md"

    charged_lepton_probe_main(
        [
            "--output",
            "json",
            "--json-output-path",
            str(json_path),
            "--report-output-path",
            str(report_path),
        ],
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert stdout_payload == saved_payload
    assert saved_payload["physics_tests"]["passed"]
    assert all(saved_payload["physics_tests"]["checks"].values())
    assert saved_payload["lepton_inputs"]["mode"] == "diagonal_charged_lepton_probe"
    assert saved_payload["probes"][2]["e_c_population"] > saved_payload["probes"][1]["e_c_population"]
    assert saved_payload["probes"][1]["e_c_population"] > saved_payload["probes"][0]["e_c_population"]
    assert "QCA_SMv0 Charged-Lepton Carrier Probe" in report
    assert "overall passed: `True`" in report
    assert "L(weak=1)" in report


def test_canonical_charged_lepton_probe_artifact_passes_physics_tests() -> None:
    payload = json.loads(
        Path("src/clifford_3plus2_d5/qca_smv0/canonical/charged_lepton_probe.json").read_text(
            encoding="utf-8",
        ),
    )

    assert payload["physics_tests"]["passed"]
    assert all(payload["physics_tests"]["checks"].values())
    assert payload["production_contract"]["uses_production_api"]
    assert payload["production_contract"]["lean_effective_yukawa"]
    assert payload["production_contract"]["structured_collision_cache_present"]
    assert not payload["production_contract"]["raw_yukawa_arrays_present"]
    assert payload["lepton_inputs"]["neutrino_yukawas"] == [0.0, 0.0, 0.0]
    assert payload["probes"][2]["e_c_population"] > payload["probes"][1]["e_c_population"]
    assert payload["probes"][1]["e_c_population"] > payload["probes"][0]["e_c_population"]
    assert max(probe["nu_c_population"] for probe in payload["probes"]) < payload["physics_tests"]["thresholds"]["wrong_sector_max"]


def test_neutrino_probe_payload_passes_physics_tests() -> None:
    payload = run_neutrino_probe()
    nu_populations = [probe["nu_c_population"] for probe in payload["probes"]]

    assert payload["physics_tests"]["passed"]
    assert all(payload["physics_tests"]["checks"].values())
    assert payload["probe_source"]["sector"] == "L"
    assert payload["probe_source"]["weak"] == 0
    assert payload["probe_source"]["target"] == "nu_c"
    assert payload["probe_source"]["wrong_target"] == "e_c"
    assert payload["lepton_inputs"]["mode"] == "direct_diagonal_effective_neutrino_yukawa_probe"
    assert payload["lepton_inputs"]["later_mode"] == "seesaw_schur_input_mode"
    assert payload["lepton_inputs"]["charged_lepton_yukawas"] == [0.0, 0.0, 0.0]
    assert payload["neutrino_metadata"]["hierarchy"] == "normal"
    assert payload["neutrino_metadata"]["lightest_massless"]
    assert payload["neutrino_metadata"]["m1"] == 0.0
    assert payload["neutrino_metadata"]["ratio_error"] < payload["physics_tests"]["thresholds"]["ratio_tolerance"]
    assert nu_populations[0] < payload["physics_tests"]["thresholds"]["wrong_sector_max"]
    assert nu_populations[1] > payload["physics_tests"]["thresholds"]["transfer_min"]
    assert nu_populations[2] > nu_populations[1]
    assert all(probe["e_c_population"] < payload["physics_tests"]["thresholds"]["wrong_sector_max"] for probe in payload["probes"])
    assert payload["production_contract"]["uses_production_api"]
    assert payload["production_contract"]["lean_effective_yukawa"]
    assert payload["production_contract"]["structured_collision_cache_present"]
    assert not payload["production_contract"]["raw_yukawa_arrays_present"]


def test_neutrino_probe_cli_writes_json_and_report(tmp_path, capsys) -> None:
    json_path = tmp_path / "neutrino_probe.json"
    report_path = tmp_path / "neutrino_probe.md"

    neutrino_probe_main(
        [
            "--output",
            "json",
            "--json-output-path",
            str(json_path),
            "--report-output-path",
            str(report_path),
        ],
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert stdout_payload == saved_payload
    assert saved_payload["physics_tests"]["passed"]
    assert all(saved_payload["physics_tests"]["checks"].values())
    assert saved_payload["lepton_inputs"]["mode"] == "direct_diagonal_effective_neutrino_yukawa_probe"
    assert saved_payload["neutrino_metadata"]["hierarchy"] == "normal"
    assert saved_payload["neutrino_metadata"]["lightest_massless"]
    assert saved_payload["probes"][2]["nu_c_population"] > saved_payload["probes"][1]["nu_c_population"]
    assert saved_payload["probes"][1]["nu_c_population"] > saved_payload["physics_tests"]["thresholds"]["transfer_min"]
    assert "QCA_SMv0 Neutrino Carrier Probe" in report
    assert "overall passed: `True`" in report
    assert "L(weak=0)" in report
    assert "epsilon^4" in report


def test_canonical_neutrino_probe_artifact_passes_physics_tests() -> None:
    payload = json.loads(
        Path("src/clifford_3plus2_d5/qca_smv0/canonical/neutrino_probe.json").read_text(
            encoding="utf-8",
        ),
    )

    assert payload["physics_tests"]["passed"]
    assert all(payload["physics_tests"]["checks"].values())
    assert payload["production_contract"]["uses_production_api"]
    assert payload["production_contract"]["lean_effective_yukawa"]
    assert payload["production_contract"]["structured_collision_cache_present"]
    assert not payload["production_contract"]["raw_yukawa_arrays_present"]
    assert payload["lepton_inputs"]["charged_lepton_yukawas"] == [0.0, 0.0, 0.0]
    assert payload["neutrino_metadata"]["hierarchy"] == "normal"
    assert payload["neutrino_metadata"]["lightest_massless"]
    assert payload["neutrino_metadata"]["ratio_error"] < payload["physics_tests"]["thresholds"]["ratio_tolerance"]
    assert payload["probes"][2]["nu_c_population"] > payload["probes"][1]["nu_c_population"]
    assert payload["probes"][1]["nu_c_population"] > payload["physics_tests"]["thresholds"]["transfer_min"]
    assert max(probe["e_c_population"] for probe in payload["probes"]) < payload["physics_tests"]["thresholds"]["wrong_sector_max"]


def test_pmns_direct_yukawa_and_schur_backend_recover_neutrino_spectrum() -> None:
    masses = jnp.asarray([0.0, 0.17157287525381, 1.0], dtype=jnp.float32)
    pmns = sm_default_pmns_matrix()
    direct = sm_lepton_direct_pmns_neutrino_yukawa(masses, pmns)
    weights = sm_lepton_pmns_expected_transfer_weights(masses, pmns)
    schur = sm_lepton_seesaw_schur_from_pmns(masses, pmns)

    assert float(sm_pmns_unitarity_residual(pmns)) < 1.0e-6
    assert direct.shape == (3, 3)
    assert weights.shape == (3, 3)
    assert jnp.allclose(jnp.sum(weights, axis=1), jnp.ones((3,), dtype=jnp.float32), atol=1.0e-6)
    assert float(jnp.max(jnp.abs(jnp.linalg.svd(direct, compute_uv=False) - jnp.asarray([1.0, 0.17157287525381, 0.0])))) < 1.0e-6
    assert float(schur.spectrum_residual) < 1.0e-6
    assert jnp.allclose(schur.recovered_masses, jnp.sort(masses), atol=1.0e-6)
    assert jnp.allclose(schur.effective_majorana, schur.effective_majorana.T, atol=1.0e-6)


def test_lepton_pmns_probe_payload_passes_physics_tests() -> None:
    payload = run_lepton_pmns_probe()

    assert payload["physics_tests"]["passed"]
    assert all(payload["physics_tests"]["checks"].values())
    assert payload["lepton_inputs"]["mode"] == "direct_pmns_effective_neutrino_yukawa_probe"
    assert payload["lepton_inputs"]["schur_backend"] == "type_i_seesaw_schur"
    assert payload["probe_source"]["sector"] == "L"
    assert payload["probe_source"]["weak"] == 0
    assert payload["probe_source"]["target"] == "nu_c"
    assert payload["probe_source"]["wrong_target"] == "e_c"
    assert payload["neutrino_metadata"]["m1"] == 0.0
    assert payload["neutrino_metadata"]["ratio_error"] < payload["physics_tests"]["thresholds"]["ratio_tolerance"]
    assert payload["schur_seesaw"]["spectrum_residual"] < payload["physics_tests"]["thresholds"]["schur_spectrum_tolerance"]
    assert jnp.allclose(jnp.asarray(payload["schur_seesaw"]["target_masses"]), jnp.asarray([0.0, 0.17157287525381, 1.0]))
    assert max(probe["pmns_weight_max_error"] for probe in payload["probes"]) < payload["physics_tests"]["thresholds"]["pmns_weight_tolerance"]
    assert all(probe["nu_c_population"] > payload["physics_tests"]["thresholds"]["transfer_min"] for probe in payload["probes"])
    assert all(probe["e_c_population"] < payload["physics_tests"]["thresholds"]["wrong_sector_max"] for probe in payload["probes"])
    assert any(len([value for value in probe["normalized_nu_c_family_populations"] if value > 1.0e-2]) > 1 for probe in payload["probes"])
    assert payload["production_contract"]["uses_production_api"]
    assert payload["production_contract"]["lean_effective_yukawa"]
    assert payload["production_contract"]["structured_collision_cache_present"]
    assert not payload["production_contract"]["raw_yukawa_arrays_present"]


def test_lepton_pmns_probe_cli_writes_json_and_report(tmp_path, capsys) -> None:
    json_path = tmp_path / "lepton_pmns_probe.json"
    report_path = tmp_path / "lepton_pmns_probe.md"

    lepton_pmns_probe_main(
        [
            "--output",
            "json",
            "--json-output-path",
            str(json_path),
            "--report-output-path",
            str(report_path),
        ],
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert stdout_payload == saved_payload
    assert saved_payload["physics_tests"]["passed"]
    assert all(saved_payload["physics_tests"]["checks"].values())
    assert saved_payload["lepton_inputs"]["mode"] == "direct_pmns_effective_neutrino_yukawa_probe"
    assert saved_payload["lepton_inputs"]["schur_backend"] == "type_i_seesaw_schur"
    assert saved_payload["schur_seesaw"]["spectrum_residual"] < saved_payload["physics_tests"]["thresholds"]["schur_spectrum_tolerance"]
    assert "QCA_SMv0 Lepton PMNS Probe" in report
    assert "overall passed: `True`" in report
    assert "Schur recovered masses" in report


def test_canonical_lepton_pmns_probe_artifact_passes_physics_tests() -> None:
    payload = json.loads(
        Path("src/clifford_3plus2_d5/qca_smv0/canonical/lepton_pmns_probe.json").read_text(
            encoding="utf-8",
        ),
    )

    assert payload["physics_tests"]["passed"]
    assert all(payload["physics_tests"]["checks"].values())
    assert payload["lepton_inputs"]["mode"] == "direct_pmns_effective_neutrino_yukawa_probe"
    assert payload["lepton_inputs"]["schur_backend"] == "type_i_seesaw_schur"
    assert payload["schur_seesaw"]["spectrum_residual"] < payload["physics_tests"]["thresholds"]["schur_spectrum_tolerance"]
    assert payload["neutrino_metadata"]["m1"] == 0.0
    assert payload["neutrino_metadata"]["ratio_error"] < payload["physics_tests"]["thresholds"]["ratio_tolerance"]
    assert max(probe["pmns_weight_max_error"] for probe in payload["probes"]) < payload["physics_tests"]["thresholds"]["pmns_weight_tolerance"]
    assert all(probe["nu_c_population"] > payload["physics_tests"]["thresholds"]["transfer_min"] for probe in payload["probes"])
    assert max(probe["e_c_population"] for probe in payload["probes"]) < payload["physics_tests"]["thresholds"]["wrong_sector_max"]
    assert payload["production_contract"]["uses_production_api"]
    assert payload["production_contract"]["lean_effective_yukawa"]


def test_calibrated_quark_family_response_measures_field_level_up_and_down_doors() -> None:
    lattice_shape = (1, 1, 1)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)

    response = sm_qca_calibrated_quark_family_response(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
        steps=1,
        center_fit_steps=0,
        yukawa_step_size=0.05,
    )

    assert response.verdict.passed
    assert response.steps_completed == 1
    assert response.up_amplitude.shape == (3, 3)
    assert response.down_amplitude.shape == (3, 3)
    assert response.up_expected_one_tick_amplitude.shape == (3, 3)
    assert response.down_expected_one_tick_amplitude.shape == (3, 3)
    assert response.up_population.shape == (3, 3)
    assert response.down_population.shape == (3, 3)
    assert jnp.max(jnp.abs(response.up_amplitude)) > 1e-6
    assert jnp.max(jnp.abs(response.down_amplitude)) > 1e-6
    assert jnp.max(jnp.abs(response.up_expected_one_tick_amplitude)) > 1e-6
    assert jnp.max(jnp.abs(response.down_expected_one_tick_amplitude)) > 1e-6
    assert response.up_expected_one_tick_residual < 5e-7
    assert response.down_expected_one_tick_residual < 5e-7
    assert jnp.max(response.up_population) > 1e-8
    assert jnp.max(response.down_population) > 1e-8
    assert jnp.all(jnp.abs(response.up_amplitude) ** 2 <= response.up_population + 1e-6)
    assert jnp.all(jnp.abs(response.down_amplitude) ** 2 <= response.down_population + 1e-6)
    assert jnp.max(jnp.abs(response.up_final_norm - 1.0)) < 2e-6
    assert jnp.max(jnp.abs(response.down_final_norm - 1.0)) < 2e-6


def test_prepared_quark_family_response_reuses_calibrated_production_setup() -> None:
    lattice_shape = (1, 1, 1)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)
    ckm = _benchmark_ckm()
    prepared = sm_qca_prepare_calibrated_production_rollout(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        center_fit_steps=0,
        yukawa_step_size=0.05,
    )
    from_prepared = sm_qca_prepared_quark_family_response(prepared, steps=1)
    from_wrapper = sm_qca_calibrated_quark_family_response(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        steps=1,
        center_fit_steps=0,
        yukawa_step_size=0.05,
    )

    assert prepared.setup.config.quark_yukawas is None
    assert prepared.setup.config.quark_path_readouts is None
    assert prepared.setup.config.family_yukawa_collision_cache is not None
    assert from_prepared.verdict.passed
    assert jnp.max(jnp.abs(from_prepared.up_amplitude - from_wrapper.up_amplitude)) < 2e-7
    assert jnp.max(jnp.abs(from_prepared.down_amplitude - from_wrapper.down_amplitude)) < 2e-7
    assert jnp.max(jnp.abs(from_prepared.up_population - from_wrapper.up_population)) < 2e-7
    assert jnp.max(jnp.abs(from_prepared.down_population - from_wrapper.down_population)) < 2e-7
    assert from_prepared.up_expected_one_tick_residual < 5e-7
    assert from_prepared.down_expected_one_tick_residual < 5e-7


def test_scanned_fn_dilation_rollout_is_jittable() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        collision_mode="fn_dilation",
    )

    def run(local_state):
        result = sm_run_qca_rollout(config, local_state, steps=2)
        aux = result.final_fn_path_aux_state
        return result.final_state, result.extended_norm_history, aux.up, aux.down

    expected = run(state)
    actual = jax.jit(run)(state)

    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert jnp.max(jnp.abs(actual_part - expected_part)) < 5e-7


def test_lean_rollout_skips_per_step_observables_without_changing_state() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        collision_mode="effective_yukawa",
    )

    full = sm_run_qca_rollout(config, state, steps=3)
    lean = sm_run_qca_rollout(config._replace(record_observables=False), state, steps=3)

    assert full.norm_history.shape == (4,)
    assert lean.norm_history.shape == (2,)
    assert lean.extended_norm_history.shape == (2,)
    assert lean.max_density_history.shape == (2,)
    assert lean.density_history.shape == (0, 1, 1, 1)
    assert jnp.max(jnp.abs(lean.final_state - full.final_state)) < 2e-7
    assert jnp.max(jnp.abs(lean.norm_history - jnp.asarray([full.norm_history[0], full.norm_history[-1]]))) < 5e-7
    assert (
        jnp.max(
            jnp.abs(
                lean.extended_norm_history
                - jnp.asarray([full.extended_norm_history[0], full.extended_norm_history[-1]]),
            ),
        )
        < 5e-7
    )


def test_fn_dilation_rollout_reports_visible_and_hidden_memory_footprint() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        collision_mode="fn_dilation",
    )

    prepared = sm_qca_prepare_state(state, config)
    footprint = sm_qca_state_memory_footprint(prepared)
    rollout_footprint = sm_qca_rollout_memory_footprint(prepared, config)
    result = sm_run_qca_rollout(config, state, steps=1)

    assert footprint.visible_complex_elements == state.size
    assert footprint.fn_path_aux_complex_elements > footprint.visible_complex_elements
    assert footprint.total_complex_elements == (
        footprint.visible_complex_elements + footprint.fn_path_aux_complex_elements
    )
    assert footprint.complex64_bytes == 8 * footprint.total_complex_elements
    assert footprint.complex128_bytes == 16 * footprint.total_complex_elements
    assert result.memory_footprint.visible_complex_elements == footprint.visible_complex_elements
    assert result.memory_footprint.fn_path_aux_complex_elements == footprint.fn_path_aux_complex_elements
    assert result.memory_footprint.total_complex_elements == footprint.total_complex_elements
    assert rollout_footprint.state == footprint
    assert rollout_footprint.state_array_bytes == footprint.complex64_bytes
    assert rollout_footprint.config_array_bytes > 0
    assert rollout_footprint.total_array_bytes == footprint.complex64_bytes + rollout_footprint.config_array_bytes
    assert result.rollout_memory_footprint.total_array_bytes == rollout_footprint.total_array_bytes


def test_fn_dilation_memory_budget_estimates_max_lattice_size() -> None:
    assert _max_cubic_edge(63) == 3
    assert _max_cubic_edge(64) == 4
    summary = _memory_budget_summary(
        bytes_per_site=16_896.0,
        memory_budget_gib=1.0,
        memory_safety_factor=0.75,
    )

    assert summary is not None
    assert summary["usable_bytes"] == 805306368
    assert summary["bytes_per_site"] == 16_896.0
    assert summary["fixed_bytes"] == 0
    assert summary["usable_bytes_after_fixed"] == 805306368
    assert summary["max_sites"] == 47_662
    assert summary["max_cubic_lattice_shape"] == [36, 36, 36]
    fixed_summary = _memory_budget_summary(
        bytes_per_site=3_072.0,
        fixed_bytes=2_304,
        memory_budget_gib=1.0,
        memory_safety_factor=0.75,
    )
    assert fixed_summary is not None
    assert fixed_summary["fixed_bytes"] == 2_304
    assert fixed_summary["usable_bytes_after_fixed"] == 805304064
    assert fixed_summary["max_sites"] == 262_143


def test_fn_dilation_dry_run_memory_payload_matches_prepared_state_footprint() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        yukawa_step_size=0.02,
        collision_mode="fn_dilation",
    )
    footprint = sm_qca_state_memory_footprint(sm_qca_prepare_state(state, config))
    payload = _memory_payload(
        lattice_shape=lattice_shape,
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode="fn_dilation",
        stream_mode="split_axis",
    )

    assert payload["sites"] == 1
    assert payload["stream_mode"] == "split_axis"
    assert payload["memory"]["visible_complex_elements"] == footprint.visible_complex_elements
    assert payload["memory"]["fn_path_aux_complex_elements"] == footprint.fn_path_aux_complex_elements
    assert payload["memory"]["total_complex_elements"] == footprint.total_complex_elements
    assert payload["memory"]["state_complex64_bytes"] == footprint.complex64_bytes
    assert payload["memory"]["state_complex64_bytes_per_site"] == footprint.complex64_bytes
    assert payload["memory"]["state_array_bytes"] == footprint.complex64_bytes
    assert payload["memory"]["runtime_array_bytes"] == payload["memory"]["total_array_bytes"]
    assert payload["memory"]["config_array_bytes"] > 0
    assert payload["memory"]["complex64_bytes"] > footprint.complex64_bytes


def test_benchmark_defaults_to_production_state_rollout() -> None:
    parser = build_benchmark_arg_parser()
    args = parser.parse_args([])
    lattice_shapes = _apply_benchmark_preset(args)
    args.collision_mode = args.collision_mode or "effective_yukawa"
    args.stream_mode = args.stream_mode or "split_axis"
    args.yukawa_collision_strategy = args.yukawa_collision_strategy or "fast"
    args.rollout_output = args.rollout_output or "state"
    args.memory_policy = args.memory_policy or "none"
    payload = _benchmark_sweep_payload(
        lattice_shapes=lattice_shapes,
        steps=1,
        repeats=1,
        warmup_repeats=0,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode="effective_yukawa",
        memory_budget_gib=None,
        memory_safety_factor=0.75,
        dry_run=True,
    )

    assert args.collision_mode == "effective_yukawa"
    assert args.stream_mode == "split_axis"
    assert args.yukawa_collision_strategy == "fast"
    assert args.rollout_output == "state"
    assert args.memory_policy == "none"
    assert args.donate_state
    assert payload["stream_mode"] == "split_axis"
    assert payload["yukawa_collision_strategy"] == "fast"
    assert payload["yukawa_collision_strategies"] == ["fast"]
    assert payload["rollout_output"] == "state"
    assert payload["benchmarks"][0]["benchmark_entrypoint"] == "production_api"
    assert payload["benchmarks"][0]["production_contract"]["uses_production_api"]
    assert payload["benchmarks"][0]["production_contract"]["lean_effective_yukawa"]
    assert not payload["benchmarks"][0]["production_contract"]["raw_yukawa_arrays_present"]
    assert not payload["benchmarks"][0]["production_contract"]["raw_readout_arrays_present"]


def test_benchmark_gauge_background_defaults_to_hop_sum_and_reports_link_memory() -> None:
    parser = build_benchmark_arg_parser()
    args = parser.parse_args(["--gauge-background", "identity", "--dry-run"])
    lattice_shapes = _apply_benchmark_preset(args)
    args.collision_mode = args.collision_mode or "effective_yukawa"
    args.stream_mode = args.stream_mode or ("hop_sum" if args.gauge_background != "none" else "split_axis")
    args.yukawa_collision_strategy = args.yukawa_collision_strategy or "fast"
    args.rollout_output = args.rollout_output or "state"
    args.memory_policy = args.memory_policy or "none"
    payload = _benchmark_sweep_payload(
        lattice_shapes=lattice_shapes,
        steps=1,
        repeats=1,
        warmup_repeats=0,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.0,
        higgs_vev=1.0,
        collision_mode=args.collision_mode,
        memory_budget_gib=None,
        memory_safety_factor=0.75,
        dry_run=True,
        stream_mode=args.stream_mode,
        gauge_background=args.gauge_background,
        yukawa_collision_strategy=args.yukawa_collision_strategy,
        rollout_output=args.rollout_output,
    )
    benchmark = payload["benchmarks"][0]

    assert args.stream_mode == "hop_sum"
    assert payload["gauge_background"] == "identity"
    assert benchmark["gauge_background"] == "identity"
    assert benchmark["production_contract"]["gauge_links_present"]
    assert benchmark["production_contract"]["lean_effective_yukawa"]
    assert benchmark["memory"]["config_array_bytes"] == 2 * 65_536
    assert benchmark["memory"]["runtime_array_bytes"] == 2 * (65_536 + 3_072)


def test_benchmark_rejects_gauge_background_with_split_axis() -> None:
    with pytest.raises(ValueError, match="static gauge links require stream_mode='hop_sum'"):
        _benchmark_sweep_payload(
            lattice_shapes=((1, 1, 1),),
            steps=1,
            repeats=1,
            warmup_repeats=0,
            donate_state=False,
            lambda_rec=0.225,
            yukawa_step_size=0.0,
            higgs_vev=1.0,
            collision_mode="effective_yukawa",
            memory_budget_gib=None,
            memory_safety_factor=0.75,
            dry_run=True,
            stream_mode="split_axis",
            gauge_background="identity",
            rollout_output="state",
        )


def test_benchmark_production_scaling_preset_selects_hot_path_sweep() -> None:
    parser = build_benchmark_arg_parser()
    args = parser.parse_args(["--preset", "production_scaling", "--dry-run"])
    lattice_shapes = _apply_benchmark_preset(args)

    assert lattice_shapes == PRODUCTION_SCALING_PRESET_SHAPES
    assert args.collision_mode == "effective_yukawa"
    assert args.stream_mode == "split_axis"
    assert args.yukawa_collision_strategy == "fast"
    assert args.rollout_output == "state"
    assert args.memory_policy == "skip"
    assert args.steps == 4
    assert args.repeats == 3
    assert args.warmup_repeats == 1

    payload = _benchmark_sweep_payload(
        lattice_shapes=lattice_shapes,
        steps=args.steps,
        repeats=args.repeats,
        warmup_repeats=args.warmup_repeats,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode=args.collision_mode,
        memory_budget_gib=None,
        memory_safety_factor=0.75,
        dry_run=True,
        memory_policy=args.memory_policy,
        preset=args.preset,
        stream_mode=args.stream_mode,
        yukawa_collision_strategy=args.yukawa_collision_strategy,
        rollout_output=args.rollout_output,
    )

    assert payload["preset"] == "production_scaling"
    assert payload["memory_policy"] == "skip"
    assert [tuple(item["lattice_shape"]) for item in payload["benchmarks"]] == list(PRODUCTION_SCALING_PRESET_SHAPES)
    assert all(item["benchmark_entrypoint"] == "production_api" for item in payload["benchmarks"])
    assert all(item["production_contract"]["lean_effective_yukawa"] for item in payload["benchmarks"])


def test_benchmark_can_compare_yukawa_collision_strategies() -> None:
    parser = build_benchmark_arg_parser()
    args = parser.parse_args(["--yukawa-collision-strategy", "both"])
    payload = _benchmark_sweep_payload(
        lattice_shapes=((1, 1, 1),),
        steps=1,
        repeats=1,
        warmup_repeats=0,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode="effective_yukawa",
        memory_budget_gib=None,
        memory_safety_factor=0.75,
        dry_run=True,
        yukawa_collision_strategy="both",
    )

    assert args.yukawa_collision_strategy == "both"
    assert payload["yukawa_collision_strategy"] == "both"
    assert payload["yukawa_collision_strategies"] == ["memory", "fast"]
    assert [benchmark["yukawa_collision_strategy"] for benchmark in payload["benchmarks"]] == ["memory", "fast"]
    assert all(benchmark["benchmark_entrypoint"] == "production_api" for benchmark in payload["benchmarks"])
    assert all(benchmark["production_contract"]["lean_effective_yukawa"] for benchmark in payload["benchmarks"])
    comparison = payload["strategy_comparisons"][0]
    assert comparison["collision_mode"] == "effective_yukawa"
    assert comparison["runtime_bytes_memory"] == comparison["runtime_bytes_fast"]
    assert comparison["runtime_bytes_delta_fast_minus_memory"] == 0
    assert comparison["complex64_bytes_memory"] == comparison["complex64_bytes_fast"]
    assert comparison["complex64_bytes_delta_fast_minus_memory"] == 0


def test_benchmark_payload_reports_state_memory_scaling_for_real_runs() -> None:
    payload = _benchmark_payload(
        lattice_shape=(1, 1, 1),
        steps=1,
        repeats=1,
        warmup_repeats=0,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.0,
        higgs_vev=1.0,
        collision_mode="effective_yukawa",
        stream_mode="split_axis",
        yukawa_collision_strategy="memory",
        rollout_output="state",
    )

    assert payload["benchmark_entrypoint"] == "production_api"
    assert payload["production_contract"]["lean_effective_yukawa"]
    assert payload["production_contract"]["uses_production_api"]
    assert not payload["production_contract"]["structured_collision_cache_present"]
    assert not payload["production_contract"]["raw_yukawa_arrays_present"]
    assert not payload["production_contract"]["raw_readout_arrays_present"]
    assert not payload["production_contract"]["higgs_field_present"]
    assert payload["memory"]["state_complex64_bytes"] == 3072
    assert payload["memory"]["state_complex64_bytes_per_site"] == 3072
    assert payload["memory"]["state_complex128_bytes_per_site"] == 6144
    assert payload["memory"]["state_array_bytes"] == 3072
    assert payload["memory"]["state_array_bytes_per_site"] == 3072
    assert payload["memory"]["runtime_array_bytes"] == payload["memory"]["total_array_bytes"]
    assert payload["memory"]["runtime_array_bytes_per_site"] == payload["memory"]["complex64_bytes_per_site"]


def test_benchmark_diagnostic_output_uses_rollout_config_entrypoint() -> None:
    payload = _memory_payload(
        lattice_shape=(1, 1, 1),
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode="effective_yukawa",
        stream_mode="split_axis",
        rollout_output="diagnostic",
    )

    assert payload["benchmark_entrypoint"] == "rollout_config"
    assert not payload["production_contract"]["uses_production_api"]
    assert not payload["production_contract"]["lean_effective_yukawa"]
    assert payload["production_contract"]["raw_yukawa_arrays_present"]


def test_fn_path_hidden_dim_estimate_matches_exact_readout_networks() -> None:
    dims = sm_family_quark_path_hidden_dims()
    readouts = sm_family_recirculated_quark_path_readouts()

    assert dims.up == readouts.up.network.unitary.shape[0]
    assert dims.down == readouts.down.network.unitary.shape[0]
    assert dims.up + dims.down == 72


def test_fn_dilation_dry_run_flags_shapes_that_exceed_memory_budget() -> None:
    payload = _benchmark_sweep_payload(
        lattice_shapes=((1, 1, 1), (64, 64, 64)),
        steps=1,
        repeats=1,
        warmup_repeats=0,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode="fn_dilation",
        memory_budget_gib=1.0,
        memory_safety_factor=0.75,
        dry_run=True,
        stream_mode="hop_sum",
    )

    small_fit = payload["benchmarks"][0]["memory_budget_fit"]
    large_fit = payload["benchmarks"][1]["memory_budget_fit"]
    assert small_fit["fits_memory_budget"] is True
    assert small_fit["budget_fraction"] < 1.0
    assert small_fit["runtime_bytes_margin"] > 0
    assert small_fit["complex64_bytes_margin"] > 0
    assert large_fit["fits_memory_budget"] is False
    assert large_fit["budget_fraction"] > 1.0
    assert large_fit["runtime_bytes_margin"] < 0
    assert large_fit["complex64_bytes_margin"] < 0


def test_benchmark_memory_policy_skip_avoids_oversized_jit() -> None:
    payload = _benchmark_sweep_payload(
        lattice_shapes=((1, 1, 1),),
        steps=1,
        repeats=1,
        warmup_repeats=0,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode="effective_yukawa",
        memory_budget_gib=4_000 / 1024**3,
        memory_safety_factor=1.0,
        dry_run=False,
        memory_policy="skip",
        stream_mode="split_axis",
        rollout_output="state",
    )

    benchmark = payload["benchmarks"][0]
    assert payload["memory_policy"] == "skip"
    assert benchmark["skipped"] is True
    assert benchmark["skip_reason"] == "exceeds_memory_budget"
    assert benchmark["memory_budget_fit"]["fits_memory_budget"] is False
    assert "compile_seconds" not in benchmark
    assert payload["mode_comparisons"] == []


def test_benchmark_memory_policy_fail_rejects_oversized_before_jit() -> None:
    with pytest.raises(ValueError, match="exceeds memory budget"):
        _benchmark_sweep_payload(
            lattice_shapes=((1, 1, 1),),
            steps=1,
            repeats=1,
            warmup_repeats=0,
            donate_state=False,
            lambda_rec=0.225,
            yukawa_step_size=0.01,
            higgs_vev=1.0,
            collision_mode="effective_yukawa",
            memory_budget_gib=4_000 / 1024**3,
            memory_safety_factor=1.0,
            dry_run=False,
            memory_policy="fail",
            stream_mode="split_axis",
            rollout_output="state",
        )


def test_benchmark_dry_run_compares_exact_and_compressed_fn_memory() -> None:
    payload = _benchmark_sweep_payload(
        lattice_shapes=((1, 1, 1),),
        steps=1,
        repeats=1,
        warmup_repeats=0,
        donate_state=False,
        lambda_rec=0.225,
        yukawa_step_size=0.01,
        higgs_vev=1.0,
        collision_mode="both",
        memory_budget_gib=1.0,
        memory_safety_factor=0.75,
        dry_run=True,
        stream_mode="split_axis",
        rollout_output="state",
    )

    assert payload["collision_modes"] == ["fn_dilation", "effective_yukawa"]
    assert payload["stream_mode"] == "split_axis"
    assert payload["rollout_output"] == "state"
    assert [benchmark["collision_mode"] for benchmark in payload["benchmarks"]] == [
        "fn_dilation",
        "effective_yukawa",
    ]
    assert [benchmark["stream_mode"] for benchmark in payload["benchmarks"]] == ["split_axis", "split_axis"]
    comparison = payload["mode_comparisons"][0]
    assert comparison["reference_collision_mode"] == "fn_dilation"
    assert comparison["compressed_collision_mode"] == "effective_yukawa"
    assert comparison["runtime_bytes_reference"] > comparison["runtime_bytes_compressed"]
    assert comparison["complex64_bytes_reference"] > comparison["complex64_bytes_compressed"]
    assert payload["benchmarks"][0]["memory"]["state_complex64_bytes"] == 16_896
    assert payload["benchmarks"][1]["memory"]["state_complex64_bytes"] == 3_072
    assert payload["benchmarks"][0]["memory"]["state_array_bytes"] == 16_896
    assert payload["benchmarks"][1]["memory"]["state_array_bytes"] == 3_072
    assert payload["benchmarks"][0]["memory"]["config_array_bytes"] > 0
    assert payload["benchmarks"][1]["memory"]["config_array_bytes"] > 0


def test_benchmark_mode_comparison_reports_compiled_memory_savings() -> None:
    comparison = _mode_comparisons(
        [
            {
                "lattice_shape": [2, 2, 2],
                "collision_mode": "fn_dilation",
                "memory": {"complex64_bytes": 16_000},
                "compiled_memory": {
                    "available": True,
                    "device_runtime_size_floor_bytes": 80_000,
                    "temp_size_in_bytes": 64_000,
                },
            },
            {
                "lattice_shape": [2, 2, 2],
                "collision_mode": "effective_yukawa",
                "memory": {"complex64_bytes": 4_000},
                "compiled_memory": {
                    "available": True,
                    "device_runtime_size_floor_bytes": 20_000,
                    "temp_size_in_bytes": 16_000,
                },
            },
        ],
    )[0]

    assert comparison["complex64_bytes_saved"] == 12_000
    assert comparison["memory_ratio_reference_to_compressed"] == 4.0
    assert comparison["compiled_runtime_floor_bytes_saved"] == 60_000
    assert comparison["compiled_runtime_floor_ratio_reference_to_compressed"] == 4.0
    assert comparison["compiled_temp_bytes_saved"] == 48_000
    assert comparison["compiled_temp_ratio_reference_to_compressed"] == 4.0


def test_benchmark_strategy_comparison_reports_runtime_and_temp_delta() -> None:
    comparison = _strategy_comparisons(
        [
            {
                "lattice_shape": [2, 2, 2],
                "collision_mode": "effective_yukawa",
                "yukawa_collision_strategy": "memory",
                "memory": {"complex64_bytes": 4_000},
                "mean_run_seconds": 0.02,
                "compiled_memory": {
                    "available": True,
                    "device_runtime_size_floor_bytes": 20_000,
                    "temp_size_in_bytes": 12_000,
                },
            },
            {
                "lattice_shape": [2, 2, 2],
                "collision_mode": "effective_yukawa",
                "yukawa_collision_strategy": "fast",
                "memory": {"complex64_bytes": 4_000},
                "mean_run_seconds": 0.01,
                "compiled_memory": {
                    "available": True,
                    "device_runtime_size_floor_bytes": 24_000,
                    "temp_size_in_bytes": 16_000,
                },
            },
        ],
    )[0]

    assert comparison["runtime_ratio_memory_to_fast"] == 2.0
    assert comparison["compiled_runtime_floor_bytes_delta_fast_minus_memory"] == 4_000
    assert comparison["compiled_temp_bytes_delta_fast_minus_memory"] == 4_000
    assert comparison["compiled_temp_ratio_memory_to_fast"] == 0.75


def test_calibrated_fn_dilation_and_effective_yukawa_readouts_agree() -> None:
    lattice_shape = (1, 1, 1)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)

    calibrated = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
        center_fit_steps=0,
        collision_mode="fn_dilation",
    )
    effective = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
        center_fit_steps=0,
        collision_mode="effective_yukawa",
    )

    assert calibrated.config.quark_yukawas is not None
    assert calibrated.config.quark_path_readouts is not None
    assert calibrated.config.family_yukawa_collision_cache is None
    assert effective.config.quark_yukawas is not None
    assert effective.config.quark_path_readouts is None
    assert effective.config.family_yukawa_collision_cache is not None
    assert effective.config.family_yukawa_collision_cache.cos_blocks is None
    assert effective.config.family_yukawa_collision_cache.cos_left_blocks is not None
    assert effective.config.family_yukawa_collision_cache.cos_left_blocks.shape == (4, 3, 3)
    assert effective.config.family_yukawa_collision_cache.cos_internal.shape == (0, 0)
    assert effective.config.family_yukawa_collision_cache.block_indices is None
    assert effective.config.family_yukawa_collision_cache.internal_pair_indices is None
    assert jnp.max(jnp.abs(calibrated.config.quark_yukawas.up - calibrated.config.quark_path_readouts.up.transfer)) < 2e-7
    assert (
        jnp.max(jnp.abs(calibrated.config.quark_yukawas.down - calibrated.config.quark_path_readouts.down.transfer))
        < 2e-7
    )
    assert jnp.max(jnp.abs(calibrated.config.quark_yukawas.up - effective.config.quark_yukawas.up)) < 2e-7
    assert jnp.max(jnp.abs(calibrated.config.quark_yukawas.down - effective.config.quark_yukawas.down)) < 2e-7
    assert (
        jnp.max(
            jnp.abs(
                fn_singular_masses(calibrated.config.quark_path_readouts.up.transfer)
                - fn_singular_masses(effective.config.quark_yukawas.up),
            ),
        )
        < 2e-7
    )
    assert (
        jnp.max(
            jnp.abs(
                jnp.abs(
                    fn_ckm_from_yukawas(
                        calibrated.config.quark_path_readouts.up.transfer,
                        calibrated.config.quark_path_readouts.down.transfer,
                    ),
                )
                - jnp.abs(fn_ckm_from_yukawas(effective.config.quark_yukawas.up, effective.config.quark_yukawas.down)),
            ),
        )
        < 2e-7
    )

    state = deterministic_qca_family_state(lattice_shape)
    cached = sm_run_qca_rollout(effective.config, state, steps=1)
    uncached = sm_run_qca_rollout(
        effective.config._replace(
            higgs=sm_constant_higgs(lattice_shape),
            family_yukawa_collision_cache=None,
        ),
        state,
        steps=1,
    )
    assert jnp.max(jnp.abs(cached.final_state - uncached.final_state)) < 5e-7


def test_effective_yukawa_block_cache_requires_unitary_gauge_higgs() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    unitary_higgs = sm_constant_higgs(lattice_shape)
    non_unitary_higgs = unitary_higgs.at[..., 0].set(jnp.asarray(0.125 - 0.25j, dtype=jnp.complex64))

    compressed = sm_family_yukawa_collision_cache(
        unitary_higgs,
        step_size=0.01,
        assume_uniform=True,
        use_unitary_gauge_blocks=True,
    )
    dense = sm_family_yukawa_collision_cache(
        non_unitary_higgs,
        step_size=0.01,
        assume_uniform=True,
        use_unitary_gauge_blocks=True,
    )

    assert compressed.cos_blocks is None
    assert compressed.cos_left_blocks is not None
    assert compressed.cos_left_blocks.shape == (4, 3, 3)
    assert compressed.cos_internal.shape == (0, 0)
    assert dense.cos_blocks is None
    assert dense.cos_left_blocks is None
    assert dense.cos_internal.shape == (96, 96)

    cached = sm_apply_family_yukawa_collision_from_cache(state, dense)
    uncached = sm_apply_family_yukawa_collision(state, non_unitary_higgs, step_size=0.01)
    assert jnp.max(jnp.abs(cached - uncached)) < 2e-7


def test_rollout_builders_default_to_effective_yukawa() -> None:
    lattice_shape = (1, 1, 1)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)

    center_config = sm_qca_center_cp_rollout_config(lattice_shape)
    calibrated = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
    )
    cli_config, _output = config_from_args([])

    assert center_config.collision_mode == "effective_yukawa"
    assert center_config.stream_mode == "split_axis"
    assert center_config.yukawa_collision_strategy == "fast"
    assert center_config.higgs is None
    assert center_config.quark_path_readouts is None
    assert center_config.family_yukawa_collision_cache is not None
    assert calibrated.config.collision_mode == "effective_yukawa"
    assert calibrated.config.stream_mode == "split_axis"
    assert calibrated.config.yukawa_collision_strategy == "fast"
    assert calibrated.config.higgs is None
    assert calibrated.config.quark_path_readouts is None
    assert calibrated.config.family_yukawa_collision_cache is not None
    assert cli_config.collision_mode == "effective_yukawa"
    assert cli_config.stream_mode == "split_axis"
    assert cli_config.yukawa_collision_strategy == "fast"
    assert sm_qca_config_memory_footprint(center_config).array_bytes == 1152


def test_phenomenology_cli_accepts_scale_masses_lambda_and_charges() -> None:
    config, output = config_from_args(
        [
            "--output",
            "json",
            "--scale-label",
            "MZ",
            "--up-masses",
            "0.00127,0.62,170.0",
            "--down-masses",
            "0.0029,0.055,2.86",
            "--mass-mode",
            "absolute",
            "--ckm-angles",
            "0.22501,0.04183,0.003732,1.147",
            "--lambda",
            "0.22501",
            "--q-charges",
            "3,2,0",
            "--u-charges",
            "5,2,0",
            "--d-charges",
            "1,0,0",
            "--lattice-shape",
            "1,1,1",
            "--steps",
            "1",
            "--collision-mode",
            "both",
            "--stream-mode",
            "split_axis",
            "--yukawa-collision-strategy",
            "memory",
            "--memory-budget-gib",
            "24",
            "--memory-safety-factor",
            "0.75",
            "--memory-policy",
            "auto",
        ],
    )

    assert output == "json"
    assert config.scale_label == "MZ"
    assert config.up_masses == (0.00127, 0.62, 170.0)
    assert config.down_masses == (0.0029, 0.055, 2.86)
    assert config.neutrino_yukawas is None
    assert config.charged_lepton_yukawas is None
    assert config.mass_mode == "absolute"
    assert config.lambda_rec == 0.22501
    assert config.charges.q == (3, 2, 0)
    assert config.charges.u == (5, 2, 0)
    assert config.charges.d == (1, 0, 0)
    assert config.lattice_shape == (1, 1, 1)
    assert config.steps == 1
    assert config.collision_mode == "both"
    assert config.stream_mode == "split_axis"
    assert config.gauge_background == "none"
    assert config.yukawa_collision_strategy == "memory"
    assert config.memory_budget_gib == 24.0
    assert config.memory_safety_factor == 0.75
    assert config.memory_policy == "auto"


def test_phenomenology_cli_loads_json_config_and_applies_overrides(tmp_path) -> None:
    config_path = tmp_path / "phenomenology.json"
    config_path.write_text(
        json.dumps(
            {
                "scale_label": "mt",
                "lattice_shape": [1, 1, 1],
                "steps": 7,
                "lambda": 0.224744871,
                "charges": {"q": [4, 2, 0], "u": [6, 2, 0], "d": [2, 1, 0]},
                "up_masses": [0.000007, 0.0036, 1.0],
                "down_masses": [0.001, 0.02, 1.0],
                "neutrino_yukawas": [0.0, 0.001, 0.02],
                "charged_lepton_yukawas": [0.000003, 0.0006, 0.01],
                "mass_mode": "ratios",
                "ckm_angles": [0.225, 0.041, 0.0037, 1.1],
                "ckm_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center_fit_steps": 2,
                "yukawa_step_size": 0.02,
                "higgs_vev": 0.9,
                "collision_mode": "effective_yukawa",
                "stream_mode": "split_axis",
                "gauge_background": "none",
                "yukawa_collision_strategy": "memory",
                "memory_budget_gib": 1.0,
                "memory_safety_factor": 0.5,
                "memory_policy": "fail",
            },
        ),
        encoding="utf-8",
    )

    config, output = config_from_args(
        [
            "--config",
            str(config_path),
            "--steps",
            "1",
            "--ckm-angles",
            "0.225,0.041,0.0037,1.1",
            "--collision-mode",
            "fn_dilation",
            "--memory-policy",
            "none",
            "--yukawa-collision-strategy",
            "fast",
            "--output",
            "json",
        ],
    )

    assert output == "json"
    assert config.scale_label == "mt"
    assert config.steps == 1
    assert config.lambda_rec == 0.224744871
    assert config.charges.q == (4, 2, 0)
    assert config.charges.u == (6, 2, 0)
    assert config.charges.d == (2, 1, 0)
    assert config.neutrino_yukawas == (0.0, 0.001, 0.02)
    assert config.charged_lepton_yukawas == (0.000003, 0.0006, 0.01)
    assert config.ckm_matrix is None
    assert config.mass_mode == "ratios"
    assert config.center_fit_steps == 2
    assert config.yukawa_step_size == 0.02
    assert config.higgs_vev == 0.9
    assert config.collision_mode == "fn_dilation"
    assert config.stream_mode == "split_axis"
    assert config.gauge_background == "none"
    assert config.yukawa_collision_strategy == "fast"
    assert config.memory_budget_gib == 1.0
    assert config.memory_safety_factor == 0.5
    assert config.memory_policy == "none"


def test_phenomenology_prepare_uses_lean_observable_rollout_config() -> None:
    prepared = _prepare_phenomenology_rollout(
        PhenomenologyRunConfig(lattice_shape=(1, 1, 1), steps=2),
        calibration_collision_mode="effective_yukawa",
    )

    assert not prepared.calibrated.config.record_observables
    assert not prepared.calibrated.config.record_density


def test_phenomenology_gauge_background_defaults_to_hop_sum_and_reports_link_memory() -> None:
    config, _ = config_from_args(
        [
            "--gauge-background",
            "identity",
            "--lattice-shape",
            "1,1,1",
            "--steps",
            "1",
        ],
    )
    memory = _mode_memory_estimate(config, "effective_yukawa")
    summary = run_phenomenology_rollout(config)

    assert config.gauge_background == "identity"
    assert config.stream_mode == "hop_sum"
    assert memory["gauge_background"] == "identity"
    assert memory["config_array_bytes"] == 65_536 + 1_152
    assert memory["runtime_array_bytes"] == 65_536 + 1_152 + 3_072
    assert summary.gauge_background == "identity"
    assert summary.stream_mode == "hop_sum"
    assert summary.lepton_mode == "zero"
    assert summary.production_contract["uses_production_api"]
    assert summary.production_contract["gauge_links_present"]
    assert summary.production_contract["lean_effective_yukawa"]
    assert summary.config_array_bytes == 65_536 + 1_152
    assert summary.runtime_array_bytes == 65_536 + 1_152 + 3_072
    assert len(summary.carrier_sector_final) == 6
    assert len(summary.carrier_chirality_final) == 2
    assert len(summary.carrier_family_final) == 3
    assert len(summary.carrier_copy_final) == 2
    assert abs(sum(summary.carrier_sector_final) - summary.norm_final) < 2e-6
    assert abs(sum(summary.carrier_family_final) - summary.norm_final) < 2e-6


def test_phenomenology_rejects_gauge_background_with_split_axis() -> None:
    with pytest.raises(ValueError, match="static gauge links require stream_mode='hop_sum'"):
        run_phenomenology_rollout(
            PhenomenologyRunConfig(
                lattice_shape=(1, 1, 1),
                steps=1,
                stream_mode="split_axis",
                gauge_background="identity",
            ),
        )


def test_phenomenology_lepton_yukawa_inputs_reach_compressed_cache_and_stay_lean() -> None:
    config = PhenomenologyRunConfig(
        lattice_shape=(1, 1, 1),
        steps=1,
        neutrino_yukawas=(0.0, 0.0, 0.0),
        charged_lepton_yukawas=(0.0, 0.0, 0.0),
    )
    prepared = _prepare_phenomenology_rollout(config, calibration_collision_mode="effective_yukawa")
    cache = prepared.calibrated.config.family_yukawa_collision_cache
    summary = run_phenomenology_rollout(config)

    assert cache is not None
    identity3 = jnp.eye(3, dtype=jnp.complex64)
    assert jnp.max(jnp.abs(cache.cos_left_blocks[2] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.cos_right_blocks[2] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_left_right_blocks[2])) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_right_left_blocks[2])) < 2e-7
    assert jnp.max(jnp.abs(cache.cos_left_blocks[3] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.cos_right_blocks[3] - identity3)) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_left_right_blocks[3])) < 2e-7
    assert jnp.max(jnp.abs(cache.sin_right_left_blocks[3])) < 2e-7
    assert summary.lepton_mode == "diagonal_input"
    assert summary.neutrino_yukawas == (0.0, 0.0, 0.0)
    assert summary.charged_lepton_yukawas == (0.0, 0.0, 0.0)
    assert summary.production_contract["lean_effective_yukawa"]


def test_phenomenology_cli_json_reports_lepton_inputs_and_lean_contract(tmp_path, capsys) -> None:
    json_path = tmp_path / "leptons.json"

    phenomenology_main(
        [
            "--lattice-shape",
            "1,1,1",
            "--steps",
            "1",
            "--neutrino-yukawas",
            "0.0,0.001,0.01",
            "--charged-lepton-yukawas",
            "0.000003,0.0006,0.01",
            "--output",
            "json",
            "--json-output-path",
            str(json_path),
        ],
    )
    payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload == saved_payload
    assert payload["lepton_inputs"]["mode"] == "diagonal_input"
    assert payload["lepton_inputs"]["neutrino_yukawas"] == [0.0, 0.001, 0.01]
    assert payload["lepton_inputs"]["charged_lepton_yukawas"] == [0.000003, 0.0006, 0.01]
    assert payload["rollout"]["production_contract"]["uses_production_api"]
    assert payload["rollout"]["production_contract"]["structured_collision_cache_present"]
    assert payload["rollout"]["production_contract"]["lean_effective_yukawa"]
    assert not payload["rollout"]["production_contract"]["raw_yukawa_arrays_present"]
    assert payload["rollout"]["memory"]["fn_path_aux_complex_elements"] == 0


def test_phenomenology_rollout_reports_coefficient_diagnostics() -> None:
    summary = run_phenomenology_rollout(
        PhenomenologyRunConfig(lattice_shape=(1, 1, 1), steps=1),
    )

    assert summary.passed_center_cp
    assert summary.selected_label == "wilson_flux_rule"
    assert summary.scale_label == "benchmark"
    assert summary.magnitude_min > 0.0
    assert summary.magnitude_max < 10.0
    assert summary.up_center_powers == ((2, 1, 1), (1, 0, 0), (0, 2, 0))
    assert summary.down_center_powers == ((1, 1, 1), (2, 0, 0), (1, 2, 0))
    assert len(summary.up_magnitudes) == 3
    assert len(summary.down_magnitudes) == 3
    assert summary.used_higgs_fn_collision
    assert not summary.used_fn_dilation_collision
    assert summary.collision_mode == "effective_yukawa"
    assert summary.rollout_entrypoint == "calibrated_production_api"
    assert summary.stream_mode == "split_axis"
    assert summary.gauge_background == "none"
    assert summary.yukawa_collision_strategy == "fast"
    assert summary.lepton_mode == "zero"
    assert summary.production_contract["uses_production_api"]
    assert summary.production_contract["state_only"]
    assert summary.production_contract["structured_collision_cache_present"]
    assert not summary.production_contract["raw_yukawa_arrays_present"]
    assert not summary.production_contract["raw_readout_arrays_present"]
    assert not summary.production_contract["higgs_field_present"]
    assert summary.production_contract["lean_effective_yukawa"]
    assert summary.steps_completed == 1
    assert summary.extended_norm_drift < 1e-4
    assert summary.visible_complex_elements > 0
    assert summary.fn_path_aux_complex_elements == 0
    assert summary.total_complex_elements == summary.visible_complex_elements + summary.fn_path_aux_complex_elements
    assert summary.state_complex64_bytes == 8 * summary.total_complex_elements
    assert summary.state_complex128_bytes == 16 * summary.total_complex_elements
    assert summary.state_array_bytes == summary.state_complex64_bytes
    assert summary.state_array_bytes_per_site == summary.state_array_bytes
    assert summary.config_array_bytes > 0
    assert summary.runtime_array_bytes == summary.state_array_bytes + summary.config_array_bytes
    assert summary.runtime_array_bytes_per_site == summary.runtime_array_bytes
    assert summary.total_array_bytes == summary.state_complex64_bytes + summary.config_array_bytes
    assert summary.complex64_bytes == summary.total_array_bytes


def test_phenomenology_cli_writes_front_door_json_and_report(tmp_path, capsys) -> None:
    json_path = tmp_path / "canonical.json"
    report_path = tmp_path / "canonical.md"

    phenomenology_main(
        [
            "--lattice-shape",
            "1,1,1",
            "--steps",
            "1",
            "--output",
            "json",
            "--json-output-path",
            str(json_path),
            "--report-output-path",
            str(report_path),
        ],
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert stdout_payload == saved_payload
    assert saved_payload["rollout"]["rollout_entrypoint"] == "calibrated_production_api"
    assert saved_payload["rollout"]["production_contract"]["uses_production_api"]
    assert saved_payload["rollout"]["production_contract"]["lean_effective_yukawa"]
    assert saved_payload["rollout"]["memory"]["fn_path_aux_complex_elements"] == 0
    assert saved_payload["physics_tests"]["passed"]
    assert all(saved_payload["physics_tests"]["checks"].values())
    assert "coefficient_diagnostics" in saved_payload
    assert "up_magnitudes" in saved_payload["coefficient_diagnostics"]
    assert "down_center_powers" in saved_payload["coefficient_diagnostics"]
    assert "carrier_populations" in saved_payload["rollout"]
    assert "QCA_SMv0 Canonical Phenomenology Benchmark" in report
    assert "Physics Tests" in report
    assert "overall passed: `True`" in report
    assert "Coefficient Diagnostics" in report
    assert "Field Rollout" in report
    assert "hidden FN aux complex elements: `0`" in report


def test_canonical_phenomenology_benchmark_artifact_passes_physics_tests() -> None:
    payload = json.loads(
        Path("src/clifford_3plus2_d5/qca_smv0/canonical/phenomenology_benchmark.json").read_text(
            encoding="utf-8",
        ),
    )

    assert payload["physics_tests"]["passed"]
    assert all(payload["physics_tests"]["checks"].values())
    assert payload["center_cp_passed"]
    assert payload["residuals"]["objective"] < payload["physics_tests"]["thresholds"]["objective_max"]
    assert payload["residuals"]["up_mass_log_rms"] < payload["physics_tests"]["thresholds"]["mass_log_rms_max"]
    assert payload["residuals"]["down_mass_log_rms"] < payload["physics_tests"]["thresholds"]["mass_log_rms_max"]
    assert payload["residuals"]["ckm_abs"] < payload["physics_tests"]["thresholds"]["ckm_abs_max"]
    assert payload["residuals"]["jarlskog_relative"] < payload["physics_tests"]["thresholds"]["jarlskog_relative_max"]
    assert (
        payload["coefficient_diagnostics"]["magnitude_min"]
        >= payload["physics_tests"]["thresholds"]["magnitude_min"]
    )
    assert (
        payload["coefficient_diagnostics"]["magnitude_max"]
        <= payload["physics_tests"]["thresholds"]["magnitude_max"]
    )
    assert payload["rollout"]["norm_drift"] < payload["physics_tests"]["thresholds"]["norm_drift_max"]
    assert payload["rollout"]["extended_norm_drift"] < payload["physics_tests"]["thresholds"]["norm_drift_max"]
    assert payload["rollout"]["carrier_populations"]["sector_drift"][0] < 0.0
    assert payload["rollout"]["carrier_populations"]["sector_drift"][1] > 0.0
    assert payload["rollout"]["memory"]["fn_path_aux_complex_elements"] == 0
    assert payload["rollout"]["production_contract"]["uses_production_api"]
    assert payload["rollout"]["production_contract"]["lean_effective_yukawa"]


def test_phenomenology_effective_memory_preflight_matches_production_setup() -> None:
    config = PhenomenologyRunConfig(lattice_shape=(1, 1, 1), steps=1)
    memory = _mode_memory_estimate(config, "effective_yukawa")
    setup = sm_qca_prepare_production_rollout(
        config.lattice_shape,
        lambda_rec=config.lambda_rec,
        charges=config.charges,
        yukawa_step_size=config.yukawa_step_size,
        higgs_vev=config.higgs_vev,
        stream_mode=config.stream_mode,
        yukawa_collision_strategy=config.yukawa_collision_strategy,
    )

    assert memory["rollout_entrypoint"] == "production_api"
    assert memory["gauge_background"] == "none"
    assert memory["yukawa_collision_strategy"] == "fast"
    assert memory["state_complex64_bytes"] == setup.rollout_memory_footprint.state.complex64_bytes
    assert memory["state_complex128_bytes"] == setup.rollout_memory_footprint.state.complex128_bytes
    assert memory["state_array_bytes"] == setup.rollout_memory_footprint.state_array_bytes
    assert memory["runtime_array_bytes"] == setup.rollout_memory_footprint.total_array_bytes
    assert memory["config_array_bytes"] == setup.rollout_memory_footprint.config_array_bytes
    assert memory["complex64_bytes"] == setup.rollout_memory_footprint.total_array_bytes
    assert memory["fn_path_aux_complex_elements"] == 0


def test_phenomenology_rollout_modes_compare_exact_and_compressed_memory() -> None:
    summaries = run_phenomenology_rollout_modes(
        PhenomenologyRunConfig(lattice_shape=(1, 1, 1), steps=1, collision_mode="both"),
    )
    by_mode = {summary.collision_mode: summary for summary in summaries}

    exact = by_mode["fn_dilation"]
    compressed = by_mode["effective_yukawa"]
    assert exact.stream_mode == "split_axis"
    assert compressed.stream_mode == "split_axis"
    assert exact.yukawa_collision_strategy == "fast"
    assert compressed.yukawa_collision_strategy == "fast"
    assert exact.rollout_entrypoint == "rollout_config"
    assert compressed.rollout_entrypoint == "calibrated_production_api"
    assert not exact.production_contract["uses_production_api"]
    assert not exact.production_contract["lean_effective_yukawa"]
    assert compressed.production_contract["uses_production_api"]
    assert compressed.production_contract["lean_effective_yukawa"]
    assert exact.state_complex64_bytes == 16_896
    assert compressed.state_complex64_bytes == 3_072
    assert exact.runtime_array_bytes > compressed.runtime_array_bytes
    assert exact.complex64_bytes > compressed.complex64_bytes
    assert exact.config_array_bytes > 0
    assert compressed.config_array_bytes > 0
    assert exact.fn_path_aux_complex_elements > 0
    assert compressed.fn_path_aux_complex_elements == 0
    assert exact.extended_norm_drift < 1e-4
    assert compressed.extended_norm_drift < 1e-4


def test_phenomenology_memory_policy_auto_selects_compressed_when_exact_exceeds_budget() -> None:
    summaries = run_phenomenology_rollout_modes(
        PhenomenologyRunConfig(
            lattice_shape=(1, 1, 1),
            steps=1,
            collision_mode="fn_dilation",
            memory_budget_gib=40_000 / 1024**3,
            memory_safety_factor=1.0,
            memory_policy="auto",
        ),
    )

    assert [summary.collision_mode for summary in summaries] == ["effective_yukawa"]
    assert summaries[0].stream_mode == "split_axis"
    assert summaries[0].yukawa_collision_strategy == "fast"
    assert summaries[0].rollout_entrypoint == "calibrated_production_api"
    assert summaries[0].state_complex64_bytes == 3_072
    assert summaries[0].runtime_array_bytes == summaries[0].complex64_bytes
    assert summaries[0].complex64_bytes <= 40_000
    assert summaries[0].fn_path_aux_complex_elements == 0


def test_phenomenology_memory_policy_fail_rejects_exact_before_rollout() -> None:
    config = PhenomenologyRunConfig(
        lattice_shape=(1, 1, 1),
        steps=1,
        collision_mode="fn_dilation",
        memory_budget_gib=8_192 / 1024**3,
        memory_safety_factor=1.0,
        memory_policy="fail",
    )

    try:
        run_phenomenology_rollout_modes(config)
    except ValueError as exc:
        assert "exceeds memory budget" in str(exc)
    else:  # pragma: no cover - defensive assertion shape for old pytest helpers.
        raise AssertionError("expected memory policy to reject exact FN dilation")
