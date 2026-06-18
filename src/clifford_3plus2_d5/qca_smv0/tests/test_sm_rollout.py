"""Tests for the compact QCA_SMv0 rollout runner."""

import json

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_step
from clifford_3plus2_d5.qca_smv0.scripts.benchmark_fn_dilation_rollout import (
    _benchmark_sweep_payload,
    _max_cubic_edge,
    _memory_budget_summary,
    _memory_payload,
)
from clifford_3plus2_d5.qca_smv0.scripts.phenomenology_rollout import (
    PhenomenologyRunConfig,
    config_from_args,
    run_phenomenology_rollout_modes,
    run_phenomenology_rollout,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    sm_family_quark_path_hidden_dims,
    sm_family_recirculated_quark_path_readouts,
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
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    QCASMRolloutConfig,
    deterministic_qca_family_state,
    deterministic_qca_sm_state,
    sm_qca_center_cp_rollout_config,
    sm_qca_rollout_memory_footprint,
    sm_qca_prepare_state,
    sm_qca_rollout_config_from_masses_ckm,
    sm_qca_state_memory_footprint,
    sm_qca_total_norm,
    sm_qca_state_step,
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


def test_center_cp_higgs_fn_rollout_preserves_norm_and_changes_family_state() -> None:
    state = deterministic_qca_family_state((1, 1, 1))
    config = sm_qca_center_cp_rollout_config((1, 1, 1), yukawa_step_size=0.02, record_density=True)

    result = sm_run_qca_rollout(config, state, steps=1)
    stream_only = sm_run_qca_rollout(QCASMRolloutConfig(record_density=True), state, steps=1)

    assert result.used_higgs_fn_collision
    assert result.used_fn_dilation_collision
    assert result.final_fn_path_aux_state is not None
    assert result.density_history.shape == (2, 1, 1, 1)
    assert jnp.abs(result.extended_norm_history[-1] - result.extended_norm_history[0]) < 1e-4
    assert sm_qca_total_norm(result.final_state) < sm_qca_total_norm(state)
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
    assert result.used_higgs_fn_collision
    assert result.used_fn_dilation_collision
    assert result.final_fn_path_aux_state is not None
    assert jnp.abs(result.extended_norm_history[-1] - result.extended_norm_history[0]) < 1e-4


def test_scanned_rollout_matches_repeated_persistent_state_steps() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(lattice_shape, yukawa_step_size=0.02, record_density=True)

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


def test_scanned_fn_dilation_rollout_is_jittable() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(lattice_shape, yukawa_step_size=0.02)

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
    config = sm_qca_center_cp_rollout_config(lattice_shape, yukawa_step_size=0.02)

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
    assert summary["max_sites"] == 47_662
    assert summary["max_cubic_lattice_shape"] == [36, 36, 36]


def test_fn_dilation_dry_run_memory_payload_matches_prepared_state_footprint() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(lattice_shape, yukawa_step_size=0.02)
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
    assert payload["memory"]["config_array_bytes"] > 0
    assert payload["memory"]["complex64_bytes"] > footprint.complex64_bytes


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
    assert small_fit["complex64_bytes_margin"] > 0
    assert large_fit["fits_memory_budget"] is False
    assert large_fit["budget_fraction"] > 1.0
    assert large_fit["complex64_bytes_margin"] < 0


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
    assert comparison["complex64_bytes_reference"] > comparison["complex64_bytes_compressed"]
    assert payload["benchmarks"][0]["memory"]["state_complex64_bytes"] == 16_896
    assert payload["benchmarks"][1]["memory"]["state_complex64_bytes"] == 3_072
    assert payload["benchmarks"][0]["memory"]["config_array_bytes"] > 0
    assert payload["benchmarks"][1]["memory"]["config_array_bytes"] > 0


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
    assert effective.config.family_yukawa_collision_cache.cos_blocks is not None
    assert effective.config.family_yukawa_collision_cache.cos_blocks.shape == (16, 6, 6)
    assert effective.config.family_yukawa_collision_cache.cos_internal.shape == (0, 0)
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
        effective.config._replace(family_yukawa_collision_cache=None),
        state,
        steps=1,
    )
    assert jnp.max(jnp.abs(cached.final_state - uncached.final_state)) < 2e-7


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
    assert config.mass_mode == "absolute"
    assert config.lambda_rec == 0.22501
    assert config.charges.q == (3, 2, 0)
    assert config.charges.u == (5, 2, 0)
    assert config.charges.d == (1, 0, 0)
    assert config.lattice_shape == (1, 1, 1)
    assert config.steps == 1
    assert config.collision_mode == "both"
    assert config.stream_mode == "split_axis"
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
                "mass_mode": "ratios",
                "ckm_angles": [0.225, 0.041, 0.0037, 1.1],
                "ckm_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center_fit_steps": 2,
                "yukawa_step_size": 0.02,
                "higgs_vev": 0.9,
                "collision_mode": "effective_yukawa",
                "stream_mode": "split_axis",
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
    assert config.ckm_matrix is None
    assert config.mass_mode == "ratios"
    assert config.center_fit_steps == 2
    assert config.yukawa_step_size == 0.02
    assert config.higgs_vev == 0.9
    assert config.collision_mode == "fn_dilation"
    assert config.stream_mode == "split_axis"
    assert config.memory_budget_gib == 1.0
    assert config.memory_safety_factor == 0.5
    assert config.memory_policy == "none"


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
    assert summary.used_fn_dilation_collision
    assert summary.collision_mode == "fn_dilation"
    assert summary.stream_mode == "hop_sum"
    assert summary.steps_completed == 1
    assert summary.extended_norm_drift < 1e-4
    assert summary.visible_complex_elements > 0
    assert summary.fn_path_aux_complex_elements > summary.visible_complex_elements
    assert summary.total_complex_elements == summary.visible_complex_elements + summary.fn_path_aux_complex_elements
    assert summary.state_complex64_bytes == 8 * summary.total_complex_elements
    assert summary.state_complex128_bytes == 16 * summary.total_complex_elements
    assert summary.config_array_bytes > 0
    assert summary.total_array_bytes == summary.state_complex64_bytes + summary.config_array_bytes
    assert summary.complex64_bytes == summary.total_array_bytes


def test_phenomenology_rollout_modes_compare_exact_and_compressed_memory() -> None:
    summaries = run_phenomenology_rollout_modes(
        PhenomenologyRunConfig(lattice_shape=(1, 1, 1), steps=1, collision_mode="both"),
    )
    by_mode = {summary.collision_mode: summary for summary in summaries}

    exact = by_mode["fn_dilation"]
    compressed = by_mode["effective_yukawa"]
    assert exact.stream_mode == "hop_sum"
    assert compressed.stream_mode == "hop_sum"
    assert exact.state_complex64_bytes == 16_896
    assert compressed.state_complex64_bytes == 3_072
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
    assert summaries[0].stream_mode == "hop_sum"
    assert summaries[0].state_complex64_bytes == 3_072
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
