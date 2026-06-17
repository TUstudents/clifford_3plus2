"""Tests for the compact QCA_SMv0 rollout runner."""

import json

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_step
from clifford_3plus2_d5.qca_smv0.scripts.phenomenology_rollout import (
    PhenomenologyRunConfig,
    config_from_args,
    run_phenomenology_rollout,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import fn_ckm_from_yukawas, fn_singular_masses
from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_free_dirac_internal_step, sm_identity_links
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    QCASMRolloutConfig,
    deterministic_qca_family_state,
    deterministic_qca_sm_state,
    sm_qca_center_cp_rollout_config,
    sm_qca_rollout_config_from_masses_ckm,
    sm_qca_total_norm,
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
    assert effective.config.quark_yukawas is not None
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
    assert summary.steps_completed == 1
    assert summary.extended_norm_drift < 1e-4
