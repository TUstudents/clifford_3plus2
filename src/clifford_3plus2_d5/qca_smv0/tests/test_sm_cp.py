"""Tests for QCA_SMv0 center-holonomy CP coefficients."""

import math

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_cp import (
    DEFAULT_CENTER_HOLONOMY_POWERS,
    SM_COLOR_CENTER_ORDER,
    VERDICT_CENTER_HOLONOMY_POWERS,
    sm_center_coefficients,
    sm_center_cp_diagnostics,
    sm_center_cp_order_one_coefficients,
    sm_center_cp_order_one_fit_residuals,
    sm_center_cp_quark_yukawas,
    sm_center_holonomy_phases,
    sm_center_phase_closure_residual,
    sm_center_phase_unit_modulus_residual,
    sm_center_power_matrix,
    sm_ckm_jarlskog,
    sm_factor_coefficients_to_center_phases,
    sm_fit_center_cp_order_one_magnitudes,
    sm_search_center_cp_powers,
    sm_quark_antiquark_mass_residual,
    sm_yukawa_commutator_cp_trace,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    FNQuarkCoefficientMatrices,
    fn_ckm_from_yukawas,
    fn_default_coefficients,
    fn_quark_yukawas_from_masses_ckm,
    fn_quark_yukawa_matrices,
    fn_singular_masses,
    fn_unitarity_residual,
)


def test_center_holonomy_phases_are_su3_center_valued() -> None:
    phases = sm_center_holonomy_phases(DEFAULT_CENTER_HOLONOMY_POWERS.up)

    assert phases.shape == (3, 3)
    assert sm_center_phase_unit_modulus_residual(phases) < 2e-7
    assert sm_center_phase_closure_residual(phases) < 1e-6
    assert jnp.max(jnp.abs(phases**SM_COLOR_CENTER_ORDER - 1.0)) < 1e-6


def test_center_power_matrix_reduces_modulo_center_order() -> None:
    powers = jnp.asarray([[0, 1, 2], [3, 4, 5], [-1, -2, -3]], dtype=jnp.int32)
    reduced = sm_center_power_matrix(powers)

    assert jnp.array_equal(reduced, jnp.asarray([[0, 1, 2], [0, 1, 2], [2, 1, 0]], dtype=jnp.int32))


def test_center_coefficients_preserve_order_one_magnitudes() -> None:
    up = sm_center_coefficients("up")
    down = sm_center_coefficients("down")

    assert jnp.max(jnp.abs(jnp.abs(up) - jnp.abs(fn_default_coefficients("up")))) < 2e-7
    assert jnp.max(jnp.abs(jnp.abs(down) - jnp.abs(fn_default_coefficients("down")))) < 2e-7


def test_factor_coefficients_to_center_phases_is_exact_for_center_coefficients() -> None:
    coefficients = FNQuarkCoefficientMatrices(
        up=sm_center_coefficients("up"),
        down=sm_center_coefficients("down"),
    )

    factorization = sm_factor_coefficients_to_center_phases(coefficients)

    assert factorization.coefficient_residual < 1e-6
    assert factorization.phase_residual < 1e-6
    assert jnp.max(jnp.abs(factorization.reconstructed_coefficients.up - coefficients.up)) < 1e-6
    assert jnp.max(jnp.abs(factorization.reconstructed_coefficients.down - coefficients.down)) < 1e-6
    assert sm_center_phase_closure_residual(factorization.center_phases.up) < 1e-6
    assert sm_center_phase_closure_residual(factorization.center_phases.down) < 1e-6


def test_factor_coefficients_to_center_phases_preserves_magnitudes_and_reports_projection_error() -> None:
    up = jnp.asarray(
        [
            [1.0, 2.0 * jnp.exp(0.41j), 0.0],
            [0.8 * jnp.exp(-0.62j), 1.5 * jnp.exp(1.3j), 0.7],
            [1.1, 0.9 * jnp.exp(-1.9j), 1.2 * jnp.exp(2.6j)],
        ],
        dtype=jnp.complex64,
    )
    down = jnp.asarray(
        [
            [0.5 * jnp.exp(0.2j), 1.0, 1.4 * jnp.exp(-0.8j)],
            [0.0, 0.6 * jnp.exp(1.7j), 1.1],
            [1.3 * jnp.exp(-2.4j), 0.9, 0.75 * jnp.exp(0.95j)],
        ],
        dtype=jnp.complex64,
    )

    factorization = sm_factor_coefficients_to_center_phases(FNQuarkCoefficientMatrices(up=up, down=down))

    assert jnp.max(jnp.abs(jnp.abs(factorization.reconstructed_coefficients.up) - jnp.abs(up))) < 2e-7
    assert jnp.max(jnp.abs(jnp.abs(factorization.reconstructed_coefficients.down) - jnp.abs(down))) < 2e-7
    assert factorization.coefficient_residual > 0.1
    assert factorization.phase_residual > 0.1
    assert factorization.magnitudes.up[0, 2] == 0.0
    assert factorization.magnitudes.down[1, 0] == 0.0
    assert factorization.center_powers.up[0, 2] == 0
    assert factorization.center_powers.down[1, 0] == 0
    assert sm_center_phase_closure_residual(factorization.center_phases.up) < 1e-6
    assert sm_center_phase_closure_residual(factorization.center_phases.down) < 1e-6


def test_search_center_cp_powers_never_worsens_nearest_projection() -> None:
    up = jnp.asarray(
        [
            [1.0, 0.8 * jnp.exp(0.31j), 1.2 * jnp.exp(-0.77j)],
            [0.9 * jnp.exp(1.1j), 1.1, 0.7 * jnp.exp(-1.4j)],
            [1.3 * jnp.exp(2.0j), 0.95 * jnp.exp(-2.2j), 1.0],
        ],
        dtype=jnp.complex64,
    )
    down = jnp.asarray(
        [
            [0.9 * jnp.exp(-0.4j), 1.2, 0.8 * jnp.exp(1.5j)],
            [1.0, 0.85 * jnp.exp(-1.8j), 1.15 * jnp.exp(0.9j)],
            [1.1 * jnp.exp(2.4j), 0.7, 0.95 * jnp.exp(-2.8j)],
        ],
        dtype=jnp.complex64,
    )
    coefficients = FNQuarkCoefficientMatrices(up=up, down=down)
    target = fn_quark_yukawa_matrices(up_coefficients=up, down_coefficients=down)

    search = sm_search_center_cp_powers(coefficients, target, max_sweeps=1)

    assert search.best_residuals.objective <= search.initial_residuals.objective + 1e-7
    assert search.best_residuals.ckm_abs_residual <= search.initial_residuals.objective + 1e-7
    assert sm_center_phase_closure_residual(search.best_factorization.center_phases.up) < 1e-6
    assert sm_center_phase_closure_residual(search.best_factorization.center_phases.down) < 1e-6
    assert jnp.isfinite(search.best_residuals.jarlskog_relative_residual)
    assert search.sweeps_completed >= 1


def _pdg_2025_quark_benchmark() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    lambda_rec = 0.22501
    s13 = 0.003732
    s23 = 0.04183
    delta = 1.147
    c12 = math.sqrt(1.0 - lambda_rec * lambda_rec)
    c13 = math.sqrt(1.0 - s13 * s13)
    c23 = math.sqrt(1.0 - s23 * s23)
    phase_p = complex(math.cos(delta), math.sin(delta))
    phase_m = complex(math.cos(delta), -math.sin(delta))
    ckm = jnp.asarray(
        [
            [c12 * c13, lambda_rec * c13, s13 * phase_m],
            [
                -lambda_rec * c23 - c12 * s23 * s13 * phase_p,
                c12 * c23 - lambda_rec * s23 * s13 * phase_p,
                s23 * c13,
            ],
            [
                lambda_rec * s23 - c12 * c23 * s13 * phase_p,
                -c12 * s23 - lambda_rec * c23 * s13 * phase_p,
                c23 * c13,
            ],
        ],
        dtype=jnp.complex64,
    )
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)
    return up_masses, down_masses, ckm, lambda_rec


def _verdict_magnitudes() -> FNQuarkCoefficientMatrices:
    return FNQuarkCoefficientMatrices(
        up=jnp.asarray(
            [
                [1.272654, 0.785760, 1.852192],
                [0.785760, 1.272654, 0.458693],
                [3.961016, 3.418301, 0.987706],
            ],
            dtype=jnp.float32,
        ),
        down=jnp.asarray(
            [
                [0.351990, 0.316821, 0.601507],
                [0.574091, 0.565377, 0.295292],
                [1.404124, 0.883492, 0.344245],
            ],
            dtype=jnp.float32,
        ),
    )


def test_order_one_center_cp_texture_fits_quark_benchmark() -> None:
    up_masses, down_masses, ckm, lambda_rec = _pdg_2025_quark_benchmark()
    target = fn_quark_yukawas_from_masses_ckm(up_masses, down_masses, ckm)
    coefficients = sm_center_cp_order_one_coefficients(_verdict_magnitudes(), powers=VERDICT_CENTER_HOLONOMY_POWERS)
    residuals = sm_center_cp_order_one_fit_residuals(coefficients, target, lambda_rec=lambda_rec)

    assert residuals.magnitude_min > 0.1
    assert residuals.magnitude_max < 10.0
    assert residuals.up_mass_log_rms < 0.003
    assert residuals.down_mass_log_rms < 0.003
    assert residuals.ckm_abs_residual < 0.0025
    assert residuals.jarlskog_relative_residual < 0.001


def test_order_one_center_cp_fit_optimizes_bounded_magnitudes() -> None:
    up_masses, down_masses, ckm, lambda_rec = _pdg_2025_quark_benchmark()
    target = fn_quark_yukawas_from_masses_ckm(up_masses, down_masses, ckm)
    unit_magnitudes = FNQuarkCoefficientMatrices(
        up=jnp.ones((3, 3), dtype=jnp.float32),
        down=jnp.ones((3, 3), dtype=jnp.float32),
    )
    unit_coefficients = sm_center_cp_order_one_coefficients(unit_magnitudes, powers=VERDICT_CENTER_HOLONOMY_POWERS)
    initial = sm_center_cp_order_one_fit_residuals(unit_coefficients, target, lambda_rec=lambda_rec)

    fit = sm_fit_center_cp_order_one_magnitudes(
        target,
        powers=VERDICT_CENTER_HOLONOMY_POWERS,
        lambda_rec=lambda_rec,
        steps=12,
        learning_rate=0.035,
    )

    assert fit.residuals.objective < initial.objective
    assert fit.residuals.magnitude_min >= 0.1
    assert fit.residuals.magnitude_max <= 10.0
    assert sm_center_phase_closure_residual(fit.factorization.center_phases.up) < 1e-6
    assert sm_center_phase_closure_residual(fit.factorization.center_phases.down) < 1e-6


def test_antiparticle_conjugation_preserves_singular_masses() -> None:
    yukawas = sm_center_cp_quark_yukawas()

    assert sm_quark_antiquark_mass_residual(yukawas) < 1e-7
    assert jnp.max(jnp.abs(fn_singular_masses(yukawas.up) - fn_singular_masses(jnp.conj(yukawas.up)))) < 1e-7
    assert (
        jnp.max(jnp.abs(fn_singular_masses(yukawas.down) - fn_singular_masses(jnp.conj(yukawas.down)))) < 1e-7
    )


def test_center_holonomy_gives_cp_odd_signal_while_real_control_does_not() -> None:
    center_yukawas = sm_center_cp_quark_yukawas()
    real_yukawas = fn_quark_yukawa_matrices()
    center_ckm = fn_ckm_from_yukawas(center_yukawas.up, center_yukawas.down)
    real_ckm = fn_ckm_from_yukawas(real_yukawas.up, real_yukawas.down)

    assert fn_unitarity_residual(center_ckm) < 3e-6
    assert jnp.abs(sm_ckm_jarlskog(center_ckm)) > 1e-7
    assert jnp.abs(sm_ckm_jarlskog(real_ckm)) < 1e-10
    assert jnp.abs(sm_yukawa_commutator_cp_trace(center_yukawas.up, center_yukawas.down)) > 1e-12
    assert jnp.abs(sm_yukawa_commutator_cp_trace(real_yukawas.up, real_yukawas.down)) < 1e-18


def test_center_cp_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_center_cp_diagnostics()

    assert diagnostics.center_phase_unit_modulus_residual < 2e-7
    assert diagnostics.center_phase_closure_residual < 1e-6
    assert diagnostics.coefficient_magnitude_residual < 2e-7
    assert diagnostics.quark_antiquark_mass_residual < 1e-7
    assert diagnostics.ckm_unitarity_residual < 3e-6
    assert diagnostics.center_jarlskog_abs > 1e-7
    assert diagnostics.real_control_jarlskog_abs < 1e-10
    assert diagnostics.commutator_cp_abs > 1e-12
    assert diagnostics.jit_delta < 1e-6


def test_center_holonomy_phases_are_jittable() -> None:
    powers = sm_center_power_matrix(DEFAULT_CENTER_HOLONOMY_POWERS.up)
    expected = sm_center_holonomy_phases(powers)
    actual = jax.jit(sm_center_holonomy_phases)(powers)

    assert jnp.max(jnp.abs(actual - expected)) < 1e-6
