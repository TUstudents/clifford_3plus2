"""Tests for QCA_SMv0 center-holonomy CP coefficients."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_cp import (
    DEFAULT_CENTER_HOLONOMY_POWERS,
    SM_COLOR_CENTER_ORDER,
    sm_center_coefficients,
    sm_center_cp_diagnostics,
    sm_center_cp_quark_yukawas,
    sm_center_holonomy_phases,
    sm_center_phase_closure_residual,
    sm_center_phase_unit_modulus_residual,
    sm_center_power_matrix,
    sm_ckm_jarlskog,
    sm_factor_coefficients_to_center_phases,
    sm_quark_antiquark_mass_residual,
    sm_yukawa_commutator_cp_trace,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    FNQuarkCoefficientMatrices,
    fn_ckm_from_yukawas,
    fn_default_coefficients,
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
