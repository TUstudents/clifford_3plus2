"""Center-holonomy CP coefficients for QCA_SMv0.

Stage 6 adds a constructive CP rule to the FN simulator layer.  Closed color
recirculation histories can contribute phases in the center of ``SU(3)_c``.
The phases enter only the order-one FN coefficient matrices.  Antiparticle
sectors use the conjugate Yukawas, so singular masses are unchanged while the
relative left-frame mismatch can carry a CP-odd Jarlskog invariant.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FN_LAMBDA_WOLFENSTEIN,
    FNQuarkCharges,
    FNQuarkYukawas,
    SM_FAMILY_DIM,
    fn_ckm_from_yukawas,
    fn_default_coefficients,
    fn_effective_yukawa,
    fn_quark_yukawa_matrices,
    fn_singular_masses,
    fn_unitarity_residual,
)

SM_COLOR_CENTER_ORDER = 3
SM_COLOR_CENTER_OMEGA = jnp.exp(2j * jnp.pi / SM_COLOR_CENTER_ORDER)
DEFAULT_UP_CENTER_POWERS = ((0, 1, 0), (2, 0, 1), (1, 2, 0))
DEFAULT_DOWN_CENTER_POWERS = ((0, 0, 1), (1, 0, 2), (2, 1, 0))


class CenterHolonomyPowers(NamedTuple):
    """Integer color-center powers for up/down coefficient matrices."""

    up: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = DEFAULT_UP_CENTER_POWERS
    down: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = DEFAULT_DOWN_CENTER_POWERS


class CenterCPDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 6 center-holonomy CP."""

    center_phase_unit_modulus_residual: jnp.ndarray
    center_phase_closure_residual: jnp.ndarray
    coefficient_magnitude_residual: jnp.ndarray
    quark_antiquark_mass_residual: jnp.ndarray
    ckm_unitarity_residual: jnp.ndarray
    center_jarlskog_abs: jnp.ndarray
    real_control_jarlskog_abs: jnp.ndarray
    commutator_cp_abs: jnp.ndarray
    jit_delta: jnp.ndarray


DEFAULT_CENTER_HOLONOMY_POWERS = CenterHolonomyPowers()


def _validate_family_matrix(matrix: jnp.ndarray, name: str) -> jnp.ndarray:
    arr = jnp.asarray(matrix)
    if arr.shape != (SM_FAMILY_DIM, SM_FAMILY_DIM):
        raise ValueError(f"{name} must have shape (3,3)")
    return arr


def sm_center_power_matrix(
    powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | jnp.ndarray,
) -> jnp.ndarray:
    """Return center powers reduced modulo the ``SU(3)`` center order."""

    arr = jnp.asarray(powers, dtype=jnp.int32)
    if arr.shape != (SM_FAMILY_DIM, SM_FAMILY_DIM):
        raise ValueError("center powers must have shape (3,3)")
    return jnp.mod(arr, SM_COLOR_CENTER_ORDER)


def sm_center_holonomy_phases(
    powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | jnp.ndarray,
) -> jnp.ndarray:
    """Return ``omega**powers`` for closed color-center holonomies."""

    reduced = sm_center_power_matrix(powers)
    return (SM_COLOR_CENTER_OMEGA ** reduced).astype(jnp.complex64)


def sm_center_phase_unit_modulus_residual(phases: jnp.ndarray) -> jnp.ndarray:
    """Return ``max_abs(|phase|-1)``."""

    arr = _validate_family_matrix(phases, "phases")
    return jnp.max(jnp.abs(jnp.abs(arr) - 1.0))


def sm_center_phase_closure_residual(phases: jnp.ndarray) -> jnp.ndarray:
    """Return ``max_abs(phase^3-1)`` for center-valued phases."""

    arr = _validate_family_matrix(phases, "phases")
    return jnp.max(jnp.abs(arr**SM_COLOR_CENTER_ORDER - 1.0))


def sm_center_coefficients(
    kind: str,
    *,
    powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | jnp.ndarray | None = None,
    base_coefficients: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return FN order-one coefficients decorated by center holonomies."""

    base = fn_default_coefficients(kind) if base_coefficients is None else _validate_family_matrix(base_coefficients, "base_coefficients")
    if powers is None:
        powers = DEFAULT_CENTER_HOLONOMY_POWERS.up if kind == "up" else DEFAULT_CENTER_HOLONOMY_POWERS.down
    return base.astype(jnp.complex64) * sm_center_holonomy_phases(powers)


def sm_center_cp_quark_yukawas(
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    powers: CenterHolonomyPowers = DEFAULT_CENTER_HOLONOMY_POWERS,
) -> FNQuarkYukawas:
    """Return quark Yukawas with color-center phases in FN coefficients."""

    up_coeffs = sm_center_coefficients("up", powers=powers.up)
    down_coeffs = sm_center_coefficients("down", powers=powers.down)
    return FNQuarkYukawas(
        up=fn_effective_yukawa(lambda_rec, charges.q, charges.u, coefficients=up_coeffs),
        down=fn_effective_yukawa(lambda_rec, charges.q, charges.d, coefficients=down_coeffs),
    )


def sm_antiparticle_yukawas(yukawas: FNQuarkYukawas) -> FNQuarkYukawas:
    """Return CPT-conjugate Yukawas for antiparticle sectors."""

    return FNQuarkYukawas(up=jnp.conj(yukawas.up), down=jnp.conj(yukawas.down))


def sm_quark_antiquark_mass_residual(yukawas: FNQuarkYukawas) -> jnp.ndarray:
    """Return max singular-mass difference between quarks and antiquarks."""

    anti = sm_antiparticle_yukawas(yukawas)
    up_residual = jnp.max(jnp.abs(fn_singular_masses(yukawas.up) - fn_singular_masses(anti.up)))
    down_residual = jnp.max(jnp.abs(fn_singular_masses(yukawas.down) - fn_singular_masses(anti.down)))
    return jnp.maximum(up_residual, down_residual)


def sm_ckm_jarlskog(ckm: jnp.ndarray) -> jnp.ndarray:
    """Return a CKM Jarlskog quartet ``Im(V00 V11 V01* V10*)``."""

    matrix = _validate_family_matrix(ckm, "ckm")
    return jnp.imag(matrix[0, 0] * matrix[1, 1] * jnp.conj(matrix[0, 1]) * jnp.conj(matrix[1, 0]))


def sm_yukawa_commutator_cp_trace(up_yukawa: jnp.ndarray, down_yukawa: jnp.ndarray) -> jnp.ndarray:
    """Return ``Im Tr([Yu Yu^dag, Yd Yd^dag]^3)``."""

    up = _validate_family_matrix(up_yukawa, "up_yukawa")
    down = _validate_family_matrix(down_yukawa, "down_yukawa")
    h_up = up @ jnp.swapaxes(jnp.conj(up), -1, -2)
    h_down = down @ jnp.swapaxes(jnp.conj(down), -1, -2)
    commutator = h_up @ h_down - h_down @ h_up
    return jnp.imag(jnp.trace(commutator @ commutator @ commutator))


def sm_center_cp_diagnostics(lambda_rec: float = FN_LAMBDA_WOLFENSTEIN) -> CenterCPDiagnostics:
    """Return focused Stage 6 center-holonomy CP diagnostics."""

    phases = sm_center_holonomy_phases(DEFAULT_CENTER_HOLONOMY_POWERS.up)
    center_coeffs = sm_center_coefficients("up")
    real_coeffs = fn_default_coefficients("up")
    center_yukawas = sm_center_cp_quark_yukawas(lambda_rec=lambda_rec)
    real_yukawas = fn_quark_yukawa_matrices(lambda_rec=lambda_rec)
    center_ckm = fn_ckm_from_yukawas(center_yukawas.up, center_yukawas.down)
    real_ckm = fn_ckm_from_yukawas(real_yukawas.up, real_yukawas.down)
    jitted_phases = jax.jit(sm_center_holonomy_phases)(sm_center_power_matrix(DEFAULT_CENTER_HOLONOMY_POWERS.up))

    return CenterCPDiagnostics(
        center_phase_unit_modulus_residual=sm_center_phase_unit_modulus_residual(phases),
        center_phase_closure_residual=sm_center_phase_closure_residual(phases),
        coefficient_magnitude_residual=jnp.max(jnp.abs(jnp.abs(center_coeffs) - jnp.abs(real_coeffs))),
        quark_antiquark_mass_residual=sm_quark_antiquark_mass_residual(center_yukawas),
        ckm_unitarity_residual=fn_unitarity_residual(center_ckm),
        center_jarlskog_abs=jnp.abs(sm_ckm_jarlskog(center_ckm)),
        real_control_jarlskog_abs=jnp.abs(sm_ckm_jarlskog(real_ckm)),
        commutator_cp_abs=jnp.abs(sm_yukawa_commutator_cp_trace(center_yukawas.up, center_yukawas.down)),
        jit_delta=jnp.max(jnp.abs(jitted_phases - phases)),
    )
