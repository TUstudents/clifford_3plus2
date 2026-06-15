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
    FNQuarkCoefficientMatrices,
    FNQuarkYukawas,
    SM_FAMILY_DIM,
    fn_ckm_from_yukawas,
    fn_default_coefficients,
    fn_effective_yukawa,
    fn_quark_coefficients_from_yukawas,
    fn_quark_yukawas_from_masses_ckm,
    fn_quark_yukawa_matrices,
    fn_singular_masses,
    fn_unitarity_residual,
)

SM_COLOR_CENTER_ORDER = 3
SM_COLOR_CENTER_OMEGA = jnp.exp(2j * jnp.pi / SM_COLOR_CENTER_ORDER)
DEFAULT_UP_CENTER_POWERS = ((0, 1, 0), (2, 0, 1), (1, 2, 0))
DEFAULT_DOWN_CENTER_POWERS = ((0, 0, 1), (1, 0, 2), (2, 1, 0))
VERDICT_UP_CENTER_POWERS = ((2, 1, 1), (1, 0, 0), (0, 2, 0))
VERDICT_DOWN_CENTER_POWERS = ((1, 1, 1), (2, 0, 0), (1, 2, 0))
ZERO_CENTER_POWERS = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
VERDICT_UP_MAGNITUDES = ((1.272654, 0.785760, 1.852192), (0.785760, 1.272654, 0.458693), (3.961016, 3.418301, 0.987706))
VERDICT_DOWN_MAGNITUDES = (
    (0.351990, 0.316821, 0.601507),
    (0.574091, 0.565377, 0.295292),
    (1.404124, 0.883492, 0.344245),
)


class CenterHolonomyPowers(NamedTuple):
    """Integer color-center powers for up/down coefficient matrices."""

    up: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = DEFAULT_UP_CENTER_POWERS
    down: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] = DEFAULT_DOWN_CENTER_POWERS


class CenterPowerMatrixAnalysis(NamedTuple):
    """Gauge-invariant ``Z3`` structure of one center-power matrix."""

    powers: jnp.ndarray
    rank_mod3: int
    base_power: int
    row_gradient: jnp.ndarray
    column_gradient: jnp.ndarray
    coboundary: jnp.ndarray
    curvature: jnp.ndarray
    elementary_fluxes: jnp.ndarray
    elementary_wilson_phases: jnp.ndarray
    curvature_nonzero_count: int
    elementary_flux_nonzero_count: int
    is_pure_coboundary: bool
    is_single_flux_defect: bool


class CenterPowerPairAnalysis(NamedTuple):
    """Relative ``Z3`` structure of the fitted up/down center powers."""

    up: CenterPowerMatrixAnalysis
    down: CenterPowerMatrixAnalysis
    down_minus_up: CenterPowerMatrixAnalysis
    relative_nonzero_rows: jnp.ndarray
    relative_nonzero_columns: jnp.ndarray
    relative_nonzero_row_count: int
    relative_nonzero_column_count: int
    relative_is_rank_one: bool
    relative_is_single_column_defect: bool


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


class CenterCPCoefficientFactorization(NamedTuple):
    """Projection of calibrated FN coefficients onto center-holonomy phases."""

    magnitudes: FNQuarkCoefficientMatrices
    center_powers: CenterHolonomyPowers
    center_phases: FNQuarkCoefficientMatrices
    reconstructed_coefficients: FNQuarkCoefficientMatrices
    coefficient_residual: jnp.ndarray
    phase_residual: jnp.ndarray


class CenterCPObjectiveWeights(NamedTuple):
    """Weights for the constrained center-CP fit objective."""

    up_mass: float = 1.0
    down_mass: float = 1.0
    ckm_abs: float = 1.0
    jarlskog: float = 1.0


class CenterCPOrderOneFitWeights(NamedTuple):
    """Weights for bounded order-one center-CP texture fitting."""

    up_mass_log: float = 1.0
    down_mass_log: float = 1.0
    ckm_abs: float = 200.0
    jarlskog: float = 5.0
    magnitude_log: float = 0.002


class CenterCPPhenomenologyThresholds(NamedTuple):
    """Pass/fail thresholds for the compact center-CP phenomenology verdict."""

    up_mass_log_rms: float = 0.02
    down_mass_log_rms: float = 0.02
    ckm_abs: float = 0.01
    jarlskog_relative: float = 0.05
    magnitude_min: float = 0.1
    magnitude_max: float = 10.0


class CenterCPResidualBreakdown(NamedTuple):
    """Residuals for one center-holonomy coefficient assignment."""

    objective: jnp.ndarray
    up_mass_relative_residual: jnp.ndarray
    down_mass_relative_residual: jnp.ndarray
    ckm_abs_residual: jnp.ndarray
    jarlskog_relative_residual: jnp.ndarray
    yukawa_relative_residual: jnp.ndarray
    candidate_jarlskog: jnp.ndarray
    target_jarlskog: jnp.ndarray


class CenterCPOrderOneFitResiduals(NamedTuple):
    """Physics residuals for an order-one center-phase FN texture."""

    objective: jnp.ndarray
    up_mass_log_rms: jnp.ndarray
    down_mass_log_rms: jnp.ndarray
    ckm_abs_residual: jnp.ndarray
    jarlskog_relative_residual: jnp.ndarray
    candidate_jarlskog: jnp.ndarray
    target_jarlskog: jnp.ndarray
    magnitude_min: jnp.ndarray
    magnitude_max: jnp.ndarray
    magnitude_mean: jnp.ndarray


class CenterCPOrderOneTextureFit(NamedTuple):
    """Bounded-magnitude fit for one discrete center-power texture."""

    factorization: CenterCPCoefficientFactorization
    residuals: CenterCPOrderOneFitResiduals
    steps_completed: int
    learning_rate: float
    magnitude_bounds: tuple[float, float]


class CenterCPTextureSeed(NamedTuple):
    """One candidate center-power texture and optional magnitude warm start."""

    powers: CenterHolonomyPowers
    initial_magnitudes: FNQuarkCoefficientMatrices | None = None
    label: str = ""


class CenterCPPhenomenologyVerdict(NamedTuple):
    """Compact quark flavor verdict for one FN center-CP fit."""

    fit: CenterCPOrderOneTextureFit
    target_yukawas: FNQuarkYukawas
    selected_powers: CenterHolonomyPowers
    selected_label: str
    charges: FNQuarkCharges
    lambda_rec: float
    passed: bool
    status: str
    failure_reasons: tuple[str, ...]


class CenterCPInputVariation(NamedTuple):
    """Deterministic benchmark variation for robustness scans."""

    up_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    down_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ckm: jnp.ndarray | None = None
    lambda_rec: float | None = None
    label: str = "baseline"


class CenterCPRobustnessReport(NamedTuple):
    """Robustness summary over input variations and charge choices."""

    verdicts: tuple[CenterCPPhenomenologyVerdict, ...]
    pass_count: int
    total_count: int
    pass_fraction: float
    all_passed: bool
    best_objective: jnp.ndarray
    worst_objective: jnp.ndarray


class CenterCPCoefficientSearchResult(NamedTuple):
    """Coordinate-descent search result over discrete center powers."""

    initial_factorization: CenterCPCoefficientFactorization
    best_factorization: CenterCPCoefficientFactorization
    initial_residuals: CenterCPResidualBreakdown
    best_residuals: CenterCPResidualBreakdown
    sweeps_completed: int


DEFAULT_CENTER_HOLONOMY_POWERS = CenterHolonomyPowers()
VERDICT_CENTER_HOLONOMY_POWERS = CenterHolonomyPowers(
    up=VERDICT_UP_CENTER_POWERS,
    down=VERDICT_DOWN_CENTER_POWERS,
)
DEFAULT_CENTER_CP_OBJECTIVE_WEIGHTS = CenterCPObjectiveWeights()
DEFAULT_CENTER_CP_ORDER_ONE_FIT_WEIGHTS = CenterCPOrderOneFitWeights()
DEFAULT_CENTER_CP_PHENOMENOLOGY_THRESHOLDS = CenterCPPhenomenologyThresholds()


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


def sm_center_power_rank_mod3(
    powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | jnp.ndarray,
) -> int:
    """Return the rank of a center-power matrix over the finite field ``Z3``."""

    matrix = [[int(value) % SM_COLOR_CENTER_ORDER for value in row] for row in sm_center_power_matrix(powers).tolist()]
    rank = 0
    rows = len(matrix)
    cols = len(matrix[0])
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if matrix[row][col] % SM_COLOR_CENTER_ORDER:
                pivot = row
                break
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = 1 if matrix[rank][col] == 1 else 2
        matrix[rank] = [(inverse * value) % SM_COLOR_CENTER_ORDER for value in matrix[rank]]
        for row in range(rows):
            if row == rank:
                continue
            factor = matrix[row][col] % SM_COLOR_CENTER_ORDER
            if factor:
                matrix[row] = [
                    (matrix[row][entry] - factor * matrix[rank][entry]) % SM_COLOR_CENTER_ORDER
                    for entry in range(cols)
                ]
        rank += 1
        if rank == rows:
            break
    return rank


def sm_center_power_coboundary(
    powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | jnp.ndarray,
) -> tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split center powers into anchored row/column coboundary plus curvature.

    A removable family rephasing has the form ``base + r_i + c_j`` over ``Z3``.
    The returned curvature is the invariant remainder after subtracting that
    anchored row/column part.
    """

    matrix = sm_center_power_matrix(powers)
    base = int(matrix[0, 0].item())
    row_gradient = jnp.mod(matrix[:, 0] - base, SM_COLOR_CENTER_ORDER)
    column_gradient = jnp.mod(matrix[0, :] - base, SM_COLOR_CENTER_ORDER)
    coboundary = jnp.mod(
        base + row_gradient[:, None] + column_gradient[None, :],
        SM_COLOR_CENTER_ORDER,
    )
    curvature = jnp.mod(matrix - coboundary, SM_COLOR_CENTER_ORDER)
    return base, row_gradient, column_gradient, coboundary, curvature


def sm_center_power_plaquette_fluxes(
    powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | jnp.ndarray,
) -> jnp.ndarray:
    """Return oriented elementary plaquette fluxes of a ``3x3`` power matrix."""

    matrix = sm_center_power_matrix(powers)
    fluxes = matrix[:-1, :-1] - matrix[1:, :-1] - matrix[:-1, 1:] + matrix[1:, 1:]
    return jnp.mod(fluxes, SM_COLOR_CENTER_ORDER)


def sm_analyze_center_power_matrix(
    powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | jnp.ndarray,
) -> CenterPowerMatrixAnalysis:
    """Analyze one fitted color-center power matrix as a ``Z3`` lattice field."""

    matrix = sm_center_power_matrix(powers)
    base, row_gradient, column_gradient, coboundary, curvature = sm_center_power_coboundary(matrix)
    elementary_fluxes = sm_center_power_plaquette_fluxes(matrix)
    curvature_nonzero_count = int(jnp.count_nonzero(curvature).item())
    elementary_flux_nonzero_count = int(jnp.count_nonzero(elementary_fluxes).item())
    return CenterPowerMatrixAnalysis(
        powers=matrix,
        rank_mod3=sm_center_power_rank_mod3(matrix),
        base_power=base,
        row_gradient=row_gradient,
        column_gradient=column_gradient,
        coboundary=coboundary,
        curvature=curvature,
        elementary_fluxes=elementary_fluxes,
        elementary_wilson_phases=(SM_COLOR_CENTER_OMEGA ** elementary_fluxes).astype(jnp.complex64),
        curvature_nonzero_count=curvature_nonzero_count,
        elementary_flux_nonzero_count=elementary_flux_nonzero_count,
        is_pure_coboundary=curvature_nonzero_count == 0 and elementary_flux_nonzero_count == 0,
        is_single_flux_defect=elementary_flux_nonzero_count == 1,
    )


def sm_analyze_center_power_pair(powers: CenterHolonomyPowers) -> CenterPowerPairAnalysis:
    """Analyze the fitted up/down center powers and their relative defect."""

    up = sm_analyze_center_power_matrix(powers.up)
    down = sm_analyze_center_power_matrix(powers.down)
    relative_matrix = jnp.mod(down.powers - up.powers, SM_COLOR_CENTER_ORDER)
    relative = sm_analyze_center_power_matrix(relative_matrix)
    relative_nonzero_rows = jnp.any(relative_matrix != 0, axis=1)
    relative_nonzero_columns = jnp.any(relative_matrix != 0, axis=0)
    relative_nonzero_row_count = int(jnp.count_nonzero(relative_nonzero_rows).item())
    relative_nonzero_column_count = int(jnp.count_nonzero(relative_nonzero_columns).item())
    return CenterPowerPairAnalysis(
        up=up,
        down=down,
        down_minus_up=relative,
        relative_nonzero_rows=relative_nonzero_rows,
        relative_nonzero_columns=relative_nonzero_columns,
        relative_nonzero_row_count=relative_nonzero_row_count,
        relative_nonzero_column_count=relative_nonzero_column_count,
        relative_is_rank_one=relative.rank_mod3 == 1,
        relative_is_single_column_defect=relative.rank_mod3 == 1 and relative_nonzero_column_count == 1,
    )


def sm_analyze_verdict_center_powers() -> CenterPowerPairAnalysis:
    """Return the ``Z3`` structure of the successful fitted center powers."""

    return sm_analyze_center_power_pair(VERDICT_CENTER_HOLONOMY_POWERS)


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


def _factor_matrix_to_center_phases(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    coeffs = _validate_family_matrix(matrix, "coefficients").astype(jnp.complex64)
    magnitudes = jnp.abs(coeffs)
    nonzero = magnitudes > 1e-30
    safe_magnitudes = jnp.where(nonzero, magnitudes, 1.0)
    normalized = jnp.where(nonzero, coeffs / safe_magnitudes, 1.0 + 0.0j)
    centers = (SM_COLOR_CENTER_OMEGA ** jnp.arange(SM_COLOR_CENTER_ORDER, dtype=jnp.int32)).astype(jnp.complex64)
    distances = jnp.abs(normalized[..., None] - centers)
    powers = jnp.where(nonzero, jnp.argmin(distances, axis=-1), 0).astype(jnp.int32)
    phases = sm_center_holonomy_phases(powers)
    reconstructed = magnitudes.astype(jnp.complex64) * phases
    coefficient_residual = jnp.max(jnp.abs(reconstructed - coeffs))
    phase_residual = jnp.max(jnp.where(nonzero, jnp.abs(normalized - phases), 0.0))
    return magnitudes, powers, phases, reconstructed, coefficient_residual, phase_residual


def sm_factor_coefficients_to_center_phases(
    coefficients: FNQuarkCoefficientMatrices,
) -> CenterCPCoefficientFactorization:
    """Factor calibrated FN coefficients as magnitude times nearest center phase.

    This is the constrained CP readout used by the simulator mode: arbitrary
    calibrated coefficient phases are projected to the nearest element of the
    ``SU(3)`` center while their magnitudes are kept fixed.
    """

    up_magnitudes, up_powers, up_phases, up_reconstructed, up_coefficient_residual, up_phase_residual = _factor_matrix_to_center_phases(
        coefficients.up,
    )
    down_magnitudes, down_powers, down_phases, down_reconstructed, down_coefficient_residual, down_phase_residual = (
        _factor_matrix_to_center_phases(coefficients.down)
    )
    return CenterCPCoefficientFactorization(
        magnitudes=FNQuarkCoefficientMatrices(up=up_magnitudes, down=down_magnitudes),
        center_powers=CenterHolonomyPowers(up=up_powers, down=down_powers),
        center_phases=FNQuarkCoefficientMatrices(up=up_phases, down=down_phases),
        reconstructed_coefficients=FNQuarkCoefficientMatrices(up=up_reconstructed, down=down_reconstructed),
        coefficient_residual=jnp.maximum(up_coefficient_residual, down_coefficient_residual),
        phase_residual=jnp.maximum(up_phase_residual, down_phase_residual),
    )


def _factor_coefficients_with_center_powers(
    coefficients: FNQuarkCoefficientMatrices,
    magnitudes: FNQuarkCoefficientMatrices,
    powers: CenterHolonomyPowers,
) -> CenterCPCoefficientFactorization:
    up_powers = sm_center_power_matrix(powers.up)
    down_powers = sm_center_power_matrix(powers.down)
    up_phases = sm_center_holonomy_phases(up_powers)
    down_phases = sm_center_holonomy_phases(down_powers)
    up_reconstructed = jnp.asarray(magnitudes.up, dtype=jnp.complex64) * up_phases
    down_reconstructed = jnp.asarray(magnitudes.down, dtype=jnp.complex64) * down_phases
    up_coeffs = _validate_family_matrix(coefficients.up, "coefficients.up").astype(jnp.complex64)
    down_coeffs = _validate_family_matrix(coefficients.down, "coefficients.down").astype(jnp.complex64)
    up_nonzero = jnp.asarray(magnitudes.up) > 1e-30
    down_nonzero = jnp.asarray(magnitudes.down) > 1e-30
    up_normalized = jnp.where(up_nonzero, up_coeffs / jnp.where(up_nonzero, jnp.asarray(magnitudes.up), 1.0), 1.0 + 0.0j)
    down_normalized = jnp.where(down_nonzero, down_coeffs / jnp.where(down_nonzero, jnp.asarray(magnitudes.down), 1.0), 1.0 + 0.0j)
    return CenterCPCoefficientFactorization(
        magnitudes=magnitudes,
        center_powers=CenterHolonomyPowers(up=up_powers, down=down_powers),
        center_phases=FNQuarkCoefficientMatrices(up=up_phases, down=down_phases),
        reconstructed_coefficients=FNQuarkCoefficientMatrices(up=up_reconstructed, down=down_reconstructed),
        coefficient_residual=jnp.maximum(jnp.max(jnp.abs(up_reconstructed - up_coeffs)), jnp.max(jnp.abs(down_reconstructed - down_coeffs))),
        phase_residual=jnp.maximum(
            jnp.max(jnp.where(up_nonzero, jnp.abs(up_normalized - up_phases), 0.0)),
            jnp.max(jnp.where(down_nonzero, jnp.abs(down_normalized - down_phases), 0.0)),
        ),
    )


def _center_cp_yukawas_from_coefficients(
    coefficients: FNQuarkCoefficientMatrices,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
) -> FNQuarkYukawas:
    return FNQuarkYukawas(
        up=fn_effective_yukawa(lambda_rec, charges.q, charges.u, coefficients=coefficients.up),
        down=fn_effective_yukawa(lambda_rec, charges.q, charges.d, coefficients=coefficients.down),
    )


def _relative_matrix_residual(candidate: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.maximum(jnp.abs(target), 1.0)
    return jnp.max(jnp.abs(candidate - target) / scale)


def sm_center_cp_residual_breakdown(
    coefficients: FNQuarkCoefficientMatrices,
    target_yukawas: FNQuarkYukawas,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    weights: CenterCPObjectiveWeights = DEFAULT_CENTER_CP_OBJECTIVE_WEIGHTS,
) -> CenterCPResidualBreakdown:
    """Return residuals for a center-constrained FN coefficient assignment."""

    candidate_yukawas = _center_cp_yukawas_from_coefficients(coefficients, lambda_rec=lambda_rec, charges=charges)
    target_up_masses = fn_singular_masses(target_yukawas.up)
    target_down_masses = fn_singular_masses(target_yukawas.down)
    candidate_up_masses = fn_singular_masses(candidate_yukawas.up)
    candidate_down_masses = fn_singular_masses(candidate_yukawas.down)
    target_ckm = fn_ckm_from_yukawas(target_yukawas.up, target_yukawas.down)
    candidate_ckm = fn_ckm_from_yukawas(candidate_yukawas.up, candidate_yukawas.down)
    target_jarlskog = sm_ckm_jarlskog(target_ckm)
    candidate_jarlskog = sm_ckm_jarlskog(candidate_ckm)
    up_mass = jnp.max(jnp.abs(candidate_up_masses - target_up_masses) / jnp.maximum(jnp.abs(target_up_masses), 1e-30))
    down_mass = jnp.max(jnp.abs(candidate_down_masses - target_down_masses) / jnp.maximum(jnp.abs(target_down_masses), 1e-30))
    ckm_abs = jnp.max(jnp.abs(jnp.abs(candidate_ckm) - jnp.abs(target_ckm)))
    jarlskog = jnp.abs(candidate_jarlskog - target_jarlskog) / jnp.maximum(jnp.abs(target_jarlskog), 1e-12)
    yukawa = jnp.maximum(
        _relative_matrix_residual(candidate_yukawas.up, target_yukawas.up),
        _relative_matrix_residual(candidate_yukawas.down, target_yukawas.down),
    )
    objective = (
        jnp.asarray(weights.up_mass, dtype=jnp.float32) * up_mass
        + jnp.asarray(weights.down_mass, dtype=jnp.float32) * down_mass
        + jnp.asarray(weights.ckm_abs, dtype=jnp.float32) * ckm_abs
        + jnp.asarray(weights.jarlskog, dtype=jnp.float32) * jarlskog
    )
    return CenterCPResidualBreakdown(
        objective=objective,
        up_mass_relative_residual=up_mass,
        down_mass_relative_residual=down_mass,
        ckm_abs_residual=ckm_abs,
        jarlskog_relative_residual=jarlskog,
        yukawa_relative_residual=yukawa,
        candidate_jarlskog=candidate_jarlskog,
        target_jarlskog=target_jarlskog,
    )


def sm_center_cp_order_one_coefficients(
    magnitudes: FNQuarkCoefficientMatrices,
    *,
    powers: CenterHolonomyPowers = VERDICT_CENTER_HOLONOMY_POWERS,
) -> FNQuarkCoefficientMatrices:
    """Return bounded-magnitude coefficients with discrete center phases."""

    up_magnitudes = _validate_family_matrix(magnitudes.up, "magnitudes.up").astype(jnp.float32)
    down_magnitudes = _validate_family_matrix(magnitudes.down, "magnitudes.down").astype(jnp.float32)
    return FNQuarkCoefficientMatrices(
        up=up_magnitudes.astype(jnp.complex64) * sm_center_holonomy_phases(powers.up),
        down=down_magnitudes.astype(jnp.complex64) * sm_center_holonomy_phases(powers.down),
    )


def _center_cp_order_one_magnitude_stats(coefficients: FNQuarkCoefficientMatrices) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    magnitudes = jnp.concatenate((jnp.ravel(jnp.abs(coefficients.up)), jnp.ravel(jnp.abs(coefficients.down))))
    return jnp.min(magnitudes), jnp.max(magnitudes), jnp.mean(magnitudes)


def sm_center_cp_order_one_fit_residuals(
    coefficients: FNQuarkCoefficientMatrices,
    target_yukawas: FNQuarkYukawas,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    weights: CenterCPOrderOneFitWeights = DEFAULT_CENTER_CP_ORDER_ONE_FIT_WEIGHTS,
) -> CenterCPOrderOneFitResiduals:
    """Return mass, CKM, and CP residuals for a bounded center-phase texture."""

    candidate_yukawas = _center_cp_yukawas_from_coefficients(coefficients, lambda_rec=lambda_rec, charges=charges)
    target_up_masses = jnp.maximum(fn_singular_masses(target_yukawas.up), 1e-30)
    target_down_masses = jnp.maximum(fn_singular_masses(target_yukawas.down), 1e-30)
    candidate_up_masses = jnp.maximum(fn_singular_masses(candidate_yukawas.up), 1e-30)
    candidate_down_masses = jnp.maximum(fn_singular_masses(candidate_yukawas.down), 1e-30)
    target_ckm = fn_ckm_from_yukawas(target_yukawas.up, target_yukawas.down)
    candidate_ckm = fn_ckm_from_yukawas(candidate_yukawas.up, candidate_yukawas.down)
    target_jarlskog = sm_ckm_jarlskog(target_ckm)
    candidate_jarlskog = sm_ckm_jarlskog(candidate_ckm)
    up_log_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.log(candidate_up_masses / target_up_masses))))
    down_log_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.log(candidate_down_masses / target_down_masses))))
    ckm_delta = jnp.abs(candidate_ckm) - jnp.abs(target_ckm)
    ckm_abs = jnp.max(jnp.abs(ckm_delta))
    jarlskog_relative = jnp.abs(candidate_jarlskog - target_jarlskog) / jnp.maximum(jnp.abs(target_jarlskog), 1e-12)
    magnitude_min, magnitude_max, magnitude_mean = _center_cp_order_one_magnitude_stats(coefficients)
    log_magnitudes = jnp.log(jnp.maximum(jnp.abs(jnp.concatenate((jnp.ravel(coefficients.up), jnp.ravel(coefficients.down)))), 1e-30))
    objective = (
        jnp.asarray(weights.up_mass_log, dtype=jnp.float32) * up_log_rms**2
        + jnp.asarray(weights.down_mass_log, dtype=jnp.float32) * down_log_rms**2
        + jnp.asarray(weights.ckm_abs, dtype=jnp.float32) * jnp.mean(jnp.square(ckm_delta))
        + jnp.asarray(weights.jarlskog, dtype=jnp.float32) * jarlskog_relative**2
        + jnp.asarray(weights.magnitude_log, dtype=jnp.float32) * jnp.mean(jnp.square(log_magnitudes))
    )
    return CenterCPOrderOneFitResiduals(
        objective=objective,
        up_mass_log_rms=up_log_rms,
        down_mass_log_rms=down_log_rms,
        ckm_abs_residual=ckm_abs,
        jarlskog_relative_residual=jarlskog_relative,
        candidate_jarlskog=candidate_jarlskog,
        target_jarlskog=target_jarlskog,
        magnitude_min=magnitude_min,
        magnitude_max=magnitude_max,
        magnitude_mean=magnitude_mean,
    )


def _validate_magnitude_bounds(bounds: tuple[float, float]) -> tuple[jnp.ndarray, jnp.ndarray]:
    low, high = bounds
    if low <= 0.0 or high <= low:
        raise ValueError("magnitude_bounds must satisfy 0 < low < high")
    return jnp.log(jnp.asarray(low, dtype=jnp.float32)), jnp.log(jnp.asarray(high, dtype=jnp.float32))


def sm_center_cp_verdict_magnitudes() -> FNQuarkCoefficientMatrices:
    """Return the order-one magnitudes from the first successful verdict fit."""

    return FNQuarkCoefficientMatrices(
        up=jnp.asarray(VERDICT_UP_MAGNITUDES, dtype=jnp.float32),
        down=jnp.asarray(VERDICT_DOWN_MAGNITUDES, dtype=jnp.float32),
    )


def sm_default_center_cp_texture_seeds() -> tuple[CenterCPTextureSeed, ...]:
    """Return production candidate seeds for compact center-CP phenomenology."""

    return (
        CenterCPTextureSeed(
            powers=VERDICT_CENTER_HOLONOMY_POWERS,
            initial_magnitudes=sm_center_cp_verdict_magnitudes(),
            label="verdict",
        ),
        CenterCPTextureSeed(
            powers=DEFAULT_CENTER_HOLONOMY_POWERS,
            initial_magnitudes=None,
            label="default",
        ),
        CenterCPTextureSeed(
            powers=CenterHolonomyPowers(up=ZERO_CENTER_POWERS, down=ZERO_CENTER_POWERS),
            initial_magnitudes=None,
            label="all_zero",
        ),
    )


def _bounded_magnitudes_from_raw(raw: jnp.ndarray, magnitude_bounds: tuple[float, float]) -> FNQuarkCoefficientMatrices:
    log_low, log_high = _validate_magnitude_bounds(magnitude_bounds)
    logs = log_low + jax.nn.sigmoid(jnp.asarray(raw, dtype=jnp.float32)) * (log_high - log_low)
    mags = jnp.exp(logs).reshape((2, SM_FAMILY_DIM, SM_FAMILY_DIM))
    return FNQuarkCoefficientMatrices(up=mags[0], down=mags[1])


def _raw_from_magnitudes(magnitudes: FNQuarkCoefficientMatrices, magnitude_bounds: tuple[float, float]) -> jnp.ndarray:
    log_low, log_high = _validate_magnitude_bounds(magnitude_bounds)
    stacked = jnp.stack(
        (
            _validate_family_matrix(magnitudes.up, "initial_magnitudes.up").astype(jnp.float32),
            _validate_family_matrix(magnitudes.down, "initial_magnitudes.down").astype(jnp.float32),
        ),
    )
    scaled = (jnp.log(jnp.clip(stacked, min=jnp.exp(log_low), max=jnp.exp(log_high))) - log_low) / (log_high - log_low)
    clipped = jnp.clip(scaled, 1e-6, 1.0 - 1e-6)
    return jnp.log(clipped / (1.0 - clipped)).reshape((-1,))


def sm_fit_center_cp_order_one_magnitudes(
    target_yukawas: FNQuarkYukawas,
    *,
    powers: CenterHolonomyPowers = VERDICT_CENTER_HOLONOMY_POWERS,
    initial_magnitudes: FNQuarkCoefficientMatrices | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    magnitude_bounds: tuple[float, float] = (0.1, 10.0),
    weights: CenterCPOrderOneFitWeights = DEFAULT_CENTER_CP_ORDER_ONE_FIT_WEIGHTS,
    steps: int = 600,
    learning_rate: float = 0.035,
) -> CenterCPOrderOneTextureFit:
    """Fit order-one magnitudes for fixed discrete center powers.

    This is the production form of the quark-flavor existence test: powers are
    restricted to the ``SU(3)_c`` center, magnitudes are bounded in an explicit
    order-one interval, and the objective compares singular masses, CKM moduli,
    and the CP-odd Jarlskog invariant.
    """

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    _validate_magnitude_bounds(magnitude_bounds)
    if initial_magnitudes is None:
        raw = jnp.zeros((2 * SM_FAMILY_DIM * SM_FAMILY_DIM,), dtype=jnp.float32)
    else:
        raw = _raw_from_magnitudes(initial_magnitudes, magnitude_bounds)

    def coefficients_from_raw(raw_values: jnp.ndarray) -> tuple[FNQuarkCoefficientMatrices, FNQuarkCoefficientMatrices]:
        magnitudes = _bounded_magnitudes_from_raw(raw_values, magnitude_bounds)
        return magnitudes, sm_center_cp_order_one_coefficients(magnitudes, powers=powers)

    def loss(raw_values: jnp.ndarray) -> jnp.ndarray:
        _, coefficients = coefficients_from_raw(raw_values)
        return sm_center_cp_order_one_fit_residuals(
            coefficients,
            target_yukawas,
            lambda_rec=lambda_rec,
            charges=charges,
            weights=weights,
        ).objective

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    beta1 = jnp.asarray(0.9, dtype=jnp.float32)
    beta2 = jnp.asarray(0.999, dtype=jnp.float32)
    eps = jnp.asarray(1e-8, dtype=jnp.float32)
    step_size = jnp.asarray(learning_rate, dtype=jnp.float32)
    first_moment = jnp.zeros_like(raw)
    second_moment = jnp.zeros_like(raw)
    for step in range(steps):
        _, grad = value_and_grad(raw)
        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * jnp.square(grad)
        step_count = jnp.asarray(step + 1, dtype=jnp.float32)
        first_hat = first_moment / (1.0 - beta1**step_count)
        second_hat = second_moment / (1.0 - beta2**step_count)
        raw = raw - step_size * first_hat / (jnp.sqrt(second_hat) + eps)

    magnitudes, coefficients = coefficients_from_raw(raw)
    reference = fn_quark_coefficients_from_yukawas(target_yukawas, lambda_rec=lambda_rec, charges=charges)
    factorization = _factor_coefficients_with_center_powers(reference, magnitudes, powers)
    residuals = sm_center_cp_order_one_fit_residuals(
        coefficients,
        target_yukawas,
        lambda_rec=lambda_rec,
        charges=charges,
        weights=weights,
    )
    return CenterCPOrderOneTextureFit(
        factorization=factorization,
        residuals=residuals,
        steps_completed=steps,
        learning_rate=learning_rate,
        magnitude_bounds=magnitude_bounds,
    )


def _center_cp_verdict_failures(
    residuals: CenterCPOrderOneFitResiduals,
    thresholds: CenterCPPhenomenologyThresholds,
) -> tuple[str, ...]:
    failures = []
    if float(residuals.up_mass_log_rms) > thresholds.up_mass_log_rms:
        failures.append("up_mass")
    if float(residuals.down_mass_log_rms) > thresholds.down_mass_log_rms:
        failures.append("down_mass")
    if float(residuals.ckm_abs_residual) > thresholds.ckm_abs:
        failures.append("ckm")
    if float(residuals.jarlskog_relative_residual) > thresholds.jarlskog_relative:
        failures.append("jarlskog")
    if float(residuals.magnitude_min) < thresholds.magnitude_min:
        failures.append("magnitude_min")
    if float(residuals.magnitude_max) > thresholds.magnitude_max:
        failures.append("magnitude_max")
    return tuple(failures)


def sm_center_cp_phenomenology_verdict(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    candidate_seeds: tuple[CenterCPTextureSeed, ...] | None = None,
    magnitude_bounds: tuple[float, float] = (0.1, 10.0),
    thresholds: CenterCPPhenomenologyThresholds = DEFAULT_CENTER_CP_PHENOMENOLOGY_THRESHOLDS,
    weights: CenterCPOrderOneFitWeights = DEFAULT_CENTER_CP_ORDER_ONE_FIT_WEIGHTS,
    steps: int = 200,
    learning_rate: float = 0.035,
) -> CenterCPPhenomenologyVerdict:
    """Fit a compact FN center-CP quark flavor verdict from masses and CKM.

    The selected texture is the candidate seed whose bounded-magnitude fit has
    the smallest physics objective.  The returned pass/fail status is based on
    explicit mass, CKM, Jarlskog, and coefficient-magnitude thresholds.
    """

    seeds = sm_default_center_cp_texture_seeds() if candidate_seeds is None else candidate_seeds
    if not seeds:
        raise ValueError("candidate_seeds must contain at least one texture")
    target = fn_quark_yukawas_from_masses_ckm(up_masses, down_masses, ckm)
    best_fit: CenterCPOrderOneTextureFit | None = None
    best_seed: CenterCPTextureSeed | None = None
    for seed in seeds:
        fit = sm_fit_center_cp_order_one_magnitudes(
            target,
            powers=seed.powers,
            initial_magnitudes=seed.initial_magnitudes,
            lambda_rec=lambda_rec,
            charges=charges,
            magnitude_bounds=magnitude_bounds,
            weights=weights,
            steps=steps,
            learning_rate=learning_rate,
        )
        if best_fit is None or float(fit.residuals.objective) < float(best_fit.residuals.objective):
            best_fit = fit
            best_seed = seed
    if best_fit is None or best_seed is None:
        raise RuntimeError("center-CP verdict fitting failed to produce a candidate")
    failures = _center_cp_verdict_failures(best_fit.residuals, thresholds)
    passed = len(failures) == 0
    return CenterCPPhenomenologyVerdict(
        fit=best_fit,
        target_yukawas=target,
        selected_powers=best_fit.factorization.center_powers,
        selected_label=best_seed.label,
        charges=charges,
        lambda_rec=lambda_rec,
        passed=passed,
        status="pass" if passed else "fail",
        failure_reasons=failures,
    )


def sm_center_cp_robustness_scan(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges_options: tuple[FNQuarkCharges, ...] = (DEFAULT_FN_QUARK_CHARGES,),
    variations: tuple[CenterCPInputVariation, ...] = (CenterCPInputVariation(),),
    candidate_seeds: tuple[CenterCPTextureSeed, ...] | None = None,
    magnitude_bounds: tuple[float, float] = (0.1, 10.0),
    thresholds: CenterCPPhenomenologyThresholds = DEFAULT_CENTER_CP_PHENOMENOLOGY_THRESHOLDS,
    weights: CenterCPOrderOneFitWeights = DEFAULT_CENTER_CP_ORDER_ONE_FIT_WEIGHTS,
    steps: int = 200,
    learning_rate: float = 0.035,
) -> CenterCPRobustnessReport:
    """Run the compact verdict over input variations and charge choices."""

    if not charges_options:
        raise ValueError("charges_options must contain at least one charge assignment")
    if not variations:
        raise ValueError("variations must contain at least one case")
    base_up = jnp.asarray(up_masses, dtype=jnp.float32)
    base_down = jnp.asarray(down_masses, dtype=jnp.float32)
    verdicts = []
    for variation in variations:
        varied_up = base_up * jnp.asarray(variation.up_scale, dtype=jnp.float32)
        varied_down = base_down * jnp.asarray(variation.down_scale, dtype=jnp.float32)
        varied_ckm = ckm if variation.ckm is None else variation.ckm
        varied_lambda = lambda_rec if variation.lambda_rec is None else variation.lambda_rec
        for charges in charges_options:
            verdicts.append(
                sm_center_cp_phenomenology_verdict(
                    varied_up,
                    varied_down,
                    varied_ckm,
                    lambda_rec=varied_lambda,
                    charges=charges,
                    candidate_seeds=candidate_seeds,
                    magnitude_bounds=magnitude_bounds,
                    thresholds=thresholds,
                    weights=weights,
                    steps=steps,
                    learning_rate=learning_rate,
                ),
            )
    objectives = jnp.asarray([verdict.fit.residuals.objective for verdict in verdicts], dtype=jnp.float32)
    pass_count = sum(1 for verdict in verdicts if verdict.passed)
    total_count = len(verdicts)
    return CenterCPRobustnessReport(
        verdicts=tuple(verdicts),
        pass_count=pass_count,
        total_count=total_count,
        pass_fraction=pass_count / total_count,
        all_passed=pass_count == total_count,
        best_objective=jnp.min(objectives),
        worst_objective=jnp.max(objectives),
    )


def sm_search_center_cp_powers(
    coefficients: FNQuarkCoefficientMatrices,
    target_yukawas: FNQuarkYukawas,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    weights: CenterCPObjectiveWeights = DEFAULT_CENTER_CP_OBJECTIVE_WEIGHTS,
    max_sweeps: int = 2,
) -> CenterCPCoefficientSearchResult:
    """Coordinate-search center powers with fixed calibrated magnitudes."""

    if max_sweeps < 0:
        raise ValueError("max_sweeps must be nonnegative")
    initial = sm_factor_coefficients_to_center_phases(coefficients)
    best = initial
    best_residuals = sm_center_cp_residual_breakdown(
        initial.reconstructed_coefficients,
        target_yukawas,
        lambda_rec=lambda_rec,
        charges=charges,
        weights=weights,
    )
    initial_residuals = best_residuals
    sweeps_completed = 0
    for sweep in range(max_sweeps):
        improved = False
        for sector in ("up", "down"):
            for row in range(SM_FAMILY_DIM):
                for col in range(SM_FAMILY_DIM):
                    for power in range(SM_COLOR_CENTER_ORDER):
                        up_powers = jnp.asarray(best.center_powers.up, dtype=jnp.int32)
                        down_powers = jnp.asarray(best.center_powers.down, dtype=jnp.int32)
                        if sector == "up":
                            up_powers = up_powers.at[row, col].set(power)
                        else:
                            down_powers = down_powers.at[row, col].set(power)
                        candidate = _factor_coefficients_with_center_powers(
                            coefficients,
                            initial.magnitudes,
                            CenterHolonomyPowers(up=up_powers, down=down_powers),
                        )
                        candidate_residuals = sm_center_cp_residual_breakdown(
                            candidate.reconstructed_coefficients,
                            target_yukawas,
                            lambda_rec=lambda_rec,
                            charges=charges,
                            weights=weights,
                        )
                        if float(candidate_residuals.objective) < float(best_residuals.objective) - 1e-12:
                            best = candidate
                            best_residuals = candidate_residuals
                            improved = True
        sweeps_completed = sweep + 1
        if not improved:
            break
    return CenterCPCoefficientSearchResult(
        initial_factorization=initial,
        best_factorization=best,
        initial_residuals=initial_residuals,
        best_residuals=best_residuals,
        sweeps_completed=sweeps_completed,
    )


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
