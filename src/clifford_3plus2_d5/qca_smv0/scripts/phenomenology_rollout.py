"""Production-facing phenomenology rollout for QCA_SMv0.

This command is the constructive front door for the simulator: quark masses,
CKM data, ``lambda``, and FN charges define a calibrated center-CP FN texture;
that texture drives the local Higgs/FN collision inside a compact BCC field-QCA
rollout.  The script reports both rollout observables and coefficient
diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    sm_family_quark_path_hidden_dims,
    sm_family_yukawa_collision_cache,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FNQuarkCharges,
    SM_FAMILY_DIM,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_INTERNAL_DIM
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    deterministic_qca_family_state,
    sm_qca_center_cp_rollout_config,
    sm_qca_config_memory_footprint,
    sm_qca_rollout_config_from_masses_ckm,
    sm_run_qca_rollout,
)


DEFAULT_SCALE_LABEL = "benchmark"
DEFAULT_UP_MASSES_GEV = (0.00216, 1.2730, 172.57)
DEFAULT_DOWN_MASSES_GEV = (0.00467, 0.0935, 4.183)
DEFAULT_CKM_ANGLES = (0.22501, 0.04183, 0.003732, 1.147)
BYTES_PER_GIB = 1024**3


class PhenomenologyRunConfig(NamedTuple):
    """External inputs for one QCA_SMv0 phenomenology rollout."""

    scale_label: str = DEFAULT_SCALE_LABEL
    lattice_shape: tuple[int, int, int] = (2, 1, 1)
    steps: int = 4
    lambda_rec: float = DEFAULT_CKM_ANGLES[0]
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES
    up_masses: tuple[float, float, float] = DEFAULT_UP_MASSES_GEV
    down_masses: tuple[float, float, float] = DEFAULT_DOWN_MASSES_GEV
    mass_mode: str = "absolute"
    ckm_angles: tuple[float, float, float, float] = DEFAULT_CKM_ANGLES
    ckm_matrix: tuple[tuple[complex, complex, complex], tuple[complex, complex, complex], tuple[complex, complex, complex]] | None = None
    center_fit_steps: int = 0
    yukawa_step_size: float = 0.01
    higgs_vev: float = 1.0
    collision_mode: str = "fn_dilation"
    stream_mode: str = "hop_sum"
    memory_budget_gib: float | None = None
    memory_safety_factor: float = 0.8
    memory_policy: str = "none"


class PhenomenologyRolloutSummary(NamedTuple):
    """Compact numerical summary for one QCA_SMv0 phenomenology rollout."""

    scale_label: str
    selected_label: str
    passed_center_cp: bool
    status: str
    failure_reasons: tuple[str, ...]
    lambda_rec: float
    charges_q: tuple[int, int, int]
    charges_u: tuple[int, int, int]
    charges_d: tuple[int, int, int]
    up_input_masses: tuple[float, float, float]
    down_input_masses: tuple[float, float, float]
    up_fn_masses: tuple[float, float, float]
    down_fn_masses: tuple[float, float, float]
    up_mass_normalization: float
    down_mass_normalization: float
    objective: float
    up_mass_log_rms: float
    down_mass_log_rms: float
    ckm_abs_residual: float
    jarlskog_relative_residual: float
    jarlskog_candidate: float
    jarlskog_target: float
    magnitude_min: float
    magnitude_max: float
    magnitude_mean: float
    coefficient_residual: float
    phase_residual: float
    up_magnitudes: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    down_magnitudes: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    up_center_powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    down_center_powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    norm_initial: float
    norm_final: float
    norm_drift: float
    extended_norm_initial: float
    extended_norm_final: float
    extended_norm_drift: float
    max_density_initial: float
    max_density_final: float
    used_higgs_fn_collision: bool
    used_fn_dilation_collision: bool
    collision_mode: str
    stream_mode: str
    steps_completed: int
    visible_complex_elements: int
    fn_path_aux_complex_elements: int
    total_complex_elements: int
    state_complex64_bytes: int
    state_complex128_bytes: int
    config_array_elements: int
    config_array_bytes: int
    total_array_bytes: int
    complex64_bytes: int
    complex128_bytes: int


class _PreparedPhenomenologyRollout(NamedTuple):
    """Shared calibrated inputs for one or more rollout collision modes."""

    up_fn_masses: jnp.ndarray
    down_fn_masses: jnp.ndarray
    up_normalization: float
    down_normalization: float
    state: jnp.ndarray
    calibrated: Any


def _parse_float_triplet(text: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return values  # type: ignore[return-value]


def _parse_int_triplet(text: str) -> tuple[int, int, int]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return values  # type: ignore[return-value]


def _parse_ckm_angles(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected s12,s23,s13,delta")
    return values  # type: ignore[return-value]


def _as_python_tuple3(values: Any, *, label: str, cast: type = float) -> tuple[Any, Any, Any]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{label} must be a length-3 list")
    return tuple(cast(value) for value in values)  # type: ignore[return-value]


def _as_python_tuple4(values: Any, *, label: str, cast: type = float) -> tuple[Any, Any, Any, Any]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{label} must be a length-4 list")
    return tuple(cast(value) for value in values)  # type: ignore[return-value]


def _complex_from_json(value: Any) -> complex:
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, str):
        return complex(value.replace("i", "j"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, dict):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    raise ValueError(f"cannot parse CKM matrix entry {value!r}")


def _ckm_matrix_from_json(values: Any) -> tuple[tuple[complex, complex, complex], tuple[complex, complex, complex], tuple[complex, complex, complex]]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("ckm_matrix must be a 3x3 nested list")
    rows = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("ckm_matrix must be a 3x3 nested list")
        rows.append(tuple(_complex_from_json(entry) for entry in row))
    return tuple(rows)  # type: ignore[return-value]


def ckm_from_angles(angles: tuple[float, float, float, float]) -> jnp.ndarray:
    """Return a CKM matrix from ``s12,s23,s13,delta``."""

    s12, s23, s13, delta = (jnp.asarray(value, dtype=jnp.float32) for value in angles)
    c12 = jnp.sqrt(1.0 - s12 * s12)
    c23 = jnp.sqrt(1.0 - s23 * s23)
    c13 = jnp.sqrt(1.0 - s13 * s13)
    exp_pos = jnp.exp(1j * delta)
    exp_neg = jnp.exp(-1j * delta)
    return jnp.asarray(
        [
            [c12 * c13, s12 * c13, s13 * exp_neg],
            [
                -s12 * c23 - c12 * s23 * s13 * exp_pos,
                c12 * c23 - s12 * s23 * s13 * exp_pos,
                s23 * c13,
            ],
            [
                s12 * s23 - c12 * c23 * s13 * exp_pos,
                -c12 * s23 - s12 * c23 * s13 * exp_pos,
                c23 * c13,
            ],
        ],
        dtype=jnp.complex64,
    )


def benchmark_ckm() -> jnp.ndarray:
    """Return the default benchmark CKM matrix."""

    return ckm_from_angles(DEFAULT_CKM_ANGLES)


def _normalize_masses(
    masses: tuple[float, float, float],
    mass_mode: str,
    *,
    label: str,
) -> tuple[jnp.ndarray, float]:
    if mass_mode not in ("absolute", "ratios"):
        raise ValueError("mass_mode must be 'absolute' or 'ratios'")
    arr = jnp.asarray(masses, dtype=jnp.float32)
    if bool(jnp.any(arr <= 0.0)):
        raise ValueError(f"{label} masses must be positive")
    if mass_mode == "ratios":
        return arr, 1.0
    normalization = float(arr[-1])
    if normalization <= 0.0:
        raise ValueError(f"{label} heaviest mass normalization must be positive")
    return arr / normalization, normalization


def _matrix3_to_float_tuples(matrix: jnp.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    arr = jnp.asarray(matrix)
    return tuple(tuple(float(arr[i, j]) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _powers_to_tuples(powers: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return tuple(tuple(int(value) for value in row) for row in powers)  # type: ignore[return-value]


def load_phenomenology_config(path: str | Path) -> PhenomenologyRunConfig:
    """Load a run config from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    charges_data = data.get("charges", {})
    charges = FNQuarkCharges(
        q=_as_python_tuple3(charges_data.get("q", DEFAULT_FN_QUARK_CHARGES.q), label="charges.q", cast=int),
        u=_as_python_tuple3(charges_data.get("u", DEFAULT_FN_QUARK_CHARGES.u), label="charges.u", cast=int),
        d=_as_python_tuple3(charges_data.get("d", DEFAULT_FN_QUARK_CHARGES.d), label="charges.d", cast=int),
    )
    ckm_matrix = _ckm_matrix_from_json(data["ckm_matrix"]) if "ckm_matrix" in data else None
    ckm_angles = _as_python_tuple4(data["ckm_angles"], label="ckm_angles", cast=float) if "ckm_angles" in data else DEFAULT_CKM_ANGLES
    return PhenomenologyRunConfig(
        scale_label=str(data.get("scale_label", DEFAULT_SCALE_LABEL)),
        lattice_shape=_as_python_tuple3(data.get("lattice_shape", (2, 1, 1)), label="lattice_shape", cast=int),
        steps=int(data.get("steps", 4)),
        lambda_rec=float(data.get("lambda", data.get("lambda_rec", DEFAULT_CKM_ANGLES[0]))),
        charges=charges,
        up_masses=_as_python_tuple3(data.get("up_masses", DEFAULT_UP_MASSES_GEV), label="up_masses", cast=float),
        down_masses=_as_python_tuple3(data.get("down_masses", DEFAULT_DOWN_MASSES_GEV), label="down_masses", cast=float),
        mass_mode=str(data.get("mass_mode", "absolute")),
        ckm_angles=ckm_angles,  # type: ignore[arg-type]
        ckm_matrix=ckm_matrix,
        center_fit_steps=int(data.get("center_fit_steps", 0)),
        yukawa_step_size=float(data.get("yukawa_step_size", 0.01)),
        higgs_vev=float(data.get("higgs_vev", 1.0)),
        collision_mode=str(data.get("collision_mode", "fn_dilation")),
        stream_mode=str(data.get("stream_mode", "hop_sum")),
        memory_budget_gib=float(data["memory_budget_gib"]) if "memory_budget_gib" in data else None,
        memory_safety_factor=float(data.get("memory_safety_factor", 0.8)),
        memory_policy=str(data.get("memory_policy", "none")),
    )


def _summary_to_dict(summary: PhenomenologyRolloutSummary) -> dict[str, Any]:
    return {
        "scale_label": summary.scale_label,
        "selected_center_cp_texture": summary.selected_label,
        "center_cp_passed": summary.passed_center_cp,
        "status": summary.status,
        "failure_reasons": list(summary.failure_reasons),
        "lambda": summary.lambda_rec,
        "charges": {
            "q": list(summary.charges_q),
            "u": list(summary.charges_u),
            "d": list(summary.charges_d),
        },
        "input_masses": {
            "up": list(summary.up_input_masses),
            "down": list(summary.down_input_masses),
            "up_normalization": summary.up_mass_normalization,
            "down_normalization": summary.down_mass_normalization,
        },
        "fn_masses": {
            "up": list(summary.up_fn_masses),
            "down": list(summary.down_fn_masses),
        },
        "residuals": {
            "objective": summary.objective,
            "up_mass_log_rms": summary.up_mass_log_rms,
            "down_mass_log_rms": summary.down_mass_log_rms,
            "ckm_abs": summary.ckm_abs_residual,
            "jarlskog_relative": summary.jarlskog_relative_residual,
            "jarlskog_candidate": summary.jarlskog_candidate,
            "jarlskog_target": summary.jarlskog_target,
        },
        "coefficient_diagnostics": {
            "magnitude_min": summary.magnitude_min,
            "magnitude_max": summary.magnitude_max,
            "magnitude_mean": summary.magnitude_mean,
            "coefficient_residual": summary.coefficient_residual,
            "phase_residual": summary.phase_residual,
            "up_magnitudes": [list(row) for row in summary.up_magnitudes],
            "down_magnitudes": [list(row) for row in summary.down_magnitudes],
            "up_center_powers": [list(row) for row in summary.up_center_powers],
            "down_center_powers": [list(row) for row in summary.down_center_powers],
        },
        "rollout": {
            "collision_mode": summary.collision_mode,
            "stream_mode": summary.stream_mode,
            "norm_initial": summary.norm_initial,
            "norm_final": summary.norm_final,
            "norm_drift": summary.norm_drift,
            "extended_norm_initial": summary.extended_norm_initial,
            "extended_norm_final": summary.extended_norm_final,
            "extended_norm_drift": summary.extended_norm_drift,
            "max_density_initial": summary.max_density_initial,
            "max_density_final": summary.max_density_final,
            "used_higgs_fn_collision": summary.used_higgs_fn_collision,
            "used_fn_dilation_collision": summary.used_fn_dilation_collision,
            "steps_completed": summary.steps_completed,
            "memory": {
                "visible_complex_elements": summary.visible_complex_elements,
                "fn_path_aux_complex_elements": summary.fn_path_aux_complex_elements,
                "total_complex_elements": summary.total_complex_elements,
                "state_complex64_bytes": summary.state_complex64_bytes,
                "state_complex128_bytes": summary.state_complex128_bytes,
                "config_array_elements": summary.config_array_elements,
                "config_array_bytes": summary.config_array_bytes,
                "total_array_bytes": summary.total_array_bytes,
                "complex64_bytes": summary.complex64_bytes,
                "complex128_bytes": summary.complex128_bytes,
            },
        },
    }


def _prepare_phenomenology_rollout(
    config: PhenomenologyRunConfig,
    *,
    calibration_collision_mode: str,
) -> _PreparedPhenomenologyRollout:
    """Build the shared calibrated texture once for one or more rollout modes."""

    up_fn_masses, up_normalization = _normalize_masses(config.up_masses, config.mass_mode, label="up")
    down_fn_masses, down_normalization = _normalize_masses(config.down_masses, config.mass_mode, label="down")
    ckm = (
        jnp.asarray(config.ckm_matrix, dtype=jnp.complex64)
        if config.ckm_matrix is not None
        else ckm_from_angles(config.ckm_angles)
    )
    state = deterministic_qca_family_state(config.lattice_shape)
    calibrated = sm_qca_rollout_config_from_masses_ckm(
        up_fn_masses,
        down_fn_masses,
        ckm,
        config.lattice_shape,
        lambda_rec=config.lambda_rec,
        charges=config.charges,
        center_fit_steps=config.center_fit_steps,
        yukawa_step_size=config.yukawa_step_size,
        higgs_vev=config.higgs_vev,
        collision_mode=calibration_collision_mode,
        stream_mode=config.stream_mode,
        record_density=True,
    )
    return _PreparedPhenomenologyRollout(
        up_fn_masses=up_fn_masses,
        down_fn_masses=down_fn_masses,
        up_normalization=up_normalization,
        down_normalization=down_normalization,
        state=state,
        calibrated=calibrated,
    )


def _run_prepared_phenomenology_rollout(
    config: PhenomenologyRunConfig,
    prepared: _PreparedPhenomenologyRollout,
    *,
    collision_mode: str,
) -> PhenomenologyRolloutSummary:
    """Run one collision mode from an already-calibrated phenomenology texture."""

    collision_cache = (
        sm_family_yukawa_collision_cache(
            prepared.calibrated.config.higgs,
            step_size=config.yukawa_step_size,
            quark_yukawas=prepared.calibrated.config.quark_yukawas,
            lepton_yukawas=prepared.calibrated.config.lepton_yukawas,
            assume_uniform=True,
            use_unitary_gauge_blocks=True,
        )
        if collision_mode == "effective_yukawa"
        and prepared.calibrated.config.higgs is not None
        and config.yukawa_step_size != 0.0
        else None
    )
    rollout_config = prepared.calibrated.config._replace(
        collision_mode=collision_mode,
        stream_mode=config.stream_mode,
        quark_path_readouts=prepared.calibrated.config.quark_path_readouts
        if collision_mode == "fn_dilation"
        else None,
        family_yukawa_collision_cache=collision_cache,
    )
    result = sm_run_qca_rollout(rollout_config, prepared.state, steps=config.steps)
    calibrated = prepared.calibrated
    residuals = calibrated.verdict.fit.residuals
    factorization = calibrated.verdict.fit.factorization
    norm_initial = float(result.norm_history[0])
    norm_final = float(result.norm_history[-1])
    extended_norm_initial = float(result.extended_norm_history[0])
    extended_norm_final = float(result.extended_norm_history[-1])
    footprint = result.memory_footprint
    rollout_footprint = result.rollout_memory_footprint
    return PhenomenologyRolloutSummary(
        scale_label=config.scale_label,
        selected_label=calibrated.verdict.selected_label,
        passed_center_cp=bool(calibrated.verdict.passed),
        status=calibrated.verdict.status,
        failure_reasons=tuple(calibrated.verdict.failure_reasons),
        lambda_rec=float(config.lambda_rec),
        charges_q=tuple(int(value) for value in config.charges.q),
        charges_u=tuple(int(value) for value in config.charges.u),
        charges_d=tuple(int(value) for value in config.charges.d),
        up_input_masses=tuple(float(value) for value in config.up_masses),
        down_input_masses=tuple(float(value) for value in config.down_masses),
        up_fn_masses=tuple(float(value) for value in prepared.up_fn_masses),
        down_fn_masses=tuple(float(value) for value in prepared.down_fn_masses),
        up_mass_normalization=float(prepared.up_normalization),
        down_mass_normalization=float(prepared.down_normalization),
        objective=float(residuals.objective),
        up_mass_log_rms=float(residuals.up_mass_log_rms),
        down_mass_log_rms=float(residuals.down_mass_log_rms),
        ckm_abs_residual=float(residuals.ckm_abs_residual),
        jarlskog_relative_residual=float(residuals.jarlskog_relative_residual),
        jarlskog_candidate=float(residuals.candidate_jarlskog),
        jarlskog_target=float(residuals.target_jarlskog),
        magnitude_min=float(residuals.magnitude_min),
        magnitude_max=float(residuals.magnitude_max),
        magnitude_mean=float(residuals.magnitude_mean),
        coefficient_residual=float(factorization.coefficient_residual),
        phase_residual=float(factorization.phase_residual),
        up_magnitudes=_matrix3_to_float_tuples(factorization.magnitudes.up),
        down_magnitudes=_matrix3_to_float_tuples(factorization.magnitudes.down),
        up_center_powers=_powers_to_tuples(factorization.center_powers.up),
        down_center_powers=_powers_to_tuples(factorization.center_powers.down),
        norm_initial=norm_initial,
        norm_final=norm_final,
        norm_drift=abs(norm_final - norm_initial),
        extended_norm_initial=extended_norm_initial,
        extended_norm_final=extended_norm_final,
        extended_norm_drift=abs(extended_norm_final - extended_norm_initial),
        max_density_initial=float(result.max_density_history[0]),
        max_density_final=float(result.max_density_history[-1]),
        used_higgs_fn_collision=bool(result.used_higgs_fn_collision),
        used_fn_dilation_collision=bool(result.used_fn_dilation_collision),
        collision_mode=result.collision_mode,
        stream_mode=config.stream_mode,
        steps_completed=int(result.steps_completed),
        visible_complex_elements=int(footprint.visible_complex_elements),
        fn_path_aux_complex_elements=int(footprint.fn_path_aux_complex_elements),
        total_complex_elements=int(footprint.total_complex_elements),
        state_complex64_bytes=int(footprint.complex64_bytes),
        state_complex128_bytes=int(footprint.complex128_bytes),
        config_array_elements=int(rollout_footprint.config_array_elements),
        config_array_bytes=int(rollout_footprint.config_array_bytes),
        total_array_bytes=int(rollout_footprint.total_array_bytes),
        complex64_bytes=int(rollout_footprint.total_array_bytes),
        complex128_bytes=int(footprint.complex128_bytes + rollout_footprint.config_array_bytes),
    )


def run_phenomenology_rollout(config: PhenomenologyRunConfig = PhenomenologyRunConfig()) -> PhenomenologyRolloutSummary:
    """Run masses/CKM through the compact QCA_SMv0 rollout."""

    if config.collision_mode == "both":
        raise ValueError("run_phenomenology_rollout expects one collision mode; use run_phenomenology_rollout_modes for 'both'")
    prepared = _prepare_phenomenology_rollout(config, calibration_collision_mode=config.collision_mode)
    return _run_prepared_phenomenology_rollout(
        config,
        prepared,
        collision_mode=config.collision_mode,
    )


def _collision_modes(collision_mode: str) -> tuple[str, ...]:
    if collision_mode == "both":
        return ("fn_dilation", "effective_yukawa")
    if collision_mode in ("fn_dilation", "effective_yukawa"):
        return (collision_mode,)
    raise ValueError("collision_mode must be 'fn_dilation', 'effective_yukawa', or 'both'")


def _mode_memory_estimate(config: PhenomenologyRunConfig, collision_mode: str) -> dict[str, Any]:
    if collision_mode not in ("fn_dilation", "effective_yukawa"):
        raise ValueError("collision_mode must be 'fn_dilation' or 'effective_yukawa'")
    sites = int(config.lattice_shape[0] * config.lattice_shape[1] * config.lattice_shape[2])
    visible_complex_elements = sites * 4 * SM_INTERNAL_DIM * SM_FAMILY_DIM
    fn_path_aux_complex_elements = 0
    if collision_mode == "fn_dilation":
        dims = sm_family_quark_path_hidden_dims(charges=config.charges)
        hidden_dim = int(dims.up + dims.down)
        fn_path_aux_complex_elements = sites * 4 * 3 * 2 * hidden_dim
    state_complex_elements = visible_complex_elements + fn_path_aux_complex_elements
    mode_config = sm_qca_center_cp_rollout_config(
        config.lattice_shape,
        lambda_rec=config.lambda_rec,
        charges=config.charges,
        yukawa_step_size=config.yukawa_step_size,
        higgs_vev=config.higgs_vev,
        collision_mode=collision_mode,
        stream_mode=config.stream_mode,
        record_density=False,
    )
    config_footprint = sm_qca_config_memory_footprint(mode_config)
    state_complex64_bytes = 8 * state_complex_elements
    state_complex128_bytes = 16 * state_complex_elements
    complex64_bytes = state_complex64_bytes + config_footprint.array_bytes
    complex128_bytes = state_complex128_bytes + config_footprint.array_bytes
    return {
        "collision_mode": collision_mode,
        "stream_mode": config.stream_mode,
        "lattice_shape": list(config.lattice_shape),
        "sites": sites,
        "visible_complex_elements": visible_complex_elements,
        "fn_path_aux_complex_elements": fn_path_aux_complex_elements,
        "state_complex_elements": state_complex_elements,
        "config_array_elements": config_footprint.array_elements,
        "config_array_bytes": config_footprint.array_bytes,
        "total_complex_elements": state_complex_elements,
        "state_complex64_bytes": state_complex64_bytes,
        "state_complex128_bytes": state_complex128_bytes,
        "total_array_bytes": complex64_bytes,
        "complex64_bytes": complex64_bytes,
        "complex128_bytes": complex128_bytes,
        "complex64_bytes_per_site": complex64_bytes / sites,
        "complex128_bytes_per_site": complex128_bytes / sites,
    }


def _memory_fit(config: PhenomenologyRunConfig, memory: dict[str, Any]) -> dict[str, Any] | None:
    if config.memory_budget_gib is None:
        return None
    if config.memory_budget_gib <= 0.0:
        raise ValueError("memory_budget_gib must be positive")
    if not 0.0 < config.memory_safety_factor <= 1.0:
        raise ValueError("memory_safety_factor must be in (0, 1]")
    usable_bytes = int(config.memory_budget_gib * BYTES_PER_GIB * config.memory_safety_factor)
    complex64_bytes = int(memory["complex64_bytes"])
    return {
        "estimate_scope": "runtime_complex64_state_plus_selected_config_arrays",
        "memory_budget_gib": float(config.memory_budget_gib),
        "memory_safety_factor": float(config.memory_safety_factor),
        "usable_bytes": usable_bytes,
        "complex64_bytes": complex64_bytes,
        "fits_memory_budget": complex64_bytes <= usable_bytes,
        "budget_fraction": float(complex64_bytes / usable_bytes),
        "complex64_bytes_margin": usable_bytes - complex64_bytes,
    }


def _memory_preflight_payload(config: PhenomenologyRunConfig) -> dict[str, Any]:
    probe_modes = list(_collision_modes(config.collision_mode))
    if config.memory_policy == "auto" and "effective_yukawa" not in probe_modes:
        probe_modes.append("effective_yukawa")
    estimates = []
    for mode in probe_modes:
        memory = _mode_memory_estimate(config, mode)
        fit = _memory_fit(config, memory)
        estimates.append(
            {
                "collision_mode": mode,
                "memory": memory,
                "memory_budget_fit": fit,
            },
        )
    return {
        "requested_collision_mode": config.collision_mode,
        "stream_mode": config.stream_mode,
        "memory_policy": config.memory_policy,
        "memory_budget_gib": config.memory_budget_gib,
        "memory_safety_factor": config.memory_safety_factor,
        "estimates": estimates,
    }


def _selected_collision_modes(config: PhenomenologyRunConfig) -> tuple[str, ...]:
    if config.memory_policy not in ("none", "fail", "auto"):
        raise ValueError("memory_policy must be 'none', 'fail', or 'auto'")
    requested_modes = _collision_modes(config.collision_mode)
    if config.memory_budget_gib is None or config.memory_policy == "none":
        return requested_modes

    fits = {
        mode: _memory_fit(config, _mode_memory_estimate(config, mode))
        for mode in ("fn_dilation", "effective_yukawa")
    }
    requested_fits = {mode: fits[mode] for mode in requested_modes}
    if config.memory_policy == "fail":
        failed_modes = [
            mode
            for mode, fit in requested_fits.items()
            if fit is not None and not fit["fits_memory_budget"]
        ]
        if failed_modes:
            raise ValueError(
                "requested collision mode exceeds memory budget: "
                + ", ".join(failed_modes),
            )
        return requested_modes

    selected_modes = []
    for mode in requested_modes:
        fit = fits[mode]
        if fit is not None and fit["fits_memory_budget"]:
            if mode not in selected_modes:
                selected_modes.append(mode)
            continue
        if mode == "fn_dilation":
            fallback_fit = fits["effective_yukawa"]
            if fallback_fit is not None and fallback_fit["fits_memory_budget"]:
                if "effective_yukawa" not in selected_modes:
                    selected_modes.append("effective_yukawa")
                continue
        raise ValueError(f"collision mode {mode!r} exceeds memory budget and no compressed fallback fits")
    return tuple(selected_modes)


def run_phenomenology_rollout_modes(
    config: PhenomenologyRunConfig = PhenomenologyRunConfig(),
) -> tuple[PhenomenologyRolloutSummary, ...]:
    """Run one or both calibrated rollout modes from the same phenomenology input."""

    selected_modes = _selected_collision_modes(config)
    prepared = _prepare_phenomenology_rollout(
        config,
        calibration_collision_mode=selected_modes[0],
    )
    return tuple(
        _run_prepared_phenomenology_rollout(config, prepared, collision_mode=mode)
        for mode in selected_modes
    )


def _rollout_mode_comparison(summaries: tuple[PhenomenologyRolloutSummary, ...]) -> dict[str, Any] | None:
    by_mode = {summary.collision_mode: summary for summary in summaries}
    exact = by_mode.get("fn_dilation")
    compressed = by_mode.get("effective_yukawa")
    if exact is None or compressed is None:
        return None
    return {
        "reference_collision_mode": "fn_dilation",
        "compressed_collision_mode": "effective_yukawa",
        "complex64_bytes_reference": exact.complex64_bytes,
        "complex64_bytes_compressed": compressed.complex64_bytes,
        "complex64_bytes_saved": exact.complex64_bytes - compressed.complex64_bytes,
        "memory_ratio_reference_to_compressed": float(exact.complex64_bytes / compressed.complex64_bytes),
        "extended_norm_drift_reference": exact.extended_norm_drift,
        "extended_norm_drift_compressed": compressed.extended_norm_drift,
    }


def _summaries_to_payload(
    config: PhenomenologyRunConfig,
    summaries: tuple[PhenomenologyRolloutSummary, ...],
) -> dict[str, Any]:
    if len(summaries) == 1:
        payload = _summary_to_dict(summaries[0])
        payload["requested_collision_mode"] = config.collision_mode
        payload["stream_mode"] = config.stream_mode
        payload["selected_collision_modes"] = [summary.collision_mode for summary in summaries]
        payload["memory_preflight"] = _memory_preflight_payload(config)
        return payload
    comparison = _rollout_mode_comparison(summaries)
    return {
        "scale_label": config.scale_label,
        "lambda": config.lambda_rec,
        "charges": {
            "q": list(config.charges.q),
            "u": list(config.charges.u),
            "d": list(config.charges.d),
        },
        "requested_collision_mode": config.collision_mode,
        "stream_mode": config.stream_mode,
        "collision_modes": [summary.collision_mode for summary in summaries],
        "memory_preflight": _memory_preflight_payload(config),
        "rollouts": {summary.collision_mode: _summary_to_dict(summary) for summary in summaries},
        "mode_comparison": comparison,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the production-facing rollout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON config file. CLI flags override matching fields.")
    parser.add_argument("--scale-label", help="Label for the input mass/CKM scale, e.g. MZ or mt.")
    parser.add_argument("--up-masses", type=_parse_float_triplet, help="Comma triple for up,c,t masses or ratios.")
    parser.add_argument("--down-masses", type=_parse_float_triplet, help="Comma triple for d,s,b masses or ratios.")
    parser.add_argument("--mass-mode", choices=("absolute", "ratios"), help="Interpret masses as absolute values or already-normalized ratios.")
    parser.add_argument("--ckm-angles", type=_parse_ckm_angles, help="Comma tuple s12,s23,s13,delta.")
    parser.add_argument("--lambda", dest="lambda_rec", type=float, help="FN recirculation attenuation lambda.")
    parser.add_argument("--q-charges", type=_parse_int_triplet, help="Comma triple for Q charges.")
    parser.add_argument("--u-charges", type=_parse_int_triplet, help="Comma triple for U charges.")
    parser.add_argument("--d-charges", type=_parse_int_triplet, help="Comma triple for D charges.")
    parser.add_argument("--lattice-shape", type=_parse_int_triplet, help="Comma triple nx,ny,nz.")
    parser.add_argument("--steps", type=int, help="Number of QCA rollout ticks.")
    parser.add_argument("--center-fit-steps", type=int, help="Bounded magnitude fit steps.")
    parser.add_argument("--yukawa-step-size", type=float, help="Local Higgs/FN collision step size.")
    parser.add_argument("--higgs-vev", type=float, help="Constant Higgs vev used by the local collision.")
    parser.add_argument(
        "--collision-mode",
        choices=("fn_dilation", "effective_yukawa", "both"),
        help=(
            "Use exact hidden FN path dilation as a reference, compressed "
            "effective Yukawa for production-scale runs, or both side by side."
        ),
    )
    parser.add_argument(
        "--stream-mode",
        choices=("hop_sum", "split_axis"),
        help="Use the fast free-stream kernel or the lower-temporary-memory split-axis kernel.",
    )
    parser.add_argument("--memory-budget-gib", type=float, help="Optional GPU memory budget in GiB.")
    parser.add_argument("--memory-safety-factor", type=float, help="Fraction of GPU budget treated as usable.")
    parser.add_argument(
        "--memory-policy",
        choices=("none", "fail", "auto"),
        help="Memory-budget policy: report only, fail before allocation, or auto-select compressed FN.",
    )
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser


def config_from_args(argv: list[str] | None = None) -> tuple[PhenomenologyRunConfig, str]:
    """Build a run config from CLI arguments and an optional JSON config."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_phenomenology_config(args.config) if args.config is not None else PhenomenologyRunConfig()
    charges = FNQuarkCharges(
        q=args.q_charges if args.q_charges is not None else config.charges.q,
        u=args.u_charges if args.u_charges is not None else config.charges.u,
        d=args.d_charges if args.d_charges is not None else config.charges.d,
    )
    return (
        PhenomenologyRunConfig(
            scale_label=args.scale_label if args.scale_label is not None else config.scale_label,
            lattice_shape=args.lattice_shape if args.lattice_shape is not None else config.lattice_shape,
            steps=args.steps if args.steps is not None else config.steps,
            lambda_rec=args.lambda_rec if args.lambda_rec is not None else config.lambda_rec,
            charges=charges,
            up_masses=args.up_masses if args.up_masses is not None else config.up_masses,
            down_masses=args.down_masses if args.down_masses is not None else config.down_masses,
            mass_mode=args.mass_mode if args.mass_mode is not None else config.mass_mode,
            ckm_angles=args.ckm_angles if args.ckm_angles is not None else config.ckm_angles,
            ckm_matrix=None if args.ckm_angles is not None else config.ckm_matrix,
            center_fit_steps=args.center_fit_steps if args.center_fit_steps is not None else config.center_fit_steps,
            yukawa_step_size=args.yukawa_step_size if args.yukawa_step_size is not None else config.yukawa_step_size,
            higgs_vev=args.higgs_vev if args.higgs_vev is not None else config.higgs_vev,
            collision_mode=args.collision_mode if args.collision_mode is not None else config.collision_mode,
            stream_mode=args.stream_mode if args.stream_mode is not None else config.stream_mode,
            memory_budget_gib=args.memory_budget_gib if args.memory_budget_gib is not None else config.memory_budget_gib,
            memory_safety_factor=(
                args.memory_safety_factor if args.memory_safety_factor is not None else config.memory_safety_factor
            ),
            memory_policy=args.memory_policy if args.memory_policy is not None else config.memory_policy,
        ),
        args.output,
    )


def _print_text_summary(summary: PhenomenologyRolloutSummary) -> None:
    print("QCA_SMv0 phenomenology rollout")
    print(f"  scale_label: {summary.scale_label}")
    print(f"  lambda: {summary.lambda_rec:.9g}")
    print(f"  charges_q/u/d: {summary.charges_q} / {summary.charges_u} / {summary.charges_d}")
    print(f"  selected_center_cp_texture: {summary.selected_label}")
    print(f"  center_cp_status: {summary.status}")
    print(f"  center_cp_passed: {summary.passed_center_cp}")
    if summary.failure_reasons:
        print(f"  failure_reasons: {','.join(summary.failure_reasons)}")
    print("  residuals:")
    print(f"    objective: {summary.objective:.6g}")
    print(f"    up_mass_log_rms: {summary.up_mass_log_rms:.6g}")
    print(f"    down_mass_log_rms: {summary.down_mass_log_rms:.6g}")
    print(f"    ckm_abs: {summary.ckm_abs_residual:.6g}")
    print(f"    jarlskog_relative: {summary.jarlskog_relative_residual:.6g}")
    print(f"    jarlskog_candidate: {summary.jarlskog_candidate:.6g}")
    print(f"    jarlskog_target: {summary.jarlskog_target:.6g}")
    print("  coefficient_diagnostics:")
    print(f"    magnitude_min/max/mean: {summary.magnitude_min:.6g} / {summary.magnitude_max:.6g} / {summary.magnitude_mean:.6g}")
    print(f"    coefficient_residual: {summary.coefficient_residual:.6g}")
    print(f"    phase_residual: {summary.phase_residual:.6g}")
    print(f"    up_center_powers: {summary.up_center_powers}")
    print(f"    down_center_powers: {summary.down_center_powers}")
    print(f"    up_magnitudes: {summary.up_magnitudes}")
    print(f"    down_magnitudes: {summary.down_magnitudes}")
    print("  rollout:")
    print(f"    collision_mode: {summary.collision_mode}")
    print(f"    stream_mode: {summary.stream_mode}")
    print(f"    norm_initial: {summary.norm_initial:.9g}")
    print(f"    norm_final: {summary.norm_final:.9g}")
    print(f"    norm_drift: {summary.norm_drift:.6g}")
    print(f"    extended_norm_initial: {summary.extended_norm_initial:.9g}")
    print(f"    extended_norm_final: {summary.extended_norm_final:.9g}")
    print(f"    extended_norm_drift: {summary.extended_norm_drift:.6g}")
    print(f"    max_density_initial: {summary.max_density_initial:.9g}")
    print(f"    max_density_final: {summary.max_density_final:.9g}")
    print(f"    used_higgs_fn_collision: {summary.used_higgs_fn_collision}")
    print(f"    used_fn_dilation_collision: {summary.used_fn_dilation_collision}")
    print(f"    steps_completed: {summary.steps_completed}")
    print("    memory:")
    print(f"      visible_complex_elements: {summary.visible_complex_elements}")
    print(f"      fn_path_aux_complex_elements: {summary.fn_path_aux_complex_elements}")
    print(f"      total_complex_elements: {summary.total_complex_elements}")
    print(f"      state_complex64_bytes: {summary.state_complex64_bytes}")
    print(f"      config_array_bytes: {summary.config_array_bytes}")
    print(f"      total_array_bytes: {summary.total_array_bytes}")
    print(f"      complex64_bytes: {summary.complex64_bytes}")
    print(f"      complex128_bytes: {summary.complex128_bytes}")


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""

    config, output = config_from_args(argv)
    summaries = run_phenomenology_rollout_modes(config)
    if output == "json":
        print(json.dumps(_summaries_to_payload(config, summaries), indent=2, sort_keys=True))
    else:
        for index, summary in enumerate(summaries):
            if index:
                print()
            _print_text_summary(summary)
        comparison = _rollout_mode_comparison(summaries)
        if comparison is not None:
            print()
            print("QCA_SMv0 rollout mode comparison")
            for key, value in comparison.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
