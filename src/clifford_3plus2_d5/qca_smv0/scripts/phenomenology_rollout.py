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

from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FNQuarkCharges,
)
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    deterministic_qca_family_state,
    sm_qca_rollout_config_from_masses_ckm,
    sm_run_qca_rollout,
)


DEFAULT_SCALE_LABEL = "benchmark"
DEFAULT_UP_MASSES_GEV = (0.00216, 1.2730, 172.57)
DEFAULT_DOWN_MASSES_GEV = (0.00467, 0.0935, 4.183)
DEFAULT_CKM_ANGLES = (0.22501, 0.04183, 0.003732, 1.147)


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
    steps_completed: int


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
        },
    }


def run_phenomenology_rollout(config: PhenomenologyRunConfig = PhenomenologyRunConfig()) -> PhenomenologyRolloutSummary:
    """Run masses/CKM through the compact QCA_SMv0 rollout."""

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
        collision_mode=config.collision_mode,
        record_density=True,
    )
    result = sm_run_qca_rollout(calibrated.config, state, steps=config.steps)
    residuals = calibrated.verdict.fit.residuals
    factorization = calibrated.verdict.fit.factorization
    norm_initial = float(result.norm_history[0])
    norm_final = float(result.norm_history[-1])
    extended_norm_initial = float(result.extended_norm_history[0])
    extended_norm_final = float(result.extended_norm_history[-1])
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
        up_fn_masses=tuple(float(value) for value in up_fn_masses),
        down_fn_masses=tuple(float(value) for value in down_fn_masses),
        up_mass_normalization=float(up_normalization),
        down_mass_normalization=float(down_normalization),
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
        steps_completed=int(result.steps_completed),
    )


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
        choices=("fn_dilation", "effective_yukawa"),
        help="Use explicit hidden FN path dilation or the old effective Yukawa collision.",
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


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""

    config, output = config_from_args(argv)
    summary = run_phenomenology_rollout(config)
    if output == "json":
        print(json.dumps(_summary_to_dict(summary), indent=2, sort_keys=True))
    else:
        _print_text_summary(summary)


if __name__ == "__main__":
    main()
