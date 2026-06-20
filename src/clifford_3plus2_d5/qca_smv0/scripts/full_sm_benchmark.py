"""Canonical full Standard-Model-like QCA_SMv0 benchmark.

This benchmark activates the current constructive simulator stack in one
short production run:

* quark FN effective Yukawas calibrated from masses and CKM;
* charged-lepton diagonal Yukawas;
* PMNS neutrino Yukawas;
* optional Schur/seesaw certificate for the neutrino input.

The field rollout is one compressed production hot-path evolution.  The
attached quark and lepton response diagnostics use the same public production
APIs to make the run interpretable as physics, not just a norm smoke test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.scripts.lepton_phenomenology import (
    DEFAULT_CHARGED_LEPTON_MASSES_GEV,
    DEFAULT_NEUTRINO_SPLITTINGS,
    DEFAULT_YUKAWA_STEP_SIZE,
    _lepton_yukawa_inputs,
    run_lepton_phenomenology,
)
from clifford_3plus2_d5.qca_smv0.scripts.neutrino_probe import (
    WRONG_SECTOR_MAX,
    _parse_float_triplet,
    _parse_int_triplet,
)
from clifford_3plus2_d5.qca_smv0.scripts.phenomenology_rollout import (
    DEFAULT_DOWN_MASSES_GEV,
    DEFAULT_SCALE_LABEL,
    DEFAULT_UP_MASSES_GEV,
    _normalize_masses,
    benchmark_ckm,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import DEFAULT_FN_QUARK_CHARGES, FNQuarkCharges
from clifford_3plus2_d5.qca_smv0.sm_lepton import DEFAULT_PMNS_ANGLES, DEFAULT_STERILE_MAJORANA_MASSES
from clifford_3plus2_d5.qca_smv0.sm_lepton import sm_pmns_matrix
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    sm_qca_calibrated_coefficient_diagnostics,
    sm_qca_prepared_quark_family_response,
    sm_qca_prepare_calibrated_production_rollout,
    sm_run_jitted_qca_calibrated_production_rollout_with_observables,
)


DEFAULT_FULL_SM_LATTICE_SHAPE = (2, 1, 1)
DEFAULT_FULL_SM_STEPS = 2
FULL_SM_NORM_DRIFT_MAX = 2.0e-6
QUARK_TRANSFER_MIN = 1.0e-8


def _parse_float_pair(text: str) -> tuple[float, float]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated floats")
    return values  # type: ignore[return-value]


def _parse_float_quad(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated floats")
    return values  # type: ignore[return-value]


def _array_payload(array: Any) -> Any:
    arr = jnp.asarray(array)
    if arr.ndim == 0:
        value = arr.item()
        return float(value) if isinstance(value, float) else int(value)
    return arr.tolist()


def _namedtuple_payload(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {key: _namedtuple_payload(item) for key, item in value._asdict().items()}
    if isinstance(value, tuple):
        return [_namedtuple_payload(item) for item in value]
    if isinstance(value, list):
        return [_namedtuple_payload(item) for item in value]
    if hasattr(value, "shape"):
        return _array_payload(value)
    return value


def _matrix_abs_payload(matrix: jnp.ndarray) -> list[list[float]]:
    arr = jnp.abs(jnp.asarray(matrix))
    return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]


def _jarlskog(matrix: jnp.ndarray) -> float:
    arr = jnp.asarray(matrix)
    invariant = arr[0, 0] * arr[1, 1] * jnp.conj(arr[0, 1]) * jnp.conj(arr[1, 0])
    return float(jnp.imag(invariant))


def run_full_sm_benchmark(
    *,
    mode: str = "schur",
    scale_label: str = DEFAULT_SCALE_LABEL,
    lattice_shape: tuple[int, int, int] = DEFAULT_FULL_SM_LATTICE_SHAPE,
    steps: int = DEFAULT_FULL_SM_STEPS,
    charged_lepton_masses: tuple[float, float, float] = DEFAULT_CHARGED_LEPTON_MASSES_GEV,
    neutrino_splittings: tuple[float, float] = DEFAULT_NEUTRINO_SPLITTINGS,
    pmns_angles: tuple[float, float, float, float] = DEFAULT_PMNS_ANGLES,
    sterile_majorana_masses: tuple[float, float, float] = DEFAULT_STERILE_MAJORANA_MASSES,
    up_masses: tuple[float, float, float] = DEFAULT_UP_MASSES_GEV,
    down_masses: tuple[float, float, float] = DEFAULT_DOWN_MASSES_GEV,
    mass_mode: str = "absolute",
    lambda_rec: float = 0.22501,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    yukawa_step_size: float = DEFAULT_YUKAWA_STEP_SIZE,
) -> dict[str, Any]:
    """Run the canonical short full-SM benchmark and return a JSON payload."""

    up_fn_masses, up_normalization = _normalize_masses(up_masses, mass_mode, label="up")
    down_fn_masses, down_normalization = _normalize_masses(down_masses, mass_mode, label="down")
    lepton_yukawas, lepton_metadata = _lepton_yukawa_inputs(
        mode=mode,
        charged_lepton_masses=charged_lepton_masses,
        neutrino_masses=None,
        neutrino_splittings=neutrino_splittings,
        pmns_angles=pmns_angles,
        sterile_majorana_masses=sterile_majorana_masses,
    )
    pmns = sm_pmns_matrix(*pmns_angles)
    ckm = benchmark_ckm()
    setup = sm_qca_prepare_calibrated_production_rollout(
        up_fn_masses,
        down_fn_masses,
        ckm,
        lattice_shape,
        lepton_yukawas=lepton_yukawas,
        lambda_rec=lambda_rec,
        charges=charges,
        center_fit_steps=0,
        yukawa_step_size=yukawa_step_size,
    )
    observed = sm_run_jitted_qca_calibrated_production_rollout_with_observables(
        setup,
        steps=steps,
        donate_state=False,
    )
    quark_response = sm_qca_prepared_quark_family_response(setup, steps=1)
    coefficient_diagnostics = sm_qca_calibrated_coefficient_diagnostics(setup.verdict)
    lepton_summary = run_lepton_phenomenology(
        mode=mode,
        scale_label=scale_label,
        lattice_shape=(1, 1, 1),
        steps=1,
        charged_lepton_masses=charged_lepton_masses,
        neutrino_splittings=neutrino_splittings,
        pmns_angles=pmns_angles,
        sterile_majorana_masses=sterile_majorana_masses,
        up_masses=up_masses,
        down_masses=down_masses,
        mass_mode=mass_mode,
        lambda_rec=lambda_rec,
        charges=charges,
        yukawa_step_size=yukawa_step_size,
    )
    norm_drift = float(abs(observed.production_observables.norm_final - observed.production_observables.norm_initial))
    extended_norm_drift = float(
        abs(observed.production_observables.extended_norm_final - observed.production_observables.extended_norm_initial),
    )
    max_quark_up_population = float(jnp.max(quark_response.up_population))
    max_quark_down_population = float(jnp.max(quark_response.down_population))
    pmns_ckm_abs_mismatch = float(jnp.linalg.norm(jnp.abs(pmns) - jnp.abs(ckm)))
    pmns_offdiag_norm = float(jnp.linalg.norm(jnp.abs(pmns - jnp.diag(jnp.diag(pmns)))))
    ckm_offdiag_norm = float(jnp.linalg.norm(jnp.abs(ckm - jnp.diag(jnp.diag(ckm)))))
    default_jarlskog = abs(_jarlskog(pmns))
    cp_even_pmns = sm_pmns_matrix(pmns_angles[0], pmns_angles[1], pmns_angles[2], 0.0)
    cp_even_jarlskog = abs(_jarlskog(cp_even_pmns))
    checks = {
        "quark_fit": bool(setup.verdict.passed),
        "quark_carrier_transfer": max_quark_up_population > QUARK_TRANSFER_MIN
        and max_quark_down_population > QUARK_TRANSFER_MIN,
        "lepton_carrier_transfer": bool(lepton_summary["physics_tests"]["checks"]["charged_carrier_transfer"])
        and bool(lepton_summary["physics_tests"]["checks"]["neutrino_carrier_transfer"]),
        "pmns_response": bool(lepton_summary["physics_tests"]["checks"]["pmns_mixing"]),
        "schur_spectrum": bool(lepton_summary["physics_tests"]["checks"]["schur_spectrum"]),
        "norm_conservation": norm_drift < FULL_SM_NORM_DRIFT_MAX and extended_norm_drift < FULL_SM_NORM_DRIFT_MAX,
        "no_cross_lepton_leakage": bool(lepton_summary["physics_tests"]["checks"]["charged_no_nu_c_leakage"])
        and bool(lepton_summary["physics_tests"]["checks"]["neutrino_no_e_c_leakage"]),
        "memory_footprint": int(observed.production_observables.production.final_rollout_memory_footprint.total_array_bytes) > 0,
        "cp_phase_sensitivity": default_jarlskog > 1.0e-3 and cp_even_jarlskog < 1.0e-6,
        "pmns_ckm_frame_mismatch": pmns_ckm_abs_mismatch > 1.0 and pmns_offdiag_norm > ckm_offdiag_norm,
        "production_contract": bool(observed.production_contract.uses_production_api)
        and bool(observed.production_contract.lean_effective_yukawa)
        and not bool(observed.production_contract.raw_yukawa_arrays_present),
    }
    initial_memory = observed.production_observables.production.initial_rollout_memory_footprint
    final_memory = observed.production_observables.production.final_rollout_memory_footprint
    return {
        "benchmark": "full_sm_benchmark",
        "scale_label": scale_label,
        "mode": mode,
        "lattice_shape": list(lattice_shape),
        "steps": steps,
        "lambda": lambda_rec,
        "charges": {"q": list(charges.q), "u": list(charges.u), "d": list(charges.d)},
        "input_masses": {
            "up": list(up_masses),
            "down": list(down_masses),
            "up_normalization": up_normalization,
            "down_normalization": down_normalization,
        },
        "lepton_inputs": {
            "mode": lepton_metadata["mode"],
            "production_input": lepton_metadata["production_input"],
            "charged_lepton_yukawas": lepton_metadata["charged_lepton_yukawas"],
            "neutrino_masses": lepton_metadata["neutrino_masses"],
            "pmns_angles": lepton_metadata["pmns_angles"],
        },
        "rollout": {
            "collision_mode": observed.production_observables.collision_mode,
            "steps_completed": int(observed.production_observables.steps_completed),
            "norm_initial": float(observed.production_observables.norm_initial),
            "norm_final": float(observed.production_observables.norm_final),
            "norm_drift": norm_drift,
            "extended_norm_drift": extended_norm_drift,
            "carrier_sector_initial": [float(value) for value in observed.production_observables.carrier_populations_initial.sector],
            "carrier_sector_final": [float(value) for value in observed.production_observables.carrier_populations_final.sector],
        },
        "quark": {
            "coefficient_diagnostics": _namedtuple_payload(coefficient_diagnostics),
            "up_population_max": max_quark_up_population,
            "down_population_max": max_quark_down_population,
            "up_expected_one_tick_residual": float(quark_response.up_expected_one_tick_residual),
            "down_expected_one_tick_residual": float(quark_response.down_expected_one_tick_residual),
        },
        "leptons": {
            "diagnostics": lepton_summary["diagnostics"],
            "charged_probes": lepton_summary["charged_probes"],
            "neutrino_probes": lepton_summary["neutrino_probes"],
            "schur_backend": lepton_summary["lepton_inputs"]["schur_backend"],
        },
        "frame_diagnostics": {
            "ckm_abs": _matrix_abs_payload(ckm),
            "pmns_abs": _matrix_abs_payload(pmns),
            "pmns_ckm_abs_mismatch_norm": pmns_ckm_abs_mismatch,
            "pmns_offdiag_norm": pmns_offdiag_norm,
            "ckm_offdiag_norm": ckm_offdiag_norm,
            "pmns_jarlskog_abs": default_jarlskog,
            "pmns_delta_zero_jarlskog_abs": cp_even_jarlskog,
        },
        "memory": {
            "initial": _namedtuple_payload(initial_memory),
            "final": _namedtuple_payload(final_memory),
            "runtime_array_bytes": int(final_memory.total_array_bytes),
            "runtime_array_bytes_per_site": float(final_memory.total_array_bytes / math.prod(lattice_shape)),
        },
        "production_contract": _namedtuple_payload(observed.production_contract),
        "physics_tests": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "norm_drift_max": FULL_SM_NORM_DRIFT_MAX,
                "quark_transfer_min": QUARK_TRANSFER_MIN,
                "wrong_sector_max": WRONG_SECTOR_MAX,
            },
        },
    }


def payload_to_markdown(payload: dict[str, Any]) -> str:
    """Return a Markdown report for the full SM benchmark."""

    check_rows = "\n".join(
        f"| {label} | {passed} |" for label, passed in payload["physics_tests"]["checks"].items()
    )
    return f"""# QCA_SMv0 Full SM Benchmark

This canonical short rollout activates quark FN effective Yukawas, charged
lepton Yukawas, and PMNS neutrino Yukawas on the compressed production fibre.
It reports the field rollout, quark carrier response, lepton carrier response,
PMNS response, Schur spectrum check, norm conservation, and memory footprint.

## Inputs

- mode: `{payload["mode"]}`
- scale label: `{payload["scale_label"]}`
- lattice shape: `{tuple(payload["lattice_shape"])}`
- steps: `{payload["steps"]}`
- lepton production input: `{payload["lepton_inputs"]["production_input"]}`

## Rollout

- collision mode: `{payload["rollout"]["collision_mode"]}`
- norm drift: `{payload["rollout"]["norm_drift"]:.3g}`
- extended norm drift: `{payload["rollout"]["extended_norm_drift"]:.3g}`
- runtime array bytes: `{payload["memory"]["runtime_array_bytes"]}`
- runtime array bytes/site: `{payload["memory"]["runtime_array_bytes_per_site"]:.6g}`

## Physics Tests

- overall passed: `{payload["physics_tests"]["passed"]}`

| check | passed |
|---|---:|
{check_rows}

## Quark Response

- max up transfer population: `{payload["quark"]["up_population_max"]:.6g}`
- max down transfer population: `{payload["quark"]["down_population_max"]:.6g}`
- up one-tick residual: `{payload["quark"]["up_expected_one_tick_residual"]:.3g}`
- down one-tick residual: `{payload["quark"]["down_expected_one_tick_residual"]:.3g}`

## Lepton Response

- charged hierarchy input: `{tuple(payload["lepton_inputs"]["charged_lepton_yukawas"])}`
- neutrino masses: `{tuple(payload["lepton_inputs"]["neutrino_masses"])}`
- PMNS ratio error: `{payload["leptons"]["diagnostics"]["ratio_error"]:.3g}`
- Schur spectrum residual: `{payload["leptons"]["diagnostics"]["schur_spectrum_residual"]:.3g}`
- PMNS/CKM abs-frame mismatch norm: `{payload["frame_diagnostics"]["pmns_ckm_abs_mismatch_norm"]:.6g}`
- PMNS Jarlskog abs: `{payload["frame_diagnostics"]["pmns_jarlskog_abs"]:.6g}`
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "schur"), default="schur")
    parser.add_argument("--scale-label", default=DEFAULT_SCALE_LABEL)
    parser.add_argument("--lattice-shape", type=_parse_int_triplet, default=DEFAULT_FULL_SM_LATTICE_SHAPE)
    parser.add_argument("--steps", type=int, default=DEFAULT_FULL_SM_STEPS)
    parser.add_argument("--charged-lepton-masses", type=_parse_float_triplet, default=DEFAULT_CHARGED_LEPTON_MASSES_GEV)
    parser.add_argument("--neutrino-splittings", type=_parse_float_pair, default=DEFAULT_NEUTRINO_SPLITTINGS)
    parser.add_argument("--pmns-angles", type=lambda text: tuple(float(item.strip()) for item in text.split(",")), default=DEFAULT_PMNS_ANGLES)
    parser.add_argument("--yukawa-step-size", type=float, default=DEFAULT_YUKAWA_STEP_SIZE)
    parser.add_argument("--json-output-path", type=Path)
    parser.add_argument("--report-output-path", type=Path)
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _print_text(payload: dict[str, Any]) -> None:
    print("QCA_SMv0 full SM benchmark")
    print(f"  mode: {payload['mode']}")
    print(f"  physics_tests_passed: {payload['physics_tests']['passed']}")
    print(f"  norm_drift: {payload['rollout']['norm_drift']:.3g}")
    print(f"  runtime_array_bytes: {payload['memory']['runtime_array_bytes']}")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if len(args.pmns_angles) != 4:
        raise SystemExit("--pmns-angles must contain four comma-separated floats")
    payload = run_full_sm_benchmark(
        mode=args.mode,
        scale_label=args.scale_label,
        lattice_shape=args.lattice_shape,
        steps=args.steps,
        charged_lepton_masses=args.charged_lepton_masses,
        neutrino_splittings=args.neutrino_splittings,
        pmns_angles=args.pmns_angles,
        yukawa_step_size=args.yukawa_step_size,
    )
    json_payload = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output_path is not None:
        _write_text(args.json_output_path, json_payload + "\n")
    if args.report_output_path is not None:
        _write_text(args.report_output_path, payload_to_markdown(payload))
    if args.output == "json":
        print(json_payload)
    else:
        _print_text(payload)


if __name__ == "__main__":
    main()
