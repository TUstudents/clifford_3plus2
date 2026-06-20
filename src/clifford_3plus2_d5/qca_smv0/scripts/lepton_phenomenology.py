"""Production-facing lepton phenomenology command for QCA_SMv0.

The command builds explicit lepton Yukawa inputs from charged-lepton masses,
neutrino masses or splittings, and PMNS angles.  It supports two input modes:

``direct``
    The production collision consumes ``Y_nu = U_PMNS diag(m_nu)`` directly.

``schur``
    A type-I heavy-sterile Schur/seesaw block is built and checked first; the
    production collision then consumes the equivalent low-energy PMNS effective
    Yukawa matrix.  The heavy block is a backend certificate, not an extra hot
    path register.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.scripts.lepton_pmns_probe import (
    DEFAULT_YUKAWA_STEP_SIZE,
    NU_C_INTERNAL_INDEX,
    PMNS_NORM_DRIFT_MAX,
    PMNS_TRANSFER_MIN,
    PMNS_WEIGHT_TOLERANCE,
    SCHUR_SPECTRUM_TOLERANCE,
    _complex_matrix_payload,
    _normalized,
)
from clifford_3plus2_d5.qca_smv0.scripts.neutrino_probe import (
    DEFAULT_LATTICE_SHAPE,
    DEFAULT_NEUTRINO_PROBE_YUKAWAS,
    DEFAULT_STEPS,
    EPSILON,
    MASSLESS_TOLERANCE,
    RATIO_TOLERANCE,
    WRONG_SECTOR_MAX,
    _contract_dict,
    _mass_squared_ratio,
    _parse_float_triplet,
    _parse_int_triplet,
    _production_contract_passed,
)
from clifford_3plus2_d5.qca_smv0.scripts.phenomenology_rollout import (
    DEFAULT_DOWN_MASSES_GEV,
    DEFAULT_SCALE_LABEL,
    DEFAULT_UP_MASSES_GEV,
    _normalize_masses,
    benchmark_ckm,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_fn import DEFAULT_FN_QUARK_CHARGES, FNQuarkCharges
from clifford_3plus2_d5.qca_smv0.sm_lepton import (
    DEFAULT_PMNS_ANGLES,
    DEFAULT_STERILE_MAJORANA_MASSES,
    sm_lepton_direct_pmns_neutrino_yukawa,
    sm_lepton_pmns_expected_transfer_weights,
    sm_lepton_seesaw_schur_from_pmns,
    sm_pmns_matrix,
    sm_pmns_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_rollout import sm_run_jitted_qca_calibrated_carrier_basis_probe


DEFAULT_CHARGED_LEPTON_MASSES_GEV = (0.00051099895, 0.1056583755, 1.77686)
DEFAULT_NEUTRINO_SPLITTINGS = (0.02943725152285944, 1.0)
CHARGED_TRANSFER_MIN = 1.0e-12
CHARGED_MONOTONIC_TOLERANCE = 1.0e-15
CHARGED_LEPTON_INTERNAL_INDEX = 14


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


def _normalize_by_heaviest(values: tuple[float, float, float], *, allow_zero: bool, label: str) -> tuple[float, float, float]:
    if values[-1] <= 0.0:
        raise ValueError(f"{label} heaviest value must be positive")
    if allow_zero:
        if any(value < 0.0 for value in values):
            raise ValueError(f"{label} values must be non-negative")
    elif any(value <= 0.0 for value in values):
        raise ValueError(f"{label} values must be positive")
    return tuple(float(value / values[-1]) for value in values)  # type: ignore[return-value]


def _neutrino_masses_from_splittings(splittings: tuple[float, float]) -> tuple[float, float, float]:
    dm21, dm31 = splittings
    if dm21 <= 0.0 or dm31 <= 0.0 or dm21 >= dm31:
        raise ValueError("normal-ordering splittings must satisfy 0 < dm21 < dm31")
    return (0.0, math.sqrt(dm21 / dm31), 1.0)


def _family_populations(final_state: jnp.ndarray, internal_index: int, *, internal_copy: int = 0) -> tuple[float, float, float]:
    internal = internal_index + 16 * internal_copy
    populations = jnp.sum(jnp.abs(final_state[..., :, internal, :]) ** 2, axis=(0, 1, 2, 3))
    return tuple(float(value) for value in populations)  # type: ignore[return-value]


def _lepton_yukawa_inputs(
    *,
    mode: str,
    charged_lepton_masses: tuple[float, float, float],
    neutrino_masses: tuple[float, float, float] | None,
    neutrino_splittings: tuple[float, float] | None,
    pmns_angles: tuple[float, float, float, float],
    sterile_majorana_masses: tuple[float, float, float],
) -> tuple[FamilyLeptonYukawas, dict[str, Any]]:
    if mode not in ("direct", "schur"):
        raise ValueError("mode must be 'direct' or 'schur'")
    charged_yukawas = _normalize_by_heaviest(charged_lepton_masses, allow_zero=False, label="charged_lepton_masses")
    if neutrino_masses is None:
        resolved_neutrino_masses = (
            _neutrino_masses_from_splittings(neutrino_splittings)
            if neutrino_splittings is not None
            else DEFAULT_NEUTRINO_PROBE_YUKAWAS
        )
        neutrino_input = "splittings" if neutrino_splittings is not None else "default_eta"
    else:
        resolved_neutrino_masses = _normalize_by_heaviest(neutrino_masses, allow_zero=True, label="neutrino_masses")
        neutrino_input = "masses"
    pmns = sm_pmns_matrix(*pmns_angles)
    neutrino_yukawa = sm_lepton_direct_pmns_neutrino_yukawa(resolved_neutrino_masses, pmns)
    schur = sm_lepton_seesaw_schur_from_pmns(
        resolved_neutrino_masses,
        pmns,
        sterile_majorana_masses=sterile_majorana_masses,
    )
    lepton_yukawas = FamilyLeptonYukawas(
        neutrino=neutrino_yukawa,
        electron=jnp.diag(jnp.asarray(charged_yukawas, dtype=jnp.complex64)),
    )
    metadata = {
        "mode": mode,
        "production_input": "direct_pmns_effective_yukawa"
        if mode == "direct"
        else "schur_reduced_pmns_effective_yukawa",
        "neutrino_input": neutrino_input,
        "charged_lepton_masses": list(charged_lepton_masses),
        "charged_lepton_yukawas": list(charged_yukawas),
        "neutrino_masses": list(resolved_neutrino_masses),
        "neutrino_splittings": list(neutrino_splittings) if neutrino_splittings is not None else None,
        "pmns_angles": list(pmns_angles),
        "pmns_matrix": _complex_matrix_payload(pmns),
        "neutrino_yukawa_matrix": _complex_matrix_payload(neutrino_yukawa),
        "pmns_unitarity_residual": float(sm_pmns_unitarity_residual(pmns)),
        "schur_backend": {
            "enabled": mode == "schur",
            "kind": "type_i_seesaw_schur",
            "sterile_majorana_masses": [float(value) for value in sterile_majorana_masses],
            "target_masses": [float(value) for value in schur.target_masses],
            "recovered_masses": [float(value) for value in schur.recovered_masses],
            "spectrum_residual": float(schur.spectrum_residual),
            "effective_majorana": _complex_matrix_payload(schur.effective_majorana),
        },
    }
    return lepton_yukawas, metadata


def run_lepton_phenomenology(
    *,
    mode: str = "direct",
    scale_label: str = DEFAULT_SCALE_LABEL,
    lattice_shape: tuple[int, int, int] = DEFAULT_LATTICE_SHAPE,
    steps: int = DEFAULT_STEPS,
    charged_lepton_masses: tuple[float, float, float] = DEFAULT_CHARGED_LEPTON_MASSES_GEV,
    neutrino_masses: tuple[float, float, float] | None = None,
    neutrino_splittings: tuple[float, float] | None = DEFAULT_NEUTRINO_SPLITTINGS,
    pmns_angles: tuple[float, float, float, float] = DEFAULT_PMNS_ANGLES,
    sterile_majorana_masses: tuple[float, float, float] = DEFAULT_STERILE_MAJORANA_MASSES,
    up_masses: tuple[float, float, float] = DEFAULT_UP_MASSES_GEV,
    down_masses: tuple[float, float, float] = DEFAULT_DOWN_MASSES_GEV,
    mass_mode: str = "absolute",
    lambda_rec: float = 0.22501,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    yukawa_step_size: float = DEFAULT_YUKAWA_STEP_SIZE,
) -> dict[str, Any]:
    """Run a production-facing lepton phenomenology carrier benchmark."""

    up_fn_masses, up_normalization = _normalize_masses(up_masses, mass_mode, label="up")
    down_fn_masses, down_normalization = _normalize_masses(down_masses, mass_mode, label="down")
    lepton_yukawas, lepton_metadata = _lepton_yukawa_inputs(
        mode=mode,
        charged_lepton_masses=charged_lepton_masses,
        neutrino_masses=neutrino_masses,
        neutrino_splittings=neutrino_splittings,
        pmns_angles=pmns_angles,
        sterile_majorana_masses=sterile_majorana_masses,
    )
    neutrino_vector = tuple(float(value) for value in lepton_metadata["neutrino_masses"])
    charged_vector = tuple(float(value) for value in lepton_metadata["charged_lepton_yukawas"])
    pmns = sm_pmns_matrix(*pmns_angles)
    expected_pmns_weights = sm_lepton_pmns_expected_transfer_weights(neutrino_vector, pmns)
    charged_probes = []
    neutrino_probes = []
    contracts = []
    for family, label in enumerate(("e", "mu", "tau")):
        result = sm_run_jitted_qca_calibrated_carrier_basis_probe(
            up_fn_masses,
            down_fn_masses,
            benchmark_ckm(),
            lattice_shape,
            steps=steps,
            dirac=0,
            sector="L",
            weak=1,
            family=family,
            lepton_yukawas=lepton_yukawas,
            lambda_rec=lambda_rec,
            charges=charges,
            center_fit_steps=0,
            yukawa_step_size=yukawa_step_size,
            donate_state=False,
        )
        observed = result.production_observables
        final = observed.carrier_populations_final
        contracts.append(_contract_dict(result.production_contract))
        charged_family_pops = _family_populations(observed.production.final_qca_state.visible_state, CHARGED_LEPTON_INTERNAL_INDEX)
        charged_probes.append(
            {
                "family": family,
                "label": label,
                "charged_yukawa": charged_vector[family],
                "e_c_population": float(final.sector[4]),
                "nu_c_population": float(final.sector[5]),
                "e_c_family_populations": list(charged_family_pops),
                "norm_drift": float(abs(observed.norm_final - observed.norm_initial)),
                "extended_norm_drift": float(abs(observed.extended_norm_final - observed.extended_norm_initial)),
            },
        )

    for family, label in enumerate(("e-flavor", "mu-flavor", "tau-flavor")):
        result = sm_run_jitted_qca_calibrated_carrier_basis_probe(
            up_fn_masses,
            down_fn_masses,
            benchmark_ckm(),
            lattice_shape,
            steps=steps,
            dirac=0,
            sector="L",
            weak=0,
            family=family,
            lepton_yukawas=lepton_yukawas,
            lambda_rec=lambda_rec,
            charges=charges,
            center_fit_steps=0,
            yukawa_step_size=yukawa_step_size,
            donate_state=False,
        )
        observed = result.production_observables
        final = observed.carrier_populations_final
        contracts.append(_contract_dict(result.production_contract))
        family_pops = _family_populations(observed.production.final_qca_state.visible_state, NU_C_INTERNAL_INDEX)
        normalized_family_pops = _normalized(family_pops)
        expected = tuple(float(value) for value in expected_pmns_weights[family])
        neutrino_probes.append(
            {
                "source_family": family,
                "label": label,
                "expected_pmns_weights": list(expected),
                "nu_c_family_populations": list(family_pops),
                "normalized_nu_c_family_populations": list(normalized_family_pops),
                "pmns_weight_max_error": max(abs(normalized_family_pops[i] - expected[i]) for i in range(3)),
                "nu_c_population": float(final.sector[5]),
                "e_c_population": float(final.sector[4]),
                "norm_drift": float(abs(observed.norm_final - observed.norm_initial)),
                "extended_norm_drift": float(abs(observed.extended_norm_final - observed.extended_norm_initial)),
            },
        )

    delta_ratio = _mass_squared_ratio(neutrino_vector)
    expected_ratio = EPSILON**4
    ratio_error = abs(delta_ratio - expected_ratio)
    charged_e_populations = [probe["e_c_population"] for probe in charged_probes]
    charged_wrong = [probe["nu_c_population"] for probe in charged_probes]
    neutrino_nu_populations = [probe["nu_c_population"] for probe in neutrino_probes]
    neutrino_wrong = [probe["e_c_population"] for probe in neutrino_probes]
    norm_drifts = [
        max(probe["norm_drift"], probe["extended_norm_drift"])
        for probe in [*charged_probes, *neutrino_probes]
    ]
    pmns_errors = [probe["pmns_weight_max_error"] for probe in neutrino_probes]
    checks = {
        "charged_hierarchy": charged_vector[0] < charged_vector[1] < charged_vector[2],
        "charged_carrier_transfer": charged_e_populations[2] > charged_e_populations[1] > charged_e_populations[0] + CHARGED_MONOTONIC_TOLERANCE
        and charged_e_populations[2] > CHARGED_TRANSFER_MIN,
        "charged_no_nu_c_leakage": all(value < WRONG_SECTOR_MAX for value in charged_wrong),
        "normal_ordering": neutrino_vector[0] < neutrino_vector[1] < neutrino_vector[2],
        "massless_lightest": abs(neutrino_vector[0]) < MASSLESS_TOLERANCE,
        "ratio_check": ratio_error < RATIO_TOLERANCE,
        "pmns_unitarity": lepton_metadata["pmns_unitarity_residual"] < 1.0e-6,
        "pmns_mixing": all(error < PMNS_WEIGHT_TOLERANCE for error in pmns_errors),
        "neutrino_carrier_transfer": all(value > PMNS_TRANSFER_MIN for value in neutrino_nu_populations),
        "neutrino_no_e_c_leakage": all(value < WRONG_SECTOR_MAX for value in neutrino_wrong),
        "schur_spectrum": lepton_metadata["schur_backend"]["spectrum_residual"] < SCHUR_SPECTRUM_TOLERANCE,
        "norm_conservation": all(drift < PMNS_NORM_DRIFT_MAX for drift in norm_drifts),
        "production_contract": all(_production_contract_passed(contract) for contract in contracts),
    }
    return {
        "benchmark": "lepton_phenomenology",
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
        "lepton_inputs": lepton_metadata,
        "diagnostics": {
            "charged_lepton_yukawa_range": [
                min(value for value in charged_vector if value > 0.0),
                max(charged_vector),
            ],
            "pmns_unitarity_residual": lepton_metadata["pmns_unitarity_residual"],
            "delta_m2_21_over_delta_m2_31": delta_ratio,
            "expected_epsilon_fourth": expected_ratio,
            "ratio_error": ratio_error,
            "normal_ordering": neutrino_vector[0] < neutrino_vector[1] < neutrino_vector[2],
            "massless_lightest": abs(neutrino_vector[0]) < MASSLESS_TOLERANCE,
            "schur_spectrum_residual": lepton_metadata["schur_backend"]["spectrum_residual"],
        },
        "charged_probes": charged_probes,
        "neutrino_probes": neutrino_probes,
        "production_contract": contracts[0],
        "physics_tests": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "charged_transfer_min": CHARGED_TRANSFER_MIN,
                "norm_drift_max": PMNS_NORM_DRIFT_MAX,
                "pmns_weight_tolerance": PMNS_WEIGHT_TOLERANCE,
                "ratio_tolerance": RATIO_TOLERANCE,
                "schur_spectrum_tolerance": SCHUR_SPECTRUM_TOLERANCE,
                "transfer_min": PMNS_TRANSFER_MIN,
                "wrong_sector_max": WRONG_SECTOR_MAX,
            },
        },
    }


def payload_to_markdown(payload: dict[str, Any]) -> str:
    """Return a Markdown interpretation report for a lepton phenomenology run."""

    check_rows = "\n".join(
        f"| {label} | {passed} |" for label, passed in payload["physics_tests"]["checks"].items()
    )
    charged_rows = "\n".join(
        "| {label} | {charged_yukawa:.9g} | {e_c_population:.9g} | {nu_c_population:.3g} | {norm_drift:.3g} |".format(
            **probe,
        )
        for probe in payload["charged_probes"]
    )
    neutrino_rows = "\n".join(
        "| {label} | {nu_c_population:.9g} | {e_c_population:.3g} | {pmns_weight_max_error:.3g} | {norm_drift:.3g} |".format(
            **probe,
        )
        for probe in payload["neutrino_probes"]
    )
    return f"""# QCA_SMv0 Lepton Phenomenology

This report is produced by the production-facing lepton command.  It converts
charged-lepton masses, neutrino masses/splittings, and PMNS angles into
effective lepton Yukawa matrices, then runs charged and neutrino carrier probes
on the lean QCA production fibre.

## Inputs

- mode: `{payload["mode"]}`
- scale label: `{payload["scale_label"]}`
- lattice shape: `{tuple(payload["lattice_shape"])}`
- steps: `{payload["steps"]}`
- charged-lepton masses: `{tuple(payload["lepton_inputs"]["charged_lepton_masses"])}`
- charged-lepton Yukawas: `{tuple(payload["lepton_inputs"]["charged_lepton_yukawas"])}`
- neutrino masses: `{tuple(payload["lepton_inputs"]["neutrino_masses"])}`
- PMNS angles: `{tuple(payload["lepton_inputs"]["pmns_angles"])}`
- production input: `{payload["lepton_inputs"]["production_input"]}`

## Diagnostics

- PMNS unitarity residual: `{payload["diagnostics"]["pmns_unitarity_residual"]:.3g}`
- Delta m2 ratio: `{payload["diagnostics"]["delta_m2_21_over_delta_m2_31"]:.12g}`
- expected epsilon^4: `{payload["diagnostics"]["expected_epsilon_fourth"]:.12g}`
- Schur spectrum residual: `{payload["diagnostics"]["schur_spectrum_residual"]:.3g}`

## Physics Tests

- overall passed: `{payload["physics_tests"]["passed"]}`

| check | passed |
|---|---:|
{check_rows}

## Charged-Lepton Carrier Response

| family | Yukawa | e_c population | nu_c leakage | norm drift |
|---|---:|---:|---:|---:|
{charged_rows}

## Neutrino PMNS Carrier Response

| source flavor | nu_c population | e_c leakage | PMNS weight error | norm drift |
|---|---:|---:|---:|---:|
{neutrino_rows}
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "schur"), default="direct")
    parser.add_argument("--scale-label", default=DEFAULT_SCALE_LABEL)
    parser.add_argument("--lattice-shape", type=_parse_int_triplet, default=DEFAULT_LATTICE_SHAPE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--charged-lepton-masses", type=_parse_float_triplet, default=DEFAULT_CHARGED_LEPTON_MASSES_GEV)
    parser.add_argument("--neutrino-masses", type=_parse_float_triplet)
    parser.add_argument("--neutrino-splittings", type=_parse_float_pair, default=DEFAULT_NEUTRINO_SPLITTINGS)
    parser.add_argument("--pmns-angles", type=_parse_float_quad, default=DEFAULT_PMNS_ANGLES)
    parser.add_argument("--sterile-majorana-masses", type=_parse_float_triplet, default=DEFAULT_STERILE_MAJORANA_MASSES)
    parser.add_argument("--yukawa-step-size", type=float, default=DEFAULT_YUKAWA_STEP_SIZE)
    parser.add_argument("--json-output-path", type=Path)
    parser.add_argument("--report-output-path", type=Path)
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _print_text(payload: dict[str, Any]) -> None:
    print("QCA_SMv0 lepton phenomenology")
    print(f"  mode: {payload['mode']}")
    print(f"  physics_tests_passed: {payload['physics_tests']['passed']}")
    print(f"  charged_lepton_yukawas: {payload['lepton_inputs']['charged_lepton_yukawas']}")
    print(f"  neutrino_masses: {payload['lepton_inputs']['neutrino_masses']}")
    print(f"  pmns_unitarity_residual: {payload['diagnostics']['pmns_unitarity_residual']:.3g}")
    print(f"  schur_spectrum_residual: {payload['diagnostics']['schur_spectrum_residual']:.3g}")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_lepton_phenomenology(
        mode=args.mode,
        scale_label=args.scale_label,
        lattice_shape=args.lattice_shape,
        steps=args.steps,
        charged_lepton_masses=args.charged_lepton_masses,
        neutrino_masses=args.neutrino_masses,
        neutrino_splittings=None if args.neutrino_masses is not None else args.neutrino_splittings,
        pmns_angles=args.pmns_angles,
        sterile_majorana_masses=args.sterile_majorana_masses,
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
