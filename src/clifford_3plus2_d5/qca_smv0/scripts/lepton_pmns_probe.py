"""Canonical PMNS lepton probe for QCA_SMv0.

This command combines the charged-lepton and neutrino baselines into one
lepton-sector mixing test.  Charged leptons are kept diagonal/off for the probe,
while the neutrino door receives a direct effective PMNS matrix
``Y_nu = U_PMNS diag(0, eta, 1)``.  The same run also constructs a type-I
seesaw/Schur backend and checks that its low-energy singular spectrum recovers
``(0, eta, 1)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.scripts.neutrino_probe import (
    DEFAULT_CHARGED_LEPTON_PROBE_YUKAWAS,
    DEFAULT_LATTICE_SHAPE,
    DEFAULT_NEUTRINO_PROBE_YUKAWAS,
    DEFAULT_STEPS,
    ETA,
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
    sm_default_pmns_matrix,
    sm_lepton_direct_pmns_neutrino_yukawa,
    sm_lepton_pmns_expected_transfer_weights,
    sm_lepton_seesaw_schur_from_pmns,
    sm_pmns_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_rollout import sm_run_jitted_qca_calibrated_carrier_basis_probe


DEFAULT_YUKAWA_STEP_SIZE = 0.02
PMNS_WEIGHT_TOLERANCE = 7.5e-3
SCHUR_SPECTRUM_TOLERANCE = 1.0e-6
PMNS_TRANSFER_MIN = 1.0e-6
PMNS_NORM_DRIFT_MAX = 2.0e-6
NU_C_INTERNAL_INDEX = 15


def _nu_c_family_populations(final_state: jnp.ndarray, *, internal_copy: int = 0) -> tuple[float, float, float]:
    internal = NU_C_INTERNAL_INDEX + 16 * internal_copy
    populations = jnp.sum(jnp.abs(final_state[..., :, internal, :]) ** 2, axis=(0, 1, 2, 3))
    return tuple(float(value) for value in populations)  # type: ignore[return-value]


def _normalized(values: tuple[float, float, float]) -> tuple[float, float, float]:
    total = sum(values)
    if total <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _complex_matrix_payload(matrix: jnp.ndarray) -> list[list[dict[str, float]]]:
    arr = jnp.asarray(matrix)
    return [
        [{"re": float(jnp.real(arr[i, j])), "im": float(jnp.imag(arr[i, j]))} for j in range(arr.shape[1])]
        for i in range(arr.shape[0])
    ]


def run_lepton_pmns_probe(
    *,
    scale_label: str = DEFAULT_SCALE_LABEL,
    lattice_shape: tuple[int, int, int] = DEFAULT_LATTICE_SHAPE,
    steps: int = DEFAULT_STEPS,
    neutrino_yukawas: tuple[float, float, float] = DEFAULT_NEUTRINO_PROBE_YUKAWAS,
    charged_lepton_yukawas: tuple[float, float, float] = DEFAULT_CHARGED_LEPTON_PROBE_YUKAWAS,
    pmns_angles: tuple[float, float, float, float] = DEFAULT_PMNS_ANGLES,
    up_masses: tuple[float, float, float] = DEFAULT_UP_MASSES_GEV,
    down_masses: tuple[float, float, float] = DEFAULT_DOWN_MASSES_GEV,
    mass_mode: str = "absolute",
    lambda_rec: float = 0.22501,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    yukawa_step_size: float = DEFAULT_YUKAWA_STEP_SIZE,
) -> dict[str, Any]:
    """Run clean PMNS carrier probes and return a JSON payload."""

    up_fn_masses, up_normalization = _normalize_masses(up_masses, mass_mode, label="up")
    down_fn_masses, down_normalization = _normalize_masses(down_masses, mass_mode, label="down")
    pmns = sm_default_pmns_matrix()
    if pmns_angles != DEFAULT_PMNS_ANGLES:
        from clifford_3plus2_d5.qca_smv0.sm_lepton import sm_pmns_matrix

        pmns = sm_pmns_matrix(*pmns_angles)
    neutrino_matrix = sm_lepton_direct_pmns_neutrino_yukawa(neutrino_yukawas, pmns)
    expected_weights = sm_lepton_pmns_expected_transfer_weights(neutrino_yukawas, pmns)
    schur = sm_lepton_seesaw_schur_from_pmns(neutrino_yukawas, pmns)
    lepton_yukawas = FamilyLeptonYukawas(
        neutrino=neutrino_matrix,
        electron=jnp.diag(jnp.asarray(charged_lepton_yukawas, dtype=jnp.complex64)),
    )
    probes = []
    contracts = []
    for source_family, label in enumerate(("e-flavor", "mu-flavor", "tau-flavor")):
        result = sm_run_jitted_qca_calibrated_carrier_basis_probe(
            up_fn_masses,
            down_fn_masses,
            benchmark_ckm(),
            lattice_shape,
            steps=steps,
            dirac=0,
            sector="L",
            weak=0,
            family=source_family,
            lepton_yukawas=lepton_yukawas,
            lambda_rec=lambda_rec,
            charges=charges,
            center_fit_steps=0,
            yukawa_step_size=yukawa_step_size,
            donate_state=False,
        )
        observed = result.production_observables
        initial = observed.carrier_populations_initial
        final = observed.carrier_populations_final
        contract = _contract_dict(result.production_contract)
        contracts.append(contract)
        family_pops = _nu_c_family_populations(observed.production.final_qca_state.visible_state)
        normalized_family_pops = _normalized(family_pops)
        expected = tuple(float(value) for value in expected_weights[source_family])
        probes.append(
            {
                "source_family": source_family,
                "label": label,
                "expected_pmns_weights": list(expected),
                "nu_c_family_populations": list(family_pops),
                "normalized_nu_c_family_populations": list(normalized_family_pops),
                "pmns_weight_max_error": max(abs(normalized_family_pops[i] - expected[i]) for i in range(3)),
                "initial_sector": [float(value) for value in initial.sector],
                "final_sector": [float(value) for value in final.sector],
                "nu_c_population": float(final.sector[5]),
                "e_c_population": float(final.sector[4]),
                "l_population_final": float(final.sector[3]),
                "norm_initial": float(observed.norm_initial),
                "norm_final": float(observed.norm_final),
                "norm_drift": float(abs(observed.norm_final - observed.norm_initial)),
                "extended_norm_drift": float(abs(observed.extended_norm_final - observed.extended_norm_initial)),
            },
        )

    nu_c_populations = [probe["nu_c_population"] for probe in probes]
    e_c_populations = [probe["e_c_population"] for probe in probes]
    norm_drifts = [max(probe["norm_drift"], probe["extended_norm_drift"]) for probe in probes]
    delta_ratio = _mass_squared_ratio(neutrino_yukawas)
    expected_ratio = EPSILON**4
    ratio_error = abs(delta_ratio - expected_ratio)
    pmns_weight_errors = [probe["pmns_weight_max_error"] for probe in probes]
    checks = {
        "normal_ordering": neutrino_yukawas[0] < neutrino_yukawas[1] < neutrino_yukawas[2],
        "massless_lightest": abs(neutrino_yukawas[0]) < MASSLESS_TOLERANCE,
        "ratio_check": ratio_error < RATIO_TOLERANCE,
        "pmns_unitarity": float(sm_pmns_unitarity_residual(pmns)) < 1.0e-6,
        "pmns_mixing": all(error < PMNS_WEIGHT_TOLERANCE for error in pmns_weight_errors)
        and any(len([value for value in probe["normalized_nu_c_family_populations"] if value > 1.0e-2]) > 1 for probe in probes),
        "schur_spectrum": float(schur.spectrum_residual) < SCHUR_SPECTRUM_TOLERANCE,
        "norm_conservation": all(drift < PMNS_NORM_DRIFT_MAX for drift in norm_drifts),
        "carrier_transfer": all(population > PMNS_TRANSFER_MIN for population in nu_c_populations),
        "no_e_c_leakage": all(population < WRONG_SECTOR_MAX for population in e_c_populations),
        "production_contract": all(_production_contract_passed(contract) for contract in contracts),
    }
    return {
        "benchmark": "lepton_pmns_probe",
        "scale_label": scale_label,
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
            "mode": "direct_pmns_effective_neutrino_yukawa_probe",
            "schur_backend": "type_i_seesaw_schur",
            "neutrino_yukawas": list(neutrino_yukawas),
            "charged_lepton_yukawas": list(charged_lepton_yukawas),
            "pmns_angles": list(pmns_angles),
            "pmns_matrix": _complex_matrix_payload(pmns),
            "neutrino_yukawa_matrix": _complex_matrix_payload(neutrino_matrix),
        },
        "neutrino_metadata": {
            "epsilon": EPSILON,
            "eta": ETA,
            "hierarchy": "normal",
            "lightest_mass_index": 1,
            "lightest_massless": True,
            "m1": float(neutrino_yukawas[0]),
            "m2": float(neutrino_yukawas[1]),
            "m3": float(neutrino_yukawas[2]),
            "delta_m2_21_over_delta_m2_31": delta_ratio,
            "expected_epsilon_fourth": expected_ratio,
            "ratio_error": ratio_error,
        },
        "schur_seesaw": {
            "sterile_majorana_masses": [float(value) for value in jnp.diag(schur.sterile_majorana).real],
            "target_masses": [float(value) for value in schur.target_masses],
            "recovered_masses": [float(value) for value in schur.recovered_masses],
            "spectrum_residual": float(schur.spectrum_residual),
            "effective_majorana": _complex_matrix_payload(schur.effective_majorana),
        },
        "probe_source": {
            "sector": "L",
            "weak": 0,
            "target": "nu_c",
            "wrong_target": "e_c",
        },
        "probes": probes,
        "production_contract": contracts[0],
        "physics_tests": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "norm_drift_max": PMNS_NORM_DRIFT_MAX,
                "wrong_sector_max": WRONG_SECTOR_MAX,
                "transfer_min": PMNS_TRANSFER_MIN,
                "ratio_tolerance": RATIO_TOLERANCE,
                "massless_tolerance": MASSLESS_TOLERANCE,
                "pmns_weight_tolerance": PMNS_WEIGHT_TOLERANCE,
                "schur_spectrum_tolerance": SCHUR_SPECTRUM_TOLERANCE,
            },
        },
    }


def payload_to_markdown(payload: dict[str, Any]) -> str:
    """Return a concise Markdown report for a PMNS lepton probe payload."""

    check_rows = "\n".join(
        f"| {label} | {passed} |" for label, passed in payload["physics_tests"]["checks"].items()
    )
    probe_rows = "\n".join(
        "| {label} | {nu_c_population:.9g} | {e_c_population:.3g} | {pmns_weight_max_error:.3g} | {norm_drift:.3g} |".format(
            **probe,
        )
        for probe in payload["probes"]
    )
    metadata = payload["neutrino_metadata"]
    schur = payload["schur_seesaw"]
    return f"""# QCA_SMv0 Lepton PMNS Probe

This canonical lepton-sector probe tests mixing rather than only diagonal
transfer.  The neutrino door receives `Y_nu = U_PMNS diag(0, eta, 1)`, charged
lepton Yukawas are zero, and the target-family populations are compared with
the PMNS-weighted transfer distribution.  A type-I Schur/seesaw backend is
constructed in the same payload and checked against the same low-energy
spectrum.

## Inputs

- scale label: `{payload["scale_label"]}`
- lattice shape: `{tuple(payload["lattice_shape"])}`
- steps: `{payload["steps"]}`
- neutrino Yukawas: `{tuple(payload["lepton_inputs"]["neutrino_yukawas"])}`
- PMNS angles: `{tuple(payload["lepton_inputs"]["pmns_angles"])}`
- source: `{payload["probe_source"]["sector"]}(weak={payload["probe_source"]["weak"]})`
- target: `{payload["probe_source"]["target"]}`

## Neutrino And Schur Metadata

- hierarchy: `{metadata["hierarchy"]}`
- lightest massless: `{metadata["lightest_massless"]}`
- Delta m2 ratio: `{metadata["delta_m2_21_over_delta_m2_31"]:.12g}`
- expected epsilon^4: `{metadata["expected_epsilon_fourth"]:.12g}`
- Schur recovered masses: `{tuple(schur["recovered_masses"])}`
- Schur spectrum residual: `{schur["spectrum_residual"]:.3g}`

## Physics Tests

- overall passed: `{payload["physics_tests"]["passed"]}`

| check | passed |
|---|---:|
{check_rows}

## Flavor Response

| source flavor | nu_c population | e_c leakage | PMNS weight error | norm drift |
|---|---:|---:|---:|---:|
{probe_rows}

## Interpretation

The probe verifies that the lepton production door can consume a non-diagonal
PMNS neutrino input, produce mixed `nu_c` target-family populations, conserve
norm, avoid charged-lepton leakage, and represent the same `(0, eta, 1)`
low-energy neutrino spectrum through an explicit heavy-sterile Schur backend.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-label", default=DEFAULT_SCALE_LABEL)
    parser.add_argument("--lattice-shape", type=_parse_int_triplet, default=DEFAULT_LATTICE_SHAPE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--neutrino-yukawas", type=_parse_float_triplet, default=DEFAULT_NEUTRINO_PROBE_YUKAWAS)
    parser.add_argument("--charged-lepton-yukawas", type=_parse_float_triplet, default=DEFAULT_CHARGED_LEPTON_PROBE_YUKAWAS)
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
    print("QCA_SMv0 lepton PMNS probe")
    print(f"  physics_tests_passed: {payload['physics_tests']['passed']}")
    print(f"  neutrino_yukawas: {payload['lepton_inputs']['neutrino_yukawas']}")
    print(f"  schur_spectrum_residual: {payload['schur_seesaw']['spectrum_residual']:.3g}")
    for probe in payload["probes"]:
        print(
            "  {label}: nu_c={nu_c_population:.9g}, e_c={e_c_population:.3g}, "
            "pmns_error={pmns_weight_max_error:.3g}".format(**probe),
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if len(args.pmns_angles) != 4:
        raise SystemExit("--pmns-angles must contain four comma-separated floats")
    payload = run_lepton_pmns_probe(
        scale_label=args.scale_label,
        lattice_shape=args.lattice_shape,
        steps=args.steps,
        neutrino_yukawas=args.neutrino_yukawas,
        charged_lepton_yukawas=args.charged_lepton_yukawas,
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
