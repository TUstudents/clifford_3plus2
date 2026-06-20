"""Canonical neutrino carrier probe for QCA_SMv0.

This command is a narrow lepton-sector physics test.  It uses the existing
production fibre and compressed Higgs/Yukawa collision, initializes clean
``L(weak=0)`` basis states, and checks that a direct diagonal effective
neutrino Yukawa hierarchy routes population into ``nu_c`` without ``e_c``
leakage when charged-lepton Yukawas are zero.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.scripts.phenomenology_rollout import (
    DEFAULT_DOWN_MASSES_GEV,
    DEFAULT_SCALE_LABEL,
    DEFAULT_UP_MASSES_GEV,
    _normalize_masses,
    benchmark_ckm,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_fn import DEFAULT_FN_QUARK_CHARGES, FNQuarkCharges
from clifford_3plus2_d5.qca_smv0.sm_rollout import sm_run_jitted_qca_calibrated_carrier_basis_probe


EPSILON = math.sqrt(2.0) - 1.0
ETA = EPSILON * EPSILON
DEFAULT_NEUTRINO_PROBE_YUKAWAS = (0.0, ETA, 1.0)
DEFAULT_CHARGED_LEPTON_PROBE_YUKAWAS = (0.0, 0.0, 0.0)
DEFAULT_LATTICE_SHAPE = (1, 1, 1)
DEFAULT_STEPS = 1
DEFAULT_YUKAWA_STEP_SIZE = 0.10
NORM_DRIFT_MAX = 1.0e-6
WRONG_SECTOR_MAX = 1.0e-9
TRANSFER_MIN = 1.0e-5
RATIO_TOLERANCE = 1.0e-6
MASSLESS_TOLERANCE = 1.0e-12


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


def _contract_dict(contract: Any) -> dict[str, bool]:
    return {key: bool(value) for key, value in contract._asdict().items()}


def _production_contract_passed(contract: dict[str, bool]) -> bool:
    return (
        contract["uses_production_api"]
        and contract["state_only"]
        and contract["structured_collision_cache_present"]
        and contract["lean_effective_yukawa"]
        and not contract["raw_yukawa_arrays_present"]
        and not contract["raw_readout_arrays_present"]
        and not contract["higgs_field_present"]
    )


def _mass_squared_ratio(masses: tuple[float, float, float]) -> float:
    m1, m2, m3 = masses
    denominator = m3 * m3 - m1 * m1
    if denominator == 0.0:
        return math.inf
    return (m2 * m2 - m1 * m1) / denominator


def run_neutrino_probe(
    *,
    scale_label: str = DEFAULT_SCALE_LABEL,
    lattice_shape: tuple[int, int, int] = DEFAULT_LATTICE_SHAPE,
    steps: int = DEFAULT_STEPS,
    neutrino_yukawas: tuple[float, float, float] = DEFAULT_NEUTRINO_PROBE_YUKAWAS,
    charged_lepton_yukawas: tuple[float, float, float] = DEFAULT_CHARGED_LEPTON_PROBE_YUKAWAS,
    up_masses: tuple[float, float, float] = DEFAULT_UP_MASSES_GEV,
    down_masses: tuple[float, float, float] = DEFAULT_DOWN_MASSES_GEV,
    mass_mode: str = "absolute",
    lambda_rec: float = 0.22501,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    yukawa_step_size: float = DEFAULT_YUKAWA_STEP_SIZE,
) -> dict[str, Any]:
    """Run three clean neutrino carrier probes and return a JSON payload."""

    up_fn_masses, up_normalization = _normalize_masses(up_masses, mass_mode, label="up")
    down_fn_masses, down_normalization = _normalize_masses(down_masses, mass_mode, label="down")
    lepton_yukawas = FamilyLeptonYukawas(
        neutrino=jnp.diag(jnp.asarray(neutrino_yukawas, dtype=jnp.complex64)),
        electron=jnp.diag(jnp.asarray(charged_lepton_yukawas, dtype=jnp.complex64)),
    )
    probes = []
    contracts = []
    for family, label in enumerate(("nu1", "nu2", "nu3")):
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
        initial = observed.carrier_populations_initial
        final = observed.carrier_populations_final
        contract = _contract_dict(result.production_contract)
        contracts.append(contract)
        probes.append(
            {
                "family": family,
                "label": label,
                "neutrino_yukawa": float(neutrino_yukawas[family]),
                "initial_sector": [float(value) for value in initial.sector],
                "final_sector": [float(value) for value in final.sector],
                "sector_drift": [float(value) for value in final.sector - initial.sector],
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
    checks = {
        "normal_ordering": neutrino_yukawas[0] < neutrino_yukawas[1] < neutrino_yukawas[2],
        "massless_lightest": abs(neutrino_yukawas[0]) < MASSLESS_TOLERANCE,
        "ratio_check": ratio_error < RATIO_TOLERANCE,
        "norm_conservation": all(drift < NORM_DRIFT_MAX for drift in norm_drifts),
        "carrier_transfer": nu_c_populations[0] < WRONG_SECTOR_MAX
        and nu_c_populations[1] > TRANSFER_MIN
        and nu_c_populations[2] > nu_c_populations[1],
        "no_e_c_leakage": all(population < WRONG_SECTOR_MAX for population in e_c_populations),
        "production_contract": all(_production_contract_passed(contract) for contract in contracts),
    }
    return {
        "benchmark": "neutrino_probe",
        "scale_label": scale_label,
        "lattice_shape": list(lattice_shape),
        "steps": steps,
        "lambda": lambda_rec,
        "charges": {
            "q": list(charges.q),
            "u": list(charges.u),
            "d": list(charges.d),
        },
        "input_masses": {
            "up": list(up_masses),
            "down": list(down_masses),
            "up_normalization": up_normalization,
            "down_normalization": down_normalization,
        },
        "lepton_inputs": {
            "mode": "direct_diagonal_effective_neutrino_yukawa_probe",
            "later_mode": "seesaw_schur_input_mode",
            "neutrino_yukawas": list(neutrino_yukawas),
            "charged_lepton_yukawas": list(charged_lepton_yukawas),
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
                "norm_drift_max": NORM_DRIFT_MAX,
                "wrong_sector_max": WRONG_SECTOR_MAX,
                "transfer_min": TRANSFER_MIN,
                "ratio_tolerance": RATIO_TOLERANCE,
                "massless_tolerance": MASSLESS_TOLERANCE,
            },
        },
    }


def payload_to_markdown(payload: dict[str, Any]) -> str:
    """Return a concise Markdown report for a neutrino probe payload."""

    probe_rows = "\n".join(
        "| {label} | {neutrino_yukawa:.9g} | {nu_c_population:.9g} | {e_c_population:.3g} | {norm_drift:.3g} |".format(
            **probe,
        )
        for probe in payload["probes"]
    )
    check_rows = "\n".join(
        f"| {label} | {passed} |" for label, passed in payload["physics_tests"]["checks"].items()
    )
    metadata = payload["neutrino_metadata"]
    return f"""# QCA_SMv0 Neutrino Carrier Probe

This canonical lepton-sector probe promotes the best existing neutrino result
to a simulator input.  It uses direct diagonal effective neutrino Yukawas
`(0, eta, 1)`, with `eta=(sqrt(2)-1)^2`, while charged-lepton Yukawas are set
to zero.  The Schur/seesaw input mode is recorded as the next backend, not used
in this direct probe.

## Inputs

- scale label: `{payload["scale_label"]}`
- lattice shape: `{tuple(payload["lattice_shape"])}`
- steps: `{payload["steps"]}`
- neutrino Yukawas: `{tuple(payload["lepton_inputs"]["neutrino_yukawas"])}`
- charged-lepton Yukawas: `{tuple(payload["lepton_inputs"]["charged_lepton_yukawas"])}`
- source: `{payload["probe_source"]["sector"]}(weak={payload["probe_source"]["weak"]})`
- target: `{payload["probe_source"]["target"]}`

## Neutrino Metadata

- hierarchy: `{metadata["hierarchy"]}`
- lightest massless: `{metadata["lightest_massless"]}`
- epsilon: `{metadata["epsilon"]:.12g}`
- eta: `{metadata["eta"]:.12g}`
- Delta m2 ratio: `{metadata["delta_m2_21_over_delta_m2_31"]:.12g}`
- expected epsilon^4: `{metadata["expected_epsilon_fourth"]:.12g}`
- ratio error: `{metadata["ratio_error"]:.3g}`

## Physics Tests

- overall passed: `{payload["physics_tests"]["passed"]}`

| check | passed |
|---|---:|
{check_rows}

## Family Response

| family | input Yukawa | nu_c population | e_c leakage | norm drift |
|---|---:|---:|---:|---:|
{probe_rows}

## Interpretation

The probe verifies the direct neutrino baseline on the production fibre:
`L(weak=0)` routes into `nu_c`, zero charged-lepton Yukawas prevent `e_c`
leakage, the direct input has normal ordering with one massless lightest state,
and the mass-squared ratio is the silver prediction `epsilon^4`.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-label", default=DEFAULT_SCALE_LABEL)
    parser.add_argument("--lattice-shape", type=_parse_int_triplet, default=DEFAULT_LATTICE_SHAPE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--neutrino-yukawas", type=_parse_float_triplet, default=DEFAULT_NEUTRINO_PROBE_YUKAWAS)
    parser.add_argument("--charged-lepton-yukawas", type=_parse_float_triplet, default=DEFAULT_CHARGED_LEPTON_PROBE_YUKAWAS)
    parser.add_argument("--yukawa-step-size", type=float, default=DEFAULT_YUKAWA_STEP_SIZE)
    parser.add_argument("--json-output-path", type=Path)
    parser.add_argument("--report-output-path", type=Path)
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _print_text(payload: dict[str, Any]) -> None:
    print("QCA_SMv0 neutrino carrier probe")
    print(f"  physics_tests_passed: {payload['physics_tests']['passed']}")
    print(f"  neutrino_yukawas: {payload['lepton_inputs']['neutrino_yukawas']}")
    metadata = payload["neutrino_metadata"]
    print(
        "  delta_m2_21/delta_m2_31={ratio:.9g}, expected_epsilon4={expected:.9g}".format(
            ratio=metadata["delta_m2_21_over_delta_m2_31"],
            expected=metadata["expected_epsilon_fourth"],
        ),
    )
    for probe in payload["probes"]:
        print(
            "  {label}: y={neutrino_yukawa:.9g}, nu_c={nu_c_population:.9g}, "
            "e_c={e_c_population:.3g}, norm_drift={norm_drift:.3g}".format(**probe),
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_neutrino_probe(
        scale_label=args.scale_label,
        lattice_shape=args.lattice_shape,
        steps=args.steps,
        neutrino_yukawas=args.neutrino_yukawas,
        charged_lepton_yukawas=args.charged_lepton_yukawas,
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
