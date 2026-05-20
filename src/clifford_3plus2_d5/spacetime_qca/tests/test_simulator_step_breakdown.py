"""Step-breakdown profiling tests for the spacetime-QCA simulator."""

from __future__ import annotations

import json

import pytest

from clifford_3plus2_d5.spacetime_qca.simulator import (
    default_step_breakdown_cases,
    recommend_step_breakdown_bottleneck,
    run_spacetime_step_breakdown_profile,
)
from clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown import (
    main as step_breakdown_cli,
)


def test_step_breakdown_case_names_are_unique_and_sm_scoped() -> None:
    cases = default_step_breakdown_cases()
    names = tuple(case.name for case in cases)

    assert len(names) == len(set(names))
    assert {
        "yukawa_half_kick_sm",
        "fermion_gauge_no_matter_sm",
        "higgs_leapfrog_sm",
        "yukawa_final_half_kick_sm",
        "diagnostics_sm",
        "gauss_residual_sm",
        "gauge_hamiltonian_sm",
        "left_force_sm",
    } <= set(names)
    assert {case.config.sector for case in cases} == {"sm"}
    assert {case.config.lattice_shape for case in cases} == {(1, 1, 1)}


def test_step_breakdown_higgs_case_is_json_safe() -> None:
    payload = run_spacetime_step_breakdown_profile(
        case_names=("higgs_leapfrog_sm",),
        warmup_runs=0,
        timed_runs=1,
    )

    assert payload["metadata"]["runner"] == "spacetime_qca.simulator.step_breakdown_profile"
    assert payload["metadata"]["sector"] == "sm"
    assert payload["metadata"]["warmup_runs"] == 0
    assert payload["metadata"]["timed_runs"] == 1
    assert payload["metadata"]["default_case"] is False
    assert payload["cases"][0]["label"] == "higgs_leapfrog_sm"
    assert payload["cases"][0]["all_finite"] is True
    assert payload["cases"][0]["min_seconds"] <= payload["cases"][0]["mean_seconds"]
    assert payload["recommendation"]
    json.dumps(payload)


def test_step_breakdown_default_profile_is_single_safe_case() -> None:
    payload = run_spacetime_step_breakdown_profile(warmup_runs=0, timed_runs=1)

    assert payload["metadata"]["default_case"] is True
    assert payload["metadata"]["case_count"] == 1
    assert payload["cases"][0]["label"] == "higgs_leapfrog_sm"


def test_step_breakdown_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="unknown step-breakdown profile case"):
        run_spacetime_step_breakdown_profile(case_names=("missing",), warmup_runs=0, timed_runs=1)


def test_step_breakdown_cli_prints_json(capsys) -> None:
    exit_code = step_breakdown_cli(("--warmup-runs", "0", "--timed-runs", "1"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["case_count"] == 1
    assert payload["cases"][0]["label"] == "higgs_leapfrog_sm"


def test_step_breakdown_cli_writes_output(tmp_path, capsys) -> None:
    output = tmp_path / "step_breakdown.json"
    exit_code = step_breakdown_cli(
        (
            "--case",
            "higgs_leapfrog_sm",
            "--warmup-runs",
            "0",
            "--timed-runs",
            "1",
            "--output",
            str(output),
        ),
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written


def test_step_breakdown_recommendation_flags_left_force() -> None:
    recommendation = recommend_step_breakdown_bottleneck(
        (
            {"label": "left_force_sm", "mean_seconds": 10.0},
            {"label": "fermion_gauge_no_matter_sm", "mean_seconds": 3.0},
            {"label": "yukawa_half_kick_sm", "mean_seconds": 1.0},
        ),
    )

    assert "finite-difference left-force" in recommendation
