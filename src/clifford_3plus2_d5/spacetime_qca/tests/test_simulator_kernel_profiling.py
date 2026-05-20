"""Warm kernel profiling tests for the spacetime-QCA simulator."""

from __future__ import annotations

import json

import pytest

from clifford_3plus2_d5.spacetime_qca.simulator import (
    default_kernel_profile_cases,
    recommend_kernel_bottleneck,
    run_spacetime_kernel_profile,
)
from clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_kernels import main as kernel_profile_cli


def test_kernel_profile_case_names_are_unique_and_current_is_opt_in() -> None:
    default_names = tuple(case.name for case in default_kernel_profile_cases())
    current_names = tuple(case.name for case in default_kernel_profile_cases(include_current=True))

    assert len(default_names) == len(set(default_names))
    assert {"link_field_u1_y", "link_field_su2_l", "link_field_sm"} <= set(default_names)
    assert {"step_no_matter_u1_y", "step_no_matter_su2_l", "step_no_matter_sm"} <= set(default_names)
    assert "matter_current_u1_y" not in default_names
    assert "matter_current_u1_y" in current_names


def test_kernel_profile_initial_state_case_is_json_safe() -> None:
    payload = run_spacetime_kernel_profile(
        case_names=("initial_state_u1_y",),
        warmup_runs=0,
        timed_runs=1,
    )

    assert payload["metadata"]["runner"] == "spacetime_qca.simulator.kernel_profile"
    assert payload["metadata"]["include_current"] is False
    assert payload["metadata"]["warmup_runs"] == 0
    assert payload["metadata"]["timed_runs"] == 1
    assert payload["cases"][0]["label"] == "initial_state_u1_y"
    assert payload["cases"][0]["all_finite"] is True
    assert payload["cases"][0]["min_seconds"] <= payload["cases"][0]["mean_seconds"]
    assert payload["recommendation"]
    json.dumps(payload)


def test_kernel_profile_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="unknown kernel profile case"):
        run_spacetime_kernel_profile(case_names=("missing",), warmup_runs=0, timed_runs=1)


def test_kernel_profile_cli_prints_json(capsys) -> None:
    exit_code = kernel_profile_cli(
        ("--case", "initial_state_u1_y", "--warmup-runs", "0", "--timed-runs", "1"),
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["case_count"] == 1
    assert payload["cases"][0]["label"] == "initial_state_u1_y"


def test_kernel_profile_cli_writes_output(tmp_path, capsys) -> None:
    output = tmp_path / "kernel_profile.json"
    exit_code = kernel_profile_cli(
        (
            "--case",
            "initial_state_u1_y",
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


def test_kernel_bottleneck_recommendation_flags_sm_step() -> None:
    recommendation = recommend_kernel_bottleneck(
        (
            {"label": "step_no_matter_u1_y", "mean_seconds": 1.0},
            {"label": "step_no_matter_sm", "mean_seconds": 3.0},
            {"label": "link_field_sm", "mean_seconds": 0.1},
        ),
    )

    assert "full-SM no-matter step" in recommendation
