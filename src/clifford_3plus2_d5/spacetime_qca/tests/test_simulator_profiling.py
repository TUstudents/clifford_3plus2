"""Spacetime simulator profiling tests."""

from __future__ import annotations

import json

import pytest

from clifford_3plus2_d5.spacetime_qca.simulator import (
    default_spacetime_profile_cases,
    recommend_bottleneck,
    run_spacetime_profile,
)
from clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_sim import main as profile_cli


def test_profile_case_names_are_unique() -> None:
    names = tuple(case.name for case in default_spacetime_profile_cases())

    assert len(names) == len(set(names))
    assert "runner_zero_step" in names
    assert "physics_with_matter" in names


def test_spacetime_profile_runner_zero_step_is_json_safe() -> None:
    payload = run_spacetime_profile(case_names=("runner_zero_step",), include_jit=False)

    assert payload["metadata"]["runner"] == "spacetime_qca.simulator.profiling"
    assert payload["metadata"]["include_jit"] is False
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["label"] == "runner_zero_step"
    assert payload["cases"][0]["all_finite"] is True
    assert payload["jit_cases"] == {}
    assert payload["recommendation"]
    json.dumps(payload)


def test_spacetime_profile_force_override_is_recorded() -> None:
    payload = run_spacetime_profile(
        case_names=("runner_zero_step",),
        include_jit=False,
        force_method="analytic_staple",
        force_chunk_size=32,
    )

    assert payload["metadata"]["force_method_override"] == "analytic_staple"
    assert payload["metadata"]["force_chunk_size_override"] == 32
    assert payload["cases"][0]["metadata"]["config"]["force_method"] == "analytic_staple"
    assert payload["cases"][0]["metadata"]["config"]["force_chunk_size"] == 32


def test_spacetime_profile_rejects_invalid_force_chunk_size() -> None:
    with pytest.raises(ValueError, match="force_chunk_size must be positive"):
        run_spacetime_profile(case_names=("runner_zero_step",), force_chunk_size=0)


def test_spacetime_profile_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="unknown profile case"):
        run_spacetime_profile(case_names=("missing",))


def test_profile_cli_prints_json(capsys) -> None:
    exit_code = profile_cli(("--case", "runner_zero_step", "--force-method", "analytic_staple"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["case_count"] == 1
    assert payload["metadata"]["force_method_override"] == "analytic_staple"
    assert payload["cases"][0]["label"] == "runner_zero_step"


def test_profile_cli_writes_output(tmp_path, capsys) -> None:
    output = tmp_path / "profile.json"
    exit_code = profile_cli(("--case", "runner_zero_step", "--output", str(output)))

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written


def test_bottleneck_recommendation_flags_matter_current() -> None:
    recommendation = recommend_bottleneck(
        (
            {"label": "physics_no_matter", "python_seconds": 1.0},
            {"label": "physics_with_matter", "python_seconds": 2.0},
        ),
    )

    assert "finite-difference matter current" in recommendation


def test_bottleneck_recommendation_requires_baseline_for_sm_comparison() -> None:
    recommendation = recommend_bottleneck(({"label": "sector_sm", "python_seconds": 3.0},))

    assert "split the SM sector step" in recommendation
