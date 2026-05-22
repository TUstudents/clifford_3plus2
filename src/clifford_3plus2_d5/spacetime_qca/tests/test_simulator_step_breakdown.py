"""Step-breakdown profiling tests for the spacetime-QCA simulator."""

from __future__ import annotations

import json

import pytest

from clifford_3plus2_d5.spacetime_qca.simulator import (
    default_step_breakdown_cases,
    recommend_force_chunk_tuning,
    recommend_step_breakdown_bottleneck,
    run_force_chunk_comparison,
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
        "yukawa_half_kick_eigh_sm",
        "fermion_gauge_no_matter_sm",
        "higgs_leapfrog_sm",
        "yukawa_final_half_kick_sm",
        "yukawa_final_half_kick_eigh_sm",
        "diagnostics_sm",
        "gauss_residual_sm",
        "gauge_hamiltonian_sm",
        "left_force_sm",
        "left_force_batched_sm",
        "left_force_analytic_sm",
        "gauge_leapfrog_sm",
        "gauge_leapfrog_batched_sm",
        "gauge_leapfrog_analytic_sm",
        "dirac_transport_sm",
        "momentum_update_sm",
        "first_left_force_sm",
        "first_left_force_batched_sm",
        "first_left_force_analytic_sm",
        "second_left_force_sm",
        "second_left_force_batched_sm",
        "second_left_force_analytic_sm",
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


@pytest.mark.parametrize("case_name", ("dirac_transport_sm", "momentum_update_sm"))
def test_step_breakdown_safe_microcase_is_json_safe(case_name: str) -> None:
    payload = run_spacetime_step_breakdown_profile(
        case_names=(case_name,),
        warmup_runs=0,
        timed_runs=1,
    )

    assert payload["metadata"]["case_count"] == 1
    assert payload["cases"][0]["label"] == case_name
    assert payload["cases"][0]["all_finite"] is True
    assert payload["cases"][0]["output_summary"]["type"] == "array"
    json.dumps(payload)


def test_step_breakdown_force_override_skips_non_force_case() -> None:
    payload = run_spacetime_step_breakdown_profile(
        case_names=("momentum_update_sm",),
        warmup_runs=0,
        timed_runs=1,
        force_method="finite_difference_batched",
        force_chunk_size=8,
    )

    assert payload["metadata"]["force_method_override"] == "finite_difference_batched"
    assert payload["metadata"]["force_chunk_size_override"] == 8
    assert payload["cases"][0]["metadata"]["config"]["force_method"] == "finite_difference"
    assert payload["cases"][0]["metadata"]["config"]["force_chunk_size"] is None
    json.dumps(payload)


def test_step_breakdown_rejects_invalid_force_chunk_size() -> None:
    with pytest.raises(ValueError, match="force_chunk_size must be positive"):
        run_spacetime_step_breakdown_profile(
            case_names=("higgs_leapfrog_sm",),
            warmup_runs=0,
            timed_runs=1,
            force_chunk_size=0,
        )


def test_step_breakdown_rejects_unknown_case() -> None:
    with pytest.raises(ValueError, match="unknown step-breakdown profile case"):
        run_spacetime_step_breakdown_profile(case_names=("missing",), warmup_runs=0, timed_runs=1)


def test_step_breakdown_cli_prints_json(capsys) -> None:
    exit_code = step_breakdown_cli(("--warmup-runs", "0", "--timed-runs", "1"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["case_count"] == 1
    assert payload["cases"][0]["label"] == "higgs_leapfrog_sm"


def test_step_breakdown_cli_rejects_force_comparison_with_cases() -> None:
    with pytest.raises(SystemExit, match="--force-comparison cannot be combined"):
        step_breakdown_cli(("--force-comparison", "--case", "higgs_leapfrog_sm"))


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
            {"label": "first_left_force_sm", "mean_seconds": 10.0},
            {"label": "first_left_force_batched_sm", "mean_seconds": 8.0},
            {"label": "fermion_gauge_no_matter_sm", "mean_seconds": 3.0},
            {"label": "yukawa_half_kick_sm", "mean_seconds": 1.0},
        ),
    )

    assert "SM Wilson left-force" in recommendation


def test_force_chunk_tuning_recommendations() -> None:
    assert "staple-like" in recommend_force_chunk_tuning(
        ({"best_seconds": 2.5}, {"best_seconds": 1.9}),
    )
    assert "next non-force" in recommend_force_chunk_tuning(
        ({"best_seconds": 0.8}, {"best_seconds": 0.9}),
    )
    assert "gauge_leapfrog_batched_sm" in recommend_force_chunk_tuning(
        ({"best_seconds": 1.2}, {"best_seconds": 1.4}),
    )


def test_force_chunk_comparison_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="chunk_sizes must contain"):
        run_force_chunk_comparison(chunk_sizes=(), warmup_runs=0, timed_runs=1)


def test_step_breakdown_batched_force_cases_pin_force_method() -> None:
    cases = {case.name: case for case in default_step_breakdown_cases()}

    assert cases["left_force_batched_sm"].config.force_method == "finite_difference_batched"
    assert cases["first_left_force_batched_sm"].config.force_method == "finite_difference_batched"
    assert cases["second_left_force_batched_sm"].config.force_method == "finite_difference_batched"
    assert cases["gauge_leapfrog_batched_sm"].config.force_chunk_size == 32
    assert cases["left_force_analytic_sm"].config.force_method == "analytic_staple"
    assert cases["first_left_force_analytic_sm"].config.force_method == "analytic_staple"
    assert cases["second_left_force_analytic_sm"].config.force_method == "analytic_staple"
    assert cases["gauge_leapfrog_analytic_sm"].config.force_method == "analytic_staple"


def test_step_breakdown_eigh_yukawa_cases_pin_oracle_mode() -> None:
    cases = {case.name: case for case in default_step_breakdown_cases()}

    assert cases["yukawa_half_kick_sm"].config.yukawa_mode == "unitary"
    assert cases["yukawa_final_half_kick_sm"].config.yukawa_mode == "unitary"
    assert cases["yukawa_half_kick_eigh_sm"].config.yukawa_mode == "unitary_eigh"
    assert cases["yukawa_final_half_kick_eigh_sm"].config.yukawa_mode == "unitary_eigh"


def test_step_breakdown_recommendation_flags_dirac_transport() -> None:
    recommendation = recommend_step_breakdown_bottleneck(
        (
            {"label": "dirac_transport_sm", "mean_seconds": 7.0},
            {"label": "momentum_update_sm", "mean_seconds": 3.0},
        ),
    )

    assert "Dirac transport" in recommendation


def test_step_breakdown_recommendation_flags_momentum_update() -> None:
    recommendation = recommend_step_breakdown_bottleneck(
        (
            {"label": "momentum_update_sm", "mean_seconds": 7.0},
            {"label": "dirac_transport_sm", "mean_seconds": 3.0},
        ),
    )

    assert "momentum link update" in recommendation
