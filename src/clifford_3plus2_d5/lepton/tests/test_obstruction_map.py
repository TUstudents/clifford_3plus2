"""Tests for the consolidated lepton obstruction map."""

from __future__ import annotations

import json
import subprocess

from clifford_3plus2_d5.lepton.obstruction_map import (
    ObstructionEntry,
    ObstructionMode,
    ObstructionStatus,
    obstruction_entries,
    obstruction_mode_counts,
    obstruction_status_counts,
    obstruction_summary,
)


def _entry_by_name() -> dict[str, ObstructionEntry]:
    return {entry.name: entry for entry in obstruction_entries()}


def test_obstruction_map_contains_session_milestones() -> None:
    entries = _entry_by_name()
    for name in (
        "lab_b_internal_domain_wall_r6",
        "lab_b_physical_domain_wall_r12",
        "complex_c5_discovered_synthetic_split",
        "complex_c5_monomial_search",
        "complex_c5_dense_fourier_lite_search",
        "designed_locality_aware_complex_qca",
    ):
        assert name in entries


def test_obstruction_map_separates_controls_domain_wall_and_bridge() -> None:
    entries = _entry_by_name()
    internal_wall = entries["lab_b_internal_domain_wall_r6"]
    physical_wall = entries["lab_b_physical_domain_wall_r12"]
    dense_control = entries["complex_c5_dense_conjugated_control"]

    assert internal_wall.status == ObstructionStatus.DOMAIN_WALL_POSITIVE
    assert internal_wall.load_bearing_domain_wall_candidate
    assert not internal_wall.load_bearing_qca_bridge

    assert physical_wall.status == ObstructionStatus.PHYSICAL_NEGATIVE
    assert physical_wall.obstruction == ObstructionMode.PHYSICAL_PROMOTION_EXTRA_CENTER
    assert not physical_wall.load_bearing_qca_bridge

    assert dense_control.status == ObstructionStatus.CONTROL_POSITIVE
    assert dense_control.obstruction == ObstructionMode.SYNTHETIC_ONLY
    assert not dense_control.load_bearing_qca_bridge


def test_obstruction_summary_counts_are_stable() -> None:
    summary = obstruction_summary()
    assert summary["entry_count"] == 16
    assert summary["load_bearing_qca_bridge_count"] == 0
    assert summary["domain_wall_candidate_count"] == 1
    assert summary["open_entries"] == ("designed_locality_aware_complex_qca",)
    assert dict(obstruction_status_counts()) == {
        "search_negative": 8,
        "control_positive": 4,
        "domain_wall_positive": 1,
        "open": 1,
        "physical_negative": 1,
        "toy_positive": 1,
    }
    assert dict(obstruction_mode_counts())["center_too_large"] == 3
    assert dict(obstruction_mode_counts())["rank_one_locking"] == 2


def test_obstruction_summary_cli_outputs_json() -> None:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "clifford_3plus2_d5.lepton.scripts.obstruction_summary",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["summary"]["entry_count"] == 16
    assert payload["summary"]["load_bearing_qca_bridge_count"] == 0
