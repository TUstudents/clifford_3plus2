from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clifford_3plus2_d5.obstruction_r10.explore.unseeded_projectors import (
    ProjectorDiscoveryBounds,
    primitive_sets_for_mode,
    rank_one_color_control_set,
    run_projector_discovery,
    summary_to_dict,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sanity_seeded_mode_finds_standard_projector_pair() -> None:
    run = run_projector_discovery(mode="sanity-seeded")

    assert run.summary.discovery_verdict == "projector_pair_found"
    assert run.summary.seeded_projector_pairs_found == 1
    assert run.summary.unseeded_projector_pairs_found == 0
    assert run.summary.rank_6_projectors == 1
    assert run.summary.rank_4_projectors == 1
    assert run.summary.discovery_check_passed
    assert not run.summary.load_bearing_qca_bridge


def test_identity_only_unseeded_mode_finds_no_nontrivial_pair() -> None:
    identity_only = (primitive_sets_for_mode("unseeded")[0],)
    run = run_projector_discovery(mode="unseeded", primitive_sets=identity_only)

    assert run.summary.discovery_verdict == "not_found"
    assert run.summary.unseeded_projector_pairs_found == 0
    assert run.summary.candidate_projectors == 0
    assert run.summary.discovery_check_passed


def test_clock_only_unseeded_mode_does_not_derive_standard_pair() -> None:
    clock_only = (primitive_sets_for_mode("unseeded")[1],)
    run = run_projector_discovery(mode="unseeded", primitive_sets=clock_only)

    assert run.summary.discovery_verdict == "not_found"
    assert run.summary.unseeded_projector_pairs_found == 0
    assert run.summary.rank_6_projectors == 0
    assert run.summary.rank_4_projectors == 0
    assert run.summary.discovery_check_passed


def test_block_reflection_candidate_mode_derives_complementary_pair() -> None:
    run = run_projector_discovery(mode="block-reflection-candidate")

    assert run.summary.discovery_verdict == "projector_pair_found"
    assert run.summary.block_reflection_pairs_found == 1
    assert run.summary.seeded_projector_pairs_found == 0
    assert run.summary.unseeded_projector_pairs_found == 0
    assert run.summary.discovery_check_passed
    assert any(candidate.equals_standard_p3 for candidate in run.candidates)
    assert any(candidate.equals_standard_p2 for candidate in run.candidates)


def test_unseeded_mode_flags_unsafe_rank_one_projectors() -> None:
    run = run_projector_discovery(mode="unseeded")

    assert run.summary.discovery_verdict == "not_found"
    assert run.summary.unseeded_projector_pairs_found == 0
    assert run.summary.unsafe_rank_one_projectors > 0
    assert any(candidate.unsafe_rank_one_projector for candidate in run.candidates)


def test_rank_one_color_control_falsifier_is_flagged_unsafe() -> None:
    run = run_projector_discovery(
        mode="unseeded",
        primitive_sets=(rank_one_color_control_set(),),
    )

    assert run.summary.discovery_verdict == "projector_pair_found"
    assert run.summary.unseeded_projector_pairs_found > 0
    assert run.summary.unsafe_rank_one_projectors > 0
    assert not run.summary.discovery_check_passed


def test_summary_json_is_stable_for_unseeded_mode() -> None:
    run = run_projector_discovery(mode="unseeded")
    payload = summary_to_dict(run.summary, bounds=run.bounds)

    assert payload["mode"] == "unseeded"
    assert payload["primitive_sets_scanned"] == 6
    assert payload["unseeded_projector_pairs_found"] == 0
    assert payload["discovery_verdict"] == "not_found"
    assert payload["discovery_check_passed"] is True
    assert payload["load_bearing_qca_bridge"] is False


def test_discover_projectors_cli_writes_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/discover_projectors.py",
            "--check",
            "--json",
            "--mode",
            "unseeded",
            "--expect-verdict",
            "not_found",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["discovery_check_passed"] is True
    assert payload["unseeded_projector_pairs_found"] == 0
    assert (tmp_path / "e2_summary.json").exists()
    assert (tmp_path / "e2_projector_candidates.jsonl").exists()
    assert (tmp_path / "e2_rejections.jsonl").exists()


def test_discover_projectors_expect_verdict_fails_on_mismatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/discover_projectors.py",
            "--mode",
            "unseeded",
            "--expect-verdict",
            "projector_pair_found",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1


def test_tighter_bounds_still_search_exactly() -> None:
    run = run_projector_discovery(
        mode="block-reflection-candidate",
        bounds=ProjectorDiscoveryBounds(max_word_depth=1, max_basis_size=3),
    )

    assert run.summary.discovery_verdict == "projector_pair_found"
    assert run.summary.block_reflection_pairs_found == 1
