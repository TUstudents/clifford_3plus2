from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clifford_3plus2_d5.explore.primitives import (
    default_e1_rule_space,
    global_clock_primitive,
    identity_primitive,
    rank_one_pair_primitive,
)
from clifford_3plus2_d5.explore.projector_discovery import discover_projector_pair
from clifford_3plus2_d5.explore.rule_space import (
    PrimitiveFamily,
    PrimitiveSet,
    RuleSpace,
    SearchBounds,
    summary_to_dict,
)
from clifford_3plus2_d5.explore.search_runner import run_rule_space_exploration
from clifford_3plus2_d5.search.addressability import (
    off_block_mixer_control,
    rank_one_color_projector_controls,
    standard_block_projectors,
)


ROOT = Path(__file__).resolve().parents[1]


def _single_set_rule_space(primitive_set: PrimitiveSet) -> RuleSpace:
    return RuleSpace(
        name=f"test_{primitive_set.name}",
        primitive_families=(PrimitiveFamily("test_family", primitive_set.primitives),),
        primitive_sets=(primitive_set,),
    )


def test_search_bounds_stop_enumeration() -> None:
    run = run_rule_space_exploration(
        rule_space=default_e1_rule_space(),
        bounds=SearchBounds(max_words=3),
    )

    assert run.summary.words_scanned == 3
    assert not run.summary.exploration_check_passed


def test_identity_only_rule_space_finds_no_j() -> None:
    primitive_set = PrimitiveSet(
        "identity_only",
        (identity_primitive(),),
        standard_block_projectors(),
    )
    run = run_rule_space_exploration(
        rule_space=_single_set_rule_space(primitive_set),
        bounds=SearchBounds(max_depth=4),
    )

    assert run.summary.words_scanned == 4
    assert run.summary.j_hits == 0
    assert run.summary.period_four_hits == 0


def test_standard_clock_sanity_space_detects_candidate_j() -> None:
    primitive_set = PrimitiveSet(
        "clock_sanity",
        (global_clock_primitive(),),
        standard_block_projectors(),
    )
    run = run_rule_space_exploration(
        rule_space=_single_set_rule_space(primitive_set),
        bounds=SearchBounds(max_depth=4),
    )

    assert run.summary.j_hits > 0
    assert run.summary.period_four_hits > 0
    assert run.summary.split_candidates > 0
    assert run.summary.forced_candidate_hits == 0


def test_rank_one_pair_family_is_rejected() -> None:
    primitive_set = PrimitiveSet(
        "rank_one_pair_falsifier",
        (global_clock_primitive(), rank_one_pair_primitive(0)),
        standard_block_projectors(),
    )
    run = run_rule_space_exploration(
        rule_space=_single_set_rule_space(primitive_set),
        bounds=SearchBounds(max_depth=2),
    )

    assert run.summary.rank_one_rejections == run.summary.words_scanned
    assert run.summary.surviving_candidates == 0


def test_rank_one_color_control_is_rejected() -> None:
    primitive_set = PrimitiveSet(
        "rank_one_color_control_falsifier",
        (global_clock_primitive(),),
        standard_block_projectors() + rank_one_color_projector_controls(),
    )
    run = run_rule_space_exploration(
        rule_space=_single_set_rule_space(primitive_set),
        bounds=SearchBounds(max_depth=2),
    )

    assert run.summary.rank_one_rejections == run.summary.words_scanned
    assert run.summary.surviving_candidates == 0


def test_off_block_control_is_rejected() -> None:
    primitive_set = PrimitiveSet(
        "off_block_control_falsifier",
        (global_clock_primitive(),),
        standard_block_projectors() + (off_block_mixer_control(),),
    )
    run = run_rule_space_exploration(
        rule_space=_single_set_rule_space(primitive_set),
        bounds=SearchBounds(max_depth=2),
    )

    assert run.summary.off_block_rejections == run.summary.words_scanned
    assert run.summary.surviving_candidates == 0


def test_projector_discovery_finds_seeded_standard_pair() -> None:
    discovery = discover_projector_pair(
        rule_data_operators=standard_block_projectors(),
    )

    assert discovery.projector_pair_found
    assert discovery.rank_6_projectors > 0
    assert discovery.rank_4_projectors > 0


def test_default_summary_records_real_exploration_counts() -> None:
    run = run_rule_space_exploration(
        rule_space=default_e1_rule_space(),
        bounds=SearchBounds(max_words=120),
    )
    payload = summary_to_dict(run.summary, bounds=run.bounds)

    assert payload["primitive_sets_scanned"] > 1
    assert payload["words_scanned"] == 120
    assert payload["j_hits"] > 0
    assert payload["period_four_hits"] > 0
    assert payload["forced_candidate_hits"] == 0
    assert payload["surviving_candidates"] == 0
    assert payload["exploration_check_passed"] is True
    assert payload["load_bearing_qca_bridge"] is False


def test_explore_rule_space_cli_writes_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/explore_rule_space.py",
            "--check",
            "--json",
            "--max-words",
            "120",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["exploration_check_passed"] is True
    assert payload["forced_candidate_hits"] == 0
    assert (tmp_path / "e1_summary.json").exists()
    assert (tmp_path / "e1_survivors.jsonl").exists()
    assert (tmp_path / "e1_rejections.jsonl").exists()
