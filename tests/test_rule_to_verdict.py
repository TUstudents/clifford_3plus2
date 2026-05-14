from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.explore.primitives import block_reflection_primitive
from clifford_3plus2_d5.qca.layers import minimal_period_four_update
from clifford_3plus2_d5.qca.rule_verdict import (
    EXACT_WORKING_FIELD,
    RuleLayerInput,
    layers_from_update,
    result_to_dict,
    rule_to_verdict,
)
from clifford_3plus2_d5.search.addressability import rank_one_color_projector_controls


ROOT = Path(__file__).resolve().parents[1]


def test_minimal_period_four_rule_has_j_but_no_6_4_center() -> None:
    update = minimal_period_four_update()
    result = rule_to_verdict(layers_from_update(update), rule_name=update.name)

    assert result.all_layers_real_orthogonal
    assert result.all_layers_local
    assert result.center_solved
    assert [item.rank for item in result.central_idempotents] == [0, 10]
    assert result.generated_j_solved
    assert len(result.generated_complex_structures) == 2
    assert result.complementary_rank_6_4_pairs == 0
    assert result.verdict == "falsified_no_rank_6_4_center"
    assert not result.pass_rule_to_bridge
    assert not result.load_bearing_qca_bridge


def test_block_reflection_rule_finds_pair_but_not_forced_j() -> None:
    carrier = standard_real_carrier()
    result = rule_to_verdict(
        (
            RuleLayerInput("global_clock_tick", carrier.complex_structure),
            RuleLayerInput("block_reflection_3_minus_2", block_reflection_primitive().matrix),
        ),
        rule_name="clock_block_reflection_rule",
    )

    assert result.center_solved
    assert [item.rank for item in result.central_idempotents] == [0, 4, 6, 10]
    assert result.complementary_rank_6_4_pairs == 1
    assert result.lower_rank_central_idempotents == ()
    assert result.generated_j_solved
    assert len(result.generated_complex_structures) == 4
    assert not result.compatible_j_solved
    assert not result.forced_j_found
    assert result.verdict == "candidate_only_j_not_forced"


def test_rank_one_reflection_rule_is_falsified_by_lower_center() -> None:
    carrier = standard_real_carrier()
    projector = rank_one_color_projector_controls()[0].matrix
    result = rule_to_verdict(
        (
            RuleLayerInput("global_clock_tick", carrier.complex_structure),
            RuleLayerInput("rank_one_color_reflection", identity(10) - 2 * projector),
        ),
        rule_name="clock_rank_one_color_reflection_rule",
    )

    assert [item.rank for item in result.central_idempotents] == [0, 2, 8, 10]
    assert [item.rank for item in result.lower_rank_central_idempotents] == [2]
    assert result.verdict == "falsified_rank_one_center"


def test_result_serialization_is_stable() -> None:
    update = minimal_period_four_update()
    payload = result_to_dict(rule_to_verdict(layers_from_update(update), rule_name=update.name))

    assert payload["rule_name"] == "minimal_period_four_clock_candidate"
    assert payload["exact_working_field"] == EXACT_WORKING_FIELD
    assert payload["floquet_spectrum"] == [{"eigenvalue": "1", "multiplicity": 10}]
    assert payload["central_idempotent_ranks"] == [0, 10]
    assert payload["generated_j_solved"] is True
    assert isinstance(payload["compatible_centralizer_dimension"], int)
    assert payload["compatible_j_solved"] is False
    assert payload["pass_rule_to_bridge"] is False
    assert payload["load_bearing_qca_bridge"] is False


def test_rule_to_verdict_cli_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/rule_to_verdict.py",
            "--json",
            "--case",
            "clock-block-reflection",
            "--expect-verdict",
            "candidate_only_j_not_forced",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["exact_working_field"] == EXACT_WORKING_FIELD
    assert payload["central_idempotent_ranks"] == [0, 4, 6, 10]
    assert payload["complementary_rank_6_4_pairs"] == 1
    assert isinstance(payload["compatible_centralizer_dimension"], int)
    assert payload["forced_j_found"] is False


def test_rule_to_verdict_cli_check_fails_without_bridge_candidate() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/rule_to_verdict.py",
            "--check",
            "--case",
            "minimal-period-four",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
