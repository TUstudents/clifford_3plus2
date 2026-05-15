from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.explore.primitives import block_reflection_primitive
from clifford_3plus2_d5.qca.layers import minimal_period_four_update
from clifford_3plus2_d5.qca.rule_verdict import (
    CentralIdempotent,
    EXACT_WORKING_FIELD,
    RuleBlochTerm,
    RuleLayerInput,
    layers_from_update,
    result_to_dict,
    rule_to_verdict,
    solve_complex_structures_from_idempotent_splitting,
)
from clifford_3plus2_d5.search.addressability import rank_one_color_projector_controls


ROOT = Path(__file__).resolve().parents[1]


def test_idempotent_split_j_solver_finds_four_complex_structures() -> None:
    zero = sp.zeros(4)
    one = sp.eye(4)
    p_left = sp.diag(1, 1, 0, 0)
    j_left = sp.Matrix(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    p_right = sp.diag(0, 0, 1, 1)
    j_right = sp.Matrix(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0],
        ]
    )
    idempotents = (
        CentralIdempotent((), zero, 0),
        CentralIdempotent((), p_left, 2),
        CentralIdempotent((), p_right, 2),
        CentralIdempotent((), one, 4),
    )

    solved, moduli_dimension, candidates = solve_complex_structures_from_idempotent_splitting(
        (p_left, j_left, p_right, j_right),
        idempotents,
        source="local_compatible_center",
        dimension=4,
    )

    assert solved
    assert moduli_dimension == 0
    assert len(candidates) == 4
    assert all(candidate.matrix * candidate.matrix == -one for candidate in candidates)


def test_minimal_period_four_rule_has_j_but_no_6_4_center() -> None:
    update = minimal_period_four_update()
    result = rule_to_verdict(layers_from_update(update), rule_name=update.name)

    assert result.all_layers_real_orthogonal
    assert result.all_layers_local
    assert result.locality_radius_bound == 0
    assert result.center_solved
    assert result.generated_algebra_closed
    assert [item.rank for item in result.central_idempotents] == [0, 10]
    assert result.generated_j_solved
    assert result.generated_j_moduli_dimension == 0
    assert len(result.generated_complex_structures) == 2
    assert result.compatible_j_moduli_dimension is None
    assert result.local_compatible_operator_dimension == 2
    assert result.local_compatible_j_solved
    assert result.local_compatible_j_moduli_dimension == 0
    assert len(result.local_compatible_complex_structures) == 2
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
    assert result.generated_j_moduli_dimension == 0
    assert len(result.generated_complex_structures) == 4
    assert result.compatible_j_moduli_dimension is None
    assert result.local_compatible_operator_dimension == 4
    assert result.local_compatible_j_solved
    assert result.local_compatible_j_moduli_dimension == 0
    assert len(result.local_compatible_complex_structures) == 4
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
    assert payload["natural_eigenvalue_field"] == "QQ"
    assert payload["bloch_period"] is None
    assert payload["bloch_sample_count"] == 0
    assert payload["all_layers_local"] is True
    assert payload["locality_radius_bound"] == 0
    assert payload["floquet_spectrum"] == [{"eigenvalue": "1", "multiplicity": 10}]
    assert payload["central_idempotent_ranks"] == [0, 10]
    assert payload["generated_algebra_closed"] is True
    assert payload["generated_j_solved"] is True
    assert payload["generated_j_moduli_dimension"] == 0
    assert isinstance(payload["compatible_centralizer_dimension"], int)
    assert payload["compatible_j_moduli_dimension"] is None
    assert payload["local_compatible_operator_dimension"] == 2
    assert payload["local_compatible_j_solved"] is True
    assert payload["local_compatible_j_moduli_dimension"] == 0
    assert payload["local_compatible_complex_structure_count"] == 2
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
    assert payload["compatible_j_moduli_dimension"] is None
    assert payload["local_compatible_operator_dimension"] == 4
    assert payload["local_compatible_j_moduli_dimension"] == 0
    assert payload["local_compatible_complex_structure_count"] == 4
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


def test_rule_to_verdict_samples_bloch_symbols_when_period_is_present() -> None:
    layer = RuleLayerInput(
        name="identity_shift",
        matrix=identity(10),
        locality_radius=1,
        bloch_terms=(RuleBlochTerm(shift=1, matrix=identity(10)),),
    )

    result = rule_to_verdict(
        (layer,),
        rule_name="identity_shift_bloch",
        bloch_period=4,
        max_center_solve_dimension=2,
        max_j_solve_dimension=2,
    )

    assert result.bloch_period == 4
    assert result.bloch_sample_count == 4
    assert result.natural_eigenvalue_field == "QQ(zeta_4)"
    assert result.layer_count == 1
    assert result.all_layers_real_orthogonal
    assert result.all_layers_local
    assert result.locality_radius_bound == 1
