from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.explore.primitives import block_reflection_primitive
from clifford_3plus2_d5.qca.layers import minimal_period_four_update
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleLayerInput,
    layers_from_update,
    result_to_dict,
    rule_to_verdict,
)
from clifford_3plus2_d5.search.addressability import rank_one_color_projector_controls


CASE_CHOICES = (
    "minimal-period-four",
    "clock-block-reflection",
    "clock-rank-one-color-reflection",
)
VERDICT_CHOICES = (
    "bridge_candidate",
    "falsified_no_rank_6_4_center",
    "falsified_rank_one_center",
    "candidate_only_j_not_forced",
    "not_solved",
)


def _parse_matrix(raw: list[list[str]]) -> sp.Matrix:
    return sp.Matrix([[sp.Rational(value) for value in row] for row in raw])


def _layers_from_json(path: Path) -> tuple[str, tuple[RuleLayerInput, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = []
    for raw_layer in payload["layers"]:
        layers.append(
            RuleLayerInput(
                name=raw_layer["name"],
                matrix=_parse_matrix(raw_layer["matrix"]),
                support=tuple(raw_layer.get("support", [0])),
                locality_radius=int(raw_layer.get("locality_radius", 0)),
            )
        )
    return str(payload.get("name", path.stem)), tuple(layers)


def _builtin_layers(case: str) -> tuple[str, tuple[RuleLayerInput, ...]]:
    carrier = standard_real_carrier()
    if case == "minimal-period-four":
        update = minimal_period_four_update()
        return update.name, layers_from_update(update)
    if case == "clock-block-reflection":
        return (
            "clock_block_reflection_rule",
            (
                RuleLayerInput("global_clock_tick", carrier.complex_structure),
                RuleLayerInput("block_reflection_3_minus_2", block_reflection_primitive().matrix),
            ),
        )
    if case == "clock-rank-one-color-reflection":
        projector = rank_one_color_projector_controls()[0].matrix
        return (
            "clock_rank_one_color_reflection_rule",
            (
                RuleLayerInput("global_clock_tick", carrier.complex_structure),
                RuleLayerInput("rank_one_color_reflection", identity(10) - 2 * projector),
            ),
        )
    raise ValueError(f"unknown built-in case: {case}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one-pass finite-depth rule to bridge verdict check."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero unless the run is solved.")
    parser.add_argument("--input-json", type=Path, help="Read a rule candidate JSON file.")
    parser.add_argument("--case", choices=CASE_CHOICES, default="minimal-period-four")
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    parser.add_argument("--max-center-solve-dimension", type=int, default=8)
    parser.add_argument("--max-j-solve-dimension", type=int, default=8)
    args = parser.parse_args()

    if args.input_json is not None:
        rule_name, layers = _layers_from_json(args.input_json)
    else:
        rule_name, layers = _builtin_layers(args.case)

    result = rule_to_verdict(
        layers,
        rule_name=rule_name,
        max_center_solve_dimension=args.max_center_solve_dimension,
        max_j_solve_dimension=args.max_j_solve_dimension,
    )
    payload = result_to_dict(result)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This runs the one-pass finite-depth rule to bridge verdict checker.")
        print("Spinor-16 bookkeeping is intentionally outside this verdict.")
        print(f"rule_name: {payload['rule_name']}")
        print(f"exact_working_field: {payload['exact_working_field']}")
        print(f"layer_count: {payload['layer_count']}")
        print(f"all_layers_real_orthogonal: {str(payload['all_layers_real_orthogonal']).lower()}")
        print(f"all_layers_local: {str(payload['all_layers_local']).lower()}")
        print(f"floquet_spectrum: {payload['floquet_spectrum']}")
        print(f"generated_algebra_dimension: {payload['generated_algebra_dimension']}")
        print(f"center_dimension: {payload['center_dimension']}")
        print(f"center_solved: {str(payload['center_solved']).lower()}")
        print(f"central_idempotent_ranks: {payload['central_idempotent_ranks']}")
        print(f"complementary_rank_6_4_pairs: {payload['complementary_rank_6_4_pairs']}")
        print(
            "lower_rank_central_idempotents: "
            f"{len(payload['lower_rank_central_idempotents'])}"
        )
        print(f"generated_j_solved: {str(payload['generated_j_solved']).lower()}")
        print(f"generated_complex_structures: {len(payload['generated_complex_structures'])}")
        print(f"compatible_centralizer_dimension: {payload['compatible_centralizer_dimension']}")
        print(f"compatible_j_solved: {str(payload['compatible_j_solved']).lower()}")
        print(
            "compatible_complex_structure_count: "
            f"{payload['compatible_complex_structure_count']}"
        )
        print(f"forced_j_found: {str(payload['forced_j_found']).lower()}")
        print(
            "rank_6_4_pair_commutes_with_forced_j: "
            f"{str(payload['rank_6_4_pair_commutes_with_forced_j']).lower()}"
        )
        print(f"pass_rule_to_bridge: {str(payload['pass_rule_to_bridge']).lower()}")
        print(f"verdict: {payload['verdict']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.expect_verdict is not None and result.verdict != args.expect_verdict:
        return 1
    if args.check and not result.pass_rule_to_bridge:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
