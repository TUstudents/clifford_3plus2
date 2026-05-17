from __future__ import annotations

import argparse
import json
from pathlib import Path

from clifford_3plus2_d5.obstruction_r10.explore.primitives import default_e1_rule_space
from clifford_3plus2_d5.obstruction_r10.explore.rule_space import SearchBounds, summary_to_dict
from clifford_3plus2_d5.obstruction_r10.explore.search_runner import (
    run_rule_space_exploration,
    write_exploration_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E1 bounded rule-space exploration.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if exploration fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-primitive-set-size", type=int, default=4)
    parser.add_argument("--max-primitive-sets", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=50_000)
    parser.add_argument("--max-survivors", type=int, default=50)
    parser.add_argument("--max-rejections", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("data/exploration"))
    args = parser.parse_args()

    bounds = SearchBounds(
        max_depth=args.max_depth,
        max_primitive_set_size=args.max_primitive_set_size,
        max_primitive_sets=args.max_primitive_sets,
        max_words=args.max_words,
        max_survivors=args.max_survivors,
        max_rejections=args.max_rejections,
    )
    rule_space = default_e1_rule_space()
    run = run_rule_space_exploration(rule_space=rule_space, bounds=bounds)
    summary_path, survivors_path, rejections_path = write_exploration_artifacts(
        run,
        args.output_dir,
    )
    payload = summary_to_dict(
        run.summary,
        bounds=bounds,
        primitive_sets=run.primitive_sets_scanned,
    )
    payload["artifacts"] = {
        "summary": str(summary_path),
        "survivors": str(survivors_path),
        "rejections": str(rejections_path),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This runs bounded exact rule-space exploration sprint E1.")
        print("It does not prove microscopic QCA rule data force J or P_3/P_2.")
        print(f"rule_space_name: {payload['rule_space_name']}")
        print(f"primitive_family_count: {payload['primitive_family_count']}")
        print(f"primitive_set_count: {payload['primitive_set_count']}")
        print(f"primitive_sets_scanned: {payload['primitive_sets_scanned']}")
        print(f"words_scanned: {payload['words_scanned']}")
        print(f"j_hits: {payload['j_hits']}")
        print(f"period_four_hits: {payload['period_four_hits']}")
        print(f"split_candidates: {payload['split_candidates']}")
        print(f"addressability_safe_hits: {payload['addressability_safe_hits']}")
        print(f"normalizer_candidate_hits: {payload['normalizer_candidate_hits']}")
        print(f"forced_candidate_hits: {payload['forced_candidate_hits']}")
        print(f"rank_one_rejections: {payload['rank_one_rejections']}")
        print(f"off_block_rejections: {payload['off_block_rejections']}")
        print(
            "normalizer_too_large_rejections: "
            f"{payload['normalizer_too_large_rejections']}"
        )
        top_rejections = ",".join(
            f"{item['reason']}={item['count']}"
            for item in payload["top_rejection_reasons"]
        )
        print(f"top_rejection_reasons: {top_rejections}")
        print(f"surviving_candidates: {payload['surviving_candidates']}")
        print(f"summary_artifact: {summary_path}")
        print(f"survivors_artifact: {survivors_path}")
        print(f"rejections_artifact: {rejections_path}")
        print(
            "exploration_check_passed: "
            f"{str(payload['exploration_check_passed']).lower()}"
        )
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.check and not run.summary.exploration_check_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
