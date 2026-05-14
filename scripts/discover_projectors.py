from __future__ import annotations

import argparse
import json
from pathlib import Path

from clifford_3plus2_d5.explore.unseeded_projectors import (
    ProjectorDiscoveryBounds,
    rank_one_color_control_set,
    run_projector_discovery,
    summary_to_dict,
    write_projector_discovery_artifacts,
)


MODE_CHOICES = ("unseeded", "sanity-seeded", "block-reflection-candidate")
VERDICT_CHOICES = ("projector_pair_found", "not_found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E2 projector discovery.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if discovery fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="unseeded")
    parser.add_argument("--max-word-depth", type=int, default=2)
    parser.add_argument("--max-basis-size", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument(
        "--include-rank-one-color",
        action="store_true",
        help="Run the selected mode with a rank-one color control falsifier set.",
    )
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    parser.add_argument("--output-dir", type=Path, default=Path("data/exploration"))
    args = parser.parse_args()

    bounds = ProjectorDiscoveryBounds(
        max_word_depth=args.max_word_depth,
        max_basis_size=args.max_basis_size,
        max_candidates=args.max_candidates,
    )
    primitive_sets = None
    if args.include_rank_one_color:
        primitive_sets = (rank_one_color_control_set(),)

    run = run_projector_discovery(
        mode=args.mode,
        bounds=bounds,
        primitive_sets=primitive_sets,
    )
    summary_path, candidates_path, rejections_path = write_projector_discovery_artifacts(
        run,
        args.output_dir,
    )
    payload = summary_to_dict(
        run.summary,
        bounds=run.bounds,
        primitive_results=run.primitive_results,
    )
    payload["artifacts"] = {
        "summary": str(summary_path),
        "candidates": str(candidates_path),
        "rejections": str(rejections_path),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This runs E2 projector discovery.")
        print("It separates unseeded discovery from seeded sanity checks.")
        print(f"mode: {payload['mode']}")
        print(f"primitive_sets_scanned: {payload['primitive_sets_scanned']}")
        print(f"algebra_elements_considered: {payload['algebra_elements_considered']}")
        print(f"candidate_projectors: {payload['candidate_projectors']}")
        print(f"rank_2_projectors: {payload['rank_2_projectors']}")
        print(f"rank_6_projectors: {payload['rank_6_projectors']}")
        print(f"rank_4_projectors: {payload['rank_4_projectors']}")
        print(f"complementary_pairs: {payload['complementary_pairs']}")
        print(f"unsafe_rank_one_projectors: {payload['unsafe_rank_one_projectors']}")
        print(
            "unseeded_projector_pairs_found: "
            f"{payload['unseeded_projector_pairs_found']}"
        )
        print(
            "seeded_projector_pairs_found: "
            f"{payload['seeded_projector_pairs_found']}"
        )
        print(
            "block_reflection_pairs_found: "
            f"{payload['block_reflection_pairs_found']}"
        )
        print(f"discovery_verdict: {payload['discovery_verdict']}")
        print(
            "discovery_check_passed: "
            f"{str(payload['discovery_check_passed']).lower()}"
        )
        print(f"summary_artifact: {summary_path}")
        print(f"candidates_artifact: {candidates_path}")
        print(f"rejections_artifact: {rejections_path}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.expect_verdict is not None and run.summary.discovery_verdict != args.expect_verdict:
        return 1
    if args.check and not run.summary.discovery_check_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
