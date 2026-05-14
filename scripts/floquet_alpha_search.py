from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.floquet_alpha import (
    floquet_alpha_candidates,
    floquet_alpha_rule_to_verdict,
)
from clifford_3plus2_d5.qca.rule_verdict import result_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the Floquet-alpha primitive family.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero on unexpected search failure.")
    parser.add_argument("--pattern-index", type=int, help="Run a single resonance pattern.")
    args = parser.parse_args()

    candidates = floquet_alpha_candidates()
    if args.pattern_index is not None:
        candidates = tuple(
            candidate for candidate in candidates if candidate.pattern_index == args.pattern_index
        )
        if not candidates:
            raise SystemExit(f"unknown pattern index: {args.pattern_index}")

    results = tuple(floquet_alpha_rule_to_verdict(candidate) for candidate in candidates)
    payload = {
        "family": "floquet_alpha",
        "candidate_count": len(results),
        "bridge_candidates": sum(result.pass_rule_to_bridge for result in results),
        "rank_6_4_pair_candidates": sum(
            result.complementary_rank_6_4_pairs > 0 for result in results
        ),
        "rank_one_falsified_candidates": sum(
            bool(result.lower_rank_central_idempotents) for result in results
        ),
        "verdict_counts": {
            verdict: sum(result.verdict == verdict for result in results)
            for verdict in sorted({result.verdict for result in results})
        },
        "load_bearing_qca_bridge": False,
        "results": [result_to_dict(result) for result in results],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This searches the Floquet-alpha physical primitive family.")
        print("It exposes one mandatory quantized resonance layer per candidate.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"rank_6_4_pair_candidates: {payload['rank_6_4_pair_candidates']}")
        print(f"rank_one_falsified_candidates: {payload['rank_one_falsified_candidates']}")
        print(f"bridge_candidates: {payload['bridge_candidates']}")
        print(f"verdict_counts: {payload['verdict_counts']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")
        for result in results:
            print(
                "candidate: "
                f"{result.rule_name}, "
                f"center_ranks={[item.rank for item in result.central_idempotents]}, "
                f"pairs={result.complementary_rank_6_4_pairs}, "
                f"forced_j={str(result.forced_j_found).lower()}, "
                f"verdict={result.verdict}"
            )

    if args.check and payload["candidate_count"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
