from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.qca.two_site_bloch import (
    two_site_bloch_certificate,
    two_site_split_step_search_summary,
)


def _certificate_to_dict(
    *,
    variant: str,
    max_generated_algebra_dim: int,
    max_coefficient_algebra_dim: int,
) -> dict[str, object]:
    certificate = two_site_bloch_certificate(
        variant=variant,
        max_generated_algebra_dimension=max_generated_algebra_dim,
        max_coefficient_algebra_dimension=max_coefficient_algebra_dim,
    )
    payload = asdict(certificate)
    payload["load_bearing_qca_bridge"] = certificate.load_bearing_qca_bridge
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the two-site Bloch Path-A carrier.")
    parser.add_argument("--split-step-search", action="store_true")
    parser.add_argument("--variant", choices=("winding-4-3", "uniform"), default="winding-4-3")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-generated-algebra-dim", type=int, default=16)
    parser.add_argument("--max-center-dim", type=int, default=8)
    parser.add_argument("--max-coefficient-algebra-dim", type=int, default=32)
    parser.add_argument("--split-step-coefficient-algebra-dim", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.split_step_search:
        summary = two_site_split_step_search_summary(
            max_candidates=args.max_candidates,
            max_generated_algebra_dimension=args.max_generated_algebra_dim,
            max_center_dimension=args.max_center_dim,
            max_coefficient_algebra_dimension=args.split_step_coefficient_algebra_dim,
        )
        payload = asdict(summary)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("This runs the two-site split-step coin search.")
            print(f"candidate_count: {payload['candidate_count']}")
            print(f"seed_guardrail_rejections: {payload['seed_guardrail_rejections']}")
            print(f"laurent_orthogonal_candidates: {payload['laurent_orthogonal_candidates']}")
            print(f"closed_candidates: {payload['closed_candidates']}")
            print(f"effective_6_4_candidates: {payload['effective_6_4_candidates']}")
            print(f"strict_bridge_candidates: {payload['strict_bridge_candidates']}")
            print(f"route_label: {payload['route_label']}")
            print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")
            for result in payload["candidates"]:
                print(
                    "candidate: "
                    f"{result['candidate_name']}, "
                    f"shifts={result['source_shifts']}, "
                    f"coins=({result['left_coin']},{result['right_coin']}), "
                    f"seed={str(result['seed_guardrail_passed']).lower()}, "
                    f"laurent={str(result['laurent_orthogonal']).lower()}, "
                    f"coef_dim={result['coefficient_algebra_dimension']}, "
                    f"coef_closed={str(result['coefficient_algebra_closed']).lower()}, "
                    f"dim={result['generated_algebra_dimension']}, "
                    f"closed={str(result['generated_algebra_closed']).lower()}, "
                    f"center={result['center_dimension']}, "
                    f"center_solved={str(result['center_solved']).lower()}, "
                    f"ranks={result['central_idempotent_ranks']}, "
                    f"effective_pairs={result['effective_rank_6_4_pairs']}, "
                    f"label={result['route_label']}"
                )
        if args.check:
            if summary.candidate_count == 0:
                return 1
            if summary.strict_bridge_candidates != 0:
                return 1
        return 0

    payload = _certificate_to_dict(
        variant=args.variant,
        max_generated_algebra_dim=args.max_generated_algebra_dim,
        max_coefficient_algebra_dim=args.max_coefficient_algebra_dim,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks the first two-site Bloch Path-A carrier.")
        print(f"variant: {args.variant}")
        print(f"rule_name: {payload['rule_name']}")
        print(f"dimension: {payload['dimension']}")
        print(f"term_count: {payload['term_count']}")
        print(f"shifts: {payload['shifts']}")
        print(f"laurent_orthogonal: {str(payload['laurent_orthogonal']).lower()}")
        print(f"seed_guardrail_passed: {str(payload['seed_guardrail_passed']).lower()}")
        print(f"coefficient_algebra_dimension: {payload['coefficient_algebra_dimension']}")
        print(f"coefficient_algebra_closed: {str(payload['coefficient_algebra_closed']).lower()}")
        print(f"generated_algebra_dimension: {payload['generated_algebra_dimension']}")
        print(f"generated_algebra_closed: {str(payload['generated_algebra_closed']).lower()}")
        print(f"center_dimension: {payload['center_dimension']}")
        print(f"center_solved: {str(payload['center_solved']).lower()}")
        print(f"central_idempotent_ranks: {payload['central_idempotent_ranks']}")
        print(f"effective_rank_6_4_pairs: {payload['effective_rank_6_4_pairs']}")
        print(f"route_label: {payload['route_label']}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.check:
        if not payload["coefficient_matrices_real"]:
            return 1
        if not payload["laurent_orthogonal"]:
            return 1
        if not payload["generated_algebra_closed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
