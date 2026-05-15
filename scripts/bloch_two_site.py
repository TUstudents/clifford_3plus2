from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.qca.two_site_bloch import two_site_bloch_certificate


def _certificate_to_dict(
    *,
    variant: str,
    max_generated_algebra_dim: int,
) -> dict[str, object]:
    certificate = two_site_bloch_certificate(
        variant=variant,
        max_generated_algebra_dimension=max_generated_algebra_dim,
    )
    payload = asdict(certificate)
    payload["load_bearing_qca_bridge"] = certificate.load_bearing_qca_bridge
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the two-site Bloch Path-A carrier.")
    parser.add_argument("--variant", choices=("winding-4-3", "uniform"), default="winding-4-3")
    parser.add_argument("--max-generated-algebra-dim", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = _certificate_to_dict(
        variant=args.variant,
        max_generated_algebra_dim=args.max_generated_algebra_dim,
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
