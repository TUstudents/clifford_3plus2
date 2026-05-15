from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from fractions import Fraction

from clifford_3plus2_d5.gauge_equivalence import route1_gauge_equivalence_certificate


def _json_default(value: object) -> str:
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _signs(signs: tuple[int, ...]) -> str:
    return "(" + ",".join("+" if sign > 0 else "-" for sign in signs) + ")"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the Route-1 block-sign gauge-equivalence standard."
    )
    parser.add_argument("--pattern-index", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    certificate = route1_gauge_equivalence_certificate(pattern_index=args.pattern_index)
    payload = asdict(certificate)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    else:
        print("This checks whether Route-1 compatible J signs satisfy the relaxed standard.")
        print(f"candidate_name: {certificate.candidate_name}")
        print(f"compatible_j_count: {certificate.compatible_j_count}")
        print(f"global_pm_orbit_count: {certificate.global_pm_orbit_count}")
        print(
            "intrinsic_branching_tables_match: "
            f"{str(certificate.intrinsic_branching_tables_match).lower()}"
        )
        print(
            "fixed_sm_branching_tables_match_mod_global_pm: "
            f"{str(certificate.fixed_sm_branching_tables_match_mod_global_pm).lower()}"
        )
        print(
            "rule_generated_normalizer_orbit_certified: "
            f"{str(certificate.rule_generated_normalizer_orbit_certified).lower()}"
        )
        print(f"relaxed_standard_supported: {str(certificate.relaxed_standard_supported).lower()}")
        print(f"strict_standard_required: {str(certificate.strict_standard_required).lower()}")
        print(f"verdict: {certificate.verdict}")
        print(f"load_bearing_qca_bridge: {str(certificate.load_bearing_qca_bridge).lower()}")
        for pattern in certificate.patterns:
            print(
                "pattern: "
                f"index={pattern.index}, "
                f"pair_signs={_signs(pattern.pair_orientation_signs)}, "
                f"block_flips=(alpha={pattern.alpha_flip},eta={pattern.eta_flip}), "
                f"global_pm_class={pattern.global_pm_class}, "
                "direct_even_chirality="
                f"{str(pattern.direct_hodge_preserves_even_chirality).lower()}, "
                "direct_fixed_hypercharge_preserved="
                f"{str(pattern.direct_fixed_hypercharge_preserved).lower()}, "
                "global_pm_fixed_hypercharge_preserved="
                f"{str(pattern.global_pm_fixed_hypercharge_preserved).lower()}"
            )

    check_passed = (
        certificate.compatible_j_count == 4
        and certificate.global_pm_orbit_count == 2
        and certificate.intrinsic_branching_tables_match
        and not certificate.fixed_sm_branching_tables_match_mod_global_pm
        and not certificate.rule_generated_normalizer_orbit_certified
        and not certificate.relaxed_standard_supported
        and certificate.strict_standard_required
        and certificate.verdict == "strict_standard_required"
    )
    return 0 if check_passed or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
