from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.obstruction_r10.sm.spinor16 import certificate_to_dict, spinor16_certificate


VERDICT_CHOICES = ("candidate_only", "derived_from_qca", "falsified")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the guarded Spin(10) spinor reconstruction.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the checker fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    args = parser.parse_args()

    certificate = spinor16_certificate()
    payload = certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This verifies guarded Spin(10) spinor reconstruction only.")
        print("It does not prove QCA rule data force J or the 3+2 split.")
        print(f"spinor16_dimension: {payload['spinor16_dimension']}")
        print(f"degree_dimensions: {payload['degree_dimensions']}")
        print(f"hypercharge_check_passed: {str(payload['hypercharge_check_passed']).lower()}")
        print(
            "branching_table_check_passed: "
            f"{str(payload['branching_table_check_passed']).lower()}"
        )
        print(f"uses_existing_j_and_split: {str(payload['uses_existing_j_and_split']).lower()}")
        print(
            "introduces_new_complex_structure: "
            f"{str(payload['introduces_new_complex_structure']).lower()}"
        )
        print(
            "introduces_new_3plus2_split: "
            f"{str(payload['introduces_new_3plus2_split']).lower()}"
        )
        print(f"spinor16_verdict: {payload['spinor16_verdict']}")
        print(f"spinor16_check_passed: {str(payload['spinor16_check_passed']).lower()}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.expect_verdict is not None and certificate.spinor16_verdict != args.expect_verdict:
        return 1
    if args.check and not payload["spinor16_check_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
