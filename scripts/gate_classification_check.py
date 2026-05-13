from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.sm.classification import (
    certificate_to_dict,
    gate_classification_certificate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the SM commutant gate classifier.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the checker fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    certificate = gate_classification_certificate()
    payload = certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This verifies the exact SM commutant gate classifier only.")
        print("It does not prove QCA rule data supply only safe geometric gates.")
        for gate in payload["gates"]:
            print(f"{gate['name']}: {gate['classification']}")
        print(
            "commutant_basis_matches_expected: "
            f"{str(payload['commutant_basis_matches_expected']).lower()}"
        )
        print(
            "safe_algebra_closure_passed: "
            f"{str(payload['safe_algebra_closure_passed']).lower()}"
        )
        print(
            "gate_classification_check_passed: "
            f"{str(payload['gate_classification_check_passed']).lower()}"
        )
        print(
            "qca_geometric_gate_algebra_safe: "
            f"{str(payload['qca_geometric_gate_algebra_safe']).lower()}"
        )
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if args.check and not payload["gate_classification_check_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
