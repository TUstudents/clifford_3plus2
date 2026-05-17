from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.algebra.real_carrier import carrier_certificate


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the exact J-first real carrier ansatz.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if Phase 1 checks fail.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    certificate = carrier_certificate()
    passed = bool(certificate["phase_1_real_carrier_check_passed"])

    if args.json:
        print(json.dumps(certificate, indent=2, sort_keys=True))
    else:
        print("This verifies the exact real carrier ansatz only.")
        print("It does not prove QCA dynamics force J.")
        print(f"carrier: {certificate['carrier']}")
        print(f"dimension: {certificate['dimension']}")
        print(f"mode_dimension: {certificate['mode_dimension']}")
        print(f"J_squared_minus_identity: {str(certificate['j_squared_minus_identity']).lower()}")
        print(f"J_orthogonal: {str(certificate['j_orthogonal']).lower()}")
        print(f"projector_3_rank: {certificate['projector_3_rank']}")
        print(f"projector_2_rank: {certificate['projector_2_rank']}")
        print(
            "projectors_commute_with_J: "
            f"{str(certificate['projector_3_commutes_with_j'] and certificate['projector_2_commutes_with_j']).lower()}"
        )
        print(f"phase_1_real_carrier_check_passed: {str(passed).lower()}")
        print("qca_forces_j: false")
        print("load_bearing_qca_bridge: false")

    return 0 if passed or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
