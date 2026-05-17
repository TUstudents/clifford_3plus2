from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.obstruction_r10.qca.certificates import certificate_to_dict, qca_update_certificate
from clifford_3plus2_d5.obstruction_r10.qca.gates import rank_one_color_shift_gate, rank_one_weak_shift_gate
from clifford_3plus2_d5.obstruction_r10.qca.layers import QCALayer, QCAUpdate, minimal_period_four_update


VERDICT_CHOICES = ("finite_depth_candidate", "candidate_only", "falsified")


def _update_from_args(include_rank_one_color_shift: bool, include_rank_one_weak_shift: bool) -> QCAUpdate:
    update = minimal_period_four_update()
    layers = update.layers
    if include_rank_one_color_shift:
        layers = layers + (
            QCALayer("rank_one_color_shift_layer", (rank_one_color_shift_gate(),)),
        )
    if include_rank_one_weak_shift:
        layers = layers + (
            QCALayer("rank_one_weak_shift_layer", (rank_one_weak_shift_gate(),)),
        )
    return QCAUpdate(name=update.name, layers=layers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the finite-depth QCA update candidate.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the checker fails.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--include-rank-one-color-shift",
        action="store_true",
        help="Include a spacetime shift with a rank-one color internal projector.",
    )
    parser.add_argument(
        "--include-rank-one-weak-shift",
        action="store_true",
        help="Include a spacetime shift with a rank-one weak internal projector.",
    )
    parser.add_argument("--expect-verdict", choices=VERDICT_CHOICES)
    args = parser.parse_args()

    update = _update_from_args(
        include_rank_one_color_shift=args.include_rank_one_color_shift,
        include_rank_one_weak_shift=args.include_rank_one_weak_shift,
    )
    certificate = qca_update_certificate(update)
    payload = certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This verifies the finite-depth QCA update candidate only.")
        print("It does not prove microscopic QCA rule data force this update.")
        print(f"finite_depth: {str(payload['finite_depth']).lower()}")
        print(f"layer_count: {payload['layer_count']}")
        print(f"max_locality_radius: {payload['max_locality_radius']}")
        print(f"all_layers_local: {str(payload['all_layers_local']).lower()}")
        print(f"all_layers_orthogonal: {str(payload['all_layers_orthogonal']).lower()}")
        print(f"period_four_check_passed: {str(payload['period_four_check_passed']).lower()}")
        print(f"quarter_period_is_j: {str(payload['quarter_period_is_j']).lower()}")
        print(
            "half_period_is_minus_identity: "
            f"{str(payload['half_period_is_minus_identity']).lower()}"
        )
        print(f"full_period_is_identity: {str(payload['full_period_is_identity']).lower()}")
        print(
            "all_internal_actions_safe: "
            f"{str(payload['all_internal_actions_safe']).lower()}"
        )
        print(f"unsafe_gate_witnesses: {','.join(payload['unsafe_gate_witnesses'])}")
        print(f"qca_rule_forces_update: {str(payload['qca_rule_forces_update']).lower()}")
        print(f"finite_depth_qca_verdict: {payload['finite_depth_qca_verdict']}")
        print(f"qca_update_check_passed: {str(payload['qca_update_check_passed']).lower()}")
        print(f"load_bearing_qca_bridge: {str(payload['load_bearing_qca_bridge']).lower()}")

    if (
        args.expect_verdict is not None
        and certificate.finite_depth_qca_verdict != args.expect_verdict
    ):
        return 1
    if args.check and not payload["qca_update_check_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
