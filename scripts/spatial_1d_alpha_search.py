from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.spatial_1d import (
    Spatial1DAlphaCertificate,
    spatial_1d_alpha_certificate,
)


def _certificate_to_dict(certificate: Spatial1DAlphaCertificate) -> dict[str, object]:
    return {
        "family": "spatial_1d_alpha",
        "rule_name": certificate.rule_name,
        "candidate_count": 1,
        "unitary_candidates": int(certificate.transfer_unitary_on_samples),
        "coarse_6_4_band_candidates": int(certificate.coarse_6_4_band_split),
        "period": certificate.period,
        "alpha_winding": certificate.alpha_winding,
        "eta_winding": certificate.eta_winding,
        "winding_gcd": certificate.winding_gcd,
        "winding_lcm": certificate.winding_lcm,
        "locality_radius": certificate.locality_radius,
        "sample_count": certificate.sample_count,
        "transfer_unitary_on_samples": certificate.transfer_unitary_on_samples,
        "alpha_projector_rank": certificate.alpha_projector_rank,
        "eta_projector_rank": certificate.eta_projector_rank,
        "coarse_6_4_band_split": certificate.coarse_6_4_band_split,
        "orientation_choices_before_transport": (
            certificate.orientation_choices_before_transport
        ),
        "orientation_choices_after_transport": (
            certificate.orientation_choices_after_transport
        ),
        "orientation_orbits": [
            {
                "alpha_sign": orbit.alpha_sign,
                "eta_sign": orbit.eta_sign,
                "transport_allowed": orbit.transport_allowed,
            }
            for orbit in certificate.orientation_orbits
        ],
        "sign_coupled_candidates": int(certificate.sign_coupled_to_global_pm),
        "sign_coupled_to_global_pm": certificate.sign_coupled_to_global_pm,
        "strict_bridge_candidates": certificate.strict_bridge_candidates,
        "route_label": certificate.route_label,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the sidecar 1D spatial alpha winding prototype."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the spatial sidecar diagnostics regress.",
    )
    args = parser.parse_args()

    certificate = spatial_1d_alpha_certificate()
    payload = _certificate_to_dict(certificate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This checks a sidecar 1D spatial alpha winding prototype.")
        print("It tests sign coupling only; it is not a load-bearing QCA bridge.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"unitary_candidates: {payload['unitary_candidates']}")
        print(f"coarse_6_4_band_candidates: {payload['coarse_6_4_band_candidates']}")
        print(f"period: {certificate.period}")
        print(f"alpha_winding: {certificate.alpha_winding}")
        print(f"eta_winding: {certificate.eta_winding}")
        print(f"winding_gcd: {certificate.winding_gcd}")
        print(f"winding_lcm: {certificate.winding_lcm}")
        print(f"locality_radius: {certificate.locality_radius}")
        print(f"sample_count: {certificate.sample_count}")
        print(
            "transfer_unitary_on_samples: "
            f"{str(certificate.transfer_unitary_on_samples).lower()}"
        )
        print(f"alpha_projector_rank: {certificate.alpha_projector_rank}")
        print(f"eta_projector_rank: {certificate.eta_projector_rank}")
        print(
            "orientation_choices_before_transport: "
            f"{certificate.orientation_choices_before_transport}"
        )
        print(
            "orientation_choices_after_transport: "
            f"{certificate.orientation_choices_after_transport}"
        )
        print(
            "sign_coupled_to_global_pm: "
            f"{str(certificate.sign_coupled_to_global_pm).lower()}"
        )
        print(f"strict_bridge_candidates: {certificate.strict_bridge_candidates}")
        print(f"route_label: {certificate.route_label}")
        print(f"load_bearing_qca_bridge: {str(certificate.load_bearing_qca_bridge).lower()}")
        for orbit in certificate.orientation_orbits:
            print(
                "orientation_orbit: "
                f"alpha={orbit.alpha_sign}, "
                f"eta={orbit.eta_sign}, "
                f"allowed={str(orbit.transport_allowed).lower()}"
            )

    if args.check:
        check_passed = (
            certificate.transfer_unitary_on_samples
            and certificate.coarse_6_4_band_split
            and certificate.winding_gcd == 1
            and certificate.winding_lcm == certificate.period
            and certificate.orientation_choices_before_transport == 4
            and certificate.orientation_choices_after_transport == 2
            and certificate.sign_coupled_to_global_pm
            and certificate.strict_bridge_candidates == 0
            and certificate.route_label == "spatial_signs_coupled_to_global_pm"
            and not certificate.load_bearing_qca_bridge
        )
        if not check_passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
