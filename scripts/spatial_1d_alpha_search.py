from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.qca.spatial_1d import (
    Spatial1DAlphaCertificate,
    Spatial1DCombinedRouteCertificate,
    Spatial1DLocalHoppingCertificate,
    Spatial1DLocalQCACertificate,
    spatial_1d_alpha_certificate,
    spatial_1d_combined_route_certificate,
    spatial_1d_local_hopping_certificate,
    spatial_1d_local_qca_certificate,
)


def _certificate_to_dict(
    certificate: Spatial1DAlphaCertificate,
    local_hopping: Spatial1DLocalHoppingCertificate,
    local_qca: Spatial1DLocalQCACertificate,
) -> dict[str, object]:
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
        "local_hopping": _local_hopping_to_dict(local_hopping),
        "local_qca": _local_qca_to_dict(local_qca),
    }


def _local_hopping_to_dict(
    certificate: Spatial1DLocalHoppingCertificate,
) -> dict[str, object]:
    return {
        "hopping_term_count": certificate.hopping_term_count,
        "hopping_shifts": list(certificate.hopping_shifts),
        "hopping_locality_radius": certificate.hopping_locality_radius,
        "mode_windings": list(certificate.mode_windings),
        "computed_alpha_winding": certificate.computed_alpha_winding,
        "computed_eta_winding": certificate.computed_eta_winding,
        "computed_winding_gcd": certificate.computed_winding_gcd,
        "computed_winding_lcm": certificate.computed_winding_lcm,
        "reconstructs_transfer_on_samples": certificate.reconstructs_transfer_on_samples,
        "transfer_unitary_on_samples": certificate.transfer_unitary_on_samples,
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
        "sign_coupled_to_global_pm": certificate.sign_coupled_to_global_pm,
        "route_label": certificate.route_label,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def _local_qca_to_dict(
    certificate: Spatial1DLocalQCACertificate,
) -> dict[str, object]:
    return {
        "layer_name": certificate.layer_name,
        "period": certificate.period,
        "dimension": certificate.dimension,
        "qca_term_count": certificate.qca_term_count,
        "qca_shifts": list(certificate.qca_shifts),
        "qca_locality_radius": certificate.qca_locality_radius,
        "finite_radius": certificate.finite_radius,
        "coefficient_matrices_real": certificate.coefficient_matrices_real,
        "laurent_orthogonal": certificate.laurent_orthogonal,
        "symbol_reconstructs_transfer_on_samples": (
            certificate.symbol_reconstructs_transfer_on_samples
        ),
        "symbol_unitary_on_samples": certificate.symbol_unitary_on_samples,
        "coefficient_algebra_dimension": certificate.coefficient_algebra_dimension,
        "coefficient_center_dimension": certificate.coefficient_center_dimension,
        "central_idempotent_ranks": list(certificate.central_idempotent_ranks),
        "lower_rank_central_idempotents": certificate.lower_rank_central_idempotents,
        "coarse_6_4_center_pair": certificate.coarse_6_4_center_pair,
        "mode_windings": list(certificate.mode_windings),
        "computed_alpha_winding": certificate.computed_alpha_winding,
        "computed_eta_winding": certificate.computed_eta_winding,
        "computed_winding_gcd": certificate.computed_winding_gcd,
        "computed_winding_lcm": certificate.computed_winding_lcm,
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
        "sign_coupled_to_global_pm": certificate.sign_coupled_to_global_pm,
        "strict_bridge_candidates": certificate.strict_bridge_candidates,
        "route_label": certificate.route_label,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def _combined_to_dict(
    certificate: Spatial1DCombinedRouteCertificate,
) -> dict[str, object]:
    return {
        "family": "spatial_1d_route1_route2_combined",
        "rule_name": certificate.rule_name,
        "onsite_candidate_name": certificate.onsite_candidate_name,
        "layer_name": certificate.layer_name,
        "period": certificate.period,
        "qca_term_count": certificate.qca_term_count,
        "qca_shifts": list(certificate.qca_shifts),
        "qca_locality_radius": certificate.qca_locality_radius,
        "finite_radius": certificate.finite_radius,
        "coefficient_matrices_real": certificate.coefficient_matrices_real,
        "laurent_orthogonal": certificate.laurent_orthogonal,
        "symbol_unitary_on_samples": certificate.symbol_unitary_on_samples,
        "onsite_generated_algebra_dimension": (
            certificate.onsite_generated_algebra_dimension
        ),
        "onsite_center_dimension": certificate.onsite_center_dimension,
        "onsite_central_idempotent_ranks": list(
            certificate.onsite_central_idempotent_ranks
        ),
        "onsite_lower_rank_central_idempotents": (
            certificate.onsite_lower_rank_central_idempotents
        ),
        "onsite_compatible_j_count": certificate.onsite_compatible_j_count,
        "transported_compatible_j_count": certificate.transported_compatible_j_count,
        "transported_j_commute_on_samples": certificate.transported_j_commute_on_samples,
        "sign_coupled_to_global_pm": certificate.sign_coupled_to_global_pm,
        "coefficient_algebra_dimension": certificate.coefficient_algebra_dimension,
        "coefficient_algebra_generates_alpha_eta_projectors": (
            certificate.coefficient_algebra_generates_alpha_eta_projectors
        ),
        "joint_rule_algebra_dimension": certificate.joint_rule_algebra_dimension,
        "joint_rule_generated_transported_j_count": (
            certificate.joint_rule_generated_transported_j_count
        ),
        "topological_pm_shape_candidate": certificate.topological_pm_shape_candidate,
        "strict_bridge_candidate": certificate.strict_bridge_candidate,
        "route_label": certificate.route_label,
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the sidecar 1D spatial alpha winding prototype."
    )
    parser.add_argument(
        "--variant",
        choices=("alpha", "combined"),
        default="alpha",
        help="Spatial diagnostic variant to run. Defaults to the alpha sidecar.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if the spatial sidecar diagnostics regress.",
    )
    args = parser.parse_args()

    if args.variant == "combined":
        certificate = spatial_1d_combined_route_certificate()
        payload = _combined_to_dict(certificate)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("This checks the combined Route-1 plus Route-2 spatial sidecar.")
            print("It tests topological sign coupling and strict rule-generated J separately.")
            print(f"rule_name: {certificate.rule_name}")
            print(f"onsite_candidate_name: {certificate.onsite_candidate_name}")
            print(f"layer_name: {certificate.layer_name}")
            print(f"qca_shifts: {list(certificate.qca_shifts)}")
            print(f"finite_radius: {str(certificate.finite_radius).lower()}")
            print(f"laurent_orthogonal: {str(certificate.laurent_orthogonal).lower()}")
            print(
                "symbol_unitary_on_samples: "
                f"{str(certificate.symbol_unitary_on_samples).lower()}"
            )
            print(
                "onsite_central_idempotent_ranks: "
                f"{list(certificate.onsite_central_idempotent_ranks)}"
            )
            print(f"onsite_compatible_j_count: {certificate.onsite_compatible_j_count}")
            print(
                "transported_compatible_j_count: "
                f"{certificate.transported_compatible_j_count}"
            )
            print(
                "transported_j_commute_on_samples: "
                f"{str(certificate.transported_j_commute_on_samples).lower()}"
            )
            print(
                "coefficient_algebra_generates_alpha_eta_projectors: "
                f"{str(certificate.coefficient_algebra_generates_alpha_eta_projectors).lower()}"
            )
            print(f"joint_rule_algebra_dimension: {certificate.joint_rule_algebra_dimension}")
            print(
                "joint_rule_generated_transported_j_count: "
                f"{certificate.joint_rule_generated_transported_j_count}"
            )
            print(
                "topological_pm_shape_candidate: "
                f"{str(certificate.topological_pm_shape_candidate).lower()}"
            )
            print(f"strict_bridge_candidate: {str(certificate.strict_bridge_candidate).lower()}")
            print(f"route_label: {certificate.route_label}")
            print(
                "load_bearing_qca_bridge: "
                f"{str(certificate.load_bearing_qca_bridge).lower()}"
            )
        if args.check:
            check_passed = (
                certificate.finite_radius
                and certificate.coefficient_matrices_real
                and certificate.laurent_orthogonal
                and certificate.symbol_unitary_on_samples
                and certificate.onsite_central_idempotent_ranks == (0, 4, 6, 10)
                and certificate.onsite_lower_rank_central_idempotents == 0
                and certificate.onsite_compatible_j_count == 4
                and certificate.transported_compatible_j_count == 2
                and certificate.transported_j_commute_on_samples
                and certificate.sign_coupled_to_global_pm
                and certificate.coefficient_algebra_generates_alpha_eta_projectors
                and certificate.joint_rule_generated_transported_j_count == 0
                and certificate.topological_pm_shape_candidate
                and not certificate.strict_bridge_candidate
                and certificate.route_label
                == "combined_route_signs_coupled_but_j_not_rule_generated"
                and not certificate.load_bearing_qca_bridge
            )
            if not check_passed:
                return 1
        return 0

    certificate = spatial_1d_alpha_certificate()
    local_hopping = spatial_1d_local_hopping_certificate()
    local_qca = spatial_1d_local_qca_certificate()
    payload = _certificate_to_dict(certificate, local_hopping, local_qca)

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
        print(f"local_hopping_term_count: {local_hopping.hopping_term_count}")
        print(f"local_hopping_shifts: {list(local_hopping.hopping_shifts)}")
        print(f"local_hopping_mode_windings: {list(local_hopping.mode_windings)}")
        print(
            "local_hopping_reconstructs_transfer_on_samples: "
            f"{str(local_hopping.reconstructs_transfer_on_samples).lower()}"
        )
        print(
            "local_hopping_orientation_choices_after_transport: "
            f"{local_hopping.orientation_choices_after_transport}"
        )
        print(f"local_hopping_route_label: {local_hopping.route_label}")
        print(f"local_qca_layer_name: {local_qca.layer_name}")
        print(f"local_qca_term_count: {local_qca.qca_term_count}")
        print(f"local_qca_shifts: {list(local_qca.qca_shifts)}")
        print(f"local_qca_locality_radius: {local_qca.qca_locality_radius}")
        print(f"local_qca_finite_radius: {str(local_qca.finite_radius).lower()}")
        print(
            "local_qca_laurent_orthogonal: "
            f"{str(local_qca.laurent_orthogonal).lower()}"
        )
        print(
            "local_qca_symbol_reconstructs_transfer_on_samples: "
            f"{str(local_qca.symbol_reconstructs_transfer_on_samples).lower()}"
        )
        print(
            "local_qca_symbol_unitary_on_samples: "
            f"{str(local_qca.symbol_unitary_on_samples).lower()}"
        )
        print(
            "local_qca_coefficient_algebra_dimension: "
            f"{local_qca.coefficient_algebra_dimension}"
        )
        print(
            "local_qca_coefficient_center_dimension: "
            f"{local_qca.coefficient_center_dimension}"
        )
        print(
            "local_qca_central_idempotent_ranks: "
            f"{list(local_qca.central_idempotent_ranks)}"
        )
        print(
            "local_qca_lower_rank_central_idempotents: "
            f"{local_qca.lower_rank_central_idempotents}"
        )
        print(
            "local_qca_orientation_choices_after_transport: "
            f"{local_qca.orientation_choices_after_transport}"
        )
        print(f"local_qca_route_label: {local_qca.route_label}")
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
            and local_hopping.hopping_shifts == (3, 4)
            and local_hopping.mode_windings == (4, 4, 4, 3, 3)
            and local_hopping.computed_alpha_winding == 4
            and local_hopping.computed_eta_winding == 3
            and local_hopping.computed_winding_gcd == 1
            and local_hopping.computed_winding_lcm == certificate.period
            and local_hopping.reconstructs_transfer_on_samples
            and local_hopping.orientation_choices_after_transport == 2
            and local_hopping.sign_coupled_to_global_pm
            and local_hopping.route_label == "spatial_local_hopping_signs_coupled"
            and local_qca.qca_shifts == (3, 4)
            and local_qca.finite_radius
            and local_qca.coefficient_matrices_real
            and local_qca.laurent_orthogonal
            and local_qca.symbol_reconstructs_transfer_on_samples
            and local_qca.symbol_unitary_on_samples
            and local_qca.coefficient_algebra_dimension == 2
            and local_qca.coefficient_center_dimension == 2
            and local_qca.central_idempotent_ranks == (0, 4, 6, 10)
            and local_qca.lower_rank_central_idempotents == 0
            and local_qca.coarse_6_4_center_pair
            and local_qca.mode_windings == (4, 4, 4, 3, 3)
            and local_qca.orientation_choices_after_transport == 2
            and local_qca.sign_coupled_to_global_pm
            and local_qca.strict_bridge_candidates == 0
            and local_qca.route_label == "spatial_local_qca_signs_coupled_not_load_bearing"
            and not certificate.load_bearing_qca_bridge
            and not local_qca.load_bearing_qca_bridge
        )
        if not check_passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
