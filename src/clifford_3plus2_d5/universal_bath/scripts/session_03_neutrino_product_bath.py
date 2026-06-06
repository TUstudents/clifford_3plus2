"""Run the Session 03 neutrino product-bath certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.neutrino_product import (
    cross_moments,
    diagonal_moment_differences,
    neutrino_product_bath_payload,
)


def main() -> None:
    """Print Session 03 payload."""

    payload = neutrino_product_bath_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("frozen neutrino sources =", payload.frozen_neutrino_sources)
    print("checked moment powers =", payload.checked_moment_powers)
    print("cross moments =", cross_moments(6, payload.checked_moment_powers))
    print("diagonal moment differences =", diagonal_moment_differences(6, payload.checked_moment_powers))
    print("diagonal moments equal =", payload.diagonal_moments_equal)
    print("cross moments zero =", payload.cross_moments_zero)
    print("tail value =", payload.tail_value)
    print("tail fixed-point residual =", payload.fixed_point_residual)
    print("tail value matches epsilon =", payload.tail_value_matches_epsilon)
    print("response diagonal (a,u,b) =")
    print(payload.response_diagonal)
    print("response matches target =", payload.response_matches_target)
    print("mass ratio m2/m3 =", payload.mass_ratio)
    print("mass-squared ratio =", payload.mass_squared_ratio)
    print("rank-one control has cross return =", payload.rank_one_control_has_cross_return)
    print("wrong-source control rejected =", payload.wrong_source_control_rejected)
    print("alternate-tail control rejected =", payload.alternate_tail_control_rejected)
    print("PMNS/CKM parked =", payload.pmns_ckm_parked)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
