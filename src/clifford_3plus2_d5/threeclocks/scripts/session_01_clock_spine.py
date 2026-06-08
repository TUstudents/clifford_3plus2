"""Run the Session 01 three-clock infrastructure certificate."""

from __future__ import annotations

from clifford_3plus2_d5.threeclocks.clock_spine import clock_spine_payload


def main() -> None:
    """Print Session 01 payload."""

    payload = clock_spine_payload()
    print("clock count =", payload.clock_count)
    print("default orders =", payload.default_orders)
    print("default dimension =", payload.default_dimension)
    print("default closure order =", payload.default_closure_order)
    print("independent Weyl relations pass =", payload.independent_weyl_relations_pass)
    print("word composition pass =", payload.word_composition_pass)
    print("word inverse pass =", payload.word_inverse_pass)
    print("word closure order pass =", payload.word_closure_order_pass)
    print("custom orders supported =", payload.custom_orders_supported)
    print("no quark masses claimed =", payload.no_quark_masses_claimed)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
