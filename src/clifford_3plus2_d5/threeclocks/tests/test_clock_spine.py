"""Tests for Session 01 three-clock infrastructure."""

import pytest

from clifford_3plus2_d5.threeclocks.clock_spine import (
    ClockSpec,
    ThreeClockSystem,
    clock_spine_payload,
    custom_orders_supported,
    default_three_clock_system,
    weyl_relation_holds,
)


def test_default_system_is_three_independent_z3_clocks() -> None:
    system = default_three_clock_system()

    assert system.names == ("clock_a", "clock_b", "clock_c")
    assert system.orders == (3, 3, 3)
    assert system.dimension == 27
    assert system.closure_order == 3
    assert system.independent_weyl_relations_hold()


def test_clock_words_compose_in_product_group() -> None:
    system = default_three_clock_system()
    identity = system.identity_word()
    word_a = system.word((1, 0, 0))
    word_b = system.word((0, 1, 0))
    word_ab = system.word((1, 1, 0))

    assert identity.is_identity()
    assert identity * word_a == word_a
    assert word_a * word_b == word_ab
    assert (word_ab * word_ab.inverse()).is_identity()
    assert word_a.order() == 3
    assert word_ab.order() == 3


def test_custom_clock_orders_are_supported() -> None:
    system = ThreeClockSystem(
        clocks=(
            ClockSpec("z2", 2),
            ClockSpec("z3", 3),
            ClockSpec("z5", 5),
        )
    )

    assert system.dimension == 30
    assert system.closure_order == 30
    assert system.word((1, 1, 1)).order() == 30
    assert custom_orders_supported()


def test_local_weyl_relations_hold_for_small_clock_orders() -> None:
    for order in (2, 3, 4, 5):
        assert weyl_relation_holds(order)


def test_invalid_clock_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="clock order"):
        ClockSpec("bad", 1)


def test_clock_spine_payload_passes_without_mass_claims() -> None:
    payload = clock_spine_payload()

    assert payload.final_verdict == "THREECLOCKS_INFRASTRUCTURE_PASS"
    assert payload.clock_count == 3
    assert payload.default_orders == (3, 3, 3)
    assert payload.default_dimension == 27
    assert payload.independent_weyl_relations_pass
    assert payload.word_composition_pass
    assert payload.word_inverse_pass
    assert payload.word_closure_order_pass
    assert payload.custom_orders_supported
    assert payload.no_quark_masses_claimed
