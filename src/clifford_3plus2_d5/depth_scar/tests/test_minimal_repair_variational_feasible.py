"""Feasible-set tests for V8 minimal causal repair."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.minimal_repair_variational import (
    active_ports,
    all_ports_participate,
    causal_repair_cost,
    variational_feasible_supports,
)
from clifford_3plus2_d5.depth_scar.support_classification import (
    accepted_minimal_nilpotent_supports,
    is_length_three_nilpotent,
    nonminimal_length_three_supports,
    support_key,
)


def _support_key_set(supports):
    return {support_key(support) for support in supports}


def test_variational_feasible_supports_are_length_three_rank_two_all_port_repairs() -> None:
    supports = variational_feasible_supports()

    assert len(supports) == 12
    assert all(is_length_three_nilpotent(support) for support in supports)
    assert all(support.rank() == 2 for support in supports)
    assert all(active_ports(support) == (0, 1, 2) for support in supports)
    assert all(all_ports_participate(support) for support in supports)


def test_feasible_set_contains_minimal_flags_and_shortcuts() -> None:
    feasible_keys = _support_key_set(variational_feasible_supports())

    assert _support_key_set(accepted_minimal_nilpotent_supports()).issubset(feasible_keys)
    assert _support_key_set(nonminimal_length_three_supports()).issubset(feasible_keys)
    assert sorted({causal_repair_cost(support) for support in variational_feasible_supports()}) == [
        2,
        3,
    ]
