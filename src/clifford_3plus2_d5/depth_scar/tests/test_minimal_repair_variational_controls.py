"""Negative controls for V8 minimal causal repair."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.minimal_repair_variational import (
    causal_repair_cost,
    constant_cost_control_pass,
    constant_cost_minimizers,
    cycle_relaxed_control_pass,
    minimal_causal_repair_variational_pass,
    nonminimal_shortcuts_excluded_by_cost,
    rank_one_relaxed_control_pass,
    unconstrained_nonzero_nilpotent_minimizers,
)
from clifford_3plus2_d5.depth_scar.support_classification import (
    canonical_support_key,
    directed_cycle_supports,
    nonminimal_length_three_supports,
    support_key,
)


def _support_key_set(supports):
    return {support_key(support) for support in supports}


def test_nonminimal_shortcuts_are_feasible_but_excluded_by_edge_count() -> None:
    shortcuts = nonminimal_length_three_supports()

    assert len(shortcuts) == 6
    assert all(causal_repair_cost(support) == 3 for support in shortcuts)
    assert nonminimal_shortcuts_excluded_by_cost()


def test_relaxing_rank_length_all_port_constraints_selects_rank_one_repairs() -> None:
    minimizers = unconstrained_nonzero_nilpotent_minimizers()

    assert len(minimizers) == 6
    assert all(causal_repair_cost(support) == 1 for support in minimizers)
    assert all(support.rank() == 1 for support in minimizers)
    assert all(support**2 == sp.zeros(3, 3) for support in minimizers)
    assert rank_one_relaxed_control_pass()


def test_nilpotency_rejects_directed_cycle_closures() -> None:
    cycles = directed_cycle_supports()

    assert len(cycles) == 2
    assert all(causal_repair_cost(support) == 3 for support in cycles)
    assert cycle_relaxed_control_pass()


def test_constant_cost_loses_unique_path_flag_orbit() -> None:
    minimizers = constant_cost_minimizers()
    orbit_keys = {canonical_support_key(support) for support in minimizers}

    assert len(minimizers) == 12
    assert len(orbit_keys) == 2
    assert _support_key_set(nonminimal_length_three_supports()).issubset(_support_key_set(minimizers))
    assert constant_cost_control_pass()
    assert minimal_causal_repair_variational_pass()
