"""Controls proving V7's minimality assumption is load-bearing."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import k3_laplacian
from clifford_3plus2_d5.depth_scar.support_classification import (
    broad_length_three_orbit_keys,
    directed_cycle_controls_rejected,
    directed_cycle_supports,
    flag_laplacian_from_nilpotent,
    is_nilpotent,
    minimality_assumption_is_load_bearing,
    nonminimal_length_three_supports,
    rank_one_nilpotent_supports,
    support_classification_controls_pass,
    support_edge_count,
)


def test_rank_one_nilpotents_exist_but_fail_length_three() -> None:
    supports = rank_one_nilpotent_supports()

    assert len(supports) == 12
    assert all(support**2 == sp.zeros(3, 3) for support in supports)
    assert all(support_edge_count(support) > 0 for support in supports)


def test_directed_cycles_fail_nilpotency_and_return_k3_sector() -> None:
    cycles = directed_cycle_supports()

    assert len(cycles) == 2
    assert all(not is_nilpotent(cycle) for cycle in cycles)
    assert all(cycle**3 == sp.eye(3) for cycle in cycles)
    assert all(flag_laplacian_from_nilpotent(cycle) == k3_laplacian() for cycle in cycles)
    assert directed_cycle_controls_rejected()


def test_nonminimal_shortcut_supports_make_minimality_load_bearing() -> None:
    shortcuts = nonminimal_length_three_supports()

    assert len(shortcuts) == 6
    assert all(support_edge_count(support) == 3 for support in shortcuts)
    assert len(broad_length_three_orbit_keys()) == 2
    assert minimality_assumption_is_load_bearing()
    assert support_classification_controls_pass()
