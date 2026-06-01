"""V9 tests for support minimality from one-tick locality."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    local_rank_complete_supports,
    local_support_induces_path_laplacian,
    local_support_operator,
    microscopic_support_minimality_pass,
    relaxed_monotone_rank_complete_supports,
    shortcut_admitted_when_locality_is_relaxed,
    shortcut_support_operator,
)
from clifford_3plus2_d5.depth_scar.nilpotent_flag import nilpotent_flag_operator
from clifford_3plus2_d5.depth_scar.support_classification import support_key


def test_local_rank_complete_support_is_unique_path_flag() -> None:
    supports = local_rank_complete_supports()

    assert len(supports) == 1
    assert support_key(supports[0]) == support_key(nilpotent_flag_operator())
    assert support_key(local_support_operator()) == support_key(nilpotent_flag_operator())
    assert microscopic_support_minimality_pass()


def test_relaxing_one_tick_locality_admits_shortcut_support() -> None:
    supports = relaxed_monotone_rank_complete_supports()
    keys = {support_key(support) for support in supports}

    assert len(supports) == 2
    assert support_key(local_support_operator()) in keys
    assert support_key(shortcut_support_operator()) in keys
    assert shortcut_admitted_when_locality_is_relaxed()


def test_local_support_induces_path_laplacian_and_depth_spectrum() -> None:
    assert local_support_induces_path_laplacian()
