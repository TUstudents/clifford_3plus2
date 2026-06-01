"""V11 tests for unique allowed-successor data."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.selection_no_leakage import (
    allowed_sets_are_unique_successors,
    leaky_successor_allowed_sets,
    leaky_successor_control_rejected,
    no_leakage_from_unique_successors,
    target_successor_map,
    unique_successor_allowed_sets,
)


def test_unique_successor_allowed_sets_match_targets() -> None:
    allowed = unique_successor_allowed_sets()

    assert target_successor_map() == {"a": "u", "b": "a"}
    assert allowed == {"a": ("u",), "b": ("a",)}
    assert allowed_sets_are_unique_successors(allowed)
    assert no_leakage_from_unique_successors(allowed)


def test_extra_allowed_successor_is_leakage_control() -> None:
    allowed = leaky_successor_allowed_sets()

    assert allowed == {"a": ("u", "bulk_a"), "b": ("a",)}
    assert not allowed_sets_are_unique_successors(allowed)
    assert not no_leakage_from_unique_successors(allowed)
    assert leaky_successor_control_rejected()
