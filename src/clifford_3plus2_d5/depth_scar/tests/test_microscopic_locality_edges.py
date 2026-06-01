"""V9 tests for defect-height and one-tick locality edges."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    PATH_REPAIR_EDGES,
    SHORTCUT_EDGE,
    bipartite_parity,
    boundary_distance,
    defect_height,
    is_bipartite_allowed,
    is_one_tick_local,
    is_strictly_height_lowering,
    one_tick_local_lowering_edges,
    shortcut_forbidden_by_one_tick_locality,
    strictly_lowering_edges,
)


def test_defect_height_filtration_is_three_level() -> None:
    assert tuple(defect_height(port) for port in range(3)) == (0, 1, 2)


def test_boundary_geometry_is_path_not_triangle() -> None:
    assert boundary_distance(0, 1) == 1
    assert boundary_distance(1, 2) == 1
    assert boundary_distance(0, 2) == 2


def test_bcc_bipartite_parity_forbids_shortcut() -> None:
    assert tuple(bipartite_parity(port) for port in range(3)) == (0, 1, 0)
    assert all(is_bipartite_allowed(edge) for edge in PATH_REPAIR_EDGES)
    assert not is_bipartite_allowed(SHORTCUT_EDGE)


def test_one_tick_locality_removes_only_the_monotone_shortcut() -> None:
    assert all(is_strictly_height_lowering(edge) for edge in PATH_REPAIR_EDGES)
    assert is_strictly_height_lowering(SHORTCUT_EDGE)
    assert all(is_one_tick_local(edge) for edge in PATH_REPAIR_EDGES)
    assert not is_one_tick_local(SHORTCUT_EDGE)

    assert strictly_lowering_edges() == (*PATH_REPAIR_EDGES, SHORTCUT_EDGE)
    assert one_tick_local_lowering_edges() == PATH_REPAIR_EDGES
    assert shortcut_forbidden_by_one_tick_locality()
