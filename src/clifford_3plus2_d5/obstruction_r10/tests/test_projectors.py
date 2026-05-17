from __future__ import annotations

import pytest

from clifford_3plus2_d5.algebra.matrices import identity, zero
from clifford_3plus2_d5.algebra.projectors import (
    color_axis_projectors,
    mode_axis_projector,
    projector_pair_check_passed,
    projector_pair_identities,
    weak_axis_projectors,
)
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier


def test_standard_projector_pair_identities_are_exact() -> None:
    identities = projector_pair_identities()

    assert projector_pair_check_passed(identities)
    assert identities.dimension == 10
    assert identities.projector_3_idempotent
    assert identities.projector_2_idempotent
    assert identities.projector_sum_identity
    assert identities.projectors_orthogonal
    assert identities.projector_3_rank == 6
    assert identities.projector_2_rank == 4
    assert identities.projector_3_commutes_with_j
    assert identities.projector_2_commutes_with_j


def test_projector_pair_check_rejects_wrong_rank_candidate() -> None:
    carrier = standard_real_carrier()
    wrong_p3 = identity(10)
    wrong_p2 = carrier.projector_2
    identities = projector_pair_identities(wrong_p3, wrong_p2)

    assert not projector_pair_check_passed(identities)
    assert identities.projector_3_rank == 10
    assert not identities.projector_sum_identity


def test_color_axis_projectors_are_rank_two_real_subprojectors_of_p3() -> None:
    carrier = standard_real_carrier()
    projectors = color_axis_projectors()

    assert len(projectors) == 3
    for projector in projectors:
        assert projector * projector == projector
        assert projector.rank() == 2
        assert carrier.projector_3 * projector == projector
        assert carrier.projector_2 * projector == zero(10)


def test_weak_axis_projectors_are_rank_two_real_subprojectors_of_p2() -> None:
    carrier = standard_real_carrier()
    projectors = weak_axis_projectors()

    assert len(projectors) == 2
    for projector in projectors:
        assert projector * projector == projector
        assert projector.rank() == 2
        assert carrier.projector_2 * projector == projector
        assert carrier.projector_3 * projector == zero(10)


def test_mode_axis_projector_rejects_invalid_index() -> None:
    with pytest.raises(ValueError):
        mode_axis_projector(5)
