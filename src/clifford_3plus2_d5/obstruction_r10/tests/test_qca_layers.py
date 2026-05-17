from __future__ import annotations

import pytest

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.obstruction_r10.qca.gates import (
    QCALocalGate,
    gate_certificate,
    gate_has_valid_locality,
    global_clock_tick_gate,
    is_real_orthogonal,
    support_radius,
)
from clifford_3plus2_d5.obstruction_r10.qca.layers import (
    QCALayer,
    layer_certificate,
    layer_operator,
    minimal_period_four_update,
    update_operator,
    update_prefix_operator,
)
from clifford_3plus2_d5.obstruction_r10.qca.locality import supports_are_disjoint


def test_global_clock_tick_gate_is_exact_safe_orthogonal_j() -> None:
    gate = global_clock_tick_gate()
    carrier = standard_real_carrier()
    certificate = gate_certificate(gate)

    assert gate.matrix == carrier.complex_structure
    assert gate.support == (0,)
    assert gate.locality_radius == 0
    assert certificate["is_real"] is True
    assert certificate["is_orthogonal"] is True
    assert certificate["internal_classification"] == "safe_sm_commutant"
    assert certificate["internal_action_safe"] is True


def test_support_radius_requires_nonempty_support() -> None:
    assert support_radius((2, 5)) == 3
    with pytest.raises(ValueError):
        support_radius(())


def test_gate_locality_rejects_negative_site_or_too_small_radius() -> None:
    gate = QCALocalGate("bad_radius", (0, 3), identity(10), locality_radius=1)
    negative_site = QCALocalGate("bad_site", (-1,), identity(10), locality_radius=0)

    assert not gate_has_valid_locality(gate)
    assert not gate_has_valid_locality(negative_site)


def test_layer_supports_must_be_disjoint() -> None:
    gate_a = QCALocalGate("a", (0, 1), identity(10), locality_radius=1)
    gate_b = QCALocalGate("b", (1, 2), identity(10), locality_radius=1)

    assert not supports_are_disjoint((gate_a, gate_b))


def test_layer_operator_and_certificate_for_clock_tick() -> None:
    layer = QCALayer("clock_tick", (global_clock_tick_gate(),))
    certificate = layer_certificate(layer)

    assert layer_operator(layer) == standard_real_carrier().complex_structure
    assert certificate["supports_are_disjoint"] is True
    assert certificate["locality_check_passed"] is True
    assert certificate["is_real_orthogonal"] is True


def test_minimal_period_four_update_products_are_exact() -> None:
    update = minimal_period_four_update()
    carrier = standard_real_carrier()

    assert len(update.layers) == 4
    assert update_prefix_operator(update, 1) == carrier.complex_structure
    assert update_prefix_operator(update, 2) == -identity(10)
    assert update_operator(update) == identity(10)
    assert is_real_orthogonal(update_operator(update))
