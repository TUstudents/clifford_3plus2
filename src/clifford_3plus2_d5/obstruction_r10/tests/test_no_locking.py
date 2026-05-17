from __future__ import annotations

from clifford_3plus2_d5.obstruction_r10.qca.certificates import qca_update_certificate
from clifford_3plus2_d5.obstruction_r10.qca.gates import (
    block_mixing_shift_gate,
    rank_one_color_shift_gate,
    rank_one_weak_shift_gate,
)
from clifford_3plus2_d5.obstruction_r10.qca.layers import QCALayer, QCAUpdate, minimal_period_four_update
from clifford_3plus2_d5.obstruction_r10.sm.classification import classify_real_gate


def _with_extra_gate(gate_name: str, gate) -> QCAUpdate:
    base = minimal_period_four_update()
    return QCAUpdate(
        name=f"{base.name}_{gate_name}",
        layers=base.layers + (QCALayer(f"{gate_name}_layer", (gate,)),),
    )


def test_rank_one_color_shift_is_falsifying() -> None:
    gate = rank_one_color_shift_gate()
    certificate = qca_update_certificate(_with_extra_gate("rank_one_color_shift", gate))

    assert classify_real_gate(gate.matrix) == "color_breaking_fail"
    assert not certificate.all_internal_actions_safe
    assert certificate.unsafe_gate_witnesses == ("rank_one_color_shift",)
    assert certificate.finite_depth_qca_verdict == "falsified"


def test_rank_one_weak_shift_is_falsifying() -> None:
    gate = rank_one_weak_shift_gate()
    certificate = qca_update_certificate(_with_extra_gate("rank_one_weak_shift", gate))

    assert classify_real_gate(gate.matrix) == "weak_breaking_fail"
    assert not certificate.all_internal_actions_safe
    assert certificate.unsafe_gate_witnesses == ("rank_one_weak_shift",)
    assert certificate.finite_depth_qca_verdict == "falsified"


def test_block_mixing_shift_is_falsifying() -> None:
    gate = block_mixing_shift_gate()
    certificate = qca_update_certificate(_with_extra_gate("block_mixing_shift", gate))

    assert classify_real_gate(gate.matrix) == "block_mixing_fail"
    assert not certificate.all_internal_actions_safe
    assert certificate.unsafe_gate_witnesses == ("block_mixing_shift",)
    assert certificate.finite_depth_qca_verdict == "falsified"
