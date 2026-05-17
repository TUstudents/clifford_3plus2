from __future__ import annotations

from clifford_3plus2_d5.obstruction_r10.qca.certificates import certificate_to_dict, qca_update_certificate
from clifford_3plus2_d5.obstruction_r10.qca.layers import QCAUpdate, minimal_period_four_update


def test_default_qca_update_certificate_is_candidate_only() -> None:
    certificate = qca_update_certificate()

    assert certificate.finite_depth
    assert certificate.layer_count == 4
    assert certificate.max_locality_radius == 0
    assert certificate.all_layers_local
    assert certificate.all_layers_orthogonal
    assert certificate.quarter_period_is_j
    assert certificate.half_period_is_minus_identity
    assert certificate.full_period_is_identity
    assert certificate.period_four_check_passed
    assert certificate.all_internal_actions_safe
    assert certificate.unsafe_gate_witnesses == ()
    assert not certificate.qca_rule_forces_update
    assert certificate.finite_depth_qca_verdict == "candidate_only"
    assert not certificate.load_bearing_qca_bridge


def test_effective_hamiltonian_only_update_is_falsified() -> None:
    base = minimal_period_four_update()
    update = QCAUpdate(
        name="effective_only",
        layers=base.layers,
        effective_hamiltonian_only=True,
    )
    certificate = qca_update_certificate(update)

    assert not certificate.finite_depth
    assert certificate.finite_depth_qca_verdict == "falsified"


def test_source_backed_rule_forced_update_gets_finite_depth_candidate_verdict() -> None:
    base = minimal_period_four_update()
    update = QCAUpdate(
        name=base.name,
        layers=base.layers,
        qca_rule_forces_update=True,
        microscopic_rule_source="synthetic_test_rule",
    )
    certificate = qca_update_certificate(update)

    assert certificate.qca_rule_forces_update
    assert certificate.finite_depth_qca_verdict == "finite_depth_candidate"
    assert not certificate.load_bearing_qca_bridge


def test_rule_forced_flag_without_source_remains_candidate_only() -> None:
    base = minimal_period_four_update()
    update = QCAUpdate(name=base.name, layers=base.layers, qca_rule_forces_update=True)
    certificate = qca_update_certificate(update)

    assert certificate.finite_depth_qca_verdict == "candidate_only"


def test_qca_update_certificate_serialization_is_stable() -> None:
    payload = certificate_to_dict(qca_update_certificate())

    assert payload["finite_depth"] is True
    assert payload["layer_count"] == 4
    assert payload["max_locality_radius"] == 0
    assert payload["all_layers_orthogonal"] is True
    assert payload["period_four_check_passed"] is True
    assert payload["quarter_period_is_j"] is True
    assert payload["half_period_is_minus_identity"] is True
    assert payload["full_period_is_identity"] is True
    assert payload["all_internal_actions_safe"] is True
    assert payload["unsafe_gate_witnesses"] == []
    assert payload["finite_depth_qca_verdict"] == "candidate_only"
    assert payload["qca_update_check_passed"] is True
    assert payload["load_bearing_qca_bridge"] is False
