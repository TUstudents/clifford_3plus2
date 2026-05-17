"""Certificates for finite-depth QCA update candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.obstruction_r10.qca.gates import gate_certificate
from clifford_3plus2_d5.obstruction_r10.qca.layers import (
    QCAUpdate,
    layer_certificate,
    minimal_period_four_update,
    update_operator,
    update_prefix_operator,
)
from clifford_3plus2_d5.obstruction_r10.qca.locality import finite_depth_check_passed
from clifford_3plus2_d5.obstruction_r10.search.forced_j import certificate_to_dict as j_certificate_to_dict
from clifford_3plus2_d5.obstruction_r10.search.forced_j import j_certificate


FiniteDepthQCAVerdict = Literal["finite_depth_candidate", "candidate_only", "falsified"]


@dataclass(frozen=True)
class QCAUpdateCertificate:
    update_name: str
    finite_depth: bool
    layer_count: int
    max_locality_radius: int
    all_layers_local: bool
    all_layers_orthogonal: bool
    quarter_period_operator: sp.Matrix
    half_period_operator: sp.Matrix
    full_period_operator: sp.Matrix
    quarter_period_is_j: bool
    half_period_is_minus_identity: bool
    full_period_is_identity: bool
    period_four_check_passed: bool
    all_internal_actions_safe: bool
    unsafe_gate_witnesses: tuple[str, ...]
    j_certificate: dict[str, object]
    layer_certificates: tuple[dict[str, object], ...]
    gate_certificates: tuple[dict[str, object], ...]
    qca_rule_forces_update: bool
    microscopic_rule_source: str
    finite_depth_qca_verdict: FiniteDepthQCAVerdict
    load_bearing_qca_bridge: bool


def _matrix_to_json(matrix: sp.Matrix) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix.tolist()]


def qca_update_certificate(update: QCAUpdate | None = None) -> QCAUpdateCertificate:
    update = update or minimal_period_four_update()
    carrier = standard_real_carrier()
    layers = tuple(layer_certificate(layer) for layer in update.layers)
    gates = tuple(gate for layer in update.layers for gate in layer.gates)
    gate_certificates = tuple(gate_certificate(gate) for gate in gates)
    layer_count = len(update.layers)
    finite_depth = finite_depth_check_passed(
        layer_count,
        effective_hamiltonian_only=update.effective_hamiltonian_only,
    )
    all_layers_local = all(bool(layer["locality_check_passed"]) for layer in layers)
    all_layers_orthogonal = all(bool(layer["is_real_orthogonal"]) for layer in layers)
    max_radius = max((int(layer["locality_radius"]) for layer in layers), default=0)
    quarter_period = update_prefix_operator(update, 1)
    half_period = update_prefix_operator(update, 2)
    full_period = update_operator(update)
    quarter_is_j = quarter_period == carrier.complex_structure
    half_is_minus_identity = half_period == -identity(10)
    full_is_identity = full_period == identity(10)
    period_four = quarter_is_j and half_is_minus_identity and full_is_identity
    all_internal_actions_safe = all(
        certificate["internal_action_safe"] is True for certificate in gate_certificates
    )
    unsafe_witnesses = tuple(
        str(certificate["name"])
        for certificate in gate_certificates
        if certificate["internal_action_safe"] is not True
    )
    falsified = (
        not finite_depth
        or not all_layers_local
        or not all_layers_orthogonal
        or not period_four
        or not all_internal_actions_safe
    )
    source_backed = update.microscopic_rule_source != "candidate_only"
    if falsified:
        verdict: FiniteDepthQCAVerdict = "falsified"
    elif update.qca_rule_forces_update and source_backed:
        verdict = "finite_depth_candidate"
    else:
        verdict = "candidate_only"

    return QCAUpdateCertificate(
        update_name=update.name,
        finite_depth=finite_depth,
        layer_count=layer_count,
        max_locality_radius=max_radius,
        all_layers_local=all_layers_local,
        all_layers_orthogonal=all_layers_orthogonal,
        quarter_period_operator=quarter_period,
        half_period_operator=half_period,
        full_period_operator=full_period,
        quarter_period_is_j=quarter_is_j,
        half_period_is_minus_identity=half_is_minus_identity,
        full_period_is_identity=full_is_identity,
        period_four_check_passed=period_four,
        all_internal_actions_safe=all_internal_actions_safe,
        unsafe_gate_witnesses=unsafe_witnesses,
        j_certificate=j_certificate_to_dict(j_certificate()),
        layer_certificates=layers,
        gate_certificates=gate_certificates,
        qca_rule_forces_update=update.qca_rule_forces_update,
        microscopic_rule_source=update.microscopic_rule_source,
        finite_depth_qca_verdict=verdict,
        load_bearing_qca_bridge=False,
    )


def certificate_to_dict(certificate: QCAUpdateCertificate) -> dict[str, object]:
    return {
        "update_name": certificate.update_name,
        "finite_depth": certificate.finite_depth,
        "layer_count": certificate.layer_count,
        "max_locality_radius": certificate.max_locality_radius,
        "all_layers_local": certificate.all_layers_local,
        "all_layers_orthogonal": certificate.all_layers_orthogonal,
        "quarter_period_operator": _matrix_to_json(certificate.quarter_period_operator),
        "half_period_operator": _matrix_to_json(certificate.half_period_operator),
        "full_period_operator": _matrix_to_json(certificate.full_period_operator),
        "quarter_period_is_j": certificate.quarter_period_is_j,
        "half_period_is_minus_identity": certificate.half_period_is_minus_identity,
        "full_period_is_identity": certificate.full_period_is_identity,
        "period_four_check_passed": certificate.period_four_check_passed,
        "all_internal_actions_safe": certificate.all_internal_actions_safe,
        "unsafe_gate_witnesses": list(certificate.unsafe_gate_witnesses),
        "j_certificate": certificate.j_certificate,
        "layer_certificates": list(certificate.layer_certificates),
        "gate_certificates": list(certificate.gate_certificates),
        "qca_rule_forces_update": certificate.qca_rule_forces_update,
        "microscopic_rule_source": certificate.microscopic_rule_source,
        "finite_depth_qca_verdict": certificate.finite_depth_qca_verdict,
        "qca_update_check_passed": (
            certificate.finite_depth
            and certificate.all_layers_local
            and certificate.all_layers_orthogonal
            and certificate.period_four_check_passed
            and certificate.all_internal_actions_safe
        ),
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }
