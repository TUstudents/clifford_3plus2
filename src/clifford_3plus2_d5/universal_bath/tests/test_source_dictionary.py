"""Tests for the Session 02 source dictionary."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceStatus,
    bb_first_hop_survival_operator,
    first_hop_survival_weight,
    frozen_source_anchors,
    source_dictionary_payload,
    unresolved_source_anchors,
    vector_norm_squared,
)


def _anchor_by_label(label: str):
    anchors = (*frozen_source_anchors(), *unresolved_source_anchors())
    return next(anchor for anchor in anchors if anchor.label == label)


def test_bb_first_hop_survival_operator_is_universal_half_identity() -> None:
    assert bb_first_hop_survival_operator() == sp.eye(2) / 2


def test_frozen_source_vectors_are_normalized() -> None:
    for anchor in frozen_source_anchors():
        assert anchor.port_vector is not None
        assert vector_norm_squared(anchor.port_vector) == 1
        assert first_hop_survival_weight(anchor) == sp.Rational(1, 2)
        assert not anchor.uses_flavor_data


def test_neutrino_source_dictionary_freezes_u_and_b_channels() -> None:
    collective = _anchor_by_label("neutrino_collective_u")
    edge = _anchor_by_label("neutrino_edge_b")

    assert collective.status == SourceStatus.FROZEN
    assert collective.reduction == ReductionKind.POSITIVE_JACOBI
    assert collective.normal_depth == 1
    assert collective.residual_components == {"a": 0, "u": 1, "b": 0}

    assert edge.status == SourceStatus.FROZEN
    assert edge.reduction == ReductionKind.POSITIVE_JACOBI
    assert edge.normal_depth == 0
    assert edge.residual_components == {"a": 0, "u": 0, "b": 1}


def test_charged_lepton_source_dictionary_freezes_selected_e1_port() -> None:
    charged = _anchor_by_label("charged_lepton_active_e1")

    assert charged.status == SourceStatus.FROZEN
    assert charged.reduction == ReductionKind.CMV_OPUC
    assert charged.normal_depth == 2
    assert sp.simplify(charged.residual_components["a"] - sp.sqrt(sp.Rational(2, 3))) == 0
    assert sp.simplify(charged.residual_components["u"] - 1 / sp.sqrt(3)) == 0
    assert charged.residual_components["b"] == 0


def test_quark_sources_are_recorded_but_not_frozen() -> None:
    unresolved = unresolved_source_anchors()
    labels = {anchor.label for anchor in unresolved}

    assert labels == {"up_quark_boundary_source", "down_quark_boundary_source"}
    assert all(anchor.status == SourceStatus.UNRESOLVED for anchor in unresolved)
    assert all(anchor.port_vector is None for anchor in unresolved)
    assert all(anchor.normal_depth is None for anchor in unresolved)
    assert all(not anchor.uses_flavor_data for anchor in unresolved)


def test_source_dictionary_core_pass_reports_partial_freeze_honestly() -> None:
    payload = source_dictionary_payload()

    assert payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
    assert payload.survival_operator_universal
    assert payload.all_frozen_sources_survive
    assert payload.no_flavor_data_used
    assert not payload.all_physical_sources_frozen
    assert set(payload.frozen_survival_weights.values()) == {sp.Rational(1, 2)}
    assert {anchor.sector for anchor in payload.unresolved_sources} == {"up_quark", "down_quark"}
