"""Tests for Session 15 quark source assembly audit."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.quark_source_assembly import (
    ACTIVE_COLOR_RETURN_PREMISE,
    BOTTOM_RANK_FIVE_PREMISE,
    HEIGHT_DYNAMICS_PREMISE,
    QUARK_NORMAL_DEPTH_PREMISE,
    missing_source_fields,
    quark_source_assembly_payload,
    quark_source_requirements,
    unresolved_premises,
    unresolved_quark_sources,
)
from clifford_3plus2_d5.universal_bath.source_dictionary import SourceStatus


def test_unresolved_quark_sources_have_missing_vector_components_and_depths() -> None:
    sources = unresolved_quark_sources()
    up = sources["up_quark_boundary_source"]
    down = sources["down_quark_boundary_source"]

    assert up.status == SourceStatus.UNRESOLVED
    assert down.status == SourceStatus.UNRESOLVED
    assert missing_source_fields(up) == ("port_vector", "residual_components", "normal_depth")
    assert missing_source_fields(down) == ("port_vector", "residual_components", "normal_depth")


def test_quark_source_requirements_keep_only_real_open_premises_unresolved() -> None:
    requirements = quark_source_requirements()
    derived = {requirement.label: requirement.derived for requirement in requirements}

    assert derived["write_once_quark_charge_anchors"]
    assert derived["common_residual_family_incidence"]
    assert not derived[HEIGHT_DYNAMICS_PREMISE]
    assert not derived[ACTIVE_COLOR_RETURN_PREMISE]
    assert not derived[BOTTOM_RANK_FIVE_PREMISE]
    assert not derived[QUARK_NORMAL_DEPTH_PREMISE]
    assert unresolved_premises(requirements) == (
        HEIGHT_DYNAMICS_PREMISE,
        ACTIVE_COLOR_RETURN_PREMISE,
        BOTTOM_RANK_FIVE_PREMISE,
        QUARK_NORMAL_DEPTH_PREMISE,
    )


def test_quark_source_assembly_collects_conditional_heads_without_freezing_sources() -> None:
    payload = quark_source_assembly_payload()

    assert payload.final_verdict == "QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT"
    assert payload.source_dictionary_pass
    assert payload.up_source_unresolved
    assert payload.down_source_unresolved
    assert payload.common_family_incidence_pass
    assert payload.height_door_pass
    assert payload.color_lift_pass
    assert payload.up_head_pass
    assert payload.down_head_pass
    assert payload.conditional_heads_assembled
    assert payload.up_conditional_profile == (sp.Rational(1, 4), 1 / sp.sqrt(2), sp.Integer(1))
    assert payload.down_spectator_baseline_profile == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(2, 3)),
    )
    assert payload.down_active_baseline_profile == payload.down_spectator_baseline_profile
    assert payload.down_active_candidate_profile == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(5, 6)),
    )
    assert not payload.height_mode_selected_by_dynamics
    assert not payload.active_color_return_selected
    assert not payload.bottom_rank_five_decided
    assert not payload.quark_normal_depths_frozen
    assert not payload.source_freeze_ready
    assert payload.quark_sources_remain_unfrozen
