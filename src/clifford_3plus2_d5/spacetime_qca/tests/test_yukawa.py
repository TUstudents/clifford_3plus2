"""Session 23 Higgs/Yukawa representation-audit tests."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.sm_hypercharge import (
    EXPECTED_JOINT_Y_T3L_TABLE,
    hypercharge_observable,
    joint_y_t3l_table,
    normalized_t3_l_observable,
    physical_hypercharge_generator,
)
from clifford_3plus2_d5.spacetime_qca import (
    beta_is_off_diagonal_between_chiralities,
    color_singlet_charge_shift_basis,
    conjugate_charge_shift_component,
    gauge_breaking_summary,
    has_charge_shift,
    higgs_like_charge_shift_candidate,
    higgs_like_charge_shift_pair,
    higgs_like_yukawa_audit,
    is_higgs_like_charge_shift,
    left_right_projectors,
    projector_control_mass,
    projector_control_yukawa_audit,
    same_matrix,
    scalar_internal_mass,
    universal_scalar_yukawa_audit,
    yukawa_representation_audit_payload,
    yukawa_spacetime_coupler,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import su2_l_generators_from_spin04
from clifford_3plus2_d5.lepton.patisalam_sm import su3_c_generators_from_su4


def test_beta_is_off_diagonal_between_spacetime_chiralities() -> None:
    p_right, p_left = left_right_projectors()
    assert same_matrix(p_right + p_left, sp.eye(4))
    assert beta_is_off_diagonal_between_chiralities()
    assert yukawa_spacetime_coupler().shape == (4, 4)


def test_universal_scalar_is_gauge_preserving_but_not_higgs_like() -> None:
    audit = universal_scalar_yukawa_audit()
    assert audit.preserves_color
    assert audit.preserves_su2_l
    assert audit.preserves_hypercharge
    assert audit.commutes_with_sm
    assert not audit.higgs_like_charge_shift
    assert "not Higgs-like" in audit.interpretation


def test_projector_control_breaks_at_least_one_sm_sector() -> None:
    audit = projector_control_yukawa_audit()
    assert not audit.commutes_with_sm
    assert not audit.higgs_like_charge_shift

    summary = gauge_breaking_summary(
        projector_control_mass(32, 16),
        su3=su3_c_generators_from_su4(),
        su2_l=su2_l_generators_from_spin04(),
        hypercharge=physical_hypercharge_generator(),
    )
    assert not all(
        (
            summary["preserves_color"],
            summary["preserves_su2_l"],
            summary["preserves_hypercharge"],
        ),
    )


def test_joint_y_t3l_table_is_available_from_lepton() -> None:
    assert joint_y_t3l_table() == EXPECTED_JOINT_Y_T3L_TABLE


def test_higgs_like_charge_shift_basis_exists_and_is_color_singlet() -> None:
    basis = color_singlet_charge_shift_basis(sp.Rational(1, 2), sp.Rational(1, 2))
    assert len(basis) == 4
    candidate = basis[0]
    assert candidate.shape == (32, 32)
    assert not same_matrix(candidate, sp.zeros(32))


def test_higgs_like_candidate_has_expected_charge_shifts() -> None:
    candidate = higgs_like_charge_shift_candidate(sp.Rational(1, 2), sp.Rational(1, 2))
    assert has_charge_shift(candidate, hypercharge_observable(), sp.Rational(1, 2))
    assert has_charge_shift(candidate, normalized_t3_l_observable(), sp.Rational(1, 2))
    assert is_higgs_like_charge_shift(candidate)


def test_transpose_component_has_opposite_higgs_like_charge_shifts() -> None:
    positive, negative = higgs_like_charge_shift_pair()
    assert same_matrix(negative, conjugate_charge_shift_component(positive))
    assert has_charge_shift(negative, hypercharge_observable(), sp.Rational(-1, 2))
    assert has_charge_shift(negative, normalized_t3_l_observable(), sp.Rational(-1, 2))
    assert is_higgs_like_charge_shift(
        negative,
        hypercharge_shift=sp.Rational(-1, 2),
        t3_l_shift=sp.Rational(-1, 2),
    )


def test_higgs_like_candidate_classification() -> None:
    audit = higgs_like_yukawa_audit()
    assert audit.preserves_color
    assert not audit.preserves_su2_l
    assert not audit.preserves_hypercharge
    assert not audit.commutes_with_sm
    assert audit.higgs_like_charge_shift
    assert audit.hypercharge_shift == sp.Rational(1, 2)
    assert audit.t3_l_shift == sp.Rational(1, 2)
    assert "Representation-level" in audit.interpretation


def test_yukawa_representation_payload_records_boundary() -> None:
    payload = yukawa_representation_audit_payload()
    assert payload["joint_y_t3l_table_matches_sm"]
    assert payload["beta_off_diagonal_between_chiralities"]
    assert payload["higgs_like_shift_basis_dimension"] == 4
    assert payload["higgs_like"].higgs_like_charge_shift
    assert payload["positive_component_is_higgs_like"]
    assert payload["negative_component_is_conjugate_higgs_like"]
    assert payload["conjugate_component_construction"] == "transpose of the positive charge-shift map"
    assert not payload["universal_scalar"].higgs_like_charge_shift
    assert "representation audit" in payload["interpretation"]


def test_scalar_identity_is_not_charge_shifting() -> None:
    scalar = scalar_internal_mass(32, sp.Integer(1))
    assert not has_charge_shift(scalar, hypercharge_observable(), sp.Rational(1, 2))
