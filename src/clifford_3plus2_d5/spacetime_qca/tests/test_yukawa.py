"""Session 23 Higgs/Yukawa representation-audit tests."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.lepton.sm_hypercharge import (
    EXPECTED_JOINT_Y_T3L_TABLE,
    hypercharge_observable,
    joint_y_t3l_table,
    normalized_t3_l_observable,
    physical_hypercharge_generator,
)
from clifford_3plus2_d5.spacetime_qca import (
    audit_hermitian_yukawa_phi,
    beta_is_off_diagonal_between_chiralities,
    color_singlet_charge_shift_basis,
    conjugate_charge_shift_component,
    commutes_with_internal_j,
    electromagnetic_charge_observable,
    gauge_breaking_summary,
    has_charge_shift,
    hermitian_yukawa_hamiltonian,
    hermitian_yukawa_internal_control,
    higgs_phi_raising_map,
    higgs_doublet_map_audit_payload,
    higgs_like_doublet_map_basis,
    higgs_like_charge_shift_candidate,
    higgs_like_charge_shift_pair,
    higgs_like_yukawa_audit,
    is_higgs_like_charge_shift,
    left_right_projectors,
    matrix_is_real_symmetric,
    neutral_yukawa_hamiltonian,
    neutral_yukawa_internal_control,
    preserves_electromagnetism,
    projector_control_mass,
    projector_control_yukawa_audit,
    same_matrix,
    selected_higgs_phi_basis,
    scalar_internal_mass,
    static_higgs_doublet_hamiltonian,
    static_higgs_doublet_internal_control,
    static_neutral_higgs_vev_hamiltonian,
    static_neutral_higgs_vev_control,
    static_yukawa_hamiltonian,
    static_yukawa_internal_control,
    su2_l_lowered_higgs_like_basis,
    universal_scalar_yukawa_audit,
    yukawa_representation_audit_payload,
    yukawa_spacetime_coupler,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import su2_l_generators_from_spin04
from clifford_3plus2_d5.lepton.patisalam_sm import su3_c_generators_from_su4

pytestmark = pytest.mark.slow


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


def test_su2_l_action_generates_lower_higgs_like_map_space() -> None:
    upper, lower = higgs_like_doublet_map_basis()
    assert len(upper) == 4
    assert len(lower) == 4
    assert lower == su2_l_lowered_higgs_like_basis()
    assert all(is_higgs_like_charge_shift(matrix) for matrix in upper)
    assert all(
        is_higgs_like_charge_shift(
            matrix,
            hypercharge_shift=sp.Rational(1, 2),
            t3_l_shift=sp.Rational(-1, 2),
        )
        for matrix in lower
    )


def test_higgs_like_maps_are_not_individually_internal_j_linear() -> None:
    upper, lower = higgs_like_doublet_map_basis()
    assert not all(commutes_with_internal_j(matrix) for matrix in upper)
    assert not all(commutes_with_internal_j(matrix) for matrix in lower)


def test_static_yukawa_control_and_hamiltonian_are_hermitian() -> None:
    control = static_yukawa_internal_control()
    hamiltonian = static_yukawa_hamiltonian()
    assert control.shape == (32, 32)
    assert hamiltonian.shape == (128, 128)
    assert matrix_is_real_symmetric(control)
    assert same_matrix(hamiltonian, hamiltonian.H)
    assert not has_charge_shift(control, hypercharge_observable(), sp.Rational(1, 2))


def test_static_higgs_doublet_control_is_hermitian() -> None:
    control = static_higgs_doublet_internal_control()
    hamiltonian = static_higgs_doublet_hamiltonian()
    assert control.shape == (32, 32)
    assert hamiltonian.shape == (128, 128)
    assert matrix_is_real_symmetric(control)
    assert same_matrix(hamiltonian, hamiltonian.H)


def test_neutral_higgs_vev_preserves_electromagnetism_only() -> None:
    control = static_neutral_higgs_vev_control()
    hamiltonian = static_neutral_higgs_vev_hamiltonian()
    assert matrix_is_real_symmetric(control)
    assert same_matrix(hamiltonian, hamiltonian.H)
    assert preserves_electromagnetism(control)
    assert same_matrix(
        hypercharge_observable() + normalized_t3_l_observable(),
        electromagnetic_charge_observable(),
    )
    assert not same_matrix(
        hypercharge_observable() * control - control * hypercharge_observable(),
        sp.zeros(32),
    )
    assert not same_matrix(
        normalized_t3_l_observable() * control - control * normalized_t3_l_observable(),
        sp.zeros(32),
    )


def test_hermitian_yukawa_phi_zero_and_selected_basis_construction() -> None:
    zero = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(0), sp.Integer(0)),
        phi_zero=(sp.Integer(0), sp.Integer(0)),
    )
    assert same_matrix(zero, sp.zeros(32))
    assert same_matrix(hermitian_yukawa_hamiltonian((0, 0), (0, 0)), sp.zeros(128))

    phi_plus = (sp.Integer(2), sp.Integer(-3))
    phi_zero = (sp.Rational(5, 2), sp.Rational(7, 3))
    upper_re, upper_im, lower_re, lower_im = selected_higgs_phi_basis()
    raising = (
        phi_plus[0] * upper_re
        + phi_plus[1] * upper_im
        + phi_zero[0] * lower_re
        + phi_zero[1] * lower_im
    ).applyfunc(sp.simplify)
    assert same_matrix(higgs_phi_raising_map(phi_plus, phi_zero), raising)
    assert same_matrix(hermitian_yukawa_internal_control(phi_plus, phi_zero), raising + raising.T)


def test_hermitian_yukawa_phi_is_linear_and_hermitian() -> None:
    left = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(1), sp.Integer(2)),
        phi_zero=(sp.Integer(3), sp.Integer(4)),
    )
    right = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(5), sp.Integer(6)),
        phi_zero=(sp.Integer(7), sp.Integer(8)),
    )
    combined = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(6), sp.Integer(8)),
        phi_zero=(sp.Integer(10), sp.Integer(12)),
    )
    hamiltonian = hermitian_yukawa_hamiltonian(
        phi_plus=(sp.Integer(2), sp.Integer(-3)),
        phi_zero=(sp.Rational(5, 2), sp.Rational(7, 3)),
    )
    assert same_matrix(combined, left + right)
    assert matrix_is_real_symmetric(combined)
    assert same_matrix(hamiltonian, hamiltonian.H)


def test_neutral_yukawa_phi_preserves_electromagnetism_only() -> None:
    control = neutral_yukawa_internal_control(sp.Integer(1))
    hamiltonian = neutral_yukawa_hamiltonian(sp.Integer(1))
    assert matrix_is_real_symmetric(control)
    assert same_matrix(hamiltonian, hamiltonian.H)
    assert preserves_electromagnetism(control)
    assert not same_matrix(
        hypercharge_observable() * control - control * hypercharge_observable(),
        sp.zeros(32),
    )
    assert not same_matrix(
        normalized_t3_l_observable() * control - control * normalized_t3_l_observable(),
        sp.zeros(32),
    )


def test_charged_yukawa_phi_breaks_electromagnetism() -> None:
    control = hermitian_yukawa_internal_control(
        phi_plus=(sp.Integer(1), sp.Integer(0)),
        phi_zero=(sp.Integer(0), sp.Integer(0)),
    )
    assert not preserves_electromagnetism(control)


def test_hermitian_yukawa_phi_audit_payload_records_boundaries() -> None:
    payload = audit_hermitian_yukawa_phi()
    assert payload.phi_api == "two_complex_explicit_re_im"
    assert payload.selected_upper_dimension == 2
    assert payload.selected_lower_dimension == 2
    assert payload.zero_phi_is_zero
    assert payload.linearity_passed
    assert payload.internal_control_symmetric
    assert payload.hamiltonian_hermitian
    assert payload.neutral_preserves_color
    assert payload.neutral_preserves_electromagnetism
    assert payload.neutral_breaks_hypercharge
    assert payload.neutral_breaks_t3_l
    assert payload.charged_component_breaks_electromagnetism
    assert payload.neutral_rank > 0
    assert payload.neutral_nullity < 32
    assert "static background Yukawa layer" in payload.interpretation


def test_higgs_doublet_map_audit_payload_records_boundaries() -> None:
    payload = higgs_doublet_map_audit_payload()
    assert payload.upper_dimension == 4
    assert payload.lower_dimension == 4
    assert payload.combined_dimension == 8
    assert payload.upper_charge_shifts_valid
    assert payload.lower_charge_shifts_valid
    assert payload.lowering_generators_span_same_space
    assert not payload.upper_commutes_with_j
    assert not payload.lower_commutes_with_j
    assert payload.static_control_symmetric
    assert payload.static_hamiltonian_hermitian
    assert payload.full_doublet_control_rank > 0
    assert payload.full_doublet_control_nullity < 32
    assert payload.neutral_vev_preserves_color
    assert payload.neutral_vev_preserves_electromagnetism
    assert payload.neutral_vev_breaks_hypercharge
    assert payload.neutral_vev_breaks_t3_l
    assert payload.neutral_vev_rank > 0
    assert payload.neutral_vev_nullity < 32
    assert "not dynamical Higgs fields" in payload.interpretation


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
