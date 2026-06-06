"""Tests for the minimal exact unitary S3-defect Floquet model."""

import sympy as sp

from clifford_3plus2_d5.radial_response.unitary_defect import (
    basis_defect_vector,
    full_unitary_resolvent_p_block,
    givens_defect_coin,
    is_exact_unitary,
    minimal_unitary_defect_payload,
    minimal_unitary_s3_defect,
    self_energy_at_z_two,
    uniform_defect_vector,
    unitary_effective_resolvent,
)


def test_givens_coin_and_floquet_are_exactly_unitary() -> None:
    coin = givens_defect_coin()
    floquet = minimal_unitary_s3_defect()
    assert is_exact_unitary(coin)
    assert is_exact_unitary(floquet)


def test_unitary_schur_resolvent_matches_full_resolvent() -> None:
    z = sp.Integer(2)
    floquet = minimal_unitary_s3_defect()
    assert sp.simplify(
        unitary_effective_resolvent(z, floquet) - full_unitary_resolvent_p_block(z, floquet)
    ) == sp.zeros(1, 1)


def test_self_energy_depends_on_coin_angle_and_defect_vector() -> None:
    default = self_energy_at_z_two()
    angle_control = self_energy_at_z_two(
        cos_value=sp.Rational(3, 5),
        sin_value=sp.Rational(4, 5),
    )
    vector_control = self_energy_at_z_two(q_vector=basis_defect_vector())
    assert sp.simplify(default - angle_control) != sp.zeros(1, 1)
    assert sp.simplify(default - vector_control) != sp.zeros(1, 1)


def test_defect_vectors_are_normalized() -> None:
    uniform = uniform_defect_vector()
    basis = basis_defect_vector()
    assert (uniform.T * uniform)[0, 0] == 1
    assert (basis.T * basis)[0, 0] == 1


def test_minimal_unitary_defect_payload_passes_as_form_not_derivation() -> None:
    payload = minimal_unitary_defect_payload()
    assert payload.final_verdict == "MINIMAL_UNITARY_S3_DEFECT_FORM_PASS"
    assert payload.unitary_passes
    assert payload.schur_matches_full_resolvent
    assert payload.coin_angle_changes_self_energy
    assert payload.defect_vector_changes_self_energy
    assert not payload.phase_and_radial_values_forced_by_form
