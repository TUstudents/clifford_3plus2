from __future__ import annotations

from fractions import Fraction

from clifford_3plus2_d5.obstruction_r10.sm.spinor16 import (
    certificate_to_dict,
    degree_dimensions,
    derived_complex_modes,
    even_spinor_basis,
    hypercharge_operator,
    spinor16_certificate,
    spinor16_sectors,
    spinor16_table_rows,
)


def test_derived_complex_modes_use_phase_1_basis_and_split_order() -> None:
    modes = derived_complex_modes()

    assert len(modes) == 5
    assert modes[0].real_basis == ("x_1", "y_1")
    assert modes[2].block == "C3"
    assert modes[3].block == "C2"
    assert modes[4].real_basis == ("x_5", "y_5")


def test_even_spinor_basis_has_dimension_16_and_even_degrees() -> None:
    basis = even_spinor_basis()

    assert len(basis) == 16
    assert degree_dimensions() == {0: 1, 2: 10, 4: 5}
    assert all(state.degree in {0, 2, 4} for state in basis)


def test_spinor_basis_states_carry_exact_sector_data() -> None:
    states = even_spinor_basis()
    vacuum = states[0]
    q_state = next(state for state in states if state.sector == (1, 1))

    assert vacuum.subset == ()
    assert vacuum.hypercharge == Fraction(0)
    assert vacuum.label == "nu^c"
    assert q_state.hypercharge == Fraction(1, 6)
    assert q_state.label == "Q"


def test_hypercharge_operator_is_exact_on_even_basis() -> None:
    operator = hypercharge_operator()

    assert len(operator) == 16
    assert all(isinstance(value, Fraction) for value in operator)
    assert Fraction(1, 6) in operator
    assert Fraction(-2, 3) in operator


def test_spinor16_sector_table_matches_expected_rows() -> None:
    sectors = spinor16_sectors()
    rows = spinor16_table_rows()

    assert sum(sector.multiplicity for sector in sectors) == 16
    assert rows == (
        {"n3": 0, "n2": 0, "multiplicity": 1, "hypercharge": "0", "label": "nu^c"},
        {"n3": 0, "n2": 2, "multiplicity": 1, "hypercharge": "1", "label": "e^c"},
        {"n3": 1, "n2": 1, "multiplicity": 6, "hypercharge": "1/6", "label": "Q"},
        {"n3": 2, "n2": 0, "multiplicity": 3, "hypercharge": "-2/3", "label": "u^c"},
        {"n3": 2, "n2": 2, "multiplicity": 3, "hypercharge": "1/3", "label": "d^c"},
        {"n3": 3, "n2": 1, "multiplicity": 2, "hypercharge": "-1/2", "label": "L"},
    )


def test_spinor16_certificate_is_candidate_only_with_provenance_guard() -> None:
    certificate = spinor16_certificate()

    assert certificate.spinor16_dimension == 16
    assert certificate.degree_dimensions == {0: 1, 2: 10, 4: 5}
    assert certificate.hypercharge_check_passed
    assert certificate.branching_table_check_passed
    assert certificate.uses_phase_1_real_carrier
    assert certificate.uses_phase_1_j
    assert certificate.uses_phase_1_projectors
    assert certificate.uses_phase_3_split_candidate
    assert certificate.uses_existing_j_and_split
    assert not certificate.introduces_new_complex_structure
    assert not certificate.introduces_new_3plus2_split
    assert not certificate.qca_derives_spinor_inputs
    assert certificate.spinor16_verdict == "candidate_only"
    assert not certificate.load_bearing_qca_bridge


def test_spinor16_certificate_can_record_prior_qca_derivation_without_bridge_flip() -> None:
    certificate = spinor16_certificate(qca_derives_spinor_inputs=True)

    assert certificate.spinor16_verdict == "derived_from_qca"
    assert not certificate.load_bearing_qca_bridge


def test_spinor16_certificate_serialization_is_stable() -> None:
    payload = certificate_to_dict(spinor16_certificate())

    assert payload["spinor16_dimension"] == 16
    assert payload["degree_dimensions"] == {"0": 1, "2": 10, "4": 5}
    assert payload["hypercharge_check_passed"] is True
    assert payload["branching_table_check_passed"] is True
    assert payload["uses_existing_j_and_split"] is True
    assert payload["introduces_new_complex_structure"] is False
    assert payload["introduces_new_3plus2_split"] is False
    assert payload["spinor16_verdict"] == "candidate_only"
    assert payload["spinor16_check_passed"] is True
    assert payload["load_bearing_qca_bridge"] is False
