"""Tests for the R8 S3 scalar-shell completeness gate."""

from clifford_3plus2_d5.radial_response.s3_scalar_completeness import (
    IDENTITY,
    TRIALITY_MINUS,
    TRIALITY_PLUS,
    S3ScalarSector,
    allowed_scalar_s3_successor_elements,
    allowed_scalar_s3_successor_labels,
    is_identity,
    is_three_cycle,
    is_transposition,
    s3_scalar_census,
    scalar_s3_shell_completeness_payload,
    scalar_s3_shell_completeness_pass,
    transposition_conjugation_swaps_triality,
    transposition_elements,
)


def test_s3_scalar_census_partitions_all_six_elements() -> None:
    census = s3_scalar_census()
    assert len(census) == 6
    assert len([row for row in census if row.sector == S3ScalarSector.IDENTITY]) == 1
    assert len([row for row in census if row.sector == S3ScalarSector.SCALAR_HOLOMORPHIC]) == 2
    assert len([row for row in census if row.sector == S3ScalarSector.HERMITIAN_Z2]) == 3


def test_identity_and_transpositions_are_rejected() -> None:
    assert is_identity(IDENTITY)
    assert not is_three_cycle(IDENTITY)
    assert len(transposition_elements()) == 3
    assert all(is_transposition(element) for element in transposition_elements())
    assert all(
        not row.scalar_allowed
        for row in s3_scalar_census()
        if row.sector in {S3ScalarSector.IDENTITY, S3ScalarSector.HERMITIAN_Z2}
    )


def test_only_two_nonidentity_three_cycles_are_allowed() -> None:
    assert is_three_cycle(TRIALITY_PLUS)
    assert is_three_cycle(TRIALITY_MINUS)
    assert allowed_scalar_s3_successor_elements() == (TRIALITY_PLUS, TRIALITY_MINUS)
    assert allowed_scalar_s3_successor_labels() == ("triality_plus", "triality_minus")


def test_transposition_conjugation_swaps_triality_pair() -> None:
    assert transposition_conjugation_swaps_triality()


def test_s3_scalar_shell_payload_passes_conditionally() -> None:
    payload = scalar_s3_shell_completeness_payload()
    assert scalar_s3_shell_completeness_pass()
    assert payload.final_verdict == "SCALAR_S3_SHELL_COMPLETENESS_PASS"
    assert payload.element_count == 6
    assert payload.identity_rejected
    assert payload.allowed_successors == ("triality_plus", "triality_minus")
    assert payload.allowed_elements == (TRIALITY_PLUS, TRIALITY_MINUS)
    assert payload.transposition_count == 3
    assert payload.transpositions_rejected
    assert payload.transposition_conjugation_swaps_triality
    assert payload.r7_labels_match
    assert payload.full_qca_boundary_completeness_derived is False
