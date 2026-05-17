from __future__ import annotations

from fractions import Fraction

from clifford_3plus2_d5.obstruction_r10.gate_algebra import (
    NamedMatrix,
    audit_one_particle_gate_algebra,
    diagonal_matrix,
    has_off_block_entries,
    is_block_scalar,
    is_even_fock_number_function,
    number_function_on_even_fock,
    parse_fraction_matrix,
)


def test_off_block_generator_fails() -> None:
    off_block = parse_fraction_matrix(
        [
            ["0", "0", "0", "1", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
        ]
    )

    audit = audit_one_particle_gate_algebra([NamedMatrix("mix", off_block)])

    assert has_off_block_entries(off_block)
    assert audit.off_block_generators_present
    assert not audit.block_diagonal_gate_algebra
    assert not audit.sm_commutant_gate_algebra


def test_block_scalar_generator_passes() -> None:
    block_scalar = diagonal_matrix([2, 2, 2, -1, -1])

    audit = audit_one_particle_gate_algebra([NamedMatrix("N3_N2", block_scalar)])

    assert is_block_scalar(block_scalar)
    assert audit.block_diagonal_gate_algebra
    assert audit.sm_commutant_gate_algebra


def test_function_of_n3_n2_passes_on_even_fock_representation() -> None:
    matrix = number_function_on_even_fock(
        {
            (0, 0): Fraction(0),
            (0, 2): Fraction(2),
            (1, 1): Fraction(3),
            (2, 0): Fraction(4),
            (2, 2): Fraction(6),
            (3, 1): Fraction(7),
        }
    )

    assert is_even_fock_number_function(matrix)


def test_individual_basis_projector_inside_v3_fails() -> None:
    projector = diagonal_matrix([1, 0, 0, 0, 0])

    audit = audit_one_particle_gate_algebra([NamedMatrix("color_projector", projector)])

    assert not has_off_block_entries(projector)
    assert not is_block_scalar(projector)
    assert audit.block_diagonal_gate_algebra
    assert not audit.sm_commutant_gate_algebra


def test_block_diagonal_non_scalar_inside_v3_fails() -> None:
    non_scalar = diagonal_matrix([1, 2, 1, 0, 0])

    audit = audit_one_particle_gate_algebra([NamedMatrix("color_cartan", non_scalar)])

    assert not has_off_block_entries(non_scalar)
    assert not is_block_scalar(non_scalar)
    assert audit.block_diagonal_gate_algebra
    assert not audit.sm_commutant_gate_algebra


def test_off_block_v2_to_v3_generator_fails() -> None:
    off_block = parse_fraction_matrix(
        [
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "1", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
        ]
    )

    audit = audit_one_particle_gate_algebra([NamedMatrix("mix_reverse", off_block)])

    assert has_off_block_entries(off_block)
    assert audit.off_block_generators_present
    assert not audit.block_diagonal_gate_algebra
    assert not audit.sm_commutant_gate_algebra
