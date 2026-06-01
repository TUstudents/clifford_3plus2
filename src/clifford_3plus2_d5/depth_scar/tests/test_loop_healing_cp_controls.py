"""Control tests for the complex loop-healing CP deformation."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.loop_healing import (
    complex_healed_laplacian,
    complex_healed_laplacian_is_hermitian,
    conjugated_loop_phase,
    conjugation_flips_loop_phase,
    path_limit_control_pass,
    phi_zero_control_pass,
    real_healed_laplacian,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import path_laplacian


def test_magnetic_healed_laplacian_is_hermitian() -> None:
    delta, phi = sp.symbols("delta phi", real=True)
    matrix = complex_healed_laplacian(delta, phi)

    assert sp.simplify(matrix - matrix.H) == sp.zeros(3, 3)
    assert complex_healed_laplacian_is_hermitian()


def test_phi_zero_reduces_to_real_healed_laplacian() -> None:
    delta = sp.symbols("delta")

    assert sp.simplify(
        complex_healed_laplacian(delta, sp.Integer(0)) - real_healed_laplacian(delta)
    ) == sp.zeros(3, 3)
    assert phi_zero_control_pass()


def test_delta_zero_recovers_path_for_any_loop_phase() -> None:
    phi = sp.symbols("phi")

    assert sp.simplify(
        complex_healed_laplacian(sp.Integer(0), phi) - path_laplacian()
    ) == sp.zeros(3, 3)
    assert path_limit_control_pass()


def test_complex_conjugation_flips_loop_phase() -> None:
    phi = sp.symbols("phi")

    assert conjugated_loop_phase(phi) == -phi
    assert conjugation_flips_loop_phase()
