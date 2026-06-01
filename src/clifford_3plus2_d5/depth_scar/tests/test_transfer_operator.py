"""Tests for the depth-scar transfer operator."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.transfer import epsilon
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    defect_eigenbasis_matrix,
    transfer_operator_eigenbasis,
    transfer_operator_eigenvalues,
    transfer_operator_port_basis,
    transition_depth_differences,
)


def test_transfer_operator_is_diagonal_in_defect_eigenbasis() -> None:
    eps = epsilon()
    assert transfer_operator_eigenbasis() == sp.diag(1, eps**2, eps**6)
    assert transfer_operator_eigenvalues() == (
        sp.Integer(1),
        sp.simplify(eps**2),
        sp.simplify(eps**6),
    )


def test_transfer_operator_port_basis_reconstructs_from_eigenbasis() -> None:
    basis = defect_eigenbasis_matrix()
    port_transfer = transfer_operator_port_basis()
    assert sp.simplify(
        basis.T * port_transfer * basis - transfer_operator_eigenbasis()
    ) == sp.zeros(3, 3)


def test_transition_depth_differences_match_ckm_scaling_exponents() -> None:
    assert transition_depth_differences() == {
        (1, 2): 2,
        (2, 3): 4,
        (1, 3): 6,
    }
