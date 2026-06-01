"""Tests for the exact port-space transfer predictions."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.prediction_ledger import (
    ckm_depth_exponents,
    ckm_lambda_exponents,
    port_transfer_relations_hold,
    transfer_kernel_from_projectors,
    transfer_kernel_matches_v1,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import transfer_operator_port_basis


def test_projector_transfer_kernel_matches_v1_transfer_operator() -> None:
    assert sp.simplify(
        transfer_kernel_from_projectors() - transfer_operator_port_basis()
    ) == sp.zeros(3, 3)
    assert transfer_kernel_matches_v1()


def test_port_space_transfer_relations_are_fixed() -> None:
    transfer = transfer_kernel_from_projectors()
    assert sp.simplify(transfer[0, 0] - transfer[2, 2]) == 0
    assert sp.simplify(transfer[0, 1] - transfer[1, 2]) == 0
    assert sp.simplify(transfer[0, 2] - transfer[0, 1]) != 0
    assert port_transfer_relations_hold()


def test_ckm_exponents_are_wolfenstein_pattern() -> None:
    assert ckm_depth_exponents() == {(1, 2): 2, (2, 3): 4, (1, 3): 6}
    assert ckm_lambda_exponents() == {(1, 2): 1, (2, 3): 2, (1, 3): 3}

