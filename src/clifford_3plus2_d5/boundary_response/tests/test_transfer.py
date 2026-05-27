"""Tests for the residual transfer invariant."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    epsilon_fourth,
    transfer_matrix,
    transfer_polynomial,
    transfer_verdict,
)


def test_epsilon_solves_decaying_transfer_polynomial() -> None:
    eps = epsilon()
    assert sp.simplify(transfer_polynomial(eps)) == 0
    assert 0 < float(eps) < 1


def test_transfer_matrix_eigenvalues_are_pinned() -> None:
    vals = transfer_matrix().eigenvals()
    assert vals == {1 + sp.sqrt(2): 1, 1 - sp.sqrt(2): 1}


def test_epsilon_fourth_simplifies_to_note_value() -> None:
    assert sp.simplify(epsilon_fourth() - (17 - 12 * sp.sqrt(2))) == 0


def test_transfer_verdict_passes() -> None:
    assert transfer_verdict() == "TRANSFER_PASS"
