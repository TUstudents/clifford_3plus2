"""Tests for exact Schur complements and the finite K3 tail kill."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.k3_tail import (
    finite_k3_tail_hamiltonian,
    k3_adjacency,
    k3_tail_self_energy,
    tail_boundary_coupling,
    tail_is_s3_equivariant,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_s3_invariant,
    k_nu_operator,
)
from clifford_3plus2_d5.boundary_response.schur import (
    matrix_equal,
    projected_resolvent_denominator,
    self_energy,
)


def test_k3_adjacency_has_singlet_and_degenerate_doublet_spectrum() -> None:
    assert k3_adjacency().eigenvals() == {2: 1, -1: 2}


def test_finite_k3_tail_is_s3_equivariant() -> None:
    h_q = finite_k3_tail_hamiltonian(shells=2)
    assert h_q.shape == (6, 6)
    assert tail_is_s3_equivariant(h_q, shells=2)


def test_tail_boundary_coupling_shape() -> None:
    coupling = tail_boundary_coupling(shells=3)
    assert coupling.shape == (9, 3)
    assert matrix_equal(coupling[0:3, 0:3], sp.eye(3))
    assert matrix_equal(coupling[3:9, 0:3], sp.zeros(6, 3))


def test_schur_self_energy_shape_and_exactness() -> None:
    z = sp.Integer(3)
    h_q = finite_k3_tail_hamiltonian(shells=1)
    v = tail_boundary_coupling(shells=1)
    sigma = self_energy(z, h_q, v)
    assert sigma.shape == (3, 3)
    assert sigma == k3_tail_self_energy(z, shells=1)


def test_projected_resolvent_denominator_shape() -> None:
    z = sp.Integer(3)
    h_p = sp.zeros(3, 3)
    h_q = finite_k3_tail_hamiltonian(shells=1)
    v = tail_boundary_coupling(shells=1)
    denominator = projected_resolvent_denominator(z, h_p, h_q, v)
    assert denominator.shape == (3, 3)


def test_s3_symmetric_tail_self_energy_remains_s3_invariant() -> None:
    sigma = k3_tail_self_energy(sp.Integer(3), shells=1)
    assert is_s3_invariant(sigma)


def test_s3_symmetric_tail_cannot_equal_k_nu_target() -> None:
    sigma = k3_tail_self_energy(sp.Integer(3), shells=1)
    assert not matrix_equal(sigma, k_nu_operator())
