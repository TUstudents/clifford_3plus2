"""Tests for W5 — the S3/Schur obstruction (the core closure result)."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_hop_walsh.s3_schur_obstruction import (
    commutant_dimension,
    depth_breaking_spurion,
    depth_is_s3_invariant,
    doubled_k3_spectrum,
    invariant_ops_have_at_most_two_distinct_eigenvalues,
    k3_laplacian_spectrum,
    s3_schur_obstruction_payload,
    schur_spectrum,
)


def test_schur_commutant_is_two_dimensional() -> None:
    # 3 = 1 + 2 under S3 -> commutant spanned by P_1, P_2 (dim 2).
    assert commutant_dimension() == 2
    assert invariant_ops_have_at_most_two_distinct_eigenvalues()


def test_schur_spectrum_is_alpha_beta_beta() -> None:
    a, b = sp.symbols("a b")
    spectrum = dict(schur_spectrum(a, b))
    assert spectrum[a] == 1  # singlet, multiplicity 1
    assert spectrum[b] == 2  # doublet, multiplicity 2


def test_unbroken_k3_gives_0_3_3_not_0_2_6() -> None:
    # The residual K3 Laplacian spectrum is {0, 3, 3}; doubled it is {0, 6} != {0,2,6}.
    spectrum = k3_laplacian_spectrum()
    assert spectrum[sp.Integer(0)] == 1
    assert spectrum[sp.Integer(3)] == 2
    assert doubled_k3_spectrum() == {sp.Integer(0), sp.Integer(6)}
    assert doubled_k3_spectrum() != {sp.Integer(0), sp.Integer(2), sp.Integer(6)}


def test_depths_026_are_an_s3_breaking_spurion() -> None:
    # diag(0,2,6) has 3 distinct eigenvalues -> cannot be S3-invariant; its breaking
    # spurion diag(-8/3,-2/3,10/3) ~ (-4,-1,5) is nonzero.
    assert depth_is_s3_invariant() is False
    spurion = depth_breaking_spurion()
    assert [spurion[i, i] for i in range(3)] == [
        sp.Rational(-8, 3),
        sp.Rational(-2, 3),
        sp.Rational(10, 3),
    ]
    assert sp.simplify(spurion.trace()) == 0  # traceless -> doublet sector


def test_obstruction_verdict() -> None:
    payload = s3_schur_obstruction_payload()
    assert payload.final_verdict == "DEPTH_HIERARCHY_REQUIRES_S3_BREAKING"
