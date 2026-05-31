"""Tests for W1 — coefficient-Walsh decomposition of the BCC hop shell."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_hop_walsh.hop_walsh_decomposition import (
    WALSH_SUBSETS,
    a1g_baseline,
    a2u_component,
    is_c3_covariant,
    is_zero_symbolic,
    walsh_coefficient,
    walsh_coefficients,
)
from clifford_3plus2_d5.depth_hop_walsh.reuse import bialynicki_birula_s_matrices


def test_walsh_reproduces_s_matrices_for_t1u_block() -> None:
    # s_i = sum_v v_i H_v = 8 * Hhat_i (the T1u/degree-1 Walsh coefficients).
    s_x, s_y, s_z = bialynicki_birula_s_matrices()
    for axis, s in zip(("x", "y", "z"), (s_x, s_y, s_z), strict=True):
        residual = 8 * walsh_coefficient(axis) - s
        assert all(sp.simplify(entry) == 0 for entry in residual)


def test_left_helicity_is_parity_signed_right() -> None:
    # Left hops are W(-v), so Hhat_S^L = (-1)^|S| Hhat_S^R.
    right = walsh_coefficients("right")
    left = walsh_coefficients("left")
    for subset in WALSH_SUBSETS:
        sign = (-1) ** len(subset)
        residual = left[subset] - sign * right[subset]
        assert all(sp.simplify(entry) == 0 for entry in residual)


def test_baseline_and_a2u_values() -> None:
    # A1g baseline = (1/8) I; A2u (right) = (i/8) I; A2u (left) = -(i/8) I.
    assert all(sp.simplify(e) == 0 for e in (a1g_baseline("right") - sp.Rational(1, 8) * sp.eye(2)))
    assert all(sp.simplify(e) == 0 for e in (a2u_component("right") - (sp.I / 8) * sp.eye(2)))
    assert all(sp.simplify(e) == 0 for e in (a2u_component("left") + (sp.I / 8) * sp.eye(2)))


def test_a2u_present_both_helicities() -> None:
    # The depth-6 mode IS present (nonzero), for both helicities.
    assert not is_zero_symbolic(a2u_component("right"))
    assert not is_zero_symbolic(a2u_component("left"))


def test_covariance_check_is_false() -> None:
    # The BB lattice hop symbol is NOT C3-covariant about [111] (only the IR limit
    # sigma.k restores rotation symmetry). Locks the honest diagnostic finding.
    assert is_c3_covariant("right") is False
    assert is_c3_covariant("left") is False
