"""Tests for W4 — covariant O-decomposition (the escape-hatch resolution)."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_hop_walsh.covariant_o_decomposition import (
    covariant_irrep_norms,
    covariant_o_decomposition_payload,
    covariant_support_verdict,
    group_is_octahedral,
    octahedral_group,
    reconstruction_is_exact,
)


def test_group_is_octahedral() -> None:
    # 24 elements and R = Ad(U) for every element (validates the spin pairing).
    assert len(octahedral_group()) == 24
    assert group_is_octahedral()


def test_projectors_reconstruct_the_source() -> None:
    # Irrep norms sum exactly to the source norm (projectors complete/orthogonal).
    assert reconstruction_is_exact("right")
    assert reconstruction_is_exact("left")


def test_covariant_norms_are_the_computed_spectrum() -> None:
    # A1=1, A2=1/3, E=2/3, T1=T2=0 (both helicities). Locks the experiment.
    for helicity in ("right", "left"):
        norms = dict(covariant_irrep_norms(helicity))
        assert sp.simplify(norms["A1"] - 1) == 0
        assert sp.simplify(norms["A2"] - sp.Rational(1, 3)) == 0
        assert sp.simplify(norms["E"] - sp.Rational(2, 3)) == 0
        assert sp.simplify(norms["T1"]) == 0
        assert sp.simplify(norms["T2"]) == 0


def test_escape_hatch_is_closed() -> None:
    payload = covariant_o_decomposition_payload()
    # Covariant T2 reassembles to zero (escape hatch's mechanism is real), but a
    # forbidden E quadrupole is present -> still a kill -> escape hatch closed.
    assert payload.final_verdict == "COVARIANT_KILL_FORBIDDEN_QUADRUPOLE"
    assert payload.escape_hatch_closed is True
    assert payload.reconstruction_exact
    assert payload.group_valid


def test_covariant_verdict_taxonomy() -> None:
    # Pure decision helper.
    clean = {"A1": sp.Integer(1), "A2": sp.Integer(1), "E": sp.Integer(0), "T1": sp.Integer(1), "T2": sp.Integer(0)}
    assert covariant_support_verdict(clean) == "COVARIANT_SUPPORT_PASS"
    forbidden = {**clean, "E": sp.Rational(2, 3)}
    assert covariant_support_verdict(forbidden) == "COVARIANT_KILL_FORBIDDEN_QUADRUPOLE"
    no_vector = {**clean, "T1": sp.Integer(0)}
    assert covariant_support_verdict(no_vector) == "COVARIANT_KILL_MISSING_VECTOR"
