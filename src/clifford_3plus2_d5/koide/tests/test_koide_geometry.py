"""Tests for ``koide_geometry.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.koide.koide_geometry import (
    KOIDE_K_TARGET,
    PDG_M_E,
    PDG_M_MU,
    PDG_M_TAU,
    angle_form_holds_for_pdg,
    angle_form_projection_ratio,
    equipartition_holds_for_pdg,
    equipartition_ratio,
    koide_geometry_payload,
    koide_holds_empirically,
    koide_K_from_masses,
    pdg_charged_lepton_masses,
    pdg_koide_K,
    pdg_koide_deviation_from_two_thirds,
    pdg_sqrt_mass_vector,
    trace_direction,
    trace_part,
    trace_projector,
    traceless_part,
    traceless_projector,
)


def test_pdg_masses_match_constants() -> None:
    me, mmu, mtau = pdg_charged_lepton_masses()
    assert me == PDG_M_E
    assert mmu == PDG_M_MU
    assert mtau == PDG_M_TAU


def test_pdg_sqrt_mass_vector_is_3x1() -> None:
    v = pdg_sqrt_mass_vector()
    assert v.shape == (3, 1)


def test_koide_K_target_is_2_over_3() -> None:
    assert KOIDE_K_TARGET == sp.Rational(2, 3)


def test_pdg_koide_K_close_to_two_thirds() -> None:
    K = float(pdg_koide_K())
    assert abs(K - 2 / 3) < 1e-4


def test_pdg_koide_deviation_within_tolerance() -> None:
    dev = float(pdg_koide_deviation_from_two_thirds())
    assert abs(dev) < 1e-4


def test_koide_holds_empirically() -> None:
    assert koide_holds_empirically()


def test_koide_K_for_equal_masses_is_one_third() -> None:
    # For m1 = m2 = m3 = m, K = 3m / (3√m)² = 3m / 9m = 1/3 (LHS minimum).
    K = koide_K_from_masses((sp.Integer(1), sp.Integer(1), sp.Integer(1)))
    assert K == sp.Rational(1, 3)


def test_koide_K_for_dominant_mass_approaches_one() -> None:
    # For m1 >> m2 + m3, K → 1 (LHS maximum).
    masses = (sp.Float("1e10"), sp.Float("1.0"), sp.Float("1.0"))
    K = float(koide_K_from_masses(masses))
    assert K > 0.99


def test_trace_direction_is_normalized() -> None:
    n = trace_direction()
    assert sp.simplify((n.T * n)[0, 0]) == 1


def test_trace_and_traceless_projectors_sum_to_identity() -> None:
    p_t = trace_projector()
    p_o = traceless_projector()
    assert (p_t + p_o).applyfunc(sp.simplify) == sp.eye(3)


def test_trace_projector_is_idempotent() -> None:
    p = trace_projector()
    assert (p * p - p).applyfunc(sp.simplify) == sp.zeros(3, 3)


def test_traceless_projector_is_idempotent() -> None:
    p = traceless_projector()
    assert (p * p - p).applyfunc(sp.simplify) == sp.zeros(3, 3)


def test_trace_and_traceless_parts_sum_to_input() -> None:
    v = pdg_sqrt_mass_vector()
    reconstructed = trace_part(v) + traceless_part(v)
    diff = (v - reconstructed).applyfunc(sp.simplify)
    # Numerical zero: each entry magnitude < 1e-10.
    for i in range(3):
        assert abs(float(diff[i, 0])) < 1e-10


def test_angle_form_pdg_close_to_one_half() -> None:
    ratio = float(angle_form_projection_ratio(pdg_sqrt_mass_vector()))
    assert abs(ratio - 0.5) < 1e-4


def test_equipartition_pdg_close_to_unity() -> None:
    ratio = float(equipartition_ratio(pdg_sqrt_mass_vector()))
    assert abs(ratio - 1.0) < 1e-4


def test_angle_form_holds_for_pdg() -> None:
    assert angle_form_holds_for_pdg()


def test_equipartition_holds_for_pdg() -> None:
    assert equipartition_holds_for_pdg()


def test_geometry_payload_consistent() -> None:
    p = koide_geometry_payload()
    assert p.koide_holds_empirically
    assert p.angle_form_holds
    assert p.equipartition_form_holds
    assert "VERIFIED" in p.verdict
