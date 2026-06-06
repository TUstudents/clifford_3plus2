"""Tests for the up-sector radial stacking fork."""

import sympy as sp

from clifford_3plus2_d5.radial_response.up_stacking import (
    exponential_stack_vector,
    geometric_stack_vector,
    stack_invariant,
    up_stacking_payload,
)


def test_exponential_stack_gives_factorial_relation() -> None:
    x = sp.Symbol("x", positive=True)
    vector = exponential_stack_vector(x)
    assert vector == (x**2 / 2, x, sp.Integer(1))
    assert stack_invariant(vector) == sp.Rational(1, 2)


def test_geometric_stack_gives_different_relation() -> None:
    x = sp.Symbol("x", positive=True)
    vector = geometric_stack_vector(x)
    assert vector == (x**2, x, sp.Integer(1))
    assert stack_invariant(vector) == 1


def test_up_stacking_payload_passes() -> None:
    payload = up_stacking_payload()
    assert payload.final_verdict == "UP_STACKING_LAW_EXPONENTIAL_FAVORED"
    assert payload.exponential_invariant == sp.Rational(1, 2)
    assert payload.geometric_invariant == 1
    assert payload.stacking_laws_distinct
