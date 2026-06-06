"""Session 01 audit for the universal bath sidecar."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.universal_bath.jacobi import (
    DiscreteMeasure,
    measure_is_positive,
    moment_round_trip,
    response_round_trip,
)
from clifford_3plus2_d5.universal_bath.opuc import opuc_free_tail_payload
from clifford_3plus2_d5.universal_bath.reduction import reduction_taxonomy_payload
from clifford_3plus2_d5.universal_bath.schur import (
    finite_head_continued_fraction,
    jacobi_matrix,
    schur_boundary_response,
)
from clifford_3plus2_d5.universal_bath.tail import (
    period_one_tail,
    silver_selected_z,
    silver_tail_payload,
)


@dataclass(frozen=True)
class UniversalBathAuditPayload:
    """Combined Session 01 verdict."""

    final_verdict: str
    tail_fixed_point: bool
    silver_value: bool
    toy_measure_positive: bool
    moment_round_trip: bool
    response_round_trip: bool
    finite_head_schur_match: bool
    alternate_tail_changes_response: bool
    reduction_taxonomy_pass: bool
    opuc_free_tail_pass: bool
    interpretation: str


def toy_measure() -> DiscreteMeasure:
    """Return a simple two-point positive measure for the Jacobi certificate."""

    return DiscreteMeasure(
        "session_01_two_point_toy",
        (sp.Integer(1), sp.Integer(3)),
        (sp.Rational(1, 4), sp.Rational(3, 4)),
    )


def finite_head_schur_matches_continued_fraction() -> bool:
    """Return whether a finite head agrees with its continued fraction."""

    z = sp.Symbol("z")
    a_values = (sp.Rational(3, 2), sp.Rational(1, 2))
    b_values = (sp.sqrt(3) / 2,)
    jacobi = jacobi_matrix(a_values, b_values)
    continued_fraction = finite_head_continued_fraction(z, a_values, b_values)
    return sp.simplify(continued_fraction - schur_boundary_response(jacobi, z)) == 0


def alternate_tail_changes_response() -> bool:
    """Return whether replacing the silver tail changes a terminated head."""

    z = sp.Symbol("z")
    a_values = (sp.Integer(0),)
    b_values = (sp.Integer(1),)
    silver_response = finite_head_continued_fraction(
        z,
        a_values,
        b_values,
        terminator=period_one_tail(z),
    )
    constant_response = finite_head_continued_fraction(
        z,
        a_values,
        b_values,
        terminator=sp.Rational(1, 3),
    )
    probe = silver_selected_z()
    return sp.simplify(silver_response.subs(z, probe) - constant_response.subs(z, probe)) != 0


def universal_bath_audit_payload() -> UniversalBathAuditPayload:
    """Return Session 01 payload."""

    tail = silver_tail_payload()
    reduction = reduction_taxonomy_payload()
    opuc = opuc_free_tail_payload()
    measure = toy_measure()
    toy_positive = measure_is_positive(measure)
    moments_ok = moment_round_trip(measure)
    response_ok = response_round_trip(measure)
    schur_ok = finite_head_schur_matches_continued_fraction()
    alternate_changes = alternate_tail_changes_response()
    checks_pass = (
        tail.fixed_point_residual == 0
        and tail.selected_value_matches_epsilon
        and tail.alternate_tail_changes_value
        and toy_positive
        and moments_ok
        and response_ok
        and schur_ok
        and alternate_changes
        and reduction.final_verdict == "BATH_REDUCTION_TAXONOMY_PASS"
        and opuc.all_verblunsky_zero
        and opuc.free_remainder_after_head
    )

    if checks_pass:
        final_verdict = "UNIVERSAL_BATH_SPINE_PASS"
        interpretation = (
            "Session 01 certifies the common bath spine only: scalar finite "
            "moments round-trip to a Jacobi head; the finite head Schur "
            "response equals its continued fraction; the period-one retarded "
            "tail satisfies t=1/(z-t); and at the BB marginal probe it gives "
            "epsilon=sqrt(2)-1. Replacing the terminator changes the response, "
            "so universal silver is a physical closure principle, not a "
            "continued-fraction tautology. The audit also fixes the reduction "
            "taxonomy: positive sectors use scalar Jacobi, real non-positive "
            "shells require indefinite look-ahead Lanczos, and chiral unitary "
            "sectors use CMV/OPUC with free tail alpha_n=0 after the finite "
            "head. Sector source heads remain future work."
        )
    else:
        final_verdict = "UNIVERSAL_BATH_SPINE_KILL"
        interpretation = "At least one Session 01 universal-bath spine check failed."

    return UniversalBathAuditPayload(
        final_verdict=final_verdict,
        tail_fixed_point=tail.fixed_point_residual == 0,
        silver_value=tail.selected_value_matches_epsilon,
        toy_measure_positive=toy_positive,
        moment_round_trip=moments_ok,
        response_round_trip=response_ok,
        finite_head_schur_match=schur_ok,
        alternate_tail_changes_response=alternate_changes,
        reduction_taxonomy_pass=reduction.final_verdict == "BATH_REDUCTION_TAXONOMY_PASS",
        opuc_free_tail_pass=opuc.all_verblunsky_zero and opuc.free_remainder_after_head,
        interpretation=interpretation,
    )
