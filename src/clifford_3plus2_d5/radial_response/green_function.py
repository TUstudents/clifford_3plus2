"""Boundary Green-function / Feshbach recirculation audit."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def feshbach_self_energy(z: sp.Expr, h_q: sp.Matrix, v: sp.Matrix) -> sp.Matrix:
    """Return ``Sigma(z) = V^T (z - H_Q)^-1 V`` for real symbolic matrices."""

    z = sp.sympify(z)
    return (v.T * (z * sp.eye(h_q.rows) - h_q).inv() * v).applyfunc(sp.simplify)


def p_block_resolvent(z: sp.Expr, h_p: sp.Matrix, h_q: sp.Matrix, v: sp.Matrix) -> sp.Matrix:
    """Return the P-block resolvent from the Schur complement."""

    sigma = feshbach_self_energy(z, h_q, v)
    return (z * sp.eye(h_p.rows) - h_p - sigma).inv().applyfunc(sp.simplify)


def full_resolvent_p_block(z: sp.Expr, h_p: sp.Matrix, h_q: sp.Matrix, v: sp.Matrix) -> sp.Matrix:
    """Return the P block of the full inverse resolvent."""

    upper = sp.Matrix.hstack(h_p, v.T)
    lower = sp.Matrix.hstack(v, h_q)
    h_full = sp.Matrix.vstack(upper, lower)
    full = (z * sp.eye(h_full.rows) - h_full).inv().applyfunc(sp.simplify)
    return full[: h_p.rows, : h_p.cols].applyfunc(sp.simplify)


def boundary_recirculation_series_terms(
    z: sp.Expr,
    h_q: sp.Matrix,
    v: sp.Matrix,
    max_returns: int,
) -> tuple[sp.Matrix, ...]:
    """Return first terms in ``V^T z^-1 (H_Q z^-1)^n V`` recirculation series."""

    z = sp.sympify(z)
    q_identity = sp.eye(h_q.rows)
    terms: list[sp.Matrix] = []
    for returns in range(max_returns + 1):
        term = v.T * (h_q**returns) * v / z ** (returns + 1)
        terms.append(term.applyfunc(sp.simplify))
    assert q_identity.rows == h_q.rows
    return tuple(terms)


def example_boundary_system() -> tuple[sp.Symbol, sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return a small exact system used by the audit and tests."""

    z = sp.Symbol("z")
    h_p = sp.Matrix([[0]])
    h_q = sp.Matrix([[sp.Symbol("a")]])
    v = sp.Matrix([[sp.Symbol("g")]])
    return z, h_p, h_q, v


@dataclass(frozen=True)
class GreenFunctionPayload:
    """Payload for the Feshbach recirculation gate."""

    final_verdict: str
    self_energy: sp.Matrix
    schur_matches_full_resolvent: bool
    first_recirculation_terms: tuple[sp.Matrix, ...]
    interpretation: str


def green_function_payload() -> GreenFunctionPayload:
    """Return the boundary Green-function mass-form verdict."""

    z, h_p, h_q, v = example_boundary_system()
    sigma = feshbach_self_energy(z, h_q, v)
    schur = p_block_resolvent(z, h_p, h_q, v)
    full = full_resolvent_p_block(z, h_p, h_q, v)
    terms = boundary_recirculation_series_terms(z, h_q, v, max_returns=2)
    expected_sigma = sp.Matrix([[sp.Symbol("g") ** 2 / (z - sp.Symbol("a"))]])
    schur_matches = sp.simplify(schur - full) == sp.zeros(1, 1)
    terms_match = terms == (
        sp.Matrix([[sp.Symbol("g") ** 2 / z]]),
        sp.Matrix([[sp.Symbol("a") * sp.Symbol("g") ** 2 / z**2]]),
        sp.Matrix([[sp.Symbol("a") ** 2 * sp.Symbol("g") ** 2 / z**3]]),
    )
    checks_pass = sigma == expected_sigma and schur_matches and terms_match

    if checks_pass:
        final_verdict = "MASS_AS_BOUNDARY_RECIRCULATION_PASS"
        interpretation = (
            "The P-block Green function is exactly the Schur complement with "
            "Sigma(z)=V^T(z-H_Q)^-1V. Expanding Sigma gives repeated "
            "P->Q->...->Q->P boundary returns, so radial masses are pole "
            "shifts/residues of boundary recirculation."
        )
    else:
        final_verdict = "MASS_AS_BOUNDARY_RECIRCULATION_KILL"
        interpretation = "The Schur complement or recirculation expansion failed."

    return GreenFunctionPayload(
        final_verdict=final_verdict,
        self_energy=sigma,
        schur_matches_full_resolvent=schur_matches,
        first_recirculation_terms=terms,
        interpretation=interpretation,
    )
