"""FD-11/FD-12: standard SM anomaly polynomial + BCC walk anomaly contribution.

For one generation of SM left-handed Weyl fermions, with the standard
hypercharge assignments

    Y(Q_L)   = +1/6
    Y(u_R^c) = -2/3      (3-bar of color)
    Y(d_R^c) = +1/3      (3-bar of color)
    Y(L_L)   = -1/2
    Y(e_R^c) = +1
    Y(ν_R^c) =  0

the standard anomaly conditions are

    A_grav = Σ Y                                  (gravitational anomaly)
    A_Y3   = Σ Y³                                  (U(1)_Y³ anomaly)
    A_SU2  = Σ_{SU(2) doublets} Y                  (SU(2)_L² · U(1)_Y)
    A_SU3  = Σ_{color triplets/antitriplets} Y     (SU(3)_c² · U(1)_Y)

with signed sums.  All four cancel exactly per generation for the SM
chiral-16 content.

The BCC walk's continuum limit is free Dirac.  Free Dirac fermions have
the standard SM anomaly polynomial.  The BCC lattice itself does not
contribute additional anomaly terms that depend on N_generations.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


# Per left-handed Weyl species: (name, Y, n_color, color_sign, n_su2_doublet, n_color_triplet)
# - n_color = 3 for triplets/antitriplets, 1 for singlets
# - color_sign = +1 for 3, -1 for 3-bar (relevant for SU(3)² · U(1))
# - n_su2_doublet = 2 if SU(2)_L doublet, 1 if singlet (multiplicity in SU(2))
# - n_color_triplet = 1 if color-charged (3 or 3-bar), 0 if color singlet
SM_LEFT_HANDED_CONTENT_PER_GENERATION = (
    ("Q_L",    sp.Rational(1, 6),  3, +1, 2, 1),
    ("u_R^c",  sp.Rational(-2, 3), 3, -1, 1, 1),
    ("d_R^c",  sp.Rational(1, 3),  3, -1, 1, 1),
    ("L_L",    sp.Rational(-1, 2), 1,  0, 2, 0),
    ("e_R^c",  sp.Integer(1),      1,  0, 1, 0),
    ("nu_R^c", sp.Integer(0),      1,  0, 1, 0),
)


def gravitational_anomaly(generations: int = 1) -> sp.Expr:
    """Return Σ Y over all Weyl fermions (mixed gauge-gravity anomaly)."""

    per_gen: sp.Expr = sp.Integer(0)
    for (_, Y, n_color, _, n_su2, _) in SM_LEFT_HANDED_CONTENT_PER_GENERATION:
        per_gen = per_gen + Y * n_color * n_su2
    return sp.simplify(generations * per_gen)


def u1_cubed_anomaly(generations: int = 1) -> sp.Expr:
    """Return Σ Y³ over all Weyl fermions (U(1)_Y³ anomaly)."""

    per_gen: sp.Expr = sp.Integer(0)
    for (_, Y, n_color, _, n_su2, _) in SM_LEFT_HANDED_CONTENT_PER_GENERATION:
        per_gen = per_gen + (Y ** 3) * n_color * n_su2
    return sp.simplify(generations * per_gen)


def su2_squared_u1_anomaly(generations: int = 1) -> sp.Expr:
    """Return Σ_{doublets} Y (SU(2)_L² · U(1)_Y mixed anomaly)."""

    per_gen: sp.Expr = sp.Integer(0)
    for (_, Y, n_color, _, n_su2, _) in SM_LEFT_HANDED_CONTENT_PER_GENERATION:
        if n_su2 == 2:
            per_gen = per_gen + Y * n_color
    return sp.simplify(generations * per_gen)


def su3_squared_u1_anomaly(generations: int = 1) -> sp.Expr:
    """Return Σ_{color reps} Y (SU(3)_c² · U(1)_Y mixed anomaly).

    Both color triplets and antitriplets contribute with the same trace
    factor (1/2 δ_ab); the sign in the formula above is absorbed into
    the Y assignment (the u_R^c / d_R^c hypercharges already account for
    being conjugate).
    """

    per_gen: sp.Expr = sp.Integer(0)
    for (_, Y, _, _, n_su2, n_color_triplet) in SM_LEFT_HANDED_CONTENT_PER_GENERATION:
        if n_color_triplet == 1:
            per_gen = per_gen + Y * n_su2
    return sp.simplify(generations * per_gen)


def all_anomalies_cancel(generations: int = 1) -> bool:
    """Return whether all four anomaly conditions cancel for the given N."""

    return all(
        sp.simplify(anomaly(generations)) == 0
        for anomaly in (
            gravitational_anomaly,
            u1_cubed_anomaly,
            su2_squared_u1_anomaly,
            su3_squared_u1_anomaly,
        )
    )


@dataclass(frozen=True)
class AnomalyPolynomialPayload:
    generations: int
    gravitational: sp.Expr
    u1_cubed: sp.Expr
    su2_squared_u1: sp.Expr
    su3_squared_u1: sp.Expr
    all_cancel: bool
    interpretation: str


def anomaly_polynomial_payload(generations: int = 1) -> AnomalyPolynomialPayload:
    grav = gravitational_anomaly(generations)
    y3 = u1_cubed_anomaly(generations)
    su2 = su2_squared_u1_anomaly(generations)
    su3 = su3_squared_u1_anomaly(generations)
    cancel = all_anomalies_cancel(generations)

    if cancel:
        interpretation = (
            f"All four standard SM anomaly conditions cancel for N = "
            f"{generations} generations.  The chiral-16 carries the "
            f"correct charge assignments to make U(1)_Y, SU(2)_L²·U(1)_Y, "
            f"SU(3)_c²·U(1)_Y, and gravitational anomalies vanish.  Since "
            f"cancellation is per-generation, this holds for any N ≥ 0."
        )
    else:
        interpretation = (
            f"Anomalies do NOT cancel for N = {generations}: "
            f"grav={grav}, Y³={y3}, SU(2)²Y={su2}, SU(3)²Y={su3}.  "
            f"Indicates either a hypercharge assignment bug or a "
            f"non-anomaly-free spectrum."
        )

    return AnomalyPolynomialPayload(
        generations=generations,
        gravitational=grav,
        u1_cubed=y3,
        su2_squared_u1=su2,
        su3_squared_u1=su3,
        all_cancel=cancel,
        interpretation=interpretation,
    )
