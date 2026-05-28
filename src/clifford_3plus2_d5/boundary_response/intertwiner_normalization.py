"""V18 algebraic intertwiner normalization gate.

V17 shows that an orthogonal chiral swap normalizes the odd collective mode
and therefore gives primitive ratio ``r = 1/sqrt(5)``, not the CKM ratio
``r = 1``.  V18 tests a different algebraic route: maybe the unnormalized
even-to-odd intertwiner

    T_c |e> = c (|o_1> + ... + |o_5>)

is forced by ``Cl_5`` or odd-shell ``S_5`` structure.

The exact result is a kill theorem.  Odd-shell ``S_5`` makes the intertwiner
unique only up to scale, and

    T_c.T T_c = 5 c^2 P_even.

The spectral lift ``Gamma_c = T_c + T_c.T`` has the expected Casimir only on
the two-dimensional collective subspace:

    Gamma_c^2 = 5 c^2 (P_even + P_odd_collective),

not on the full six-channel shell.  Choosing ``c = 1`` recovers the V15/CKM
primitive ratio, but that is unit component normalization, not an algebraic
consequence of ``S_5`` or the collective spectral lift.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.chiral_boundary_normalization import (
    even_projector,
    odd_collective_projector,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
    parity_preserving_generators,
    primitive_even_vector,
    primitive_odd_sum_vector,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    isotropic_quark_phase_angle,
)


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two exact matrices agree after simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def s5_even_to_odd_intertwiner_basis() -> tuple[sp.Matrix, ...]:
    """Return the odd-target invariant directions for an ``S_5`` intertwiner.

    The domain is the fixed even line.  Compatibility with odd-shell ``S_5``
    therefore means the image of ``|e>`` must lie in the invariant subspace of
    the five odd channels, with no even component.
    """

    identity = sp.eye(SHELL_DIMENSION)
    even_zero_constraint = sp.Matrix([[1, 0, 0, 0, 0, 0]])
    constraints = sp.Matrix.vstack(
        even_zero_constraint,
        *(generator - identity for generator in parity_preserving_generators()),
    )
    return tuple(constraints.nullspace())


def even_odd_intertwiner(scale: sp.Expr = 1) -> sp.Matrix:
    """Return ``T_c`` mapping ``|e>`` to ``c sum_A |o_A>``."""

    c = sp.sympify(scale)
    return c * primitive_odd_sum_vector() * primitive_even_vector().T


def intertwiner_norm_square(scale: sp.Expr = 1) -> sp.Matrix:
    """Return ``T_c.T T_c``."""

    intertwiner = even_odd_intertwiner(scale)
    return sp.simplify(intertwiner.T * intertwiner)


def intertwiner_induced_ratio(scale: sp.Expr = 1) -> sp.Expr:
    """Return the V15 primitive-channel ratio induced by ``|e> + T_c |e>``."""

    return sp.sympify(scale)


def intertwiner_induced_phase(scale: sp.Expr = 1) -> sp.Expr:
    """Return the V15 positive-branch phase induced by ``T_c``."""

    return isotropic_quark_phase_angle(intertwiner_induced_ratio(scale))


def spectral_lift_operator(scale: sp.Expr = 1) -> sp.Matrix:
    """Return the parity-odd collective lift ``Gamma_c = T_c + T_c.T``."""

    intertwiner = even_odd_intertwiner(scale)
    return sp.simplify(intertwiner + intertwiner.T)


def spectral_lift_square(scale: sp.Expr = 1) -> sp.Matrix:
    """Return ``Gamma_c^2``."""

    lift = spectral_lift_operator(scale)
    return sp.simplify(lift * lift)


def spectral_lift_has_full_casimir(scale: sp.Expr = 1) -> bool:
    """Return true if ``Gamma_c^2 = 5I`` on the full shell."""

    c = sp.sympify(scale)
    return _matrix_equal(spectral_lift_square(c), 5 * c**2 * sp.eye(SHELL_DIMENSION))


def intertwiner_is_s5_compatible(scale: sp.Expr = 1) -> bool:
    """Return true when ``T_c`` is invariant under odd-shell ``S_5`` conjugation."""

    intertwiner = even_odd_intertwiner(scale)
    return all(
        _matrix_equal(generator * intertwiner * generator.T, intertwiner)
        for generator in parity_preserving_generators()
    )


def unit_component_normalization_forces_ckm_scale() -> sp.Expr:
    """Return the extra normalization choice needed to recover CKM flatness."""

    return sp.Integer(1)


@dataclass(frozen=True)
class AlgebraicIntertwinerAuditPayload:
    """Verdict payload for the V18 algebraic intertwiner normalization gate."""

    final_verdict: str
    s5_intertwiner_dimension: int
    norm_square: sp.Matrix
    spectral_lift_full_casimir: bool
    scale_free_under_s5: bool
    ckm_scale: sp.Expr
    ckm_phase_recovered_if_scale_one: bool
    unit_component_normalization_required: bool
    interpretation: str


def algebraic_intertwiner_audit_payload() -> AlgebraicIntertwinerAuditPayload:
    """Return the V18 algebraic intertwiner normalization verdict."""

    c = sp.Symbol("c", real=True)
    basis = s5_even_to_odd_intertwiner_basis()
    expected_norm = 5 * c**2 * even_projector()
    expected_lift_square = 5 * c**2 * (even_projector() + odd_collective_projector())
    nonflat_controls = (sp.Rational(1, 2), sp.Integer(2))

    norm_matches = _matrix_equal(intertwiner_norm_square(c), expected_norm)
    lift_square_matches = _matrix_equal(spectral_lift_square(c), expected_lift_square)
    scale_free_under_s5 = (
        all(intertwiner_is_s5_compatible(value) for value in (*nonflat_controls, sp.Integer(1)))
        and all(
            sp.simplify(intertwiner_induced_phase(value) - quark_boundary_phase_angle()) != 0
            for value in nonflat_controls
        )
    )
    ckm_scale = unit_component_normalization_forces_ckm_scale()
    ckm_phase_recovered = sp.simplify(
        intertwiner_induced_phase(ckm_scale) - quark_boundary_phase_angle()
    ) == 0
    full_casimir = spectral_lift_has_full_casimir(1)

    checks_pass = (
        len(basis) == 1
        and _matrix_equal(basis[0], primitive_odd_sum_vector())
        and norm_matches
        and lift_square_matches
        and not full_casimir
        and scale_free_under_s5
        and ckm_phase_recovered
    )

    if checks_pass:
        final_verdict = "ALGEBRAIC_INTERTWINER_FREE_NORM_KILL"
        interpretation = (
            "The S5-compatible even-to-odd intertwiner exists and is unique "
            "up to scale, but its norm is 5*c^2. The spectral lift has the "
            "Casimir only on the even/odd collective subspace, not on the full "
            "six-channel shell. Choosing c=1 recovers the CKM phase, but that "
            "unit component normalization is an extra input rather than a "
            "consequence of S5 or Cl5 shell algebra."
        )
    else:
        final_verdict = "ALGEBRAIC_INTERTWINER_NORMALIZATION_UNRESOLVED"
        interpretation = (
            "The exact S5 intertwiner dimension, norm-square identity, "
            "collective spectral-lift identity, or scale controls failed."
        )

    return AlgebraicIntertwinerAuditPayload(
        final_verdict=final_verdict,
        s5_intertwiner_dimension=len(basis),
        norm_square=intertwiner_norm_square(c),
        spectral_lift_full_casimir=full_casimir,
        scale_free_under_s5=scale_free_under_s5,
        ckm_scale=ckm_scale,
        ckm_phase_recovered_if_scale_one=ckm_phase_recovered,
        unit_component_normalization_required=True,
        interpretation=interpretation,
    )
