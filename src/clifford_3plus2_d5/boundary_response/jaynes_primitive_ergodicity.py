"""V20 Jaynes primitive-ergodicity theorem.

V19 shows that maximizing entropy over the six primitive quark boundary
channels forces the CKM flat ratio ``r = 1``.  V20 rewrites that statement as
a Jaynes density-matrix theorem.

Odd-shell ``S_5`` invariance restricts a diagonal primitive-shell density to

    rho(alpha) = alpha P_even + ((1 - alpha) / 5) P_odd.

If no parity-bias observable is retained, maximizing Shannon-von Neumann
entropy fixes ``alpha = 1/6`` and hence ``rho = I_6 / 6``.  The induced
primitive amplitude ratio is then ``r = 1`` and the V15 phase is
``atan(sqrt(5))``.

The theorem remains conditional on the Jaynes input.  If ``alpha`` is fixed as
an external parity-bias constraint, the ratio remains free.  If entropy is
computed over the compressed macrochannels ``{even, odd_total}``, the maximum
is ``alpha = 1/2``, giving ``r = 1/sqrt(5)`` and phase ``pi/4``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.chiral_boundary_normalization import (
    even_projector,
    odd_projector,
)
from clifford_3plus2_d5.boundary_response.primitive_ergodicity import (
    SHELL_DIMENSION,
    parity_preserving_generators,
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


def jaynes_primitive_density(alpha: sp.Expr) -> sp.Matrix:
    """Return the odd-shell ``S_5`` invariant density ``rho(alpha)``."""

    a = sp.sympify(alpha)
    return sp.simplify(a * even_projector() + ((1 - a) / 5) * odd_projector())


def jaynes_density_trace(alpha: sp.Expr) -> sp.Expr:
    """Return ``Tr rho(alpha)``."""

    return sp.simplify(sp.trace(jaynes_primitive_density(alpha)))


def density_commutes_with_odd_s5(alpha: sp.Expr) -> bool:
    """Return true when ``rho(alpha)`` commutes with odd-shell ``S_5``."""

    density = jaynes_primitive_density(alpha)
    return all(
        _matrix_equal(generator * density, density * generator)
        for generator in parity_preserving_generators()
    )


def jaynes_primitive_entropy(alpha: sp.Expr) -> sp.Expr:
    """Return the six-atom entropy of ``rho(alpha)``."""

    a = sp.sympify(alpha)
    return sp.simplify(-a * sp.log(a) - (1 - a) * sp.log((1 - a) / 5))


def jaynes_entropy_derivative(alpha: sp.Expr) -> sp.Expr:
    """Return ``dS/dalpha`` for the six-atom entropy."""

    a = sp.sympify(alpha)
    return sp.simplify(sp.log((1 - a) / (5 * a)))


def jaynes_entropy_second_derivative(alpha: sp.Expr) -> sp.Expr:
    """Return ``d^2S/dalpha^2`` for the six-atom entropy."""

    a = sp.sympify(alpha)
    return sp.simplify(-1 / (a * (1 - a)))


def jaynes_entropy_max_alpha() -> sp.Expr:
    """Return the unconstrained positive Jaynes maximum."""

    return sp.Rational(1, 6)


def primitive_ratio_from_alpha(alpha: sp.Expr) -> sp.Expr:
    """Return the V15 amplitude ratio induced by ``rho(alpha)``."""

    a = sp.sympify(alpha)
    return sp.sqrt(sp.simplify((1 - a) / (5 * a)))


def phase_from_alpha(alpha: sp.Expr) -> sp.Expr:
    """Return the V15 phase induced by ``rho(alpha)``."""

    return isotropic_quark_phase_angle(primitive_ratio_from_alpha(alpha))


def compressed_macro_entropy_alpha(alpha: sp.Expr) -> sp.Expr:
    """Return entropy after compressing odd channels into one macrochannel."""

    a = sp.sympify(alpha)
    return sp.simplify(-a * sp.log(a) - (1 - a) * sp.log(1 - a))


def compressed_macro_entropy_alpha_derivative(alpha: sp.Expr) -> sp.Expr:
    """Return the derivative of compressed macrochannel entropy."""

    a = sp.sympify(alpha)
    return sp.simplify(sp.log((1 - a) / a))


def compressed_macro_entropy_alpha_max() -> sp.Expr:
    """Return the compressed macrochannel entropy maximum."""

    return sp.Rational(1, 2)


@dataclass(frozen=True)
class JaynesPrimitiveErgodicityAuditPayload:
    """Verdict payload for the V20 Jaynes primitive-ergodicity theorem."""

    final_verdict: str
    maximizing_alpha: sp.Expr
    induced_ratio: sp.Expr
    induced_phase: sp.Expr
    density_uniform: bool
    s5_invariant: bool
    parity_bias_controls_leave_ratio_free: bool
    compressed_max_alpha: sp.Expr
    compressed_max_ratio: sp.Expr
    compressed_max_phase: sp.Expr
    ckm_phase_recovered: bool
    compressed_control_rejected: bool
    remaining_physical_input: str
    interpretation: str


def jaynes_primitive_ergodicity_audit_payload() -> JaynesPrimitiveErgodicityAuditPayload:
    """Return the V20 Jaynes primitive-ergodicity verdict."""

    alpha = sp.Symbol("alpha", positive=True)
    max_alpha = jaynes_entropy_max_alpha()
    induced_ratio = primitive_ratio_from_alpha(max_alpha)
    induced_phase = phase_from_alpha(max_alpha)
    ckm_phase = quark_boundary_phase_angle()
    compressed_alpha = compressed_macro_entropy_alpha_max()
    compressed_ratio = primitive_ratio_from_alpha(compressed_alpha)
    compressed_phase = phase_from_alpha(compressed_alpha)

    density_at_max = jaynes_primitive_density(max_alpha)
    uniform_density = sp.eye(SHELL_DIMENSION) / 6
    density_uniform = _matrix_equal(density_at_max, uniform_density)
    s5_invariant = density_commutes_with_odd_s5(alpha)
    entropy_derivative_matches = sp.simplify(
        sp.diff(jaynes_primitive_entropy(alpha), alpha)
        - jaynes_entropy_derivative(alpha)
    ) == 0
    entropy_second_derivative_matches = sp.simplify(
        sp.diff(jaynes_entropy_derivative(alpha), alpha)
        - jaynes_entropy_second_derivative(alpha)
    ) == 0
    compressed_derivative_matches = sp.simplify(
        sp.diff(compressed_macro_entropy_alpha(alpha), alpha)
        - compressed_macro_entropy_alpha_derivative(alpha)
    ) == 0
    parity_bias_controls_leave_ratio_free = all(
        density_commutes_with_odd_s5(control)
        and sp.simplify(phase_from_alpha(control) - ckm_phase) != 0
        for control in (sp.Rational(1, 2), sp.Rational(1, 3))
    )
    ckm_phase_recovered = sp.simplify(induced_phase - ckm_phase) == 0
    compressed_control_rejected = (
        sp.simplify(compressed_ratio - 1 / sp.sqrt(5)) == 0
        and sp.simplify(compressed_phase - sp.pi / 4) == 0
        and sp.simplify(compressed_phase - ckm_phase) != 0
    )

    checks_pass = (
        sp.simplify(jaynes_density_trace(alpha) - 1) == 0
        and s5_invariant
        and entropy_derivative_matches
        and entropy_second_derivative_matches
        and compressed_derivative_matches
        and sp.simplify(jaynes_entropy_derivative(max_alpha)) == 0
        and sp.simplify(jaynes_entropy_second_derivative(max_alpha) + sp.Rational(36, 5)) == 0
        and density_uniform
        and sp.simplify(induced_ratio - 1) == 0
        and ckm_phase_recovered
        and parity_bias_controls_leave_ratio_free
        and compressed_control_rejected
    )

    if checks_pass:
        final_verdict = "JAYNES_PRIMITIVE_ERGODICITY_THEOREM_PASS"
        interpretation = (
            "Odd-shell S5 invariance reduces the primitive density to one "
            "parameter alpha. With no retained parity-bias observable, Jaynes "
            "max entropy fixes alpha=1/6, the uniform six-channel density, "
            "and primitive ratio r=1, recovering atan(sqrt(5)). Fixed "
            "parity-bias and compressed macrochannel controls show the "
            "Jaynes primitive-shell input is load-bearing."
        )
    else:
        final_verdict = "JAYNES_PRIMITIVE_ERGODICITY_THEOREM_KILL"
        interpretation = (
            "The S5-invariant density, entropy derivatives, maximum, induced "
            "CKM phase, or controls failed."
        )

    return JaynesPrimitiveErgodicityAuditPayload(
        final_verdict=final_verdict,
        maximizing_alpha=max_alpha,
        induced_ratio=induced_ratio,
        induced_phase=induced_phase,
        density_uniform=density_uniform,
        s5_invariant=s5_invariant,
        parity_bias_controls_leave_ratio_free=parity_bias_controls_leave_ratio_free,
        compressed_max_alpha=compressed_alpha,
        compressed_max_ratio=compressed_ratio,
        compressed_max_phase=compressed_phase,
        ckm_phase_recovered=ckm_phase_recovered,
        compressed_control_rejected=compressed_control_rejected,
        remaining_physical_input=(
            "Jaynes microcanonical assumption: the unresolved primitive shell "
            "retains no parity-bias observable beyond odd-shell S5 symmetry."
        ),
        interpretation=interpretation,
    )
