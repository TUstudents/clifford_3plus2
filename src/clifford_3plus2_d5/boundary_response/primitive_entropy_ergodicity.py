"""V19 primitive max-entropy ergodicity gate.

V15 proves that the CKM phase is equivalent to the primitive even/odd ratio
``r = 1``.  V16-V18 prove that the obvious symmetry, chiral-swap, and
intertwiner routes do not force that ratio.

V19 tests the remaining statistical route: maximize Shannon entropy over the
six primitive boundary channels themselves.  For

    psi(r) = (|e> + r sum_A |o_A>) / sqrt(1 + 5 r^2),

the primitive-channel probabilities are

    p_e = 1 / (1 + 5 r^2),       p_o = r^2 / (1 + 5 r^2).

Maximizing entropy over these six microscopic atoms forces the uniform
distribution, hence ``r = 1`` and ``delta_q = atan(sqrt(5))``.

The control is load-bearing: if entropy is computed over compressed
macrochannels ``{even, odd_total}``, the maximum is instead
``r = 1/sqrt(5)``, giving the V17 phase ``pi/4``.  Therefore V19 is a
conditional pass for primitive-channel max entropy, not a derivation from BCC
geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    isotropic_quark_phase_angle,
)


def primitive_entropy_probabilities(ratio: sp.Expr) -> tuple[sp.Expr, ...]:
    """Return the six primitive-channel probabilities for ratio ``r``."""

    r = sp.sympify(ratio)
    denominator = 1 + 5 * r**2
    even_probability = sp.simplify(1 / denominator)
    odd_probability = sp.simplify(r**2 / denominator)
    return (even_probability, *(odd_probability for _ in range(5)))


def primitive_shannon_entropy(ratio: sp.Expr) -> sp.Expr:
    """Return the exact six-channel Shannon entropy ``H_6(r)``."""

    r = sp.sympify(ratio)
    denominator = 1 + 5 * r**2
    return sp.simplify(
        sp.log(denominator) - (5 * r**2 / denominator) * sp.log(r**2)
    )


def primitive_entropy_derivative(ratio: sp.Expr) -> sp.Expr:
    """Return ``dH_6/dr`` in closed form."""

    r = sp.sympify(ratio)
    denominator = 1 + 5 * r**2
    return sp.simplify(-10 * r * sp.log(r**2) / denominator**2)


def primitive_entropy_max_ratio() -> sp.Expr:
    """Return the positive-ratio primitive entropy maximum."""

    return sp.Integer(1)


def compressed_macro_probabilities(ratio: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return probabilities for the compressed ``{even, odd_total}`` partition."""

    r = sp.sympify(ratio)
    denominator = 1 + 5 * r**2
    return (sp.simplify(1 / denominator), sp.simplify(5 * r**2 / denominator))


def compressed_macro_entropy(ratio: sp.Expr) -> sp.Expr:
    """Return Shannon entropy after compressing the five odd atoms."""

    r = sp.sympify(ratio)
    denominator = 1 + 5 * r**2
    return sp.simplify(
        sp.log(denominator) - (5 * r**2 / denominator) * sp.log(5 * r**2)
    )


def compressed_macro_entropy_derivative(ratio: sp.Expr) -> sp.Expr:
    """Return the exact derivative of the compressed macrochannel entropy."""

    r = sp.sympify(ratio)
    denominator = 1 + 5 * r**2
    return sp.simplify(-10 * r * sp.log(5 * r**2) / denominator**2)


def compressed_macro_entropy_max_ratio() -> sp.Expr:
    """Return the positive-ratio compressed entropy maximum."""

    return 1 / sp.sqrt(5)


@dataclass(frozen=True)
class PrimitiveEntropyErgodicityAuditPayload:
    """Verdict payload for the V19 primitive max-entropy gate."""

    final_verdict: str
    primitive_max_ratio: sp.Expr
    primitive_max_phase: sp.Expr
    compressed_max_ratio: sp.Expr
    compressed_max_phase: sp.Expr
    ckm_phase_recovered: bool
    compressed_control_rejected: bool
    entropy_partition_is_extra_principle: bool
    interpretation: str


def primitive_entropy_ergodicity_audit_payload() -> PrimitiveEntropyErgodicityAuditPayload:
    """Return the V19 primitive max-entropy verdict."""

    r = sp.Symbol("r", positive=True)
    primitive_max = primitive_entropy_max_ratio()
    compressed_max = compressed_macro_entropy_max_ratio()
    primitive_phase = isotropic_quark_phase_angle(primitive_max)
    compressed_phase = isotropic_quark_phase_angle(compressed_max)
    ckm_phase = quark_boundary_phase_angle()

    primitive_probabilities = primitive_entropy_probabilities(primitive_max)
    compressed_probabilities = compressed_macro_probabilities(compressed_max)
    primitive_derivative_matches = sp.simplify(
        sp.diff(primitive_shannon_entropy(r), r) - primitive_entropy_derivative(r)
    ) == 0
    compressed_derivative_matches = sp.simplify(
        sp.diff(compressed_macro_entropy(r), r) - compressed_macro_entropy_derivative(r)
    ) == 0
    primitive_uniform = all(
        sp.simplify(probability - sp.Rational(1, 6)) == 0
        for probability in primitive_probabilities
    )
    compressed_uniform = all(
        sp.simplify(probability - sp.Rational(1, 2)) == 0
        for probability in compressed_probabilities
    )
    ckm_phase_recovered = sp.simplify(primitive_phase - ckm_phase) == 0
    compressed_control_rejected = (
        sp.simplify(compressed_phase - sp.pi / 4) == 0
        and sp.simplify(compressed_phase - ckm_phase) != 0
    )

    checks_pass = (
        sp.simplify(sum(primitive_entropy_probabilities(r)) - 1) == 0
        and sp.simplify(sum(compressed_macro_probabilities(r)) - 1) == 0
        and primitive_derivative_matches
        and compressed_derivative_matches
        and primitive_uniform
        and compressed_uniform
        and ckm_phase_recovered
        and compressed_control_rejected
    )

    if checks_pass:
        final_verdict = "MAX_ENTROPY_PRIMITIVE_ERGODICITY_CONDITIONAL_PASS"
        interpretation = (
            "Maximizing Shannon entropy over the six primitive channels gives "
            "the uniform distribution and fixes r=1, hence the CKM phase "
            "atan(sqrt(5)). The compressed two-channel entropy control gives "
            "r=1/sqrt(5) and phase pi/4, so the primitive-channel partition is "
            "a load-bearing extra principle rather than a consequence of BCC "
            "geometry alone."
        )
    else:
        final_verdict = "MAX_ENTROPY_PRIMITIVE_ERGODICITY_KILL"
        interpretation = (
            "The primitive entropy derivative, normalization, CKM phase "
            "recovery, or compressed-channel control failed."
        )

    return PrimitiveEntropyErgodicityAuditPayload(
        final_verdict=final_verdict,
        primitive_max_ratio=primitive_max,
        primitive_max_phase=primitive_phase,
        compressed_max_ratio=compressed_max,
        compressed_max_phase=compressed_phase,
        ckm_phase_recovered=ckm_phase_recovered,
        compressed_control_rejected=compressed_control_rejected,
        entropy_partition_is_extra_principle=True,
        interpretation=interpretation,
    )
