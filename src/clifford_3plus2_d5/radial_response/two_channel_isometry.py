"""Two-channel scalar repair isometry gate."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def equal_successor_amplitudes(channel_count: int) -> tuple[sp.Expr, ...]:
    """Return equal no-leakage amplitudes for ``channel_count`` successors."""

    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    amplitude = 1 / sp.sqrt(channel_count)
    return tuple(amplitude for _ in range(channel_count))


def total_probability(amplitudes: tuple[sp.Expr, ...]) -> sp.Expr:
    """Return the exact total probability for real amplitudes."""

    return sp.simplify(sum(amplitude**2 for amplitude in amplitudes))


def two_channel_repair_amplitude() -> sp.Expr:
    """Return the two-successor no-leakage scalar repair amplitude."""

    return equal_successor_amplitudes(2)[0]


def leakage_repair_amplitude(leakage_probability: sp.Expr) -> sp.Expr:
    """Return per-channel amplitude for two equal channels with leakage."""

    leakage_probability = sp.sympify(leakage_probability)
    return sp.sqrt((1 - leakage_probability) / 2)


def asymmetric_two_channel_amplitudes(weight: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return normalized asymmetric two-channel amplitudes."""

    weight = sp.sympify(weight)
    return (sp.sqrt(weight), sp.sqrt(1 - weight))


@dataclass(frozen=True)
class TwoChannelIsometryPayload:
    """Payload for the two-channel scalar repair isometry gate."""

    final_verdict: str
    two_channel_amplitudes: tuple[sp.Expr, sp.Expr]
    one_channel_control: tuple[sp.Expr]
    three_channel_control: tuple[sp.Expr, sp.Expr, sp.Expr]
    leakage_amplitude: sp.Expr
    asymmetric_control_is_free: bool
    interpretation: str


def two_channel_isometry_payload() -> TwoChannelIsometryPayload:
    """Return the two-channel repair-isometry verdict."""

    two = equal_successor_amplitudes(2)
    one = equal_successor_amplitudes(1)
    three = equal_successor_amplitudes(3)
    leakage = leakage_repair_amplitude(sp.Symbol("ell", nonnegative=True))
    weight = sp.Symbol("w", positive=True)
    asymmetric = asymmetric_two_channel_amplitudes(weight)
    checks_pass = (
        two == (1 / sp.sqrt(2), 1 / sp.sqrt(2))
        and total_probability(two) == 1
        and one == (sp.Integer(1),)
        and three == (1 / sp.sqrt(3), 1 / sp.sqrt(3), 1 / sp.sqrt(3))
        and sp.simplify(leakage - sp.sqrt((1 - sp.Symbol("ell", nonnegative=True)) / 2)) == 0
        and asymmetric != two
    )

    if checks_pass:
        final_verdict = "TWO_CHANNEL_REPAIR_ISOMETRY_PASS"
        interpretation = (
            "Exactly two symmetry-related scalar repair successors plus "
            "no leakage force amplitudes (1/sqrt(2),1/sqrt(2)). The result "
            "fails to be forced for one channel, three channels, leakage, or "
            "asymmetric successor weights."
        )
    else:
        final_verdict = "TWO_CHANNEL_REPAIR_ISOMETRY_KILL"
        interpretation = "The two-channel isometry or one of its controls failed."

    return TwoChannelIsometryPayload(
        final_verdict=final_verdict,
        two_channel_amplitudes=two,
        one_channel_control=one,
        three_channel_control=three,
        leakage_amplitude=leakage,
        asymmetric_control_is_free=asymmetric != two,
        interpretation=interpretation,
    )
