"""V40 gauge-invariant origin of the V39 radial stabilizer.

V39 imports a Mexican-hat radial term to close the free-BB radial no-go.  V40
does not derive the Higgs sector microscopically.  It proves the more modest
Landau statement: once the selector amplitude is identified with the norm of a
site-local electroweak Higgs doublet, local gauge invariance, quartic
minimality, boundedness, and a broken-phase condition force the radial form

    V(Phi) = lambda (|Phi|^2 - v^2)^2 + constant.

The selector potential used in V39 is then the gauge-fixed pullback
``Phi(r) = (0, r)`` of the existing spacetime-QCA Higgs potential.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.boundary_response.vacuum_selector_higgs_stabilizer import (
    DEFAULT_SELECTOR_QUARTIC,
    DEFAULT_SELECTOR_VEV_SQUARED,
    RADIAL_STABILIZER_AUDIT_RADII,
    mexican_hat_radial_potential,
    higgs_backreaction_radial_stabilizer_audit_payload,
)
from clifford_3plus2_d5.spacetime_qca.jax_higgs import jax_higgs_potential_density

HiggsComponent = Literal["neutral", "charged"]

DEFAULT_BROKEN_QUADRATIC = -2.0
DEFAULT_BROKEN_QUARTIC = 1.0
DEFAULT_LANDAU_CONSTANT = 0.0
HIGGS_ORIGIN_TOLERANCE = 1.0e-8

REMAINING_DECLARED_INPUTS_AFTER_HIGGS_RADIAL_ORIGIN = (
    "higgs_backreaction_sector_enters_broken_quartic_phase",
)


def quartic_higgs_landau_potential(
    rho: float,
    *,
    quadratic: float = DEFAULT_BROKEN_QUADRATIC,
    quartic: float = DEFAULT_BROKEN_QUARTIC,
    constant: float = DEFAULT_LANDAU_CONSTANT,
) -> float:
    """Return the minimal gauge-invariant quartic ``c + a rho + b rho^2``."""

    return constant + quadratic * rho + quartic * rho * rho


def quartic_is_bounded_below(*, quartic: float = DEFAULT_BROKEN_QUARTIC) -> bool:
    """Return true when the quartic coefficient bounds the potential below."""

    return quartic > 0.0


def quartic_has_broken_phase(
    *,
    quadratic: float = DEFAULT_BROKEN_QUADRATIC,
    quartic: float = DEFAULT_BROKEN_QUARTIC,
) -> bool:
    """Return true when the bounded quartic has a nonzero radial minimum."""

    return quartic_is_bounded_below(quartic=quartic) and quadratic < 0.0


def completed_square_vev_squared(
    *,
    quadratic: float = DEFAULT_BROKEN_QUADRATIC,
    quartic: float = DEFAULT_BROKEN_QUARTIC,
) -> float:
    """Return ``v^2 = -a/(2b)`` for ``c + a rho + b rho^2``."""

    if quartic == 0.0:
        raise ValueError("quartic coefficient must be nonzero")
    return -quadratic / (2.0 * quartic)


def completed_square_constant(
    *,
    quadratic: float = DEFAULT_BROKEN_QUADRATIC,
    quartic: float = DEFAULT_BROKEN_QUARTIC,
    constant: float = DEFAULT_LANDAU_CONSTANT,
) -> float:
    """Return the additive constant in ``b(rho-v^2)^2 + constant_shift``."""

    vev_squared = completed_square_vev_squared(quadratic=quadratic, quartic=quartic)
    return constant - quartic * vev_squared * vev_squared


def completed_square_higgs_potential(
    rho: float,
    *,
    quadratic: float = DEFAULT_BROKEN_QUADRATIC,
    quartic: float = DEFAULT_BROKEN_QUARTIC,
    constant: float = DEFAULT_LANDAU_CONSTANT,
) -> float:
    """Return the completed-square form of the gauge-invariant quartic."""

    vev_squared = completed_square_vev_squared(quadratic=quadratic, quartic=quartic)
    return quartic * (rho - vev_squared) ** 2 + completed_square_constant(
        quadratic=quadratic,
        quartic=quartic,
        constant=constant,
    )


def square_completion_matches(
    *,
    rho_samples: tuple[float, ...] = tuple(radius * radius for radius in RADIAL_STABILIZER_AUDIT_RADII),
    quadratic: float = DEFAULT_BROKEN_QUADRATIC,
    quartic: float = DEFAULT_BROKEN_QUARTIC,
    constant: float = DEFAULT_LANDAU_CONSTANT,
    tolerance: float = HIGGS_ORIGIN_TOLERANCE,
) -> bool:
    """Return true when the polynomial and completed-square forms agree."""

    return all(
        abs(
            quartic_higgs_landau_potential(
                rho,
                quadratic=quadratic,
                quartic=quartic,
                constant=constant,
            )
            - completed_square_higgs_potential(
                rho,
                quadratic=quadratic,
                quartic=quartic,
                constant=constant,
            )
        )
        <= tolerance
        for rho in rho_samples
    )


def radial_term_from_broken_quartic_matches_v39(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    tolerance: float = HIGGS_ORIGIN_TOLERANCE,
) -> bool:
    """Return true when the default broken quartic has V39's radial shape."""

    vev_squared = completed_square_vev_squared()
    quartic = DEFAULT_BROKEN_QUARTIC
    constant_shift = completed_square_constant()
    return all(
        abs(
            (
                quartic_higgs_landau_potential(radius * radius)
                - constant_shift
            )
            - mexican_hat_radial_potential(
                radius,
                vev_squared=vev_squared,
                quartic=quartic,
            )
        )
        <= tolerance
        for radius in radii
    )


def gauge_fixed_higgs_doublet(
    radius: float,
    *,
    component: HiggsComponent = "neutral",
    dtype: object = jnp.complex64,
) -> jnp.ndarray:
    """Return a one-site Higgs doublet representative with norm ``radius``."""

    if component == "neutral":
        value = (0.0 + 0.0j, complex(radius, 0.0))
    elif component == "charged":
        value = (complex(radius, 0.0), 0.0 + 0.0j)
    else:
        raise ValueError(f"unknown Higgs component {component!r}")
    return jnp.asarray(value, dtype=dtype).reshape((1, 1, 1, 2))


def spacetime_higgs_potential_pullback(
    radius: float,
    *,
    component: HiggsComponent = "neutral",
    vev_squared: float = DEFAULT_SELECTOR_VEV_SQUARED,
    quartic: float = DEFAULT_SELECTOR_QUARTIC,
) -> float:
    """Return the spacetime-QCA Higgs potential on ``Phi(r)``."""

    potential = jax_higgs_potential_density(
        gauge_fixed_higgs_doublet(radius, component=component),
        vev_squared=vev_squared,
        quartic=quartic,
    )
    return float(np.asarray(potential)[0, 0, 0])


def spacetime_pullback_matches_v39_radial_term(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    component: HiggsComponent = "neutral",
    tolerance: float = HIGGS_ORIGIN_TOLERANCE,
) -> bool:
    """Return true when the spacetime Higgs potential pulls back to V39."""

    return all(
        abs(
            spacetime_higgs_potential_pullback(radius, component=component)
            - mexican_hat_radial_potential(radius)
        )
        <= tolerance
        for radius in radii
    )


def charged_and_neutral_representatives_are_degenerate(
    *,
    radii: tuple[float, ...] = RADIAL_STABILIZER_AUDIT_RADII,
    tolerance: float = HIGGS_ORIGIN_TOLERANCE,
) -> bool:
    """Return true when same-norm Higgs representatives have equal potential."""

    return all(
        abs(
            spacetime_higgs_potential_pullback(radius, component="charged")
            - spacetime_higgs_potential_pullback(radius, component="neutral")
        )
        <= tolerance
        for radius in radii
    )


def doublet_quadratic_is_gauge_radial(*, charged_coefficient: float, neutral_coefficient: float) -> bool:
    """Return true when a diagonal doublet quadratic depends only on ``|Phi|^2``."""

    return abs(charged_coefficient - neutral_coefficient) <= HIGGS_ORIGIN_TOLERANCE


def anisotropic_doublet_term_fails_radial_criterion() -> bool:
    """Return true for a sample doublet term that distinguishes components."""

    return not doublet_quadratic_is_gauge_radial(
        charged_coefficient=1.0,
        neutral_coefficient=2.0,
    )


def v39_recovered() -> bool:
    """Return true when the V39 radial-stabilizer gate still passes."""

    return (
        higgs_backreaction_radial_stabilizer_audit_payload().final_verdict
        == "HIGGS_BACKREACTION_RADIAL_STABILIZER_PASS"
    )


@dataclass(frozen=True)
class HiggsRadialLandauOriginAuditPayload:
    """Verdict payload for the V40 Higgs radial-origin theorem gate."""

    final_verdict: str
    quadratic_coefficient: float
    quartic_coefficient: float
    vev_squared: float
    completed_square_constant: float
    bounded_below: bool
    broken_phase: bool
    square_completion: bool
    radial_term_matches_v39: bool
    neutral_pullback_matches_v39: bool
    charged_pullback_matches_v39: bool
    same_norm_degeneracy: bool
    positive_quadratic_control_unbroken: bool
    negative_quartic_control_unbounded: bool
    anisotropic_control_rejected: bool
    v39_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


@cache
def higgs_radial_landau_origin_audit_payload() -> HiggsRadialLandauOriginAuditPayload:
    """Return the V40 gauge-invariant radial-origin verdict."""

    vev_squared = completed_square_vev_squared()
    constant_shift = completed_square_constant()
    bounded = quartic_is_bounded_below()
    broken = quartic_has_broken_phase()
    square_completion = square_completion_matches()
    radial_match = radial_term_from_broken_quartic_matches_v39()
    neutral_pullback = spacetime_pullback_matches_v39_radial_term(component="neutral")
    charged_pullback = spacetime_pullback_matches_v39_radial_term(component="charged")
    same_norm = charged_and_neutral_representatives_are_degenerate()
    positive_quadratic_unbroken = not quartic_has_broken_phase(quadratic=1.0, quartic=1.0)
    negative_quartic_unbounded = not quartic_is_bounded_below(quartic=-1.0)
    anisotropic_rejected = anisotropic_doublet_term_fails_radial_criterion()
    v39 = v39_recovered()

    prerequisites_pass = (
        bounded
        and broken
        and square_completion
        and radial_match
        and neutral_pullback
        and charged_pullback
        and same_norm
        and positive_quadratic_unbroken
        and negative_quartic_unbounded
        and anisotropic_rejected
        and v39
    )

    if prerequisites_pass:
        final_verdict = "HIGGS_RADIAL_LANDAU_UNIQUENESS_PASS"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_HIGGS_RADIAL_ORIGIN
        interpretation = (
            "Gauge invariance reduces the local Higgs quartic to a function "
            "of rho=|Phi|^2.  Boundedness plus the broken-phase sign completes "
            "the square to the Mexican-hat radial form, and the V39 selector "
            "term is exactly the gauge-fixed spacetime-QCA Higgs pullback."
        )
    else:
        final_verdict = "HIGGS_RADIAL_LANDAU_UNIQUENESS_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_HIGGS_RADIAL_ORIGIN
        interpretation = (
            "The gauge-invariant quartic did not uniquely reproduce the V39 "
            "radial term or failed its sign, pullback, or anisotropy controls."
        )

    return HiggsRadialLandauOriginAuditPayload(
        final_verdict=final_verdict,
        quadratic_coefficient=DEFAULT_BROKEN_QUADRATIC,
        quartic_coefficient=DEFAULT_BROKEN_QUARTIC,
        vev_squared=vev_squared,
        completed_square_constant=constant_shift,
        bounded_below=bounded,
        broken_phase=broken,
        square_completion=square_completion,
        radial_term_matches_v39=radial_match,
        neutral_pullback_matches_v39=neutral_pullback,
        charged_pullback_matches_v39=charged_pullback,
        same_norm_degeneracy=same_norm,
        positive_quadratic_control_unbroken=positive_quadratic_unbroken,
        negative_quartic_control_unbounded=negative_quartic_unbounded,
        anisotropic_control_rejected=anisotropic_rejected,
        v39_recovered=v39,
        remaining_declared_inputs=remaining_inputs,
        interpretation=interpretation,
    )
