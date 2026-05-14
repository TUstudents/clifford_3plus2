"""Sidecar exact diagnostics for a 1D spatial transfer prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd, lcm

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity


@dataclass(frozen=True)
class SpatialTransferRule1D:
    name: str
    alpha_modes: tuple[int, ...]
    eta_modes: tuple[int, ...]
    alpha_winding: int
    eta_winding: int
    period: int

    @property
    def mode_count(self) -> int:
        return len(self.alpha_modes) + len(self.eta_modes)

    @property
    def dimension(self) -> int:
        return 2 * self.mode_count

    @property
    def locality_radius(self) -> int:
        return max(abs(self.alpha_winding), abs(self.eta_winding))


@dataclass(frozen=True)
class SpatialOrientationOrbit:
    alpha_sign: int
    eta_sign: int
    transport_allowed: bool


@dataclass(frozen=True)
class Spatial1DAlphaCertificate:
    rule_name: str
    period: int
    alpha_winding: int
    eta_winding: int
    winding_gcd: int
    winding_lcm: int
    locality_radius: int
    sample_count: int
    transfer_unitary_on_samples: bool
    alpha_projector_rank: int
    eta_projector_rank: int
    coarse_6_4_band_split: bool
    orientation_choices_before_transport: int
    orientation_choices_after_transport: int
    orientation_orbits: tuple[SpatialOrientationOrbit, ...]
    sign_coupled_to_global_pm: bool
    strict_bridge_candidates: int
    route_label: str
    load_bearing_qca_bridge: bool = False


def spatial_alpha_prototype() -> SpatialTransferRule1D:
    """Return the first root-of-unity 1D sidecar prototype.

    The prototype encodes the Floquet-alpha phases as spatial winding data over
    one period-12 Brillouin cycle: alpha advances by 4 units and eta by 3.
    It is a diagnostic transfer model, not a finite-depth QCA theorem.
    """

    return SpatialTransferRule1D(
        name="spatial_1d_alpha_period_12_winding_4_3",
        alpha_modes=(0, 1, 2),
        eta_modes=(3, 4),
        alpha_winding=4,
        eta_winding=3,
        period=12,
    )


def root_of_unity(period: int, sample: int) -> sp.Expr:
    return sp.exp(2 * sp.pi * sp.I * sp.Rational(sample, period))


def transfer_matrix_at_root(rule: SpatialTransferRule1D, sample: int) -> sp.Matrix:
    zeta = root_of_unity(rule.period, sample)
    entries: list[sp.Expr] = []
    for mode in range(rule.mode_count):
        winding = rule.alpha_winding if mode in rule.alpha_modes else rule.eta_winding
        entries.append(sp.simplify(zeta**winding))
    return sp.diag(*entries, *entries)


def alpha_eta_projectors(rule: SpatialTransferRule1D) -> tuple[sp.Matrix, sp.Matrix]:
    alpha_entries = [1 if mode in rule.alpha_modes else 0 for mode in range(rule.mode_count)]
    eta_entries = [1 if mode in rule.eta_modes else 0 for mode in range(rule.mode_count)]
    return (
        sp.diag(*alpha_entries, *alpha_entries),
        sp.diag(*eta_entries, *eta_entries),
    )


def transfer_is_unitary_on_samples(rule: SpatialTransferRule1D) -> bool:
    one = identity(rule.dimension)
    for sample in range(rule.period):
        transfer = transfer_matrix_at_root(rule, sample)
        if sp.simplify(transfer.conjugate().T * transfer - one) != sp.zeros(rule.dimension):
            return False
    return True


def spatial_orientation_orbits(rule: SpatialTransferRule1D) -> tuple[SpatialOrientationOrbit, ...]:
    """Return block-sign choices allowed by the shared spatial orientation.

    This sidecar tests the concrete Route-2 hypothesis: coprime alpha/eta
    windings over one period-12 cycle impose one shared orientation, so the
    independent block signs collapse from four choices to global +/-.
    """

    coprime_common_cycle = (
        gcd(abs(rule.alpha_winding), abs(rule.eta_winding)) == 1
        and lcm(abs(rule.alpha_winding), abs(rule.eta_winding)) == rule.period
    )
    orbits = []
    for alpha_sign, eta_sign in product((1, -1), repeat=2):
        allowed = alpha_sign == eta_sign if coprime_common_cycle else True
        orbits.append(
            SpatialOrientationOrbit(
                alpha_sign=alpha_sign,
                eta_sign=eta_sign,
                transport_allowed=allowed,
            )
        )
    return tuple(orbits)


def spatial_1d_alpha_certificate(rule: SpatialTransferRule1D | None = None) -> Spatial1DAlphaCertificate:
    rule = rule if rule is not None else spatial_alpha_prototype()
    alpha_projector, eta_projector = alpha_eta_projectors(rule)
    orientation_orbits = spatial_orientation_orbits(rule)
    allowed_count = sum(orbit.transport_allowed for orbit in orientation_orbits)
    sign_coupled = allowed_count == 2 and all(
        orbit.alpha_sign == orbit.eta_sign
        for orbit in orientation_orbits
        if orbit.transport_allowed
    )
    coarse_split = alpha_projector.rank() == 6 and eta_projector.rank() == 4
    unitary = transfer_is_unitary_on_samples(rule)

    if not unitary:
        route_label = "spatial_not_unitary"
    elif not coarse_split:
        route_label = "spatial_no_coarse_6_4_band_split"
    elif sign_coupled:
        route_label = "spatial_signs_coupled_to_global_pm"
    else:
        route_label = "spatial_signs_uncoupled"

    return Spatial1DAlphaCertificate(
        rule_name=rule.name,
        period=rule.period,
        alpha_winding=rule.alpha_winding,
        eta_winding=rule.eta_winding,
        winding_gcd=gcd(abs(rule.alpha_winding), abs(rule.eta_winding)),
        winding_lcm=lcm(abs(rule.alpha_winding), abs(rule.eta_winding)),
        locality_radius=rule.locality_radius,
        sample_count=rule.period,
        transfer_unitary_on_samples=unitary,
        alpha_projector_rank=alpha_projector.rank(),
        eta_projector_rank=eta_projector.rank(),
        coarse_6_4_band_split=coarse_split,
        orientation_choices_before_transport=len(orientation_orbits),
        orientation_choices_after_transport=allowed_count,
        orientation_orbits=orientation_orbits,
        sign_coupled_to_global_pm=sign_coupled,
        strict_bridge_candidates=0,
        route_label=route_label,
    )
