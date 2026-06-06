"""R12 radial pole/residue rigidity no-go gate.

R1-R11 fix the Schur/Feshbach form, the inherited silver transfer root, and the
two-channel scalar repair head data.  This gate checks whether those data force
the radial spectral measure.  They do not: two baths with the same two
triality-head couplings and the same inherited transfer data can have different
self-energy poles and residues.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.radial_response.green_function import feshbach_self_energy
from clifford_3plus2_d5.radial_response.s3_scalar_completeness import (
    allowed_scalar_s3_successor_labels,
)
from clifford_3plus2_d5.radial_response.silver_transfer_inheritance import (
    inherited_eta,
    inherited_intensity_ratio,
    inherited_transfer_root,
    radial_silver_transfer_inheritance_pass,
)
from clifford_3plus2_d5.radial_response.two_channel_isometry import (
    total_probability,
    two_channel_repair_amplitude,
)


@dataclass(frozen=True)
class TrialityBath:
    """One exact unresolved radial bath with triality-head coupling data."""

    label: str
    h_q: sp.Matrix
    coupling: sp.Matrix


@dataclass(frozen=True)
class PoleResidueRigidityPayload:
    """Payload for the R12 radial pole/residue rigidity no-go."""

    final_verdict: str
    inherited_root: sp.Expr
    eta: sp.Expr
    intensity_ratio: sp.Expr
    allowed_successors: tuple[str, str]
    head_amplitudes: tuple[sp.Expr, sp.Expr]
    one_level_self_energy: sp.Expr
    two_level_tail_self_energy: sp.Expr
    self_energies_differ: bool
    one_level_poles: tuple[sp.Expr, ...]
    two_level_tail_poles: tuple[sp.Expr, ...]
    poles_differ: bool
    one_level_residues: tuple[sp.Expr, ...]
    two_level_tail_residues: tuple[sp.Expr, ...]
    residues_differ: bool
    coupling_norms_match: bool
    inherited_transfer_unchanged: bool
    rigidity_forced: bool
    interpretation: str


def two_triality_head_amplitudes() -> tuple[sp.Expr, sp.Expr]:
    """Return the inherited two-channel scalar repair head amplitudes."""

    amplitude = two_channel_repair_amplitude()
    return (amplitude, amplitude)


def one_level_triality_bath() -> TrialityBath:
    """Return a symmetric one-level bath over the two triality heads."""

    amplitude = two_channel_repair_amplitude()
    return TrialityBath(
        label="one_level_triality_heads",
        h_q=sp.zeros(2, 2),
        coupling=sp.Matrix([amplitude, amplitude]),
    )


def two_level_tail_triality_bath() -> TrialityBath:
    """Return a symmetric two-level radial tail attached to each triality head."""

    amplitude = two_channel_repair_amplitude()
    return TrialityBath(
        label="two_level_triality_tails",
        h_q=sp.Matrix(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ]
        ),
        coupling=sp.Matrix([amplitude, amplitude, 0, 0]),
    )


def bath_coupling_norm(bath: TrialityBath) -> sp.Expr:
    """Return the exact coupling norm squared for a bath."""

    return sp.simplify((bath.coupling.T * bath.coupling)[0, 0])


def bath_has_same_triality_head_data(bath: TrialityBath) -> bool:
    """Return whether the bath preserves the two normalized triality head couplings."""

    amplitudes = two_triality_head_amplitudes()
    return (
        bath.coupling.rows >= 2
        and sp.simplify(bath.coupling[0, 0] - amplitudes[0]) == 0
        and sp.simplify(bath.coupling[1, 0] - amplitudes[1]) == 0
        and sp.simplify(bath_coupling_norm(bath) - 1) == 0
    )


def bath_self_energy(bath: TrialityBath, z: sp.Symbol | None = None) -> sp.Expr:
    """Return the scalar Schur self-energy for a triality bath."""

    z_symbol = sp.Symbol("z") if z is None else z
    sigma = feshbach_self_energy(z_symbol, bath.h_q, bath.coupling)
    return sp.factor(sigma[0, 0])


def self_energy_poles(self_energy: sp.Expr, z: sp.Symbol | None = None) -> tuple[sp.Expr, ...]:
    """Return exact self-energy pole locations."""

    z_symbol = sp.Symbol("z") if z is None else z
    denominator = sp.denom(sp.together(self_energy))
    poles = sp.solve(sp.Eq(denominator, 0), z_symbol)
    return tuple(sorted((sp.simplify(pole) for pole in poles), key=lambda pole: float(sp.N(pole))))


def self_energy_residues(
    self_energy: sp.Expr,
    z: sp.Symbol | None = None,
) -> tuple[sp.Expr, ...]:
    """Return residues at the exact self-energy poles."""

    z_symbol = sp.Symbol("z") if z is None else z
    return tuple(
        sp.simplify(sp.residue(self_energy, z_symbol, pole))
        for pole in self_energy_poles(self_energy, z_symbol)
    )


def pole_residue_rigidity_no_go_pass() -> bool:
    """Return whether two admissible baths kill radial pole/residue rigidity."""

    one = one_level_triality_bath()
    tail = two_level_tail_triality_bath()
    one_sigma = bath_self_energy(one)
    tail_sigma = bath_self_energy(tail)
    one_poles = self_energy_poles(one_sigma)
    tail_poles = self_energy_poles(tail_sigma)
    one_residues = self_energy_residues(one_sigma)
    tail_residues = self_energy_residues(tail_sigma)
    return (
        radial_silver_transfer_inheritance_pass()
        and allowed_scalar_s3_successor_labels() == ("triality_plus", "triality_minus")
        and bath_has_same_triality_head_data(one)
        and bath_has_same_triality_head_data(tail)
        and sp.simplify(one_sigma - tail_sigma) != 0
        and one_poles != tail_poles
        and one_residues != tail_residues
    )


def pole_residue_rigidity_payload() -> PoleResidueRigidityPayload:
    """Return the R12 radial pole/residue rigidity no-go verdict."""

    one = one_level_triality_bath()
    tail = two_level_tail_triality_bath()
    one_sigma = bath_self_energy(one)
    tail_sigma = bath_self_energy(tail)
    one_poles = self_energy_poles(one_sigma)
    tail_poles = self_energy_poles(tail_sigma)
    one_residues = self_energy_residues(one_sigma)
    tail_residues = self_energy_residues(tail_sigma)
    coupling_norms_match = (
        bath_has_same_triality_head_data(one)
        and bath_has_same_triality_head_data(tail)
        and sp.simplify(total_probability(two_triality_head_amplitudes()) - 1) == 0
    )
    self_energies_differ = sp.simplify(one_sigma - tail_sigma) != 0
    poles_differ = one_poles != tail_poles
    residues_differ = one_residues != tail_residues
    checks_pass = pole_residue_rigidity_no_go_pass()

    if checks_pass:
        final_verdict = "RADIAL_POLE_RESIDUE_RIGIDITY_NO_GO_PASS"
        interpretation = (
            "The inherited silver transfer, S3 scalar successor pair, and "
            "two-channel no-leakage norm do not force the radial spectral "
            "measure. A one-level triality-head bath and a symmetric two-level "
            "tail bath preserve the same head-channel data but give different "
            "Schur self-energies, poles, and residues. Therefore mass pole "
            "rigidity needs an additional dynamical spectral principle."
        )
    else:
        final_verdict = "RADIAL_POLE_RESIDUE_RIGIDITY_INCONCLUSIVE"
        interpretation = "The no-go controls failed to distinguish admissible radial baths."

    return PoleResidueRigidityPayload(
        final_verdict=final_verdict,
        inherited_root=inherited_transfer_root(),
        eta=inherited_eta(),
        intensity_ratio=inherited_intensity_ratio(),
        allowed_successors=allowed_scalar_s3_successor_labels(),
        head_amplitudes=two_triality_head_amplitudes(),
        one_level_self_energy=one_sigma,
        two_level_tail_self_energy=tail_sigma,
        self_energies_differ=self_energies_differ,
        one_level_poles=one_poles,
        two_level_tail_poles=tail_poles,
        poles_differ=poles_differ,
        one_level_residues=one_residues,
        two_level_tail_residues=tail_residues,
        residues_differ=residues_differ,
        coupling_norms_match=coupling_norms_match,
        inherited_transfer_unchanged=radial_silver_transfer_inheritance_pass(),
        rigidity_forced=False,
        interpretation=interpretation,
    )
