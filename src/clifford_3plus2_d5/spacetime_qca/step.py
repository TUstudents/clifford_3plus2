"""Finite real-space BCC step operators."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.bcc_weyl import (
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    bcc_dirac_symbol,
    bcc_weyl_symbol,
    opposite_helicity_hops,
)
from clifford_3plus2_d5.spacetime_qca.dirac import block_diag
from clifford_3plus2_d5.spacetime_qca.lattice import PeriodicLattice3D
from clifford_3plus2_d5.spacetime_qca.links import (
    LinkField,
    bcc_link_displacements,
    constant_link_field,
    validate_link_field,
)
from clifford_3plus2_d5.spacetime_qca.state import State, combine_dirac_state, split_dirac_state


def _zero_like_state(lattice: PeriodicLattice3D, spinor_dim: int) -> State:
    return {site: sp.zeros(spinor_dim, 1) for site in lattice.sites()}


def _validate_state(state: State, lattice: PeriodicLattice3D, spinor_dim: int) -> None:
    if set(state) != set(lattice.sites()):
        raise ValueError("state support must match lattice sites")
    for spinor in state.values():
        if spinor.shape != (spinor_dim, 1):
            raise ValueError(f"state spinors must be {spinor_dim}x1")


def weyl_step(
    state: State,
    lattice: PeriodicLattice3D,
    *,
    helicity: str = "right",
) -> State:
    """Apply the pull-form BCC Weyl step ``out[x] = sum_h W(h) psi[x+h]``."""

    _validate_state(state, lattice, 2)
    if helicity == "right":
        hops = bialynicki_birula_hops()
    elif helicity == "left":
        hops = opposite_helicity_hops()
    else:
        raise ValueError("helicity must be 'right' or 'left'")

    out = _zero_like_state(lattice, 2)
    for site in lattice.sites():
        spinor = sp.zeros(2, 1)
        for direction, hop in zip(bialynicki_birula_directions(), hops, strict=True):
            source = lattice.translate(site, direction)
            spinor += hop * state[source]
        out[site] = spinor.applyfunc(sp.simplify)
    return out


def dirac_step(state: State, lattice: PeriodicLattice3D) -> State:
    """Apply blockwise right/left BCC Weyl steps to a 4-spinor state."""

    _validate_state(state, lattice, 4)
    right, left = split_dirac_state(state)
    return combine_dirac_state(
        weyl_step(right, lattice, helicity="right"),
        weyl_step(left, lattice, helicity="left"),
    )


def dirac_step_with_constant_link(
    state: State,
    lattice: PeriodicLattice3D,
    link: sp.Matrix,
) -> State:
    """Apply ``sum_h (W_D(h) x link) psi[x+h]`` with a constant internal link."""

    if link.rows != link.cols:
        raise ValueError("link must be square")
    return dirac_step_with_link_field(state, lattice, constant_link_field(lattice, link))


def dirac_step_with_link_field(
    state: State,
    lattice: PeriodicLattice3D,
    links: LinkField,
) -> State:
    """Apply ``sum_h (W_D(h) x U[x <- x+h]) psi[x+h]`` with local links."""

    internal_dim = validate_link_field(links, lattice)
    spinor_dim = 4 * internal_dim
    _validate_state(state, lattice, spinor_dim)

    right_hops = bialynicki_birula_hops()
    left_hops = opposite_helicity_hops()
    dirac_hops = tuple(
        block_diag(right, left).applyfunc(sp.simplify)
        for right, left in zip(right_hops, left_hops, strict=True)
    )

    out = _zero_like_state(lattice, spinor_dim)
    for site in lattice.sites():
        spinor = sp.zeros(spinor_dim, 1)
        for direction, hop in zip(bcc_link_displacements(), dirac_hops, strict=True):
            source = lattice.translate(site, direction)
            link = links[(site, direction)]  # U[site <- source] in pull convention.
            spinor += sp.kronecker_product(hop, link) * state[source]
        out[site] = spinor.applyfunc(sp.simplify)
    return out


def plane_wave_state(
    lattice: PeriodicLattice3D,
    momentum: tuple[sp.Expr, sp.Expr, sp.Expr],
    spinor: sp.Matrix,
) -> State:
    """Return ``psi[x] = exp(-i k.x) spinor`` on the integer coordinate lattice.

    This sign matches the package Bloch convention
    ``U(k) = sum_h exp(-i k.h) W(h)`` for the pull-form step
    ``out[x] = sum_h W(h) psi[x+h]``.
    """

    state: State = {}
    kx, ky, kz = momentum
    for site in lattice.sites():
        x, y, z = site
        phase = sp.exp(-sp.I * (kx * x + ky * y + kz * z))
        state[site] = (phase * spinor).applyfunc(sp.simplify)
    return state


def weyl_plane_wave_matches_bloch(
    lattice: PeriodicLattice3D,
    momentum: tuple[sp.Expr, sp.Expr, sp.Expr],
    spinor: sp.Matrix,
    *,
    helicity: str = "right",
) -> bool:
    """Check one exact plane-wave eigen-relation against the Weyl Bloch symbol."""

    state = plane_wave_state(lattice, momentum, spinor)
    stepped = weyl_step(state, lattice, helicity=helicity)
    symbol = bcc_weyl_symbol(sp.Integer(1), *momentum, helicity=helicity)
    expected_spinor = (symbol * spinor).applyfunc(sp.simplify)
    expected = plane_wave_state(lattice, momentum, expected_spinor)
    return all(
        (stepped[site] - expected[site]).applyfunc(sp.simplify) == sp.zeros(2, 1)
        for site in lattice.sites()
    )


def dirac_plane_wave_matches_bloch(
    lattice: PeriodicLattice3D,
    momentum: tuple[sp.Expr, sp.Expr, sp.Expr],
    spinor: sp.Matrix,
) -> bool:
    state = plane_wave_state(lattice, momentum, spinor)
    stepped = dirac_step(state, lattice)
    symbol = bcc_dirac_symbol(sp.Integer(1), *momentum)
    expected_spinor = (symbol * spinor).applyfunc(sp.simplify)
    expected = plane_wave_state(lattice, momentum, expected_spinor)
    return all(
        (stepped[site] - expected[site]).applyfunc(sp.simplify) == sp.zeros(4, 1)
        for site in lattice.sites()
    )
