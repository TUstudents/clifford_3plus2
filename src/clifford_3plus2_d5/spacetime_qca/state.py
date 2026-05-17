"""Exact finite-lattice state helpers for spacetime QCA steps."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.lattice import PeriodicLattice3D, Site

State = dict[Site, sp.Matrix]


def zero_state(lattice: PeriodicLattice3D, spinor_dim: int) -> State:
    return {site: sp.zeros(spinor_dim, 1) for site in lattice.sites()}


def delta_state(
    lattice: PeriodicLattice3D,
    site: Site,
    spinor: sp.Matrix,
) -> State:
    state = zero_state(lattice, spinor.rows)
    state[lattice.wrap(site)] = spinor
    return state


def zero_weyl_state(lattice: PeriodicLattice3D) -> State:
    return zero_state(lattice, 2)


def zero_dirac_state(lattice: PeriodicLattice3D) -> State:
    return zero_state(lattice, 4)


def state_norm_squared(state: State) -> sp.Expr:
    total = sp.Integer(0)
    for spinor in state.values():
        total += (spinor.H * spinor)[0, 0]
    return sp.simplify(total)


def states_close_exact(left: State, right: State) -> bool:
    if set(left) != set(right):
        return False
    return all(
        (left[site] - right[site]).applyfunc(sp.simplify) == sp.zeros(left[site].rows, 1)
        for site in left
    )


def split_dirac_state(state: State) -> tuple[State, State]:
    right: State = {}
    left: State = {}
    for site, spinor in state.items():
        if spinor.shape != (4, 1):
            raise ValueError("Dirac state spinors must be 4x1")
        right[site] = spinor[:2, :]
        left[site] = spinor[2:, :]
    return right, left


def combine_dirac_state(right: State, left: State) -> State:
    if set(right) != set(left):
        raise ValueError("right and left states must have the same support")
    return {
        site: sp.Matrix.vstack(right[site], left[site]).applyfunc(sp.simplify)
        for site in right
    }
