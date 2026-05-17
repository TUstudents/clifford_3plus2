"""Session 22 finite real-space BCC step tests."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca import (
    PeriodicLattice3D,
    delta_state,
    dirac_plane_wave_matches_bloch,
    dirac_step,
    dirac_step_with_constant_link,
    plane_wave_state,
    same_matrix,
    split_dirac_state,
    state_norm_squared,
    states_close_exact,
    weyl_plane_wave_matches_bloch,
    weyl_step,
    zero_weyl_state,
)
from clifford_3plus2_d5.spacetime_qca.state import combine_dirac_state


def test_periodic_lattice_wrap_translate_and_sites() -> None:
    lattice = PeriodicLattice3D((3, 4, 5))
    assert lattice.volume == 60
    assert lattice.wrap((3, -1, 6)) == (0, 3, 1)
    assert lattice.translate((0, 0, 0), (-1, 1, -1)) == (2, 1, 4)
    assert len(lattice.sites()) == 60


def test_zero_and_delta_weyl_state_shapes() -> None:
    lattice = PeriodicLattice3D((2, 2, 2))
    zero = zero_weyl_state(lattice)
    assert all(spinor.shape == (2, 1) for spinor in zero.values())

    spinor = sp.Matrix([1, sp.I])
    delta = delta_state(lattice, (3, 0, -1), spinor)
    assert delta[(1, 0, 1)] == spinor
    assert state_norm_squared(delta) == 2


def test_weyl_delta_step_has_body_diagonal_support() -> None:
    lattice = PeriodicLattice3D((5, 5, 5))
    state = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 1]))
    stepped = weyl_step(state, lattice)
    support = {site for site, spinor in stepped.items() if spinor != sp.zeros(2, 1)}
    expected = {
        lattice.wrap(tuple(-component for component in direction))  # type: ignore[arg-type]
        for direction in (
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        )
    }
    assert support == expected


def test_dirac_step_equals_blockwise_weyl_steps() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    state = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 2, 3, 4]))
    right, left = split_dirac_state(state)
    expected = combine_dirac_state(
        weyl_step(right, lattice, helicity="right"),
        weyl_step(left, lattice, helicity="left"),
    )
    assert states_close_exact(dirac_step(state, lattice), expected)


def test_plane_wave_matches_weyl_and_dirac_bloch_symbols() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    momentum = (sp.pi / 2, 0, 0)
    assert weyl_plane_wave_matches_bloch(
        lattice,
        momentum,
        sp.Matrix([1, sp.I]),
        helicity="right",
    )
    assert weyl_plane_wave_matches_bloch(
        lattice,
        momentum,
        sp.Matrix([1, -sp.I]),
        helicity="left",
    )
    assert dirac_plane_wave_matches_bloch(lattice, momentum, sp.Matrix([1, 0, 0, 1]))


def test_weyl_and_dirac_steps_preserve_norm_for_plane_waves() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    momentum = (sp.pi / 2, 0, 0)
    weyl_state = plane_wave_state(lattice, momentum, sp.Matrix([1, sp.I]))
    assert sp.simplify(state_norm_squared(weyl_step(weyl_state, lattice)) - state_norm_squared(weyl_state)) == 0

    dirac_state = plane_wave_state(lattice, momentum, sp.Matrix([1, 0, 0, 1]))
    assert sp.simplify(state_norm_squared(dirac_step(dirac_state, lattice)) - state_norm_squared(dirac_state)) == 0


def test_constant_identity_link_matches_ungauged_dirac_tensor_step() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    internal_dim = 2
    internal = sp.eye(internal_dim)
    state = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 0, 0, 1, 0, 1, 1, 0]))

    linked = dirac_step_with_constant_link(state, lattice, internal)

    dirac_part = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 0, 0, 1]))
    stepped_dirac = dirac_step(dirac_part, lattice)
    expected = {
        site: sp.kronecker_product(stepped_dirac[site], sp.Matrix([1, 0]))
            + sp.kronecker_product(
                dirac_step(
                    delta_state(lattice, (0, 0, 0), sp.Matrix([0, 1, 1, 0])),
                    lattice,
                )[site],
                sp.Matrix([0, 1]),
            )
        for site in lattice.sites()
    }
    assert all(same_matrix(linked[site], expected[site]) for site in lattice.sites())
