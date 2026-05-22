"""JAX site-local Higgs field helpers for the spacetime QCA.

Session 39 promotes the Session 38 static two-complex Higgs coordinates
``Phi = (phi_plus, phi_zero)`` to a site-local lattice field.  The field is
kept in the fundamental electroweak representation:

* ``phi.shape == (nx, ny, nz, 2)`` with complex entries;
* ``phi[..., 0]`` is the charged component ``phi_plus``;
* ``phi[..., 1]`` is the neutral component ``phi_zero``;
* gauge coordinates are ordered ``(su2_x, su2_y, su2_z, u1_y)``.

The hypercharge convention is ``Y(Phi) = +1/2``.  Thus the anti-Hermitian
fundamental generators are ``T_a = -i sigma_a / 2`` for ``SU(2)_L`` and
``T_Y = -i I_2 / 2`` for ``U(1)_Y``.  The electromagnetic observable
``i(T_3 + T_Y)`` has charges ``(+1, 0)`` on ``(phi_plus, phi_zero)``.

The BCC link convention matches the rest of ``spacetime_qca``:
``U[x,h]`` transports from source ``x+h`` to target ``x``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.links import jax_identity_link_field
from clifford_3plus2_d5.sim.state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import (
    jax_compact_lie_link_field_from_algebra,
    jax_compact_lie_link_from_algebra,
    jax_compact_lie_site_field_from_algebra,
    jax_su2_generators,
    jax_transform_link_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_step import jax_bcc_displacements
from clifford_3plus2_d5.spacetime_qca.jax_yukawa import jax_hermitian_yukawa_internal_control
from clifford_3plus2_d5.spacetime_qca.mass import beta_matrix

__all__ = [
    "jax_constant_higgs_field",
    "jax_higgs_covariant_difference",
    "jax_higgs_energy_density",
    "jax_higgs_generators",
    "jax_higgs_kinetic_energy_density",
    "jax_higgs_link_field_from_algebra",
    "jax_higgs_link_current",
    "jax_higgs_potential_density",
    "jax_higgs_pure_gauge_links_from_site_algebra",
    "jax_higgs_site_gauge_from_algebra",
    "jax_higgs_yukawa_hamiltonian_field",
    "jax_higgs_yukawa_internal_control_field",
    "jax_transform_higgs_field",
    "jax_transform_higgs_links",
]


def _validate_higgs_field(phi: jnp.ndarray) -> tuple[int, int, int]:
    if phi.ndim != 4 or phi.shape[-1] != 2:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    return int(phi.shape[0]), int(phi.shape[1]), int(phi.shape[2])


def _validate_higgs_site_gauge(site_gauge: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if site_gauge.ndim != 5 or site_gauge.shape[-2:] != (2, 2):
        raise ValueError("Higgs site gauges must have shape (nx, ny, nz, 2, 2)")
    if lattice_shape is not None and tuple(site_gauge.shape[:3]) != lattice_shape:
        raise ValueError("Higgs site gauges must match the lattice shape")


def _validate_higgs_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, 2, 2):
        raise ValueError("Higgs BCC links must have shape (nx, ny, nz, 8, 2, 2)")
    if lattice_shape is not None and tuple(links.shape[:3]) != lattice_shape:
        raise ValueError("Higgs links must match the lattice shape")


def jax_higgs_generators(dtype: Any = jnp.complex64) -> jnp.ndarray:
    """Return anti-Hermitian ``SU(2)_L x U(1)_Y`` Higgs generators.

    The returned basis has shape ``(4, 2, 2)`` and coordinate order
    ``(su2_x, su2_y, su2_z, u1_y)``.
    """

    dtype = np.dtype(dtype)
    su2 = jax_su2_generators(dtype=dtype)
    hypercharge = -0.5j * jnp.eye(2, dtype=dtype)
    return jnp.concatenate((su2, hypercharge[None, :, :]), axis=0)


def jax_constant_higgs_field(
    lattice_shape: tuple[int, int, int],
    *,
    phi_plus: Any = 0.0 + 0.0j,
    phi_zero: Any = 1.0 + 0.0j,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return a constant site-local Higgs field on a periodic 3D lattice."""

    dtype = np.dtype(dtype)
    value = jnp.asarray((phi_plus, phi_zero), dtype=dtype)
    return jnp.broadcast_to(value, (*lattice_shape, 2))


def jax_higgs_site_gauge_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return site-local finite Higgs gauge matrices from algebra coordinates."""

    if site_theta.ndim != 4 or site_theta.shape[-1] != 4:
        raise ValueError("Higgs site gauge coordinates must have shape (nx, ny, nz, 4)")
    return jax_compact_lie_site_field_from_algebra(
        site_theta,
        jax_higgs_generators(dtype=jnp.result_type(site_theta, 1j)),
    )


def jax_higgs_link_field_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return BCC Higgs links from algebra coordinates.

    ``theta`` has shape ``(nx, ny, nz, 8, 4)`` with the coordinate order
    documented in :func:`jax_higgs_generators`.
    """

    if theta.ndim != 5 or theta.shape[-2:] != (8, 4):
        raise ValueError("Higgs link coordinates must have shape (nx, ny, nz, 8, 4)")
    return jax_compact_lie_link_field_from_algebra(
        theta,
        jax_higgs_generators(dtype=jnp.result_type(theta, 1j)),
    )


def jax_higgs_pure_gauge_links_from_site_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return pure-gauge Higgs BCC links generated by site-local gauges."""

    site_gauge = jax_higgs_site_gauge_from_algebra(site_theta)
    identity_links = jax_identity_link_field(site_gauge.shape[:3], 2, dtype=site_gauge.dtype)
    return jax_transform_higgs_links(identity_links, site_gauge)


def jax_transform_higgs_field(phi: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply ``Phi[x] -> G[x] Phi[x]`` to a site-local Higgs field."""

    lattice_shape = _validate_higgs_field(phi)
    _validate_higgs_site_gauge(site_gauge, lattice_shape)
    return jnp.einsum("...ab,...b->...a", site_gauge, phi)


def jax_transform_higgs_links(links: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply finite gauge transforms to pull-convention Higgs links."""

    _validate_higgs_links(links)
    _validate_higgs_site_gauge(site_gauge, tuple(links.shape[:3]))
    return jax_transform_link_field(links, site_gauge)


def jax_higgs_covariant_difference(phi: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Return BCC pull-form covariant differences ``D_h Phi[x]``.

    The output has shape ``(nx, ny, nz, 8, 2)`` and uses
    ``D_h Phi[x] = U[x,h] Phi[x+h] - Phi[x]``.
    """

    lattice_shape = _validate_higgs_field(phi)
    _validate_higgs_links(links, lattice_shape)

    differences = []
    for index, displacement in enumerate(jax_bcc_displacements()):
        source_phi = source_roll(phi, displacement)
        transported = jnp.einsum("...ab,...b->...a", links[..., index, :, :], source_phi)
        differences.append(transported - phi)
    return jnp.stack(differences, axis=3)


def jax_higgs_kinetic_energy_density(phi: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Return sitewise ``sum_h ||D_h Phi||^2`` for BCC Higgs links."""

    differences = jax_higgs_covariant_difference(phi, links)
    return jnp.real(jnp.sum(jnp.conj(differences) * differences, axis=(-2, -1)))


def jax_higgs_potential_density(
    phi: jnp.ndarray,
    *,
    vev_squared: Any = 1.0,
    quartic: Any = 1.0,
) -> jnp.ndarray:
    """Return sitewise Mexican-hat potential ``lambda (|Phi|^2 - v^2)^2``."""

    _validate_higgs_field(phi)
    norm_squared = jnp.real(jnp.sum(jnp.conj(phi) * phi, axis=-1))
    return jnp.asarray(quartic, dtype=norm_squared.dtype) * (
        norm_squared - jnp.asarray(vev_squared, dtype=norm_squared.dtype)
    ) ** 2


def jax_higgs_energy_density(
    phi: jnp.ndarray,
    links: jnp.ndarray,
    *,
    vev_squared: Any = 1.0,
    quartic: Any = 1.0,
) -> jnp.ndarray:
    """Return sitewise Higgs kinetic plus potential energy density."""

    return jax_higgs_kinetic_energy_density(phi, links) + jax_higgs_potential_density(
        phi,
        vev_squared=vev_squared,
        quartic=quartic,
    )


def jax_higgs_link_current(
    phi: jnp.ndarray,
    links: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    vev_squared: Any = 1.0,
    quartic: Any = 1.0,
) -> jnp.ndarray:
    """Return finite-difference Higgs link-current coordinates.

    The output has shape ``(nx, ny, nz, 8, 4)`` in the Higgs generator order
    ``(su2_x, su2_y, su2_z, u1_y)``.  Each component is the negative
    left-trivialized derivative of total Higgs energy under
    ``U[x,h] -> exp(theta_a T_a) U[x,h]``.  This mirrors the gauge-force sign:
    adding ``dt * current`` to momenta gives a force-like source kick.

    This is a small-lattice audit primitive; it loops over BCC links and
    generators explicitly instead of providing a production analytic current.
    """

    lattice_shape = _validate_higgs_field(phi)
    _validate_higgs_links(links, lattice_shape)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    eps = jnp.asarray(epsilon, dtype=real_dtype)
    basis_coordinates = eps * jnp.eye(4, dtype=real_dtype)
    generators = jax_higgs_generators(dtype=links.dtype)
    plus_updates = jax_compact_lie_link_from_algebra(basis_coordinates, generators)
    minus_updates = jax_compact_lie_link_from_algebra(-basis_coordinates, generators)

    def total_energy(candidate_links: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(
            jax_higgs_energy_density(
                phi,
                candidate_links,
                vev_squared=vev_squared,
                quartic=quartic,
            ),
        )

    current = jnp.zeros((*links.shape[:4], 4), dtype=real_dtype)
    nx, ny, nz = (int(size) for size in links.shape[:3])
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for hop in range(8):
                    base_link = links[x, y, z, hop]
                    for generator_index in range(4):
                        plus_links = links.at[x, y, z, hop].set(plus_updates[generator_index] @ base_link)
                        minus_links = links.at[x, y, z, hop].set(minus_updates[generator_index] @ base_link)
                        derivative = (total_energy(plus_links) - total_energy(minus_links)) / (2 * eps)
                        current = current.at[x, y, z, hop, generator_index].set(-derivative)
    return current


def jax_higgs_yukawa_internal_control_field(phi: jnp.ndarray, *, dtype: Any | None = None) -> jnp.ndarray:
    """Return the Session 38 internal ``Y(Phi[x])`` matrix at every site."""

    _validate_higgs_field(phi)
    if dtype is None:
        dtype = phi.dtype
    return jax_hermitian_yukawa_internal_control(phi[..., 0], phi[..., 1], dtype=dtype)


@lru_cache(maxsize=8)
def _beta_numpy(dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return sympy_matrix_to_numpy(beta_matrix(), dtype=dtype)


def jax_higgs_yukawa_hamiltonian_field(phi: jnp.ndarray, *, dtype: Any | None = None) -> jnp.ndarray:
    """Return ``beta x Y(Phi[x])`` at every site.

    The output has shape ``(nx, ny, nz, 128, 128)``.  This is a bridge to the
    static Session 38 fermion Yukawa insertion, not a time-evolution rule.
    """

    _validate_higgs_field(phi)
    if dtype is None:
        dtype = phi.dtype
    dtype = np.dtype(dtype)
    internal = jax_higgs_yukawa_internal_control_field(phi, dtype=dtype)
    beta = jnp.asarray(_beta_numpy(dtype.name), dtype=dtype)
    lifted = jnp.einsum("ab,...ij->...aibj", beta, internal)
    leading_shape = internal.shape[:-2]
    return jnp.reshape(lifted, (*leading_shape, beta.shape[0] * internal.shape[-2], beta.shape[1] * internal.shape[-1]))
