"""Antiunitary singlet bridge for QCA_SMv0.

Stage 17 turns the Stage 16 convention audit into an explicit carrier bridge.
The bridge keeps left doublet labels linear in the Stage 2 transport
representation and interprets right singlet labels anti-linearly:

``T_physical = T_transport`` on ``Q,L``
``T_physical = conj(T_transport)`` on ``u^c,d^c,e^c,nu^c``.

This converts the left-handed-conjugate singlet labels of the transport carrier
into the physical-right convention required by the local Yukawa/Higgs door.
It is an exact algebraic simulator bridge, not a microscopic BCC derivation.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import (
    deterministic_yukawa_source_state,
    sm_transform_family_state,
    sm_yukawa_energy_density,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
    deterministic_sm_site_theta,
    sm_algebra_matrix_field,
    sm_generators,
    sm_site_gauge_from_algebra,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge_convention_bridge import (
    sm_left_doublet_projector,
    sm_right_singlet_projector,
    sm_yukawa_door_electroweak_generators,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    sm_higgs_site_gauge_from_algebra,
    sm_transform_higgs_field,
)


class AntiunitaryBridgeDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 17 antiunitary singlet bridge."""

    left_linear_generator_residual: jnp.ndarray
    right_antilinear_generator_residual: jnp.ndarray
    electroweak_yukawa_slice_residual: jnp.ndarray
    finite_bridge_residual: jnp.ndarray
    finite_bridge_unitarity_residual: jnp.ndarray
    transport_physical_generator_difference_norm: jnp.ndarray
    full_physical_yukawa_energy_covariance_residual: jnp.ndarray
    full_transport_yukawa_energy_noninvariance_residual: jnp.ndarray
    jit_delta_physical_covariance: jnp.ndarray
    jit_delta_transport_noninvariance: jnp.ndarray


def sm_physical_right_generators(*, dtype: Any = jnp.complex64) -> jnp.ndarray:
    """Return full SM generators in the physical-right carrier convention."""

    transport = sm_generators(dtype=dtype)
    left = sm_left_doublet_projector(dtype=dtype)
    right = sm_right_singlet_projector(dtype=dtype)
    return jnp.stack(
        [left @ generator @ left + right @ jnp.conj(generator) @ right for generator in transport],
        axis=0,
    )


def sm_physical_right_algebra_matrix_field(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return physical-right SM algebra matrices from site coordinates."""

    if site_theta.ndim != 4 or site_theta.shape[-1] != SM_GENERATOR_COUNT:
        raise ValueError("site_theta must have shape (nx, ny, nz, 12)")
    generators = sm_physical_right_generators(dtype=jnp.result_type(site_theta, 1j))
    return jnp.einsum("...a,aij->...ij", site_theta, generators)


def sm_physical_right_site_gauge_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Exponentiate a full physical-right site gauge."""

    algebra = sm_physical_right_algebra_matrix_field(site_theta)
    flat = algebra.reshape((-1, SM_INTERNAL_DIM, SM_INTERNAL_DIM))
    gauges = jax.vmap(jsp_linalg.expm)(flat)
    return gauges.reshape((*site_theta.shape[:3], SM_INTERNAL_DIM, SM_INTERNAL_DIM))


def sm_antiunitary_bridge_gauge_from_transport(transport_gauge: jnp.ndarray) -> jnp.ndarray:
    """Return the finite bridge gauge from a transport-convention gauge."""

    if transport_gauge.ndim != 5 or transport_gauge.shape[-2:] != (SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("transport_gauge must have shape (nx, ny, nz, 32, 32)")
    left = sm_left_doublet_projector(dtype=transport_gauge.dtype)
    right = sm_right_singlet_projector(dtype=transport_gauge.dtype)
    return left @ transport_gauge @ left + right @ jnp.conj(transport_gauge) @ right


def sm_antiunitary_bridge_generator_residuals() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return generator-level bridge residuals."""

    transport = sm_generators()
    physical = sm_physical_right_generators()
    left = sm_left_doublet_projector()
    right = sm_right_singlet_projector()
    left_residual = jnp.max(jnp.abs(left @ (physical - transport) @ left))
    right_residual = jnp.max(jnp.abs(right @ (physical - jnp.conj(transport)) @ right))
    ew_slice_residual = jnp.max(jnp.abs(physical[8:12] - sm_yukawa_door_electroweak_generators()))
    difference_norm = jnp.max(jnp.abs(physical - transport))
    return left_residual, right_residual, ew_slice_residual, difference_norm


def sm_physical_right_gauge_unitarity_residual(gauge: jnp.ndarray) -> jnp.ndarray:
    """Return ``max_abs(G^dag G-I)`` for a physical-right site gauge field."""

    identity = jnp.eye(SM_INTERNAL_DIM, dtype=gauge.dtype)
    return jnp.max(jnp.abs(jnp.swapaxes(jnp.conj(gauge), -1, -2) @ gauge - identity))


def sm_full_bridge_yukawa_energy_residuals() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return physical full-gauge covariance and transport non-invariance."""

    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    site_theta = deterministic_sm_site_theta(lattice_shape, scale=0.17)
    higgs_site_theta = site_theta[..., 8:12]
    higgs_gauge = sm_higgs_site_gauge_from_algebra(higgs_site_theta)
    physical_gauge = sm_physical_right_site_gauge_from_algebra(site_theta)
    transport_gauge = sm_site_gauge_from_algebra(site_theta)

    energy = sm_yukawa_energy_density(state, higgs)
    physical_energy = sm_yukawa_energy_density(
        sm_transform_family_state(state, physical_gauge),
        sm_transform_higgs_field(higgs, higgs_gauge),
    )
    transport_energy = sm_yukawa_energy_density(
        sm_transform_family_state(state, transport_gauge),
        sm_transform_higgs_field(higgs, higgs_gauge),
    )
    return jnp.abs(physical_energy - energy), jnp.abs(transport_energy - energy)


def sm_antiunitary_bridge_diagnostics() -> AntiunitaryBridgeDiagnostics:
    """Return focused Stage 17 antiunitary bridge diagnostics."""

    lattice_shape = (1, 1, 1)
    site_theta = deterministic_sm_site_theta(lattice_shape, scale=0.17)
    transport_gauge = sm_site_gauge_from_algebra(site_theta)
    physical_gauge = sm_physical_right_site_gauge_from_algebra(site_theta)
    bridge_gauge = sm_antiunitary_bridge_gauge_from_transport(transport_gauge)
    left, right, ew_slice, difference = sm_antiunitary_bridge_generator_residuals()
    physical_covariance, transport_noninvariance = sm_full_bridge_yukawa_energy_residuals()
    jitted = jax.jit(sm_full_bridge_yukawa_energy_residuals)
    jit_physical, jit_transport = jitted()

    # Exercise the physical algebra helper directly; the value is intentionally
    # not returned separately because the finite bridge residual is the stronger
    # exponentiated statement.
    _ = sm_physical_right_algebra_matrix_field(site_theta) + sm_algebra_matrix_field(site_theta)

    return AntiunitaryBridgeDiagnostics(
        left_linear_generator_residual=left,
        right_antilinear_generator_residual=right,
        electroweak_yukawa_slice_residual=ew_slice,
        finite_bridge_residual=jnp.max(jnp.abs(physical_gauge - bridge_gauge)),
        finite_bridge_unitarity_residual=sm_physical_right_gauge_unitarity_residual(physical_gauge),
        transport_physical_generator_difference_norm=difference,
        full_physical_yukawa_energy_covariance_residual=physical_covariance,
        full_transport_yukawa_energy_noninvariance_residual=transport_noninvariance,
        jit_delta_physical_covariance=jnp.abs(jit_physical - physical_covariance),
        jit_delta_transport_noninvariance=jnp.abs(jit_transport - transport_noninvariance),
    )
