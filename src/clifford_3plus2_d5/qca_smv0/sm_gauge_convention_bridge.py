"""Gauge-convention bridge audit for QCA_SMv0.

Stage 16 quantifies the convention boundary left open by the production tick.
The BCC transport/current carrier uses left-handed conjugate singlet labels
(``u^c,d^c,e^c``), while the local Yukawa/Higgs door is written in the
physical right-handed convention (``u_R,d_R,e_R``).  Those two conventions are
not related by a unitary similarity on the fixed 32-component carrier: the
hypercharge spectra differ.  They are related on right singlets by charge
conjugation, i.e. by an anti-linear interpretation of the singlet labels.

This stage is therefore an audit, not a unification.  It makes the convention
split explicit and gives tests that would fail if the Stage 15 production tick
silently treated the two gauges as the same object.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import (
    deterministic_yukawa_source_state,
    sm_site_theta_from_higgs_site_theta,
    sm_transform_family_state,
    sm_yukawa_energy_density,
    sm_yukawa_site_gauge_from_higgs_site_theta,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_CHIRAL16_DIM,
    SM_INTERNAL_DIM,
    sm_generators,
    sm_site_gauge_from_algebra,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    deterministic_higgs_site_theta,
    sm_higgs_site_gauge_from_algebra,
    sm_transform_higgs_field,
)

SM_ELECTROWEAK_GENERATOR_COUNT = 4


class GaugeConventionBridgeDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 16 gauge-convention bridge audit."""

    generator_helper_residual: jnp.ndarray
    left_doublet_generator_residual: jnp.ndarray
    right_singlet_su2_residual: jnp.ndarray
    right_hypercharge_conjugation_residual: jnp.ndarray
    full_generator_difference_norm: jnp.ndarray
    hypercharge_spectral_mismatch: jnp.ndarray
    physical_yukawa_energy_covariance_residual: jnp.ndarray
    transport_yukawa_energy_noninvariance_residual: jnp.ndarray
    jit_delta_physical_covariance: jnp.ndarray
    jit_delta_transport_noninvariance: jnp.ndarray


def _q_index(color: int, weak: int) -> int:
    return 2 * color + weak


def _u_c_index(color: int) -> int:
    return 6 + color


def _d_c_index(color: int) -> int:
    return 9 + color


def _l_index(weak: int) -> int:
    return 12 + weak


def _e_c_index() -> int:
    return 14


def _nu_c_index() -> int:
    return 15


def _copy_indices(base_indices: tuple[int, ...]) -> jnp.ndarray:
    values = []
    for copy_offset in (0, SM_CHIRAL16_DIM):
        values.extend(copy_offset + index for index in base_indices)
    return jnp.asarray(values, dtype=jnp.int32)


def _diagonal_projector(indices: jnp.ndarray, *, dtype: Any = jnp.complex64) -> jnp.ndarray:
    diagonal = jnp.zeros((SM_INTERNAL_DIM,), dtype=dtype)
    diagonal = diagonal.at[indices].set(1.0 + 0.0j)
    return jnp.diag(diagonal)


def sm_left_doublet_indices() -> jnp.ndarray:
    """Return internal indices for the duplicated left doublets ``Q,L``."""

    q = tuple(_q_index(color, weak) for color in range(3) for weak in range(2))
    leptons = tuple(_l_index(weak) for weak in range(2))
    return _copy_indices((*q, *leptons))


def sm_right_singlet_indices() -> jnp.ndarray:
    """Return internal indices for duplicated singlet labels."""

    up = tuple(_u_c_index(color) for color in range(3))
    down = tuple(_d_c_index(color) for color in range(3))
    charged = (_e_c_index(), _nu_c_index())
    return _copy_indices((*up, *down, *charged))


def sm_left_doublet_projector(*, dtype: Any = jnp.complex64) -> jnp.ndarray:
    """Return the projector onto duplicated left doublet labels."""

    return _diagonal_projector(sm_left_doublet_indices(), dtype=dtype)


def sm_right_singlet_projector(*, dtype: Any = jnp.complex64) -> jnp.ndarray:
    """Return the projector onto duplicated singlet labels."""

    return _diagonal_projector(sm_right_singlet_indices(), dtype=dtype)


def sm_transport_electroweak_generators(*, dtype: Any = jnp.complex64) -> jnp.ndarray:
    """Return Stage 2 transport ``SU(2)_L x U(1)_Y`` generators."""

    return sm_generators(dtype=dtype)[8:12]


def sm_yukawa_door_electroweak_generators(*, dtype: Any = jnp.complex64) -> jnp.ndarray:
    """Return physical-right-handed local Yukawa-door electroweak generators."""

    sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=dtype)
    sigma_y = jnp.asarray([[0, -1j], [1j, 0]], dtype=dtype)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=dtype)
    su2 = 0.5j * jnp.stack((sigma_x, sigma_y, sigma_z), axis=0)
    generators16 = jnp.zeros((SM_ELECTROWEAK_GENERATOR_COUNT, SM_CHIRAL16_DIM, SM_CHIRAL16_DIM), dtype=dtype)

    for gen_index in range(3):
        matrix = su2[gen_index]
        for color in range(3):
            for row in range(2):
                for col in range(2):
                    generators16 = generators16.at[gen_index, _q_index(color, row), _q_index(color, col)].set(
                        matrix[row, col],
                    )
        for row in range(2):
            for col in range(2):
                generators16 = generators16.at[gen_index, _l_index(row), _l_index(col)].set(matrix[row, col])

    hypercharges = jnp.zeros((SM_CHIRAL16_DIM,), dtype=jnp.float32)
    for color in range(3):
        for weak in range(2):
            hypercharges = hypercharges.at[_q_index(color, weak)].set(1.0 / 6.0)
        hypercharges = hypercharges.at[_u_c_index(color)].set(2.0 / 3.0)
        hypercharges = hypercharges.at[_d_c_index(color)].set(-1.0 / 3.0)
    hypercharges = hypercharges.at[_l_index(0)].set(-0.5)
    hypercharges = hypercharges.at[_l_index(1)].set(-0.5)
    hypercharges = hypercharges.at[_e_c_index()].set(-1.0)
    hypercharges = hypercharges.at[_nu_c_index()].set(0.0)
    generators16 = generators16.at[3].set(1j * jnp.diag(hypercharges).astype(dtype))

    generators32 = jnp.zeros((SM_ELECTROWEAK_GENERATOR_COUNT, SM_INTERNAL_DIM, SM_INTERNAL_DIM), dtype=dtype)
    generators32 = generators32.at[:, :SM_CHIRAL16_DIM, :SM_CHIRAL16_DIM].set(generators16)
    generators32 = generators32.at[:, SM_CHIRAL16_DIM:, SM_CHIRAL16_DIM:].set(generators16)
    return generators32


def sm_yukawa_door_algebra_matrix_field(higgs_site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return local Yukawa-door electroweak algebra matrices from coordinates."""

    if higgs_site_theta.ndim != 4 or higgs_site_theta.shape[-1] != SM_ELECTROWEAK_GENERATOR_COUNT:
        raise ValueError("higgs_site_theta must have shape (nx, ny, nz, 4)")
    generators = sm_yukawa_door_electroweak_generators(dtype=jnp.result_type(higgs_site_theta, 1j))
    return jnp.einsum("...a,aij->...ij", higgs_site_theta, generators)


def sm_yukawa_door_site_gauge_from_generators(higgs_site_theta: jnp.ndarray) -> jnp.ndarray:
    """Exponentiate the explicit Stage 16 Yukawa-door generators."""

    algebra = sm_yukawa_door_algebra_matrix_field(higgs_site_theta)
    flat = algebra.reshape((-1, SM_INTERNAL_DIM, SM_INTERNAL_DIM))
    gauges = jax.vmap(jsp_linalg.expm)(flat)
    return gauges.reshape((*higgs_site_theta.shape[:3], SM_INTERNAL_DIM, SM_INTERNAL_DIM))


def sm_gauge_convention_generator_residuals() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return exact generator-level bridge residuals."""

    transport = sm_transport_electroweak_generators()
    yukawa = sm_yukawa_door_electroweak_generators()
    left = sm_left_doublet_projector()
    right = sm_right_singlet_projector()
    left_residual = jnp.max(jnp.abs(left @ (transport - yukawa) @ left))
    right_su2_residual = jnp.max(jnp.abs(right @ yukawa[:3] @ right)) + jnp.max(jnp.abs(right @ transport[:3] @ right))
    right_hypercharge_conjugation = jnp.max(jnp.abs(right @ (transport[3] + yukawa[3]) @ right))
    full_difference = jnp.max(jnp.abs(transport - yukawa))
    return left_residual, right_su2_residual, right_hypercharge_conjugation, full_difference


def sm_hypercharge_spectral_mismatch() -> jnp.ndarray:
    """Return sorted-spectrum mismatch between transport and Yukawa hypercharge."""

    transport = sm_transport_electroweak_generators()[3]
    yukawa = sm_yukawa_door_electroweak_generators()[3]
    transport_charges = jnp.sort(jnp.imag(jnp.diag(transport)))
    yukawa_charges = jnp.sort(jnp.imag(jnp.diag(yukawa)))
    return jnp.max(jnp.abs(transport_charges - yukawa_charges))


def sm_yukawa_gauge_convention_energy_residuals() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return physical covariance and transport non-invariance residuals."""

    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    higgs_site_theta = deterministic_higgs_site_theta(lattice_shape, scale=0.17)
    higgs_gauge = sm_higgs_site_gauge_from_algebra(higgs_site_theta)
    physical_sm_gauge = sm_yukawa_site_gauge_from_higgs_site_theta(higgs_site_theta)
    transport_sm_gauge = sm_site_gauge_from_algebra(sm_site_theta_from_higgs_site_theta(higgs_site_theta))

    energy = sm_yukawa_energy_density(state, higgs)
    physical_energy = sm_yukawa_energy_density(
        sm_transform_family_state(state, physical_sm_gauge),
        sm_transform_higgs_field(higgs, higgs_gauge),
    )
    transport_energy = sm_yukawa_energy_density(
        sm_transform_family_state(state, transport_sm_gauge),
        sm_transform_higgs_field(higgs, higgs_gauge),
    )
    return jnp.abs(physical_energy - energy), jnp.abs(transport_energy - energy)


def sm_gauge_convention_bridge_diagnostics() -> GaugeConventionBridgeDiagnostics:
    """Return focused Stage 16 gauge-convention diagnostics."""

    lattice_shape = (1, 1, 1)
    higgs_site_theta = deterministic_higgs_site_theta(lattice_shape, scale=0.17)
    helper_gauge = sm_yukawa_site_gauge_from_higgs_site_theta(higgs_site_theta)
    explicit_gauge = sm_yukawa_door_site_gauge_from_generators(higgs_site_theta)
    left, right_su2, right_hypercharge, full_difference = sm_gauge_convention_generator_residuals()
    physical_covariance, transport_noninvariance = sm_yukawa_gauge_convention_energy_residuals()
    jitted_energy = jax.jit(sm_yukawa_gauge_convention_energy_residuals)
    jit_physical, jit_transport = jitted_energy()

    return GaugeConventionBridgeDiagnostics(
        generator_helper_residual=jnp.max(jnp.abs(helper_gauge - explicit_gauge)),
        left_doublet_generator_residual=left,
        right_singlet_su2_residual=right_su2,
        right_hypercharge_conjugation_residual=right_hypercharge,
        full_generator_difference_norm=full_difference,
        hypercharge_spectral_mismatch=sm_hypercharge_spectral_mismatch(),
        physical_yukawa_energy_covariance_residual=physical_covariance,
        transport_yukawa_energy_noninvariance_residual=transport_noninvariance,
        jit_delta_physical_covariance=jnp.abs(jit_physical - physical_covariance),
        jit_delta_transport_noninvariance=jnp.abs(jit_transport - transport_noninvariance),
    )
