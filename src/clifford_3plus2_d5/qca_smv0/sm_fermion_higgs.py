"""Local fermion backreaction into Higgs momenta for QCA_SMv0.

Stage 10 adds the first fermion-source term.  It does not yet compute the
fermion current on BCC gauge links.  Instead it couples the already-implemented
three-family Yukawa matrix back into the classical Higgs momentum through the
local Yukawa energy density

``E_Y = Re psi^dagger beta Y(H) psi``.

The resulting Higgs force is the complex-field force ``-dE_Y/dH*`` computed by
real/imaginary automatic differentiation.  This keeps the source tied to the
same local Yukawa operator used by Stage 7 instead of introducing a separate
phenomenological Higgs source.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    FamilyFNQuarkAuxState,
    FamilyFNQuarkDilations,
    FamilyFNQuarkSourceKick,
    FamilyLeptonYukawas,
    SM_FAMILY_DIM,
    _d_c_index,
    _e_c_index,
    _l_index,
    _nu_c_index,
    _q_index,
    _u_c_index,
    sm_apply_family_fn_quark_source_kick,
    sm_apply_family_yukawa_collision,
    sm_default_family_lepton_yukawas,
    sm_family_recirculated_quark_dilations,
    sm_family_recirculated_quark_yukawas,
    sm_family_yukawa_internal_matrix,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_CHIRAL16_DIM,
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM, sm_constant_higgs, sm_dirac_beta
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    deterministic_higgs_momenta,
    deterministic_higgs_site_theta,
    sm_higgs_site_gauge_from_algebra,
    sm_transform_higgs_field,
)


class FermionHiggsBackreactionDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 10 Yukawa/Higgs backreaction."""

    fn_recirculated_source_residual: jnp.ndarray
    energy_reality_residual: jnp.ndarray
    zero_state_source_norm: jnp.ndarray
    nonzero_source_norm: jnp.ndarray
    energy_gauge_invariance_residual: jnp.ndarray
    source_covariance_residual: jnp.ndarray
    kick_delta_norm: jnp.ndarray
    kick_reversibility_residual: jnp.ndarray
    collision_norm_drift: jnp.ndarray
    source_after_collision_norm: jnp.ndarray
    jit_delta_source: jnp.ndarray
    jit_delta_kick: jnp.ndarray


class FamilyFNYukawaCollisionOutput(NamedTuple):
    """Combined Higgs kick and persistent FN quark source collision output."""

    state: jnp.ndarray
    higgs_momenta: jnp.ndarray
    fn_quark: FamilyFNQuarkSourceKick


def _validate_family_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 6 or state.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("family SM Dirac state must have shape (nx, ny, nz, 4, 32, 3)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_higgs_field(higgs: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    if higgs.ndim != 4 or higgs.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and higgs.shape[:3] != lattice_shape:
        raise ValueError("Higgs field must match the state lattice")
    return int(higgs.shape[0]), int(higgs.shape[1]), int(higgs.shape[2])


def _validate_higgs_momenta(momenta: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if momenta.ndim != 4 or momenta.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs momenta must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and momenta.shape[:3] != lattice_shape:
        raise ValueError("Higgs momenta must match the Higgs lattice")


def sm_transform_family_state(state: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply an SM site gauge to a family-extended state."""

    lattice_shape = _validate_family_state(state)
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != lattice_shape or site_gauge.shape[-2:] != (
        SM_INTERNAL_DIM,
        SM_INTERNAL_DIM,
    ):
        raise ValueError("site_gauge must have shape (nx, ny, nz, 32, 32)")
    return jnp.einsum("...ab,...sbf->...saf", site_gauge, state)


def sm_site_theta_from_higgs_site_theta(higgs_site_theta: jnp.ndarray) -> jnp.ndarray:
    """Embed Higgs electroweak site-gauge coordinates into SM coordinates."""

    if higgs_site_theta.ndim != 4 or higgs_site_theta.shape[-1] != 4:
        raise ValueError("higgs_site_theta must have shape (nx, ny, nz, 4)")
    sm_theta = jnp.zeros((*higgs_site_theta.shape[:3], SM_GENERATOR_COUNT), dtype=higgs_site_theta.dtype)
    return sm_theta.at[..., 8:12].set(higgs_site_theta)


def sm_yukawa_site_gauge_from_higgs_site_theta(higgs_site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return the electroweak site gauge in the physical Yukawa-door convention.

    The Stage 2 transport carrier uses left-handed conjugate labels for the
    chiral-16.  The local Yukawa doors in Stage 4/7 are written in the physical
    right-handed convention.  This gauge helper makes that convention explicit
    for local Yukawa-source covariance audits.
    """

    if higgs_site_theta.ndim != 4 or higgs_site_theta.shape[-1] != 4:
        raise ValueError("higgs_site_theta must have shape (nx, ny, nz, 4)")

    sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex64)
    sigma_y = jnp.asarray([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)
    su2 = 0.5j * jnp.stack((sigma_x, sigma_y, sigma_z), axis=0)
    generators16 = jnp.zeros((4, SM_CHIRAL16_DIM, SM_CHIRAL16_DIM), dtype=jnp.complex64)

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
    generators16 = generators16.at[3].set(1j * jnp.diag(hypercharges).astype(jnp.complex64))

    generators32 = jnp.zeros((4, SM_INTERNAL_DIM, SM_INTERNAL_DIM), dtype=jnp.complex64)
    generators32 = generators32.at[:, :SM_CHIRAL16_DIM, :SM_CHIRAL16_DIM].set(generators16)
    generators32 = generators32.at[:, SM_CHIRAL16_DIM:, SM_CHIRAL16_DIM:].set(generators16)
    algebra = jnp.einsum("...a,aij->...ij", higgs_site_theta, generators32.astype(jnp.result_type(higgs_site_theta, 1j)))
    flat = algebra.reshape((-1, SM_INTERNAL_DIM, SM_INTERNAL_DIM))
    gauges = jax.vmap(jsp_linalg.expm)(flat)
    return gauges.reshape((*higgs_site_theta.shape[:3], SM_INTERNAL_DIM, SM_INTERNAL_DIM))


def deterministic_yukawa_source_state(lattice_shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    """Return a deterministic family state with nonzero Yukawa bilinears."""

    state = jnp.zeros((*lattice_shape, 4, SM_INTERNAL_DIM, SM_FAMILY_DIM), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, _q_index(0, 0), 0].set(0.80 + 0.10j)
    state = state.at[0, 0, 0, 2, _u_c_index(0), 1].set(-0.35 + 0.45j)
    state = state.at[0, 0, 0, 1, _q_index(1, 1), 1].set(0.55 - 0.20j)
    state = state.at[0, 0, 0, 3, _d_c_index(1), 2].set(0.40 + 0.30j)
    return state


def sm_yukawa_energy_local_density(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    *,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return local real density ``Re psi^dag beta Y(H) psi``."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if quark_yukawas is None:
        quark_yukawas = sm_family_recirculated_quark_yukawas()
    if lepton_yukawas is None:
        lepton_yukawas = sm_default_family_lepton_yukawas()

    flat_state = state.reshape((*lattice_shape, 4, SM_INTERNAL_DIM * SM_FAMILY_DIM))
    yukawa = sm_family_yukawa_internal_matrix(
        higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    ).astype(state.dtype)
    yukawa_action = jnp.einsum("...ij,...rj->...ri", yukawa, flat_state)
    beta_yukawa_action = jnp.einsum("sr,...ri->...si", sm_dirac_beta(state.dtype), yukawa_action)
    local = jnp.sum(jnp.conj(flat_state) * beta_yukawa_action, axis=(-2, -1))
    return jnp.real(local)


def sm_yukawa_energy_density(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    *,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return mean local Yukawa energy density."""

    return jnp.mean(
        sm_yukawa_energy_local_density(
            state,
            higgs,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=lepton_yukawas,
        ),
    )


def sm_yukawa_higgs_force(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    *,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return the local Yukawa source force on Higgs momenta."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)

    def total_energy(real_part: jnp.ndarray, imag_part: jnp.ndarray) -> jnp.ndarray:
        field = real_part + 1j * imag_part
        return jnp.sum(
            sm_yukawa_energy_local_density(
                state,
                field,
                quark_yukawas=quark_yukawas,
                lepton_yukawas=lepton_yukawas,
            ),
        )

    grad_real, grad_imag = jax.grad(total_energy, argnums=(0, 1))(jnp.real(higgs), jnp.imag(higgs))
    return -0.5 * (grad_real + 1j * grad_imag)


def sm_apply_yukawa_higgs_momentum_kick(
    higgs_momenta: jnp.ndarray,
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    *,
    step_size: float,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Kick Higgs momenta by the local Yukawa source."""

    lattice_shape = _validate_higgs_field(higgs)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    source = sm_yukawa_higgs_force(
        state,
        higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    return higgs_momenta + jnp.asarray(step_size, dtype=higgs_momenta.dtype) * source


def sm_family_yukawa_collision_with_higgs_kick(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
    *,
    step_size: float,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a local fermion collision and source kick for the Higgs momenta."""

    updated_momenta = sm_apply_yukawa_higgs_momentum_kick(
        higgs_momenta,
        state,
        higgs,
        step_size=step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    updated_state = sm_apply_family_yukawa_collision(
        state,
        higgs,
        step_size=step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    return updated_state, updated_momenta


def sm_family_fn_yukawa_collision_with_higgs_kick(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
    aux_state: FamilyFNQuarkAuxState | None = None,
    dilations: FamilyFNQuarkDilations | None = None,
    *,
    step_size: float,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> FamilyFNYukawaCollisionOutput:
    """Apply a Higgs kick plus persistent FN quark recirculation source.

    Quarks are advanced by the explicit hidden FN source kick.  Leptons still use
    the closed local Yukawa collision.  This is the constructive FN simulator
    path; the older matrix-only quark collision remains available separately.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    if dilations is None:
        dilations = sm_family_recirculated_quark_dilations()
    if lepton_yukawas is None:
        lepton_yukawas = sm_default_family_lepton_yukawas()

    updated_momenta = sm_apply_yukawa_higgs_momentum_kick(
        higgs_momenta,
        state,
        higgs,
        step_size=step_size,
        lepton_yukawas=lepton_yukawas,
    )
    quark_step = sm_apply_family_fn_quark_source_kick(
        state,
        higgs,
        aux_state=aux_state,
        dilations=dilations,
        step_size=step_size,
    )
    zero_quarks = FNQuarkYukawas(
        up=jnp.zeros((SM_FAMILY_DIM, SM_FAMILY_DIM), dtype=state.dtype),
        down=jnp.zeros((SM_FAMILY_DIM, SM_FAMILY_DIM), dtype=state.dtype),
    )
    updated_state = sm_apply_family_yukawa_collision(
        quark_step.state,
        higgs,
        step_size=step_size,
        quark_yukawas=zero_quarks,
        lepton_yukawas=lepton_yukawas,
    )
    return FamilyFNYukawaCollisionOutput(
        state=updated_state,
        higgs_momenta=updated_momenta,
        fn_quark=FamilyFNQuarkSourceKick(
            state=quark_step.state,
            source=quark_step.source,
            aux_state=quark_step.aux_state,
        ),
    )


def sm_fermion_higgs_backreaction_diagnostics() -> FermionHiggsBackreactionDiagnostics:
    """Return focused Stage 10 Yukawa/Higgs backreaction diagnostics."""

    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    zero_state = jnp.zeros_like(state)
    higgs = sm_constant_higgs(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    site_theta = deterministic_higgs_site_theta(lattice_shape, scale=0.17)
    higgs_gauge = sm_higgs_site_gauge_from_algebra(site_theta)
    sm_gauge = sm_yukawa_site_gauge_from_higgs_site_theta(site_theta)
    transformed_state = sm_transform_family_state(state, sm_gauge)
    transformed_higgs = sm_transform_higgs_field(higgs, higgs_gauge)
    reference_quark_yukawas = sm_family_recirculated_quark_yukawas()

    energy = sm_yukawa_energy_density(state, higgs)
    reference_energy = sm_yukawa_energy_density(state, higgs, quark_yukawas=reference_quark_yukawas)
    transformed_energy = sm_yukawa_energy_density(transformed_state, transformed_higgs)
    source = sm_yukawa_higgs_force(state, higgs)
    reference_source = sm_yukawa_higgs_force(state, higgs, quark_yukawas=reference_quark_yukawas)
    transformed_source = sm_yukawa_higgs_force(transformed_state, transformed_higgs)
    expected_source = sm_transform_higgs_field(source, higgs_gauge)
    kicked = sm_apply_yukawa_higgs_momentum_kick(higgs_momenta, state, higgs, step_size=0.03)
    unkicked = sm_apply_yukawa_higgs_momentum_kick(kicked, state, higgs, step_size=-0.03)
    updated_state, updated_momenta = sm_family_yukawa_collision_with_higgs_kick(
        state,
        higgs,
        higgs_momenta,
        step_size=0.03,
    )
    jitted_source = jax.jit(sm_yukawa_higgs_force)
    jit_source = jitted_source(state, higgs)
    jitted_kick = jax.jit(sm_apply_yukawa_higgs_momentum_kick, static_argnames=("step_size",))
    jit_kick = jitted_kick(higgs_momenta, state, higgs, step_size=0.03)

    before_norm = jnp.real(jnp.sum(jnp.conj(state) * state))
    after_norm = jnp.real(jnp.sum(jnp.conj(updated_state) * updated_state))
    return FermionHiggsBackreactionDiagnostics(
        fn_recirculated_source_residual=jnp.maximum(
            jnp.abs(energy - reference_energy),
            jnp.max(jnp.abs(source - reference_source)),
        ),
        energy_reality_residual=jnp.abs(jnp.imag(energy + 0j)),
        zero_state_source_norm=jnp.linalg.norm(sm_yukawa_higgs_force(zero_state, higgs)),
        nonzero_source_norm=jnp.linalg.norm(source),
        energy_gauge_invariance_residual=jnp.abs(transformed_energy - energy),
        source_covariance_residual=jnp.max(jnp.abs(transformed_source - expected_source)),
        kick_delta_norm=jnp.linalg.norm(kicked - higgs_momenta),
        kick_reversibility_residual=jnp.max(jnp.abs(unkicked - higgs_momenta)),
        collision_norm_drift=jnp.abs(after_norm - before_norm),
        source_after_collision_norm=jnp.linalg.norm(sm_yukawa_higgs_force(updated_state, higgs)),
        jit_delta_source=jnp.max(jnp.abs(jit_source - source)),
        jit_delta_kick=jnp.max(jnp.abs(jit_kick - kicked)),
    )
