"""Variational audit for the physical-right production tick.

Stage 25 ties the Stage 21/24 production forces back to the energy pieces they
differentiate.  The audit is deliberately narrow: it checks force
decompositions and two explicit central-difference probes on the deterministic
physical-right production state.

This stage repairs and certifies force provenance.  It does not introduce a new
boundary rule, exact full-energy conservation claim, quantized register, or
Gauss projection.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from clifford_3plus2_d5.qca_smv0.sm_dynamics import sm_left_wilson_force
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import (
    sm_yukawa_energy_density,
    sm_yukawa_higgs_force,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_average_wilson_action_density, sm_generators
from clifford_3plus2_d5.qca_smv0.sm_gauge_higgs import (
    sm_embed_higgs_to_sm_momenta,
    sm_higgs_left_gauge_force,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    sm_higgs_force,
    sm_higgs_hamiltonian_density,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_current import (
    sm_physical_right_fermion_left_gauge_current,
    sm_physical_right_streaming_energy_density,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tick import (
    sm_physical_right_production_higgs_force,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_sourced_tick import (
    sm_physical_right_sourced_link_force,
)


class PhysicalRightProductionVariationalDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 25 force provenance."""

    link_force_decomposition_residual: jnp.ndarray
    higgs_force_decomposition_residual: jnp.ndarray
    link_directional_derivative_residual: jnp.ndarray
    higgs_real_directional_derivative_residual: jnp.ndarray
    higgs_imag_directional_derivative_residual: jnp.ndarray
    vacuum_link_force_norm: jnp.ndarray
    vacuum_higgs_force_norm: jnp.ndarray
    deterministic_link_force_norm: jnp.ndarray
    deterministic_higgs_force_norm: jnp.ndarray
    all_residuals_finite: jnp.ndarray


def _left_perturb_sm_link(
    sm_links: jnp.ndarray,
    *,
    link_index: tuple[int, int, int, int],
    generator_index: int,
    epsilon: float,
) -> jnp.ndarray:
    """Return links with one left multiplication by ``exp(epsilon T_a)``."""

    generator = sm_generators(dtype=sm_links.dtype)[generator_index]
    step = jnp.asarray(epsilon, dtype=sm_links.real.dtype)
    update = jsp_linalg.expm(step * generator)
    updated_link = update @ sm_links[link_index]
    return sm_links.at[link_index].set(updated_link)


def _color_link_energy_density(
    state: PhysicalRightProductionRolloutState,
    sm_links: jnp.ndarray,
    *,
    beta: float,
) -> jnp.ndarray:
    """Return the color-link energy pieces probed by the selected coordinate."""

    return jnp.asarray(beta, dtype=sm_links.real.dtype) * sm_average_wilson_action_density(
        sm_links,
    ) + sm_physical_right_streaming_energy_density(state.family_state, sm_links)


def _higgs_position_energy_density(
    state: PhysicalRightProductionRolloutState,
    higgs: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters,
    quark_yukawas: FNQuarkYukawas | None = None,
) -> jnp.ndarray:
    """Return Higgs potential/gradient plus local Yukawa energy for a field."""

    return sm_higgs_hamiltonian_density(
        higgs,
        state.higgs_momenta,
        state.higgs_links,
        parameters=parameters,
    ) + sm_yukawa_energy_density(
        state.family_state,
        higgs,
        quark_yukawas=quark_yukawas,
    )


def sm_physical_right_production_link_force_decomposition_residual(
    state: PhysicalRightProductionRolloutState,
    *,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> jnp.ndarray:
    """Return the residual between the sourced force and its three pieces."""

    combined = sm_physical_right_sourced_link_force(
        state.family_state,
        state.higgs,
        state.sm_links,
        state.higgs_links,
        beta=beta,
        parameters=parameters,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    pieces = (
        jnp.asarray(beta, dtype=state.sm_links.real.dtype) * sm_left_wilson_force(state.sm_links, epsilon=wilson_epsilon)
        + sm_embed_higgs_to_sm_momenta(
            sm_higgs_left_gauge_force(
                state.higgs,
                state.higgs_links,
                parameters=parameters,
                epsilon=higgs_force_epsilon,
            ),
        )
        + sm_physical_right_fermion_left_gauge_current(
            state.family_state,
            state.sm_links,
            epsilon=fermion_current_epsilon,
        )
    )
    return jnp.max(jnp.abs(combined - pieces))


def sm_physical_right_production_higgs_force_decomposition_residual(
    state: PhysicalRightProductionRolloutState,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
) -> jnp.ndarray:
    """Return the residual between the production Higgs force and its two pieces."""

    combined = sm_physical_right_production_higgs_force(
        state.family_state,
        state.higgs,
        state.higgs_links,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
    )
    pieces = sm_higgs_force(state.higgs, state.higgs_links, parameters=parameters) + sm_yukawa_higgs_force(
        state.family_state,
        state.higgs,
        quark_yukawas=quark_yukawas,
    )
    return jnp.max(jnp.abs(combined - pieces))


def sm_physical_right_production_link_directional_derivative_residual(
    state: PhysicalRightProductionRolloutState,
    *,
    beta: float = 1.0,
    epsilon: float = 3e-2,
    link_index: tuple[int, int, int, int] = (0, 0, 0, 0),
    generator_index: int = 0,
) -> jnp.ndarray:
    """Compare a selected color-link force coordinate with a central difference."""

    plus_links = _left_perturb_sm_link(
        state.sm_links,
        link_index=link_index,
        generator_index=generator_index,
        epsilon=epsilon,
    )
    minus_links = _left_perturb_sm_link(
        state.sm_links,
        link_index=link_index,
        generator_index=generator_index,
        epsilon=-epsilon,
    )
    derivative = (
        _color_link_energy_density(state, plus_links, beta=beta)
        - _color_link_energy_density(state, minus_links, beta=beta)
    ) / (2.0 * jnp.asarray(epsilon, dtype=state.sm_links.real.dtype))
    source_force = sm_physical_right_sourced_link_force(
        state.family_state,
        state.higgs,
        state.sm_links,
        state.higgs_links,
        beta=beta,
        wilson_epsilon=epsilon,
        higgs_force_epsilon=epsilon,
        fermion_current_epsilon=epsilon,
    )
    return jnp.abs(derivative - source_force[(*link_index, generator_index)])


def sm_physical_right_production_higgs_directional_derivative_residuals(
    state: PhysicalRightProductionRolloutState,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    epsilon: float = 1e-3,
    component_index: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compare one Higgs force component with real and imaginary differences."""

    step = jnp.asarray(epsilon, dtype=state.higgs.real.dtype)
    real_plus = state.higgs.at[component_index].add(step)
    real_minus = state.higgs.at[component_index].add(-step)
    imag_plus = state.higgs.at[component_index].add(1j * step)
    imag_minus = state.higgs.at[component_index].add(-1j * step)
    real_derivative = (
        _higgs_position_energy_density(state, real_plus, parameters=parameters)
        - _higgs_position_energy_density(state, real_minus, parameters=parameters)
    ) / (2.0 * step)
    imag_derivative = (
        _higgs_position_energy_density(state, imag_plus, parameters=parameters)
        - _higgs_position_energy_density(state, imag_minus, parameters=parameters)
    ) / (2.0 * step)
    force = sm_physical_right_production_higgs_force(
        state.family_state,
        state.higgs,
        state.higgs_links,
        parameters=parameters,
    )
    force_component = force[component_index]
    return (
        jnp.abs(real_derivative + 2.0 * jnp.real(force_component)),
        jnp.abs(imag_derivative + 2.0 * jnp.imag(force_component)),
    )


def sm_physical_right_production_variational_diagnostics() -> PhysicalRightProductionVariationalDiagnostics:
    """Return focused Stage 25 physical-right production variational diagnostics."""

    deterministic = sm_physical_right_production_initial_state()
    vacuum = sm_physical_right_production_vacuum_state()
    link_force = sm_physical_right_sourced_link_force(
        deterministic.family_state,
        deterministic.higgs,
        deterministic.sm_links,
        deterministic.higgs_links,
    )
    higgs_force = sm_physical_right_production_higgs_force(
        deterministic.family_state,
        deterministic.higgs,
        deterministic.higgs_links,
    )
    vacuum_link_force = sm_physical_right_sourced_link_force(
        vacuum.family_state,
        vacuum.higgs,
        vacuum.sm_links,
        vacuum.higgs_links,
    )
    vacuum_higgs_force = sm_physical_right_production_higgs_force(
        vacuum.family_state,
        vacuum.higgs,
        vacuum.higgs_links,
    )
    higgs_real_residual, higgs_imag_residual = sm_physical_right_production_higgs_directional_derivative_residuals(
        deterministic,
    )
    residuals = jnp.asarray(
        [
            sm_physical_right_production_link_force_decomposition_residual(deterministic),
            sm_physical_right_production_higgs_force_decomposition_residual(deterministic),
            sm_physical_right_production_link_directional_derivative_residual(deterministic),
            higgs_real_residual,
            higgs_imag_residual,
            jnp.linalg.norm(vacuum_link_force),
            jnp.linalg.norm(vacuum_higgs_force),
            jnp.linalg.norm(link_force),
            jnp.linalg.norm(higgs_force),
        ],
    )
    return PhysicalRightProductionVariationalDiagnostics(
        link_force_decomposition_residual=residuals[0],
        higgs_force_decomposition_residual=residuals[1],
        link_directional_derivative_residual=residuals[2],
        higgs_real_directional_derivative_residual=residuals[3],
        higgs_imag_directional_derivative_residual=residuals[4],
        vacuum_link_force_norm=residuals[5],
        vacuum_higgs_force_norm=residuals[6],
        deterministic_link_force_norm=residuals[7],
        deterministic_higgs_force_norm=residuals[8],
        all_residuals_finite=jnp.all(jnp.isfinite(residuals)),
    )
