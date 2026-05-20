"""Session 43 scaling diagnostics for the coupled spacetime-QCA prototype.

This module does not add a new evolution rule.  It packages the existing
fermion/gauge/Higgs prototype into deterministic tiny-lattice probes that can
be compared across step sizes and toy lattice shapes.  The diagnostics are
stability and normalization controls, not a continuum-renormalization proof.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypedDict

import jax.numpy as jnp

from clifford_3plus2_d5.sim.benchmarks import benchmark_jitted_kernel
from clifford_3plus2_d5.sim.backend import JaxTiming
from clifford_3plus2_d5.sim.links import jax_identity_link_field
from clifford_3plus2_d5.spacetime_qca.jax_coupled_higgs import (
    HiggsCoupledSector,
    YukawaUpdateMode,
    jax_higgs_link_field_from_patisalam_sector,
    jax_patisalam_fermion_gauge_higgs_diagnostics,
    jax_patisalam_fermion_gauge_higgs_step,
)
from clifford_3plus2_d5.spacetime_qca.jax_fermion_gauge import PATISALAM_INTERNAL_DIM
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import CompactLieForceMethod
from clifford_3plus2_d5.spacetime_qca.jax_higgs import (
    jax_constant_higgs_field,
    jax_higgs_energy_density,
    jax_higgs_link_field_from_algebra,
)
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import (
    jax_patisalam_gauge_hamiltonian_density,
    jax_patisalam_link_field_from_algebra,
)
from clifford_3plus2_d5.spacetime_qca.plaquette import (
    PlaquetteShape,
    canonical_bcc_plaquette_shapes,
)


@dataclass(frozen=True)
class ScalingRunConfig:
    """Memory-safe default configuration for a tiny coupled scaling probe."""

    lattice_shape: tuple[int, int, int] = (1, 1, 1)
    sector: HiggsCoupledSector = "u1_y"
    step_size: float = 0.005
    matter_coupling: float = 1.0
    yukawa_coupling: float = 1.0
    beta: float = 1.0
    vev_squared: float = 1.0
    quartic: float = 1.0
    force_method: CompactLieForceMethod = "finite_difference"
    force_epsilon: float = 1e-3
    force_chunk_size: int | None = None
    current_epsilon: float = 1e-3
    yukawa_mode: YukawaUpdateMode = "first_order"
    shapes: tuple[PlaquetteShape, ...] | None = None


@dataclass(frozen=True)
class ScalingInitialState:
    """Deterministic tiny-lattice field data for Session 43 probes."""

    state: jnp.ndarray
    links: jnp.ndarray
    momenta: jnp.ndarray
    phi: jnp.ndarray
    higgs_momentum: jnp.ndarray
    higgs_links: jnp.ndarray


@dataclass(frozen=True)
class ScalingSnapshot:
    """Scalarized diagnostics for one coupled prototype state."""

    fermion_norm: jnp.ndarray
    gauge_hamiltonian_density: jnp.ndarray
    higgs_total_energy: jnp.ndarray
    higgs_energy_density_mean: jnp.ndarray
    gauss_residual_norm: jnp.ndarray
    yukawa_norm_drift: jnp.ndarray
    total_energy_proxy: jnp.ndarray


@dataclass(frozen=True)
class ScalingTrial:
    """Before/after one-step diagnostics and absolute drifts."""

    before: ScalingSnapshot
    after: ScalingSnapshot
    fermion_norm_drift: jnp.ndarray
    gauge_energy_drift: jnp.ndarray
    higgs_energy_drift: jnp.ndarray
    gauss_residual_drift: jnp.ndarray
    total_energy_proxy_drift: jnp.ndarray
    all_finite: jnp.ndarray


@dataclass(frozen=True)
class ScalingTrajectorySample:
    """One recorded sample in a multi-step scaling trajectory."""

    step_index: int
    snapshot: ScalingSnapshot
    fermion_norm_drift: jnp.ndarray
    gauge_energy_drift: jnp.ndarray
    higgs_energy_drift: jnp.ndarray
    gauss_residual_drift: jnp.ndarray
    total_energy_proxy_drift: jnp.ndarray


@dataclass(frozen=True)
class ScalingTrajectory:
    """Recorded tiny-lattice trajectory with drift summaries."""

    samples: tuple[ScalingTrajectorySample, ...]
    initial: ScalingTrajectorySample
    final: ScalingTrajectorySample
    max_fermion_norm_drift: jnp.ndarray
    max_gauge_energy_drift: jnp.ndarray
    max_higgs_energy_drift: jnp.ndarray
    max_gauss_residual_drift: jnp.ndarray
    max_total_energy_proxy_drift: jnp.ndarray
    all_finite: jnp.ndarray


@dataclass(frozen=True)
class YukawaModeComparison:
    """Paired first-order/unitary Yukawa trajectory comparison."""

    first_order: ScalingTrajectory
    unitary: ScalingTrajectory
    fermion_norm_drift_ratio: jnp.ndarray
    total_energy_proxy_drift_ratio: jnp.ndarray
    all_finite: jnp.ndarray


@dataclass(frozen=True)
class NeutralVacuumDensityProbe:
    """Volume-normalization probe for neutral vacuum controls."""

    lattice_shape: tuple[int, int, int]
    gauge_hamiltonian_density: jnp.ndarray
    higgs_energy_density_mean: jnp.ndarray
    total_energy_proxy: jnp.ndarray
    all_zero: jnp.ndarray


class Session43ScalingAuditPayload(TypedDict):
    """Small stable payload for the Session 43 report."""

    default_sector: str
    default_lattice_shape: tuple[int, int, int]
    neutral_probe_shapes: tuple[tuple[int, int, int], ...]
    step_sizes: tuple[float, ...]
    notes: tuple[str, ...]


def _sector_dimension(sector: HiggsCoupledSector) -> int:
    if sector == "u1_y":
        return 1
    if sector == "su2_l":
        return 3
    if sector == "sm":
        return 12
    raise ValueError(f"sector {sector!r} is not supported by Session 43 scaling probes")


def _selected_shapes(config: ScalingRunConfig) -> tuple[PlaquetteShape, ...]:
    if config.shapes is not None:
        return config.shapes
    return (tuple(canonical_bcc_plaquette_shapes())[0],)


def _volume(lattice_shape: tuple[int, int, int]) -> int:
    nx, ny, nz = lattice_shape
    return nx * ny * nz


def _validate_lattice_shape(lattice_shape: tuple[int, int, int]) -> None:
    if len(lattice_shape) != 3 or any(size <= 0 for size in lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {lattice_shape}")


def _last_site(lattice_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(size - 1 for size in lattice_shape)


def _deterministic_coordinates(
    lattice_shape: tuple[int, int, int],
    trailing_dim: int,
    *,
    scale: float,
) -> jnp.ndarray:
    count = _volume(lattice_shape) * 8 * trailing_dim
    values = jnp.arange(count, dtype=jnp.float32).reshape((*lattice_shape, 8, trailing_dim))
    return scale * (values + 1.0)


def jax_default_scaling_initial_state(config: ScalingRunConfig) -> ScalingInitialState:
    """Return deterministic initial data for a tiny Session 43 scaling run."""

    _validate_lattice_shape(config.lattice_shape)
    sector_dim = _sector_dimension(config.sector)
    state = jnp.zeros((*config.lattice_shape, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, 0].set(1.0 + 0.25j)
    state = state.at[0, 0, 0, 1, 7].set(-0.5 + 0.125j)
    if _volume(config.lattice_shape) > 1:
        last_x, last_y, last_z = _last_site(config.lattice_shape)
        state = state.at[last_x, last_y, last_z, 2, 13].set(0.25 - 0.2j)

    theta = _deterministic_coordinates(config.lattice_shape, sector_dim, scale=2.5e-4)
    higgs_theta = _deterministic_coordinates(config.lattice_shape, sector_dim, scale=2.0e-4)
    momenta = _deterministic_coordinates(config.lattice_shape, sector_dim, scale=1.0e-3)

    site_values = jnp.arange(_volume(config.lattice_shape), dtype=jnp.float32).reshape(
        config.lattice_shape,
    )
    phi = jnp.stack(
        (
            0.025 * site_values + 0.01j * (site_values + 1.0),
            1.0 + 0.015 * site_values - 0.005j * site_values,
        ),
        axis=-1,
    ).astype(jnp.complex64)
    higgs_momentum = jnp.stack(
        (
            0.002 * (site_values + 1.0) - 0.001j * site_values,
            -0.001 * site_values + 0.0015j * (site_values + 1.0),
        ),
        axis=-1,
    ).astype(jnp.complex64)

    return ScalingInitialState(
        state=state,
        links=jax_patisalam_link_field_from_algebra(theta, sector=config.sector),
        momenta=momenta,
        phi=phi,
        higgs_momentum=higgs_momentum,
        higgs_links=jax_higgs_link_field_from_patisalam_sector(higgs_theta, sector=config.sector),
    )


def jax_scaling_snapshot(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    config: ScalingRunConfig,
    reference_state: jnp.ndarray | None = None,
) -> ScalingSnapshot:
    """Return scalarized diagnostics for one coupled prototype state."""

    diagnostics = jax_patisalam_fermion_gauge_higgs_diagnostics(
        state,
        links,
        momenta,
        phi,
        higgs_momentum,
        higgs_links,
        sector=config.sector,
        beta=config.beta,
        matter_coupling=config.matter_coupling,
        vev_squared=config.vev_squared,
        quartic=config.quartic,
        shapes=_selected_shapes(config),
        reference_state=reference_state,
    )
    higgs_energy_density_mean = jnp.mean(diagnostics["higgs_energy_density"])
    total_energy_proxy = diagnostics["gauge_hamiltonian_density"] + higgs_energy_density_mean
    return ScalingSnapshot(
        fermion_norm=diagnostics["fermion_norm"],
        gauge_hamiltonian_density=diagnostics["gauge_hamiltonian_density"],
        higgs_total_energy=jnp.sum(diagnostics["higgs_energy_density"]),
        higgs_energy_density_mean=higgs_energy_density_mean,
        gauss_residual_norm=diagnostics["gauss_residual_norm"],
        yukawa_norm_drift=diagnostics["yukawa_norm_drift"],
        total_energy_proxy=total_energy_proxy,
    )


def _snapshot_values(snapshot: ScalingSnapshot) -> tuple[jnp.ndarray, ...]:
    return (
        snapshot.fermion_norm,
        snapshot.gauge_hamiltonian_density,
        snapshot.higgs_total_energy,
        snapshot.higgs_energy_density_mean,
        snapshot.gauss_residual_norm,
        snapshot.yukawa_norm_drift,
        snapshot.total_energy_proxy,
    )


def _finite_snapshot(snapshot: ScalingSnapshot) -> jnp.ndarray:
    return jnp.all(jnp.asarray([jnp.all(jnp.isfinite(value)) for value in _snapshot_values(snapshot)]))


def _advance_scaling_fields(fields: ScalingInitialState, config: ScalingRunConfig) -> ScalingInitialState:
    """Advance one coupled step, preserving exact zero-step controls cheaply."""

    if config.step_size == 0:
        return fields
    stepped = jax_patisalam_fermion_gauge_higgs_step(
        fields.state,
        fields.links,
        fields.momenta,
        fields.phi,
        fields.higgs_momentum,
        fields.higgs_links,
        sector=config.sector,
        step_size=config.step_size,
        matter_coupling=config.matter_coupling,
        yukawa_coupling=config.yukawa_coupling,
        beta=config.beta,
        vev_squared=config.vev_squared,
        quartic=config.quartic,
        shapes=_selected_shapes(config),
        force_method=config.force_method,
        force_epsilon=config.force_epsilon,
        force_chunk_size=config.force_chunk_size,
        current_epsilon=config.current_epsilon,
        yukawa_mode=config.yukawa_mode,
    )
    return ScalingInitialState(
        state=stepped[0],
        links=stepped[1],
        momenta=stepped[2],
        phi=stepped[3],
        higgs_momentum=stepped[4],
        higgs_links=stepped[5],
    )


def jax_advance_scaling_fields(fields: ScalingInitialState, config: ScalingRunConfig) -> ScalingInitialState:
    """Advance one deterministic coupled scaling state by one configured step."""

    return _advance_scaling_fields(fields, config)


def jax_coupled_scaling_trial(config: ScalingRunConfig) -> ScalingTrial:
    """Return one before/after coupled-step scaling trial."""

    fields = jax_default_scaling_initial_state(config)
    before = jax_scaling_snapshot(
        fields.state,
        fields.links,
        fields.momenta,
        fields.phi,
        fields.higgs_momentum,
        fields.higgs_links,
        config=config,
        reference_state=fields.state,
    )
    after_fields = _advance_scaling_fields(fields, config)
    after = jax_scaling_snapshot(
        after_fields.state,
        after_fields.links,
        after_fields.momenta,
        after_fields.phi,
        after_fields.higgs_momentum,
        after_fields.higgs_links,
        config=config,
        reference_state=fields.state,
    )
    return ScalingTrial(
        before=before,
        after=after,
        fermion_norm_drift=jnp.abs(after.fermion_norm - before.fermion_norm),
        gauge_energy_drift=jnp.abs(after.gauge_hamiltonian_density - before.gauge_hamiltonian_density),
        higgs_energy_drift=jnp.abs(after.higgs_total_energy - before.higgs_total_energy),
        gauss_residual_drift=jnp.abs(after.gauss_residual_norm - before.gauss_residual_norm),
        total_energy_proxy_drift=jnp.abs(after.total_energy_proxy - before.total_energy_proxy),
        all_finite=_finite_snapshot(before) & _finite_snapshot(after),
    )


def _trajectory_sample(
    step_index: int,
    snapshot: ScalingSnapshot,
    initial_snapshot: ScalingSnapshot,
) -> ScalingTrajectorySample:
    return ScalingTrajectorySample(
        step_index=step_index,
        snapshot=snapshot,
        fermion_norm_drift=jnp.abs(snapshot.fermion_norm - initial_snapshot.fermion_norm),
        gauge_energy_drift=jnp.abs(snapshot.gauge_hamiltonian_density - initial_snapshot.gauge_hamiltonian_density),
        higgs_energy_drift=jnp.abs(snapshot.higgs_total_energy - initial_snapshot.higgs_total_energy),
        gauss_residual_drift=jnp.abs(snapshot.gauss_residual_norm - initial_snapshot.gauss_residual_norm),
        total_energy_proxy_drift=jnp.abs(snapshot.total_energy_proxy - initial_snapshot.total_energy_proxy),
    )


def _max_sample_value(samples: tuple[ScalingTrajectorySample, ...], attr: str) -> jnp.ndarray:
    return jnp.max(jnp.asarray([getattr(sample, attr) for sample in samples]))


def _finite_sample(sample: ScalingTrajectorySample) -> jnp.ndarray:
    return _finite_snapshot(sample.snapshot) & jnp.all(
        jnp.asarray(
            [
                jnp.isfinite(sample.fermion_norm_drift),
                jnp.isfinite(sample.gauge_energy_drift),
                jnp.isfinite(sample.higgs_energy_drift),
                jnp.isfinite(sample.gauss_residual_drift),
                jnp.isfinite(sample.total_energy_proxy_drift),
            ],
        ),
    )


def jax_coupled_scaling_trajectory(
    config: ScalingRunConfig,
    *,
    steps: int = 4,
    record_every: int = 1,
) -> ScalingTrajectory:
    """Return a recorded tiny-lattice trajectory for the coupled prototype.

    This is a Python-level measurement helper, not a jitted simulation loop.
    Keep `steps` small and use it for bounded diagnostics.
    """

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    if record_every <= 0:
        raise ValueError(f"record_every must be positive, got {record_every}")

    initial_fields = jax_default_scaling_initial_state(config)
    fields = initial_fields
    initial_snapshot = jax_scaling_snapshot(
        fields.state,
        fields.links,
        fields.momenta,
        fields.phi,
        fields.higgs_momentum,
        fields.higgs_links,
        config=config,
        reference_state=initial_fields.state,
    )
    samples = [_trajectory_sample(0, initial_snapshot, initial_snapshot)]

    for step_index in range(1, steps + 1):
        fields = _advance_scaling_fields(fields, config)
        should_record = step_index % record_every == 0 or step_index == steps
        if should_record:
            snapshot = jax_scaling_snapshot(
                fields.state,
                fields.links,
                fields.momenta,
                fields.phi,
                fields.higgs_momentum,
                fields.higgs_links,
                config=config,
                reference_state=initial_fields.state,
            )
            samples.append(_trajectory_sample(step_index, snapshot, initial_snapshot))

    sample_tuple = tuple(samples)
    return ScalingTrajectory(
        samples=sample_tuple,
        initial=sample_tuple[0],
        final=sample_tuple[-1],
        max_fermion_norm_drift=_max_sample_value(sample_tuple, "fermion_norm_drift"),
        max_gauge_energy_drift=_max_sample_value(sample_tuple, "gauge_energy_drift"),
        max_higgs_energy_drift=_max_sample_value(sample_tuple, "higgs_energy_drift"),
        max_gauss_residual_drift=_max_sample_value(sample_tuple, "gauss_residual_drift"),
        max_total_energy_proxy_drift=_max_sample_value(sample_tuple, "total_energy_proxy_drift"),
        all_finite=jnp.all(jnp.asarray([_finite_sample(sample) for sample in sample_tuple])),
    )


def _safe_ratio(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(denominator == 0, jnp.where(numerator == 0, 0.0, jnp.inf), numerator / denominator)


def jax_compare_yukawa_modes(
    config: ScalingRunConfig,
    *,
    steps: int = 4,
    record_every: int = 1,
) -> YukawaModeComparison:
    """Compare first-order and exact-unitary Yukawa updates on the same probe."""

    first_order = jax_coupled_scaling_trajectory(
        replace(config, yukawa_mode="first_order"),
        steps=steps,
        record_every=record_every,
    )
    unitary = jax_coupled_scaling_trajectory(
        replace(config, yukawa_mode="unitary"),
        steps=steps,
        record_every=record_every,
    )
    return YukawaModeComparison(
        first_order=first_order,
        unitary=unitary,
        fermion_norm_drift_ratio=_safe_ratio(
            unitary.max_fermion_norm_drift,
            first_order.max_fermion_norm_drift,
        ),
        total_energy_proxy_drift_ratio=_safe_ratio(
            unitary.max_total_energy_proxy_drift,
            first_order.max_total_energy_proxy_drift,
        ),
        all_finite=first_order.all_finite & unitary.all_finite,
    )


def jax_scaling_timing_probe(config: ScalingRunConfig, *, steps: int = 2) -> JaxTiming:
    """Return compile/run timings for a tiny jitted coupled-step kernel.

    The returned timings are informational.  A zero `step_size` compiles only
    the field-packing/summarization path, which is useful as a cheap timing
    smoke test without entering the coupled force kernels.
    """

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")

    fields = jax_default_scaling_initial_state(config)

    def kernel(
        state: jnp.ndarray,
        links: jnp.ndarray,
        momenta: jnp.ndarray,
        phi: jnp.ndarray,
        higgs_momentum: jnp.ndarray,
        higgs_links: jnp.ndarray,
    ) -> jnp.ndarray:
        current = (state, links, momenta, phi, higgs_momentum, higgs_links)
        if config.step_size != 0:
            for _ in range(steps):
                current = jax_patisalam_fermion_gauge_higgs_step(
                    current[0],
                    current[1],
                    current[2],
                    current[3],
                    current[4],
                    current[5],
                    sector=config.sector,
                    step_size=config.step_size,
                    matter_coupling=config.matter_coupling,
                    yukawa_coupling=config.yukawa_coupling,
                    beta=config.beta,
                    vev_squared=config.vev_squared,
                    quartic=config.quartic,
                    shapes=_selected_shapes(config),
                    force_method=config.force_method,
                    force_epsilon=config.force_epsilon,
                    force_chunk_size=config.force_chunk_size,
                    current_epsilon=config.current_epsilon,
                    yukawa_mode=config.yukawa_mode,
                )
        return (
            jnp.real(jnp.sum(current[0]))
            + jnp.real(jnp.sum(current[1]))
            + jnp.sum(current[2])
            + jnp.real(jnp.sum(current[3]))
            + jnp.real(jnp.sum(current[4]))
            + jnp.real(jnp.sum(current[5]))
        )

    return benchmark_jitted_kernel(
        kernel,
        fields.state,
        fields.links,
        fields.momenta,
        fields.phi,
        fields.higgs_momentum,
        fields.higgs_links,
    )


def jax_step_size_scaling_sweep(
    config: ScalingRunConfig,
    step_sizes: tuple[float, ...],
) -> tuple[ScalingTrial, ...]:
    """Return one-step scaling trials for the supplied step sizes."""

    return tuple(jax_coupled_scaling_trial(replace(config, step_size=step_size)) for step_size in step_sizes)


def jax_neutral_vacuum_density_probe(
    lattice_shapes: tuple[tuple[int, int, int], ...],
) -> tuple[NeutralVacuumDensityProbe, ...]:
    """Probe density normalization for neutral vacuum controls."""

    probes: list[NeutralVacuumDensityProbe] = []
    for lattice_shape in lattice_shapes:
        _validate_lattice_shape(lattice_shape)
        links = jax_identity_link_field(lattice_shape, PATISALAM_INTERNAL_DIM, dtype=jnp.complex64)
        momenta = jnp.zeros((*lattice_shape, 8, 1), dtype=jnp.float32)
        phi = jax_constant_higgs_field(lattice_shape, phi_plus=0.0 + 0.0j, phi_zero=1.0 + 0.0j)
        higgs_links = jax_higgs_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, 4), dtype=jnp.float32))
        gauge_density = jax_patisalam_gauge_hamiltonian_density(
            links,
            momenta,
            sector="u1_y",
            shapes=(tuple(canonical_bcc_plaquette_shapes())[0],),
        )
        higgs_density = jnp.mean(
            jax_higgs_energy_density(
                phi,
                higgs_links,
                vev_squared=1.0,
                quartic=1.0,
            ),
        )
        total = gauge_density + higgs_density
        probes.append(
            NeutralVacuumDensityProbe(
                lattice_shape=lattice_shape,
                gauge_hamiltonian_density=gauge_density,
                higgs_energy_density_mean=higgs_density,
                total_energy_proxy=total,
                all_zero=jnp.allclose(total, 0.0),
            ),
        )
    return tuple(probes)


def session43_scaling_audit_payload() -> Session43ScalingAuditPayload:
    """Return a stable summary payload for the Session 43 report."""

    default_config = ScalingRunConfig()
    return {
        "default_sector": default_config.sector,
        "default_lattice_shape": default_config.lattice_shape,
        "neutral_probe_shapes": ((1, 1, 1), (2, 1, 1)),
        "step_sizes": (0.0, 0.0025, 0.005),
        "notes": (
            "Session 43 packages existing coupled dynamics into tiny-lattice scaling probes.",
            "The total energy proxy excludes fermion norm and is not a full Hamiltonian.",
            "These diagnostics are stability controls, not a continuum renormalization proof.",
        ),
    }
