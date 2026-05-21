"""Warm kernel profiling for spacetime-QCA simulator internals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.profiling import RepeatedCallProfile, profile_callable_repeated
from clifford_3plus2_d5.spacetime_qca.jax_coupled_higgs import (
    HiggsCoupledSector,
    jax_patisalam_fermion_gauge_higgs_step,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import CompactLieForceMethod
from clifford_3plus2_d5.spacetime_qca.jax_gauss import jax_patisalam_fermion_link_current
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import jax_patisalam_link_field_from_algebra
from clifford_3plus2_d5.spacetime_qca.jax_scaling import (
    ScalingRunConfig,
    jax_default_scaling_initial_state,
)
from clifford_3plus2_d5.spacetime_qca.simulator.config import (
    SpacetimeSimulationConfig,
    scaling_config_from_spacetime_config,
)
from clifford_3plus2_d5.spacetime_qca.simulator.fields import fields_from_scaling_state
from clifford_3plus2_d5.spacetime_qca.simulator.observables import spacetime_observables


KernelProbeKind = Literal["initial_state", "link_field", "step_no_matter", "observables", "matter_current"]


@dataclass(frozen=True)
class KernelProfileCase:
    """One bounded warm kernel profiling case."""

    name: str
    kind: KernelProbeKind
    purpose: str
    config: SpacetimeSimulationConfig


def _sector_dimension(sector: HiggsCoupledSector) -> int:
    if sector == "u1_y":
        return 1
    if sector == "su2_l":
        return 3
    if sector == "sm":
        return 12
    raise ValueError(f"unsupported kernel-profile sector: {sector!r}")


def _profile_theta(config: ScalingRunConfig) -> jnp.ndarray:
    sector_dim = _sector_dimension(config.sector)
    count = int(np.prod(np.asarray(config.lattice_shape))) * 8 * sector_dim
    values = jnp.arange(count, dtype=jnp.float32).reshape((*config.lattice_shape, 8, sector_dim))
    return 2.5e-4 * (values + 1.0)


def default_kernel_profile_cases(*, include_current: bool = False) -> tuple[KernelProfileCase, ...]:
    """Return bounded warm kernel cases for Session 50."""

    base = SpacetimeSimulationConfig(
        lattice_shape=(1, 1, 1),
        steps=1,
        record_every=1,
        step_size=0.0025,
        matter_coupling=0.0,
        use_jit=False,
        yukawa_mode="unitary",
    )
    cases: list[KernelProfileCase] = [
        KernelProfileCase(
            name="initial_state_u1_y",
            kind="initial_state",
            purpose="Construct deterministic scaling initial fields.",
            config=replace(base, sector="u1_y", label="kernel_initial_state_u1_y"),
        ),
        KernelProfileCase(
            name="observables_u1_y",
            kind="observables",
            purpose="Extract scalar simulator observables from prebuilt fields.",
            config=replace(base, sector="u1_y", label="kernel_observables_u1_y"),
        ),
    ]
    for sector in ("u1_y", "su2_l", "sm"):
        cases.append(
            KernelProfileCase(
                name=f"link_field_{sector}",
                kind="link_field",
                purpose=f"Build {sector} chiral16 BCC links from algebra coordinates.",
                config=replace(base, sector=sector, label=f"kernel_link_field_{sector}"),
            ),
        )
    for sector in ("u1_y", "su2_l", "sm"):
        cases.append(
            KernelProfileCase(
                name=f"step_no_matter_{sector}",
                kind="step_no_matter",
                purpose=f"Run one {sector} coupled step without matter-current backreaction.",
                config=replace(base, sector=sector, label=f"kernel_step_no_matter_{sector}"),
            ),
        )
    if include_current:
        cases.append(
            KernelProfileCase(
                name="matter_current_u1_y",
                kind="matter_current",
                purpose="Compute finite-difference U(1)_Y matter current on prebuilt fields.",
                config=replace(base, sector="u1_y", matter_coupling=1.0, label="kernel_matter_current_u1_y"),
            ),
        )
    return tuple(cases)


def _config_payload(config: SpacetimeSimulationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["lattice_shape"] = tuple(config.lattice_shape)
    return payload


def _validate_force_chunk_size(force_chunk_size: int | None) -> None:
    if force_chunk_size is not None and force_chunk_size <= 0:
        raise ValueError(f"force_chunk_size must be positive when set, got {force_chunk_size}")


def _with_force_overrides(
    case: KernelProfileCase,
    *,
    force_method: CompactLieForceMethod | None,
    force_chunk_size: int | None,
) -> KernelProfileCase:
    if case.kind != "step_no_matter":
        return case

    config = case.config
    if force_method is not None:
        config = replace(config, force_method=force_method)
    if force_chunk_size is not None:
        config = replace(config, force_chunk_size=force_chunk_size)
    return replace(case, config=config)


def _array_output_summary(value: object) -> dict[str, Any]:
    array = jnp.asarray(value)
    return {
        "shape": tuple(int(size) for size in array.shape),
        "dtype": str(array.dtype),
    }


def _dict_output_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    return {
        "keys": tuple(sorted(str(key) for key in value)),
        "shapes": {
            str(key): tuple(int(size) for size in jnp.asarray(item).shape)
            for key, item in value.items()
        },
    }


def _fields_output_summary(value: object) -> dict[str, Any]:
    fields = value
    return {
        "state": tuple(int(size) for size in fields.state.shape),  # type: ignore[attr-defined]
        "links": tuple(int(size) for size in fields.links.shape),  # type: ignore[attr-defined]
        "momenta": tuple(int(size) for size in fields.momenta.shape),  # type: ignore[attr-defined]
        "phi": tuple(int(size) for size in fields.phi.shape),  # type: ignore[attr-defined]
        "higgs_momentum": tuple(int(size) for size in fields.higgs_momentum.shape),  # type: ignore[attr-defined]
        "higgs_links": tuple(int(size) for size in fields.higgs_links.shape),  # type: ignore[attr-defined]
    }


def _kernel_callable_and_summary(case: KernelProfileCase) -> tuple[Any, Any]:
    scaling_config = scaling_config_from_spacetime_config(case.config)

    if case.kind == "initial_state":
        return lambda: fields_from_scaling_state(jax_default_scaling_initial_state(scaling_config)), _fields_output_summary

    if case.kind == "link_field":
        theta = _profile_theta(scaling_config)
        return (
            lambda: jax_patisalam_link_field_from_algebra(theta, sector=scaling_config.sector),
            _array_output_summary,
        )

    fields = jax_default_scaling_initial_state(scaling_config)

    if case.kind == "step_no_matter":
        return (
            lambda: jax_patisalam_fermion_gauge_higgs_step(
                fields.state,
                fields.links,
                fields.momenta,
                fields.phi,
                fields.higgs_momentum,
                fields.higgs_links,
                sector=scaling_config.sector,
                step_size=scaling_config.step_size,
                matter_coupling=0.0,
                yukawa_coupling=scaling_config.yukawa_coupling,
                beta=scaling_config.beta,
                vev_squared=scaling_config.vev_squared,
                quartic=scaling_config.quartic,
                shapes=scaling_config.shapes,
                force_method=scaling_config.force_method,
                force_epsilon=scaling_config.force_epsilon,
                force_chunk_size=scaling_config.force_chunk_size,
                current_epsilon=scaling_config.current_epsilon,
                yukawa_mode=scaling_config.yukawa_mode,
            ),
            lambda value: {"field_count": len(value), "state_shape": tuple(int(size) for size in value[0].shape)},
        )

    if case.kind == "observables":
        simulator_fields = fields_from_scaling_state(fields)
        return (
            lambda: spacetime_observables(
                simulator_fields,
                config=scaling_config,
                reference_state=simulator_fields.state,
            ),
            _dict_output_summary,
        )

    if case.kind == "matter_current":
        return (
            lambda: jax_patisalam_fermion_link_current(
                fields.state,
                fields.links,
                sector=scaling_config.sector,
                epsilon=scaling_config.current_epsilon,
            ),
            _array_output_summary,
        )

    raise ValueError(f"unsupported kernel profile kind: {case.kind!r}")


def run_kernel_profile_case(
    case: KernelProfileCase,
    *,
    warmup_runs: int = 1,
    timed_runs: int = 3,
) -> RepeatedCallProfile:
    """Run one warm/repeated kernel profile case."""

    fn, summary_fn = _kernel_callable_and_summary(case)
    return profile_callable_repeated(
        case.name,
        fn,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        metadata={
            "kind": case.kind,
            "purpose": case.purpose,
            "config": _config_payload(case.config),
        },
        output_summary_fn=summary_fn,
    )


def _mean_by_case(case_payloads: tuple[dict[str, Any], ...]) -> dict[str, float]:
    return {str(payload["label"]): float(payload["mean_seconds"]) for payload in case_payloads}


def recommend_kernel_bottleneck(case_payloads: tuple[dict[str, Any], ...]) -> str:
    """Return a deterministic next optimization target from warm kernel data."""

    means = _mean_by_case(case_payloads)
    if not means:
        return "No kernel profile cases were run; no bottleneck recommendation is available."

    current = means.get("matter_current_u1_y")
    step_u1 = means.get("step_no_matter_u1_y", 0.0)
    step_sm = means.get("step_no_matter_sm", 0.0)
    link_sm = means.get("link_field_sm", 0.0)
    obs_u1 = means.get("observables_u1_y", 0.0)

    if current is not None and current > max(step_u1, 1e-12):
        return "First target: finite-difference matter current; it dominates the U(1)_Y no-matter step."
    if "step_no_matter_sm" in means and "step_no_matter_u1_y" not in means:
        return "First target: split the SM no-matter step; no U(1)_Y baseline was included in this profile."
    if link_sm > 0.5 * max(step_sm, 1e-12):
        return "First target: SM link exponentials from algebra coordinates; link construction is a large SM-step fraction."
    if step_sm > 2.0 * max(step_u1, 1e-12):
        return "First target: full-SM no-matter step path; warm timing still scales much worse than U(1)_Y."
    if obs_u1 > max(step_u1, 1e-12):
        return "First target: observable extraction; diagnostics are more expensive than the U(1)_Y step."
    return "No single warm kernel bottleneck was isolated; repeat with include_current or larger bounded probes."


def run_spacetime_kernel_profile(
    *,
    case_names: tuple[str, ...] | None = None,
    include_current: bool = False,
    warmup_runs: int = 1,
    timed_runs: int = 3,
    force_method: CompactLieForceMethod | None = None,
    force_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Run bounded warm kernel profiling cases for Session 50."""

    _validate_force_chunk_size(force_chunk_size)
    cases = default_kernel_profile_cases(include_current=include_current)
    if case_names is not None:
        requested = set(case_names)
        known = {case.name for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown kernel profile case(s): {', '.join(unknown)}")
        cases = tuple(case for case in cases if case.name in requested)
    cases = tuple(
        _with_force_overrides(case, force_method=force_method, force_chunk_size=force_chunk_size)
        for case in cases
    )

    profiles = tuple(
        run_kernel_profile_case(case, warmup_runs=warmup_runs, timed_runs=timed_runs).as_payload()
        for case in cases
    )
    return {
        "metadata": {
            "runner": "spacetime_qca.simulator.kernel_profile",
            "lattice_shape": (1, 1, 1),
            "include_current": include_current,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "case_count": len(cases),
            "force_method_override": force_method,
            "force_chunk_size_override": force_chunk_size,
        },
        "cases": profiles,
        "recommendation": recommend_kernel_bottleneck(profiles),
    }
