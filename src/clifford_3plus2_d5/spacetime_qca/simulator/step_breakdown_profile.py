"""Step-breakdown profiling for the expensive full-SM coupled path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.sim.profiling import RepeatedCallProfile, profile_callable_repeated
from clifford_3plus2_d5.spacetime_qca.jax_coupled_higgs import (
    jax_apply_site_local_yukawa_update,
    jax_higgs_leapfrog_step,
    jax_patisalam_fermion_gauge_higgs_diagnostics,
)
from clifford_3plus2_d5.spacetime_qca.jax_fermion_gauge import (
    jax_patisalam_dirac_step,
    jax_patisalam_fermion_gauge_step,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauss import jax_patisalam_gauss_residual
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import (
    jax_patisalam_apply_momentum_update,
    jax_patisalam_gauge_hamiltonian_density,
    jax_patisalam_left_force,
    jax_patisalam_leapfrog_step,
)
from clifford_3plus2_d5.spacetime_qca.jax_scaling import jax_default_scaling_initial_state
from clifford_3plus2_d5.spacetime_qca.simulator.config import (
    SpacetimeSimulationConfig,
    scaling_config_from_spacetime_config,
)


StepBreakdownKind = Literal[
    "yukawa_half_kick",
    "fermion_gauge_no_matter",
    "higgs_leapfrog",
    "yukawa_final_half_kick",
    "diagnostics",
    "gauss_residual",
    "gauge_hamiltonian",
    "left_force",
    "gauge_leapfrog",
    "dirac_transport",
    "momentum_update",
    "first_left_force",
    "second_left_force",
]


@dataclass(frozen=True)
class StepBreakdownCase:
    """One bounded sub-kernel profile case inside the full-SM step path."""

    name: str
    kind: StepBreakdownKind
    purpose: str
    config: SpacetimeSimulationConfig


def default_step_breakdown_cases() -> tuple[StepBreakdownCase, ...]:
    """Return bounded Session 51 profiling cases for the SM coupled step."""

    base = SpacetimeSimulationConfig(
        lattice_shape=(1, 1, 1),
        sector="sm",
        steps=1,
        record_every=1,
        step_size=0.0025,
        matter_coupling=0.0,
        use_jit=False,
        yukawa_mode="unitary",
    )
    return (
        StepBreakdownCase(
            name="yukawa_half_kick_sm",
            kind="yukawa_half_kick",
            purpose="Apply the first half site-local unitary Yukawa kick.",
            config=replace(base, label="breakdown_yukawa_half_kick_sm"),
        ),
        StepBreakdownCase(
            name="fermion_gauge_no_matter_sm",
            kind="fermion_gauge_no_matter",
            purpose="Run the no-backreaction SM gauge leapfrog plus Dirac transport.",
            config=replace(base, label="breakdown_fermion_gauge_no_matter_sm"),
        ),
        StepBreakdownCase(
            name="higgs_leapfrog_sm",
            kind="higgs_leapfrog",
            purpose="Run the fixed-link Higgs leapfrog update.",
            config=replace(base, label="breakdown_higgs_leapfrog_sm"),
        ),
        StepBreakdownCase(
            name="yukawa_final_half_kick_sm",
            kind="yukawa_final_half_kick",
            purpose="Apply the final half site-local unitary Yukawa kick.",
            config=replace(base, label="breakdown_yukawa_final_half_kick_sm"),
        ),
        StepBreakdownCase(
            name="diagnostics_sm",
            kind="diagnostics",
            purpose="Compute the coupled SM diagnostics payload.",
            config=replace(base, label="breakdown_diagnostics_sm"),
        ),
        StepBreakdownCase(
            name="gauss_residual_sm",
            kind="gauss_residual",
            purpose="Compute the SM Gauss residual only.",
            config=replace(base, label="breakdown_gauss_residual_sm"),
        ),
        StepBreakdownCase(
            name="gauge_hamiltonian_sm",
            kind="gauge_hamiltonian",
            purpose="Compute the SM gauge Hamiltonian density only.",
            config=replace(base, label="breakdown_gauge_hamiltonian_sm"),
        ),
        StepBreakdownCase(
            name="left_force_sm",
            kind="left_force",
            purpose="Compute the finite-difference SM Wilson left-force only.",
            config=replace(base, label="breakdown_left_force_sm"),
        ),
        StepBreakdownCase(
            name="left_force_batched_sm",
            kind="left_force",
            purpose="Compute the batched finite-difference SM Wilson left-force only.",
            config=replace(
                base,
                force_method="finite_difference_batched",
                force_chunk_size=16,
                label="breakdown_left_force_batched_sm",
            ),
        ),
        StepBreakdownCase(
            name="gauge_leapfrog_sm",
            kind="gauge_leapfrog",
            purpose="Run the pure SM gauge leapfrog without Dirac transport.",
            config=replace(base, label="breakdown_gauge_leapfrog_sm"),
        ),
        StepBreakdownCase(
            name="gauge_leapfrog_batched_sm",
            kind="gauge_leapfrog",
            purpose="Run the pure SM gauge leapfrog with batched finite-difference forces.",
            config=replace(
                base,
                force_method="finite_difference_batched",
                force_chunk_size=16,
                label="breakdown_gauge_leapfrog_batched_sm",
            ),
        ),
        StepBreakdownCase(
            name="dirac_transport_sm",
            kind="dirac_transport",
            purpose="Run BCC Dirac transport through prebuilt SM links only.",
            config=replace(base, label="breakdown_dirac_transport_sm"),
        ),
        StepBreakdownCase(
            name="momentum_update_sm",
            kind="momentum_update",
            purpose="Apply one compact SM momentum link update only.",
            config=replace(base, label="breakdown_momentum_update_sm"),
        ),
        StepBreakdownCase(
            name="first_left_force_sm",
            kind="first_left_force",
            purpose="Compute the first finite-difference SM Wilson left-force on initial links.",
            config=replace(base, label="breakdown_first_left_force_sm"),
        ),
        StepBreakdownCase(
            name="first_left_force_batched_sm",
            kind="first_left_force",
            purpose="Compute the first batched finite-difference SM Wilson left-force on initial links.",
            config=replace(
                base,
                force_method="finite_difference_batched",
                force_chunk_size=16,
                label="breakdown_first_left_force_batched_sm",
            ),
        ),
        StepBreakdownCase(
            name="second_left_force_sm",
            kind="second_left_force",
            purpose="Compute the second finite-difference SM Wilson left-force after one momentum update.",
            config=replace(base, label="breakdown_second_left_force_sm"),
        ),
        StepBreakdownCase(
            name="second_left_force_batched_sm",
            kind="second_left_force",
            purpose="Compute the second batched finite-difference SM Wilson left-force after one momentum update.",
            config=replace(
                base,
                force_method="finite_difference_batched",
                force_chunk_size=16,
                label="breakdown_second_left_force_batched_sm",
            ),
        ),
    )


def _config_payload(config: SpacetimeSimulationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["lattice_shape"] = tuple(config.lattice_shape)
    return payload


def _block_until_ready(value: Any) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _block_until_ready(getattr(value, field.name))
        return
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _block_until_ready(item)
        return
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _array_summary(value: object) -> dict[str, Any]:
    array = jnp.asarray(value)
    return {
        "type": "array",
        "shape": tuple(int(size) for size in array.shape),
        "dtype": str(array.dtype),
    }


def _tuple_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, tuple):
        return {"type": type(value).__name__}
    return {
        "type": "tuple",
        "length": len(value),
        "shapes": tuple(tuple(int(size) for size in jnp.asarray(item).shape) for item in value),
    }


def _dict_summary(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    return {
        "type": "dict",
        "keys": tuple(sorted(str(key) for key in value)),
        "shapes": {
            str(key): tuple(int(size) for size in jnp.asarray(item).shape)
            for key, item in value.items()
        },
    }


def _step_breakdown_callable_and_summary(case: StepBreakdownCase) -> tuple[Any, Any]:
    scaling_config = scaling_config_from_spacetime_config(case.config)
    fields = jax_default_scaling_initial_state(scaling_config)
    _block_until_ready(fields)

    if case.kind == "yukawa_half_kick":
        return (
            lambda: jax_apply_site_local_yukawa_update(
                fields.state,
                fields.phi,
                step_size=0.5 * scaling_config.step_size,
                yukawa_coupling=scaling_config.yukawa_coupling,
                mode=scaling_config.yukawa_mode,
            ),
            _array_summary,
        )

    if case.kind == "fermion_gauge_no_matter":
        half_kicked_state = jax_apply_site_local_yukawa_update(
            fields.state,
            fields.phi,
            step_size=0.5 * scaling_config.step_size,
            yukawa_coupling=scaling_config.yukawa_coupling,
            mode=scaling_config.yukawa_mode,
        )
        _block_until_ready(half_kicked_state)
        return (
            lambda: jax_patisalam_fermion_gauge_step(
                half_kicked_state,
                fields.links,
                fields.momenta,
                sector=scaling_config.sector,
                step_size=scaling_config.step_size,
                beta=scaling_config.beta,
                shapes=scaling_config.shapes,
                force_method=scaling_config.force_method,
                force_epsilon=scaling_config.force_epsilon,
                force_chunk_size=scaling_config.force_chunk_size,
            ),
            _tuple_summary,
        )

    if case.kind == "higgs_leapfrog":
        return (
            lambda: jax_higgs_leapfrog_step(
                fields.phi,
                fields.higgs_momentum,
                fields.higgs_links,
                step_size=scaling_config.step_size,
                vev_squared=scaling_config.vev_squared,
                quartic=scaling_config.quartic,
            ),
            _tuple_summary,
        )

    if case.kind == "yukawa_final_half_kick":
        updated_phi, _ = jax_higgs_leapfrog_step(
            fields.phi,
            fields.higgs_momentum,
            fields.higgs_links,
            step_size=scaling_config.step_size,
            vev_squared=scaling_config.vev_squared,
            quartic=scaling_config.quartic,
        )
        _block_until_ready(updated_phi)
        return (
            lambda: jax_apply_site_local_yukawa_update(
                fields.state,
                updated_phi,
                step_size=0.5 * scaling_config.step_size,
                yukawa_coupling=scaling_config.yukawa_coupling,
                mode=scaling_config.yukawa_mode,
            ),
            _array_summary,
        )

    if case.kind == "diagnostics":
        return (
            lambda: jax_patisalam_fermion_gauge_higgs_diagnostics(
                fields.state,
                fields.links,
                fields.momenta,
                fields.phi,
                fields.higgs_momentum,
                fields.higgs_links,
                sector=scaling_config.sector,
                beta=scaling_config.beta,
                matter_coupling=scaling_config.matter_coupling,
                vev_squared=scaling_config.vev_squared,
                quartic=scaling_config.quartic,
                shapes=scaling_config.shapes,
                reference_state=fields.state,
            ),
            _dict_summary,
        )

    if case.kind == "gauss_residual":
        return (
            lambda: jax_patisalam_gauss_residual(
                fields.state,
                fields.links,
                fields.momenta,
                sector=scaling_config.sector,
                matter_coupling=scaling_config.matter_coupling,
            ),
            _array_summary,
        )

    if case.kind == "gauge_hamiltonian":
        return (
            lambda: jax_patisalam_gauge_hamiltonian_density(
                fields.links,
                fields.momenta,
                sector=scaling_config.sector,
                beta=scaling_config.beta,
                shapes=scaling_config.shapes,
            ),
            _array_summary,
        )

    if case.kind == "left_force":
        return (
            lambda: jax_patisalam_left_force(
                fields.links,
                sector=scaling_config.sector,
                epsilon=scaling_config.force_epsilon,
                shapes=scaling_config.shapes,
                method=scaling_config.force_method,
                chunk_size=scaling_config.force_chunk_size,
            ),
            _array_summary,
        )

    if case.kind == "gauge_leapfrog":
        return (
            lambda: jax_patisalam_leapfrog_step(
                fields.links,
                fields.momenta,
                sector=scaling_config.sector,
                step_size=scaling_config.step_size,
                beta=scaling_config.beta,
                shapes=scaling_config.shapes,
                force_method=scaling_config.force_method,
                force_epsilon=scaling_config.force_epsilon,
                force_chunk_size=scaling_config.force_chunk_size,
            ),
            _tuple_summary,
        )

    if case.kind == "dirac_transport":
        return (
            lambda: jax_patisalam_dirac_step(fields.state, fields.links),
            _array_summary,
        )

    if case.kind == "momentum_update":
        return (
            lambda: jax_patisalam_apply_momentum_update(
                fields.links,
                fields.momenta,
                sector=scaling_config.sector,
                step_size=scaling_config.step_size,
            ),
            _array_summary,
        )

    if case.kind == "first_left_force":
        return (
            lambda: jax_patisalam_left_force(
                fields.links,
                sector=scaling_config.sector,
                epsilon=scaling_config.force_epsilon,
                shapes=scaling_config.shapes,
                method=scaling_config.force_method,
                chunk_size=scaling_config.force_chunk_size,
            ),
            _array_summary,
        )

    if case.kind == "second_left_force":
        updated_links = jax_patisalam_apply_momentum_update(
            fields.links,
            fields.momenta,
            sector=scaling_config.sector,
            step_size=scaling_config.step_size,
        )
        _block_until_ready(updated_links)
        return (
            lambda: jax_patisalam_left_force(
                updated_links,
                sector=scaling_config.sector,
                epsilon=scaling_config.force_epsilon,
                shapes=scaling_config.shapes,
                method=scaling_config.force_method,
                chunk_size=scaling_config.force_chunk_size,
            ),
            _array_summary,
        )

    raise ValueError(f"unsupported step breakdown kind: {case.kind!r}")


def run_step_breakdown_case(
    case: StepBreakdownCase,
    *,
    warmup_runs: int = 0,
    timed_runs: int = 1,
) -> RepeatedCallProfile:
    """Run one bounded step-breakdown profile case."""

    fn, summary_fn = _step_breakdown_callable_and_summary(case)
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


def recommend_step_breakdown_bottleneck(case_payloads: tuple[dict[str, Any], ...]) -> str:
    """Return a deterministic next target from SM step-breakdown timings."""

    means = _mean_by_case(case_payloads)
    if not means:
        return "No step-breakdown cases were run; no bottleneck recommendation is available."

    dominant = max(means, key=means.get)
    if dominant in {
        "left_force_sm",
        "first_left_force_sm",
        "second_left_force_sm",
        "left_force_batched_sm",
        "first_left_force_batched_sm",
        "second_left_force_batched_sm",
    }:
        return "First target: SM finite-difference left-force; compare batched force with a staple-like Wilson force path."
    if dominant in {"fermion_gauge_no_matter_sm", "gauge_leapfrog_sm", "gauge_leapfrog_batched_sm"}:
        return "First target: SM no-backreaction fermion/gauge step; decompose leapfrog force versus Dirac transport."
    if dominant == "dirac_transport_sm":
        return "First target: BCC Dirac transport through SM links; optimize internal-link transport."
    if dominant == "momentum_update_sm":
        return "First target: compact SM momentum link update; optimize chiral16 matrix exponentials."
    if dominant in {"gauge_hamiltonian_sm", "diagnostics_sm", "gauss_residual_sm"}:
        return f"First target: {dominant}; diagnostics or constraint probes dominate this breakdown."
    if dominant in {"yukawa_half_kick_sm", "yukawa_final_half_kick_sm"}:
        return "First target: unitary Yukawa local eigensolve; cache or specialize the site-local update."
    if dominant == "higgs_leapfrog_sm":
        return "First target: Higgs leapfrog force; inspect the Higgs gradient path."
    return f"First target: {dominant}; no specialized recommendation is available."


def run_spacetime_step_breakdown_profile(
    *,
    case_names: tuple[str, ...] | None = None,
    warmup_runs: int = 0,
    timed_runs: int = 1,
) -> dict[str, Any]:
    """Run bounded step-breakdown profiling cases for Session 51."""

    cases = default_step_breakdown_cases()
    selected_names = case_names if case_names is not None else ("higgs_leapfrog_sm",)
    requested = set(selected_names)
    known = {case.name for case in cases}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown step-breakdown profile case(s): {', '.join(unknown)}")
    cases = tuple(case for case in cases if case.name in requested)

    profiles = tuple(
        run_step_breakdown_case(case, warmup_runs=warmup_runs, timed_runs=timed_runs).as_payload()
        for case in cases
    )
    return {
        "metadata": {
            "runner": "spacetime_qca.simulator.step_breakdown_profile",
            "lattice_shape": (1, 1, 1),
            "sector": "sm",
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "case_count": len(cases),
            "default_case": case_names is None,
        },
        "cases": profiles,
        "recommendation": recommend_step_breakdown_bottleneck(profiles),
    }
