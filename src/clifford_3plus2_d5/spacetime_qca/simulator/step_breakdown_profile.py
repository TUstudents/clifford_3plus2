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
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import CompactLieForceMethod
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


FORCE_RELATED_STEP_BREAKDOWN_KINDS = {
    "fermion_gauge_no_matter",
    "left_force",
    "gauge_leapfrog",
    "first_left_force",
    "second_left_force",
}
DEFAULT_FORCE_COMPARISON_CHUNK_SIZES = (4, 8, 16, 32)

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
                force_chunk_size=32,
                label="breakdown_left_force_batched_sm",
            ),
        ),
        StepBreakdownCase(
            name="left_force_analytic_sm",
            kind="left_force",
            purpose="Compute the analytic-staple SM Wilson left-force only.",
            config=replace(
                base,
                force_method="analytic_staple",
                label="breakdown_left_force_analytic_sm",
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
                force_chunk_size=32,
                label="breakdown_gauge_leapfrog_batched_sm",
            ),
        ),
        StepBreakdownCase(
            name="gauge_leapfrog_analytic_sm",
            kind="gauge_leapfrog",
            purpose="Run the pure SM gauge leapfrog with analytic-staple force.",
            config=replace(
                base,
                force_method="analytic_staple",
                label="breakdown_gauge_leapfrog_analytic_sm",
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
                force_chunk_size=32,
                label="breakdown_first_left_force_batched_sm",
            ),
        ),
        StepBreakdownCase(
            name="first_left_force_analytic_sm",
            kind="first_left_force",
            purpose="Compute the first analytic-staple SM Wilson left-force on initial links.",
            config=replace(
                base,
                force_method="analytic_staple",
                label="breakdown_first_left_force_analytic_sm",
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
                force_chunk_size=32,
                label="breakdown_second_left_force_batched_sm",
            ),
        ),
        StepBreakdownCase(
            name="second_left_force_analytic_sm",
            kind="second_left_force",
            purpose="Compute the second analytic-staple SM Wilson left-force after one momentum update.",
            config=replace(
                base,
                force_method="analytic_staple",
                label="breakdown_second_left_force_analytic_sm",
            ),
        ),
    )


def _config_payload(config: SpacetimeSimulationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["lattice_shape"] = tuple(config.lattice_shape)
    return payload


def _validate_force_chunk_size(force_chunk_size: int | None) -> None:
    if force_chunk_size is not None and force_chunk_size <= 0:
        raise ValueError(f"force_chunk_size must be positive when set, got {force_chunk_size}")


def _case_with_force_override(
    case: StepBreakdownCase,
    *,
    force_method: CompactLieForceMethod | None = None,
    force_chunk_size: int | None = None,
) -> StepBreakdownCase:
    """Return a case with force controls overridden only when force-related."""

    _validate_force_chunk_size(force_chunk_size)
    if case.kind not in FORCE_RELATED_STEP_BREAKDOWN_KINDS:
        return case
    config = case.config
    if force_method is not None:
        config = replace(config, force_method=force_method)
    if force_chunk_size is not None:
        config = replace(config, force_chunk_size=force_chunk_size)
    return replace(case, config=config)


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
        "left_force_analytic_sm",
        "first_left_force_batched_sm",
        "first_left_force_analytic_sm",
        "second_left_force_batched_sm",
        "second_left_force_analytic_sm",
    }:
        return "First target: SM Wilson left-force; compare analytic staple force with finite-difference oracles."
    if dominant in {
        "fermion_gauge_no_matter_sm",
        "gauge_leapfrog_sm",
        "gauge_leapfrog_batched_sm",
        "gauge_leapfrog_analytic_sm",
    }:
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


def _speedup_against_baseline(baseline_seconds: float, candidate_seconds: float) -> float | None:
    if baseline_seconds <= 0 or candidate_seconds <= 0:
        return None
    return baseline_seconds / candidate_seconds


def _profile_with_speedup(profile: RepeatedCallProfile, *, baseline_seconds: float | None) -> dict[str, Any]:
    payload = profile.as_payload()
    payload["speedup_vs_scalar"] = (
        None
        if baseline_seconds is None
        else _speedup_against_baseline(baseline_seconds, float(payload["mean_seconds"]))
    )
    return payload


def _force_comparison_case(kind: Literal["first_left_force", "second_left_force"]) -> StepBreakdownCase:
    case_name = "first_left_force_sm" if kind == "first_left_force" else "second_left_force_sm"
    cases = {case.name: case for case in default_step_breakdown_cases()}
    return cases[case_name]


def _batched_force_comparison_case(
    kind: Literal["first_left_force", "second_left_force"],
    *,
    chunk_size: int,
) -> StepBreakdownCase:
    _validate_force_chunk_size(chunk_size)
    base = _force_comparison_case(kind)
    name = f"{base.name}_batched_chunk_{chunk_size}"
    return StepBreakdownCase(
        name=name,
        kind=base.kind,
        purpose=f"{base.purpose} Batched comparison chunk size {chunk_size}.",
        config=replace(
            base.config,
            force_method="finite_difference_batched",
            force_chunk_size=chunk_size,
            label=f"breakdown_{name}",
        ),
    )


def _best_batched_summary(
    *,
    kind: str,
    baseline_payload: dict[str, Any],
    candidate_payloads: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    if not candidate_payloads:
        return {
            "kind": kind,
            "baseline_seconds": float(baseline_payload["mean_seconds"]),
            "best_chunk_size": None,
            "best_seconds": None,
            "best_speedup": None,
        }
    best = min(candidate_payloads, key=lambda payload: float(payload["mean_seconds"]))
    config = best["metadata"]["config"]
    return {
        "kind": kind,
        "baseline_seconds": float(baseline_payload["mean_seconds"]),
        "best_chunk_size": config["force_chunk_size"],
        "best_seconds": float(best["mean_seconds"]),
        "best_speedup": best["speedup_vs_scalar"],
    }


def recommend_force_chunk_tuning(summary_payloads: tuple[dict[str, Any], ...]) -> str:
    """Return the next optimization decision from force-chunk comparison results."""

    best_seconds = tuple(
        float(payload["best_seconds"])
        for payload in summary_payloads
        if payload.get("best_seconds") is not None
    )
    if not best_seconds:
        return "No batched force candidates were profiled; keep scalar force as the reference."
    worst_best = max(best_seconds)
    if worst_best > 2.0:
        return "Next target: implement an analytic staple-like compact Wilson force; batched force remains dominant."
    if worst_best < 1.0:
        return "Next target: keep batched force and optimize the next non-force simulator bottleneck."
    return "Next target: keep batched force for SM smoke/prototype runs and profile gauge_leapfrog_batched_sm."


def run_force_chunk_comparison(
    *,
    chunk_sizes: tuple[int, ...] = DEFAULT_FORCE_COMPARISON_CHUNK_SIZES,
    warmup_runs: int = 0,
    timed_runs: int = 1,
) -> dict[str, Any]:
    """Compare scalar and batched SM left-force timings for selected chunks."""

    if not chunk_sizes:
        raise ValueError("chunk_sizes must contain at least one value")
    for chunk_size in chunk_sizes:
        _validate_force_chunk_size(chunk_size)

    baseline_payloads = []
    candidate_payloads = []
    summary_payloads = []
    for kind in ("first_left_force", "second_left_force"):
        baseline_case = _force_comparison_case(kind)
        baseline_profile = run_step_breakdown_case(
            baseline_case,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )
        baseline_payload = _profile_with_speedup(baseline_profile, baseline_seconds=None)
        baseline_payloads.append(baseline_payload)
        baseline_seconds = float(baseline_payload["mean_seconds"])

        kind_candidates = []
        for chunk_size in chunk_sizes:
            candidate_case = _batched_force_comparison_case(kind, chunk_size=chunk_size)
            candidate_profile = run_step_breakdown_case(
                candidate_case,
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
            )
            candidate_payload = _profile_with_speedup(candidate_profile, baseline_seconds=baseline_seconds)
            candidate_payloads.append(candidate_payload)
            kind_candidates.append(candidate_payload)

        summary_payloads.append(
            _best_batched_summary(
                kind=kind,
                baseline_payload=baseline_payload,
                candidate_payloads=tuple(kind_candidates),
            ),
        )

    summaries = tuple(summary_payloads)
    return {
        "metadata": {
            "runner": "spacetime_qca.simulator.force_chunk_comparison",
            "lattice_shape": (1, 1, 1),
            "sector": "sm",
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "chunk_sizes": chunk_sizes,
            "baseline_method": "finite_difference",
            "candidate_method": "finite_difference_batched",
        },
        "baselines": tuple(baseline_payloads),
        "candidates": tuple(candidate_payloads),
        "summary": summaries,
        "recommendation": recommend_force_chunk_tuning(summaries),
    }


def run_spacetime_step_breakdown_profile(
    *,
    case_names: tuple[str, ...] | None = None,
    warmup_runs: int = 0,
    timed_runs: int = 1,
    force_method: CompactLieForceMethod | None = None,
    force_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Run bounded step-breakdown profiling cases for Session 51."""

    _validate_force_chunk_size(force_chunk_size)
    cases = default_step_breakdown_cases()
    selected_names = case_names if case_names is not None else ("higgs_leapfrog_sm",)
    requested = set(selected_names)
    known = {case.name for case in cases}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown step-breakdown profile case(s): {', '.join(unknown)}")
    cases = tuple(
        _case_with_force_override(
            case,
            force_method=force_method,
            force_chunk_size=force_chunk_size,
        )
        for case in cases
        if case.name in requested
    )

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
            "force_method_override": force_method,
            "force_chunk_size_override": force_chunk_size,
        },
        "cases": profiles,
        "recommendation": recommend_step_breakdown_bottleneck(profiles),
    }
