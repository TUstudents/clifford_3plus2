"""Memory-safe profiling for the scan-backed spacetime-QCA simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from clifford_3plus2_d5.sim.profiling import CallProfile, profile_callable
from clifford_3plus2_d5.spacetime_qca.simulator.config import SpacetimeSimulationConfig
from clifford_3plus2_d5.spacetime_qca.simulator.runner import (
    run_spacetime_simulation,
    spacetime_simulation_summary,
)


@dataclass(frozen=True)
class SpacetimeProfileCase:
    """One bounded simulator profiling case."""

    name: str
    purpose: str
    config: SpacetimeSimulationConfig


def default_spacetime_profile_cases() -> tuple[SpacetimeProfileCase, ...]:
    """Return the fixed tiny profiling suite for Session 49."""

    base = SpacetimeSimulationConfig(lattice_shape=(1, 1, 1), use_jit=False, yukawa_mode="unitary")
    return (
        SpacetimeProfileCase(
            name="runner_zero_step",
            purpose="Baseline runner construction and initial observation.",
            config=replace(base, steps=0, step_size=0.0, matter_coupling=0.0, label="profile_runner_zero_step"),
        ),
        SpacetimeProfileCase(
            name="scan_zero_step_4",
            purpose="Scan and per-step observation overhead without physics updates.",
            config=replace(
                base,
                steps=4,
                record_every=4,
                step_size=0.0,
                matter_coupling=0.0,
                label="profile_scan_zero_step_4",
            ),
        ),
        SpacetimeProfileCase(
            name="scan_zero_step_16",
            purpose="Longer zero-step scan overhead and observation scaling.",
            config=replace(
                base,
                steps=16,
                record_every=16,
                step_size=0.0,
                matter_coupling=0.0,
                label="profile_scan_zero_step_16",
            ),
        ),
        SpacetimeProfileCase(
            name="physics_no_matter",
            purpose="One coupled update without finite-difference matter current.",
            config=replace(
                base,
                steps=1,
                step_size=0.0025,
                matter_coupling=0.0,
                sector="u1_y",
                label="profile_physics_no_matter",
            ),
        ),
        SpacetimeProfileCase(
            name="physics_with_matter",
            purpose="One coupled update with finite-difference matter current enabled.",
            config=replace(
                base,
                steps=1,
                step_size=0.0025,
                matter_coupling=1.0,
                sector="u1_y",
                label="profile_physics_with_matter",
            ),
        ),
        SpacetimeProfileCase(
            name="sector_u1_y",
            purpose="Physical-hypercharge sector one-step cost without matter current.",
            config=replace(
                base,
                steps=1,
                step_size=0.0025,
                matter_coupling=0.0,
                sector="u1_y",
                label="profile_sector_u1_y",
            ),
        ),
        SpacetimeProfileCase(
            name="sector_su2_l",
            purpose="SU(2)_L sector one-step cost without matter current.",
            config=replace(
                base,
                steps=1,
                step_size=0.0025,
                matter_coupling=0.0,
                sector="su2_l",
                label="profile_sector_su2_l",
            ),
        ),
        SpacetimeProfileCase(
            name="sector_sm",
            purpose="Full SM-sector one-step cost without matter current.",
            config=replace(
                base,
                steps=1,
                step_size=0.0025,
                matter_coupling=0.0,
                sector="sm",
                label="profile_sector_sm",
            ),
        ),
    )


def _config_payload(config: SpacetimeSimulationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["lattice_shape"] = tuple(config.lattice_shape)
    return payload


def _profile_output_summary(result: object) -> dict[str, Any]:
    summary = spacetime_simulation_summary(result)  # type: ignore[arg-type]
    final_fields = result.final_fields  # type: ignore[attr-defined]
    return {
        "summary": summary,
        "final_field_shapes": {
            "state": tuple(int(size) for size in final_fields.state.shape),
            "links": tuple(int(size) for size in final_fields.links.shape),
            "momenta": tuple(int(size) for size in final_fields.momenta.shape),
            "phi": tuple(int(size) for size in final_fields.phi.shape),
            "higgs_momentum": tuple(int(size) for size in final_fields.higgs_momentum.shape),
            "higgs_links": tuple(int(size) for size in final_fields.higgs_links.shape),
        },
    }


def _run_case(case: SpacetimeProfileCase) -> CallProfile:
    return profile_callable(
        case.name,
        lambda: run_spacetime_simulation(case.config),
        metadata={
            "purpose": case.purpose,
            "config": _config_payload(case.config),
        },
        output_summary_fn=_profile_output_summary,
    )


def _run_internal_jit_case(case: SpacetimeProfileCase) -> dict[str, Any]:
    """Measure first and repeated calls with the simulator's internal JIT flag."""

    jit_config = replace(case.config, use_jit=True)
    first = profile_callable(
        f"{case.name}:jit_first",
        lambda: run_spacetime_simulation(jit_config),
        metadata={"purpose": f"{case.purpose} (internal JIT first call)", "config": _config_payload(jit_config)},
        output_summary_fn=_profile_output_summary,
    )
    second = profile_callable(
        f"{case.name}:jit_second",
        lambda: run_spacetime_simulation(jit_config),
        metadata={"purpose": f"{case.purpose} (internal JIT repeated call)", "config": _config_payload(jit_config)},
        output_summary_fn=_profile_output_summary,
    )
    return {
        "first_seconds": first.python_seconds,
        "second_seconds": second.python_seconds,
        "all_finite": first.all_finite and second.all_finite,
        "note": "Measures the simulator's internal scan JIT path, not an external jax.jit wrapper.",
    }


def _timing_by_case(case_payloads: tuple[dict[str, Any], ...]) -> dict[str, float]:
    return {str(payload["label"]): float(payload["python_seconds"]) for payload in case_payloads}


def recommend_bottleneck(case_payloads: tuple[dict[str, Any], ...]) -> str:
    """Return a deterministic first bottleneck recommendation from profile data."""

    timings = _timing_by_case(case_payloads)
    if not timings:
        return "No profiling cases were run; no bottleneck recommendation is available."

    matter_delta = timings.get("physics_with_matter", 0.0) - timings.get("physics_no_matter", 0.0)
    zero_step_4 = timings.get("scan_zero_step_4", 0.0)
    zero_step_16 = timings.get("scan_zero_step_16", 0.0)
    sector_u1 = timings.get("sector_u1_y", 0.0)
    sector_sm = timings.get("sector_sm", 0.0)

    if matter_delta > max(0.0, 0.5 * timings.get("physics_no_matter", 0.0)):
        return (
            "First target: finite-difference matter current/backreaction. "
            "The matter-enabled physics case is substantially slower than the no-matter case."
        )

    if zero_step_16 > 2.5 * max(zero_step_4, 1e-12):
        return (
            "First target: observation scheduling inside scan. "
            "Zero-step scan time grows strongly with step count because observables are computed every step."
        )

    if sector_sm > 2.0 * max(sector_u1, 1e-12):
        return (
            "First target: full-SM sector gauge force/link algebra. "
            "The SM-sector case is substantially slower than the U(1)_Y sector."
        )

    return (
        "No single dominant tiny-case bottleneck was isolated. "
        "Next profile should separate step_fn and observe_fn with larger but still bounded cases."
    )


def run_spacetime_profile(
    *,
    case_names: tuple[str, ...] | None = None,
    include_jit: bool = False,
) -> dict[str, Any]:
    """Run the bounded Session 49 spacetime simulator profiling suite."""

    cases = default_spacetime_profile_cases()
    if case_names is not None:
        requested = set(case_names)
        known = {case.name for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown profile case(s): {', '.join(unknown)}")
        cases = tuple(case for case in cases if case.name in requested)

    profiles = tuple(_run_case(case).as_payload() for case in cases)
    jit_profiles = {
        case.name: _run_internal_jit_case(case)
        for case in cases
        if include_jit
    }
    return {
        "metadata": {
            "runner": "spacetime_qca.simulator.profiling",
            "lattice_shape": (1, 1, 1),
            "include_jit": include_jit,
            "case_count": len(cases),
        },
        "cases": profiles,
        "jit_cases": jit_profiles,
        "recommendation": recommend_bottleneck(profiles),
    }
