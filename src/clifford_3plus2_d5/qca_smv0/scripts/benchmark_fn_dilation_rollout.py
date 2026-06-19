"""Benchmark the production QCA_SMv0 FN dilation rollout hot path."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import sm_family_quark_path_hidden_dims
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_INTERNAL_DIM
from clifford_3plus2_d5.qca_smv0.sm_fn import FN_LAMBDA_WOLFENSTEIN, SM_FAMILY_DIM
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    deterministic_qca_family_state,
    sm_jit_qca_state_rollout,
    sm_qca_center_cp_rollout_config,
    sm_qca_config_memory_footprint,
    sm_qca_prepare_state,
    sm_qca_rollout_memory_footprint,
    sm_qca_state_memory_footprint,
    sm_run_qca_rollout,
)


BYTES_PER_GIB = 1024**3


def _parse_int_triplet(text: str) -> tuple[int, int, int]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("lattice dimensions must be positive")
    return values  # type: ignore[return-value]


def _max_cubic_edge(max_sites: int) -> int:
    edge = int(max_sites ** (1.0 / 3.0))
    while (edge + 1) ** 3 <= max_sites:
        edge += 1
    while edge**3 > max_sites:
        edge -= 1
    return edge


def _memory_budget_summary(
    *,
    bytes_per_site: float,
    fixed_bytes: int = 0,
    memory_budget_gib: float | None,
    memory_safety_factor: float,
) -> dict[str, Any] | None:
    if memory_budget_gib is None:
        return None
    if memory_budget_gib <= 0.0:
        raise ValueError("memory_budget_gib must be positive")
    if not 0.0 < memory_safety_factor <= 1.0:
        raise ValueError("memory_safety_factor must be in (0, 1]")
    if fixed_bytes < 0:
        raise ValueError("fixed_bytes must be nonnegative")
    usable_bytes = int(memory_budget_gib * BYTES_PER_GIB * memory_safety_factor)
    usable_bytes_after_fixed = max(0, usable_bytes - int(fixed_bytes))
    max_sites = int(usable_bytes_after_fixed // bytes_per_site)
    return {
        "estimate_scope": "runtime_complex64_state_scaling_plus_fixed_config_arrays",
        "memory_budget_gib": float(memory_budget_gib),
        "memory_safety_factor": float(memory_safety_factor),
        "usable_bytes": usable_bytes,
        "fixed_bytes": int(fixed_bytes),
        "usable_bytes_after_fixed": usable_bytes_after_fixed,
        "bytes_per_site": float(bytes_per_site),
        "max_sites": max_sites,
        "max_cubic_lattice_shape": [_max_cubic_edge(max_sites)] * 3,
    }


def _memory_budget_fit(
    *,
    complex64_bytes: int,
    memory_budget_gib: float | None,
    memory_safety_factor: float,
    estimate_scope: str = "runtime_complex64_state_plus_selected_config_arrays",
) -> dict[str, Any] | None:
    if memory_budget_gib is None:
        return None
    if memory_budget_gib <= 0.0:
        raise ValueError("memory_budget_gib must be positive")
    if not 0.0 < memory_safety_factor <= 1.0:
        raise ValueError("memory_safety_factor must be in (0, 1]")
    usable_bytes = int(memory_budget_gib * BYTES_PER_GIB * memory_safety_factor)
    return {
        "estimate_scope": estimate_scope,
        "usable_bytes": usable_bytes,
        "complex64_bytes": int(complex64_bytes),
        "fits_memory_budget": int(complex64_bytes) <= usable_bytes,
        "budget_fraction": float(complex64_bytes / usable_bytes),
        "complex64_bytes_margin": int(usable_bytes - complex64_bytes),
    }


def _memory_payload(
    *,
    lattice_shape: tuple[int, int, int],
    lambda_rec: float,
    yukawa_step_size: float,
    higgs_vev: float,
    collision_mode: str,
    stream_mode: str = "split_axis",
    yukawa_collision_strategy: str = "memory",
    rollout_output: str = "diagnostic",
) -> dict[str, Any]:
    sites = int(lattice_shape[0] * lattice_shape[1] * lattice_shape[2])
    visible_complex_elements = sites * 4 * SM_INTERNAL_DIM * SM_FAMILY_DIM
    fn_path_aux_complex_elements = 0
    if collision_mode == "fn_dilation":
        dims = sm_family_quark_path_hidden_dims()
        hidden_dim = int(dims.up + dims.down)
        fn_path_aux_complex_elements = sites * 4 * 3 * 2 * hidden_dim
    state_complex_elements = visible_complex_elements + fn_path_aux_complex_elements
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        lambda_rec=lambda_rec,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        collision_mode=collision_mode,
        stream_mode=stream_mode,
        yukawa_collision_strategy=yukawa_collision_strategy,
        record_observables=False,
        record_density=False,
    )
    config_footprint = sm_qca_config_memory_footprint(config)
    state_complex64_bytes = 8 * state_complex_elements
    state_complex128_bytes = 16 * state_complex_elements
    complex64_bytes = state_complex64_bytes + config_footprint.array_bytes
    complex128_bytes = state_complex128_bytes + config_footprint.array_bytes
    return {
        "lattice_shape": list(lattice_shape),
        "sites": sites,
        "collision_mode": collision_mode,
        "stream_mode": stream_mode,
        "yukawa_collision_strategy": yukawa_collision_strategy,
        "rollout_output": rollout_output,
        "memory": {
            "visible_complex_elements": visible_complex_elements,
            "fn_path_aux_complex_elements": fn_path_aux_complex_elements,
            "state_complex_elements": state_complex_elements,
            "config_array_elements": config_footprint.array_elements,
            "config_array_bytes": config_footprint.array_bytes,
            "total_complex_elements": state_complex_elements,
            "state_complex64_bytes": state_complex64_bytes,
            "state_complex128_bytes": state_complex128_bytes,
            "state_complex64_bytes_per_site": state_complex64_bytes / sites,
            "state_complex128_bytes_per_site": state_complex128_bytes / sites,
            "total_array_bytes": complex64_bytes,
            "complex64_bytes": complex64_bytes,
            "complex128_bytes": complex128_bytes,
            "complex64_bytes_per_site": complex64_bytes / sites,
            "complex128_bytes_per_site": complex128_bytes / sites,
        },
    }


def _collision_modes(collision_mode: str) -> tuple[str, ...]:
    if collision_mode == "both":
        return ("fn_dilation", "effective_yukawa")
    if collision_mode in ("fn_dilation", "effective_yukawa"):
        return (collision_mode,)
    raise ValueError("collision_mode must be 'fn_dilation', 'effective_yukawa', or 'both'")


def _mode_comparisons(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_shape: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for benchmark in benchmarks:
        shape = tuple(int(value) for value in benchmark["lattice_shape"])
        by_shape.setdefault(shape, {})[benchmark["collision_mode"]] = benchmark

    comparisons = []
    for shape, modes in by_shape.items():
        exact = modes.get("fn_dilation")
        compressed = modes.get("effective_yukawa")
        if exact is None or compressed is None:
            continue
        exact_memory = int(exact["memory"]["complex64_bytes"])
        compressed_memory = int(compressed["memory"]["complex64_bytes"])
        comparison: dict[str, Any] = {
            "lattice_shape": list(shape),
            "reference_collision_mode": "fn_dilation",
            "compressed_collision_mode": "effective_yukawa",
            "complex64_bytes_reference": exact_memory,
            "complex64_bytes_compressed": compressed_memory,
            "complex64_bytes_saved": exact_memory - compressed_memory,
            "memory_ratio_reference_to_compressed": float(exact_memory / compressed_memory),
        }
        if "mean_run_seconds" in exact and "mean_run_seconds" in compressed:
            comparison["runtime_ratio_reference_to_compressed"] = float(
                exact["mean_run_seconds"] / compressed["mean_run_seconds"],
            )
        exact_compiled_memory = exact.get("compiled_memory")
        compressed_compiled_memory = compressed.get("compiled_memory")
        if (
            isinstance(exact_compiled_memory, dict)
            and isinstance(compressed_compiled_memory, dict)
            and exact_compiled_memory.get("available")
            and compressed_compiled_memory.get("available")
            and "device_runtime_size_floor_bytes" in exact_compiled_memory
            and "device_runtime_size_floor_bytes" in compressed_compiled_memory
        ):
            exact_floor = int(exact_compiled_memory["device_runtime_size_floor_bytes"])
            compressed_floor = int(compressed_compiled_memory["device_runtime_size_floor_bytes"])
            comparison["compiled_runtime_floor_bytes_reference"] = exact_floor
            comparison["compiled_runtime_floor_bytes_compressed"] = compressed_floor
            comparison["compiled_runtime_floor_bytes_saved"] = exact_floor - compressed_floor
            if compressed_floor:
                comparison["compiled_runtime_floor_ratio_reference_to_compressed"] = float(
                    exact_floor / compressed_floor,
                )
        if (
            isinstance(exact_compiled_memory, dict)
            and isinstance(compressed_compiled_memory, dict)
            and exact_compiled_memory.get("available")
            and compressed_compiled_memory.get("available")
            and "temp_size_in_bytes" in exact_compiled_memory
            and "temp_size_in_bytes" in compressed_compiled_memory
        ):
            exact_temp = int(exact_compiled_memory["temp_size_in_bytes"])
            compressed_temp = int(compressed_compiled_memory["temp_size_in_bytes"])
            comparison["compiled_temp_bytes_reference"] = exact_temp
            comparison["compiled_temp_bytes_compressed"] = compressed_temp
            comparison["compiled_temp_bytes_saved"] = exact_temp - compressed_temp
            if compressed_temp:
                comparison["compiled_temp_ratio_reference_to_compressed"] = float(exact_temp / compressed_temp)
        if "memory_budget_fit" in exact and "memory_budget_fit" in compressed:
            comparison["compressed_fits_when_reference_does_not"] = bool(
                compressed["memory_budget_fit"]["fits_memory_budget"]
                and not exact["memory_budget_fit"]["fits_memory_budget"],
            )
        comparisons.append(comparison)
    return comparisons


def _block_until_ready(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _compiled_memory_analysis_payload(compiled: Any) -> dict[str, Any]:
    memory_analysis = getattr(compiled, "memory_analysis", None)
    if memory_analysis is None:
        return {"available": False, "reason": "compiled executable has no memory_analysis method"}
    analysis = memory_analysis()
    if analysis is None:
        return {"available": False, "reason": "backend did not return memory analysis"}

    payload: dict[str, Any] = {"available": True}
    for field in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
        "serialized_hlo_proto_size_in_bytes",
    ):
        if hasattr(analysis, field):
            value = getattr(analysis, field)
            if value is not None:
                payload[field] = int(value)

    argument = int(payload.get("argument_size_in_bytes", 0))
    output = int(payload.get("output_size_in_bytes", 0))
    alias = int(payload.get("alias_size_in_bytes", 0))
    temp = int(payload.get("temp_size_in_bytes", 0))
    if argument or output or temp:
        payload["device_runtime_size_floor_bytes"] = max(0, argument + output + temp - alias)
    return payload


def _benchmark_payload(
    *,
    lattice_shape: tuple[int, int, int],
    steps: int,
    repeats: int,
    warmup_repeats: int,
    donate_state: bool,
    lambda_rec: float,
    yukawa_step_size: float,
    higgs_vev: float,
    collision_mode: str,
    stream_mode: str,
    yukawa_collision_strategy: str,
    rollout_output: str,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be nonnegative")
    if rollout_output not in ("diagnostic", "state"):
        raise ValueError("rollout_output must be 'diagnostic' or 'state'")

    state = deterministic_qca_family_state(lattice_shape)
    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        lambda_rec=lambda_rec,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        collision_mode=collision_mode,
        stream_mode=stream_mode,
        yukawa_collision_strategy=yukawa_collision_strategy,
        record_observables=False,
        record_density=False,
    )
    prepared = sm_qca_prepare_state(state, config)
    footprint = sm_qca_state_memory_footprint(prepared)
    rollout_footprint = sm_qca_rollout_memory_footprint(prepared, config)

    def diagnostic_rollout_kernel(
        local_state: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        result = sm_run_qca_rollout(config, local_state, steps=steps)
        aux = result.final_fn_path_aux_state
        if aux is None:
            empty = jnp.zeros((0,), dtype=local_state.dtype)
            return result.final_state, result.norm_history, result.extended_norm_history, empty, empty
        return result.final_state, result.norm_history, result.extended_norm_history, aux.up, aux.down

    if rollout_output == "state":
        initial_arg = prepared
        jitted_rollout = sm_jit_qca_state_rollout(config, steps, donate_state=donate_state)
    else:
        initial_arg = state
        jitted_rollout = jax.jit(diagnostic_rollout_kernel, donate_argnums=(0,) if donate_state else ())
    start = time.perf_counter()
    compiled_rollout = jitted_rollout.lower(initial_arg).compile()
    compile_seconds = time.perf_counter() - start
    compiled_memory = _compiled_memory_analysis_payload(compiled_rollout)

    start = time.perf_counter()
    last_output = _block_until_ready(compiled_rollout(initial_arg))
    first_call_seconds = time.perf_counter() - start
    if rollout_output == "state" and donate_state:
        run_state = last_output
    elif donate_state:
        run_state = last_output[0]
    else:
        run_state = initial_arg

    warmup_seconds = []
    for _ in range(warmup_repeats):
        start = time.perf_counter()
        last_output = _block_until_ready(compiled_rollout(run_state))
        warmup_seconds.append(time.perf_counter() - start)
        if rollout_output == "state" and donate_state:
            run_state = last_output
        elif donate_state:
            run_state = last_output[0]
        else:
            run_state = initial_arg

    run_seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        last_output = _block_until_ready(compiled_rollout(run_state))
        run_seconds.append(time.perf_counter() - start)
        if rollout_output == "state" and donate_state:
            run_state = last_output
        elif donate_state:
            run_state = last_output[0]
        else:
            run_state = initial_arg

    if rollout_output == "diagnostic":
        norm_history = last_output[1]
        extended_norm_history = last_output[2]
        norm_drift = float(jnp.abs(norm_history[-1] - norm_history[0]))
        extended_norm_drift = float(jnp.abs(extended_norm_history[-1] - extended_norm_history[0]))
    else:
        norm_drift = None
        extended_norm_drift = None
    mean_run_seconds = sum(run_seconds) / len(run_seconds)
    sorted_run_seconds = sorted(run_seconds)
    midpoint = len(sorted_run_seconds) // 2
    if len(sorted_run_seconds) % 2 == 0:
        median_run_seconds = 0.5 * (sorted_run_seconds[midpoint - 1] + sorted_run_seconds[midpoint])
    else:
        median_run_seconds = sorted_run_seconds[midpoint]
    sites = int(lattice_shape[0] * lattice_shape[1] * lattice_shape[2])
    site_steps = int(sites * steps)
    return {
        "lattice_shape": list(lattice_shape),
        "sites": sites,
        "steps": int(steps),
        "repeats": int(repeats),
        "warmup_repeats": int(warmup_repeats),
        "donate_state": bool(donate_state),
        "collision_mode": collision_mode,
        "stream_mode": stream_mode,
        "yukawa_collision_strategy": yukawa_collision_strategy,
        "rollout_output": rollout_output,
        "lambda": float(lambda_rec),
        "yukawa_step_size": float(yukawa_step_size),
        "higgs_vev": float(higgs_vev),
        "compile_seconds": compile_seconds,
        "first_call_seconds": first_call_seconds,
        "warmup_seconds": warmup_seconds,
        "run_seconds": run_seconds,
        "mean_run_seconds": mean_run_seconds,
        "median_run_seconds": median_run_seconds,
        "min_run_seconds": min(run_seconds),
        "max_run_seconds": max(run_seconds),
        "mean_seconds_per_step": mean_run_seconds / steps,
        "median_seconds_per_step": median_run_seconds / steps,
        "mean_site_steps_per_second": site_steps / mean_run_seconds,
        "median_site_steps_per_second": site_steps / median_run_seconds,
        "norm_drift": norm_drift,
        "extended_norm_drift": extended_norm_drift,
        "compiled_memory": compiled_memory,
        "memory": {
            "visible_complex_elements": footprint.visible_complex_elements,
            "fn_path_aux_complex_elements": footprint.fn_path_aux_complex_elements,
            "state_complex_elements": footprint.total_complex_elements,
            "config_array_elements": rollout_footprint.config_array_elements,
            "config_array_bytes": rollout_footprint.config_array_bytes,
            "total_complex_elements": footprint.total_complex_elements,
            "state_complex64_bytes": footprint.complex64_bytes,
            "state_complex128_bytes": footprint.complex128_bytes,
            "state_complex64_bytes_per_site": footprint.complex64_bytes / sites,
            "state_complex128_bytes_per_site": footprint.complex128_bytes / sites,
            "total_array_bytes": rollout_footprint.total_array_bytes,
            "complex64_bytes": rollout_footprint.total_array_bytes,
            "complex128_bytes": footprint.complex128_bytes + rollout_footprint.config_array_bytes,
            "complex64_bytes_per_site": rollout_footprint.total_array_bytes / sites,
            "complex128_bytes_per_site": (footprint.complex128_bytes + rollout_footprint.config_array_bytes) / sites,
        },
    }


def _benchmark_sweep_payload(
    *,
    lattice_shapes: tuple[tuple[int, int, int], ...],
    steps: int,
    repeats: int,
    warmup_repeats: int,
    donate_state: bool,
    lambda_rec: float,
    yukawa_step_size: float,
    higgs_vev: float,
    collision_mode: str,
    memory_budget_gib: float | None,
    memory_safety_factor: float,
    dry_run: bool,
    stream_mode: str = "split_axis",
    yukawa_collision_strategy: str = "memory",
    rollout_output: str = "state",
) -> dict[str, Any]:
    benchmarks = []
    collision_modes = _collision_modes(collision_mode)
    for lattice_shape in lattice_shapes:
        for mode in collision_modes:
            if dry_run:
                benchmarks.append(
                    _memory_payload(
                        lattice_shape=lattice_shape,
                        lambda_rec=lambda_rec,
                        yukawa_step_size=yukawa_step_size,
                        higgs_vev=higgs_vev,
                        collision_mode=mode,
                        stream_mode=stream_mode,
                        yukawa_collision_strategy=yukawa_collision_strategy,
                        rollout_output=rollout_output,
                    ),
                )
            else:
                benchmarks.append(
                    _benchmark_payload(
                        lattice_shape=lattice_shape,
                        steps=steps,
                        repeats=repeats,
                        warmup_repeats=warmup_repeats,
                        donate_state=donate_state,
                        lambda_rec=lambda_rec,
                        yukawa_step_size=yukawa_step_size,
                        higgs_vev=higgs_vev,
                        collision_mode=mode,
                        stream_mode=stream_mode,
                        yukawa_collision_strategy=yukawa_collision_strategy,
                        rollout_output=rollout_output,
                    ),
                )
    for benchmark in benchmarks:
        budget_fit = _memory_budget_fit(
            complex64_bytes=int(benchmark["memory"]["complex64_bytes"]),
            memory_budget_gib=memory_budget_gib,
            memory_safety_factor=memory_safety_factor,
        )
        if budget_fit is not None:
            benchmark["memory_budget_fit"] = budget_fit
        compiled_memory = benchmark.get("compiled_memory")
        if (
            isinstance(compiled_memory, dict)
            and compiled_memory.get("available")
            and "device_runtime_size_floor_bytes" in compiled_memory
        ):
            compiled_fit = _memory_budget_fit(
                complex64_bytes=int(compiled_memory["device_runtime_size_floor_bytes"]),
                memory_budget_gib=memory_budget_gib,
                memory_safety_factor=memory_safety_factor,
                estimate_scope="xla_device_runtime_size_floor_bytes",
            )
            if compiled_fit is not None:
                benchmark["compiled_memory_budget_fit"] = compiled_fit
    first_memory = benchmarks[0]["memory"]
    bytes_per_site = float(first_memory["state_complex64_bytes_per_site"])
    fixed_bytes = int(first_memory["config_array_bytes"])
    return {
        "dry_run": bool(dry_run),
        "collision_modes": list(collision_modes),
        "stream_mode": stream_mode,
        "yukawa_collision_strategy": yukawa_collision_strategy,
        "rollout_output": rollout_output,
        "benchmarks": benchmarks,
        "mode_comparisons": _mode_comparisons(benchmarks),
        "memory_budget": _memory_budget_summary(
            bytes_per_site=bytes_per_site,
            fixed_bytes=fixed_bytes,
            memory_budget_gib=memory_budget_gib,
            memory_safety_factor=memory_safety_factor,
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lattice-shape",
        type=_parse_int_triplet,
        action="append",
        dest="lattice_shapes",
        help="Comma triple nx,ny,nz. Repeat to sweep multiple shapes.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=2, help="Post-JIT runs excluded from timing statistics.")
    parser.add_argument(
        "--donate-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow XLA to donate the input state buffer to the final-state output during benchmarking.",
    )
    parser.add_argument("--lambda", dest="lambda_rec", type=float, default=FN_LAMBDA_WOLFENSTEIN)
    parser.add_argument("--yukawa-step-size", type=float, default=0.01)
    parser.add_argument("--higgs-vev", type=float, default=1.0)
    parser.add_argument(
        "--collision-mode",
        choices=("fn_dilation", "effective_yukawa", "both"),
        default="effective_yukawa",
        help=(
            "Benchmark exact FN dilation as a reference, compressed effective "
            "Yukawa for production-scale runs, or both side by side."
        ),
    )
    parser.add_argument(
        "--stream-mode",
        choices=("hop_sum", "split_axis"),
        default="split_axis",
        help="Use the lower-temp-memory split-axis stream or the expanded eight-hop stream.",
    )
    parser.add_argument(
        "--yukawa-collision-strategy",
        choices=("memory", "fast"),
        default="memory",
        help="Use the lower-temp cached Yukawa collision or the faster grouped variant.",
    )
    parser.add_argument(
        "--rollout-output",
        choices=("diagnostic", "state"),
        default="state",
        help="Compile the pure final-state rollout by default, or the diagnostic rollout with observable reductions.",
    )
    parser.add_argument("--memory-budget-gib", type=float, help="Optional GPU memory budget in GiB.")
    parser.add_argument("--memory-safety-factor", type=float, default=0.8, help="Fraction of budget treated as usable.")
    parser.add_argument("--dry-run", action="store_true", help="Report memory estimates without allocating/JIT-running a rollout.")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    lattice_shapes = tuple(args.lattice_shapes) if args.lattice_shapes is not None else ((2, 1, 1),)
    payload = _benchmark_sweep_payload(
        lattice_shapes=lattice_shapes,
        steps=args.steps,
        repeats=args.repeats,
        warmup_repeats=args.warmup_repeats,
        donate_state=args.donate_state,
        lambda_rec=args.lambda_rec,
        yukawa_step_size=args.yukawa_step_size,
        higgs_vev=args.higgs_vev,
        collision_mode=args.collision_mode,
        stream_mode=args.stream_mode,
        yukawa_collision_strategy=args.yukawa_collision_strategy,
        rollout_output=args.rollout_output,
        memory_budget_gib=args.memory_budget_gib,
        memory_safety_factor=args.memory_safety_factor,
        dry_run=args.dry_run,
    )
    if args.output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("QCA_SMv0 FN rollout benchmark")
    for benchmark in payload["benchmarks"]:
        print(f"  lattice_shape: {tuple(benchmark['lattice_shape'])}")
        print(f"    collision_mode: {benchmark['collision_mode']}")
        print(f"    stream_mode: {benchmark['stream_mode']}")
        print(f"    yukawa_collision_strategy: {benchmark['yukawa_collision_strategy']}")
        print(f"    rollout_output: {benchmark['rollout_output']}")
        if not payload["dry_run"]:
            print(f"    steps/repeats: {benchmark['steps']} / {benchmark['repeats']}")
            print(f"    warmup_repeats: {benchmark['warmup_repeats']}")
            print(f"    donate_state: {benchmark['donate_state']}")
            print(f"    compile_seconds: {benchmark['compile_seconds']:.6g}")
            print(f"    first_call_seconds: {benchmark['first_call_seconds']:.6g}")
            print(f"    warmup_seconds: {benchmark['warmup_seconds']}")
            print(f"    mean_run_seconds: {benchmark['mean_run_seconds']:.6g}")
            print(f"    median_run_seconds: {benchmark['median_run_seconds']:.6g}")
            print(f"    mean_seconds_per_step: {benchmark['mean_seconds_per_step']:.6g}")
            print(f"    median_seconds_per_step: {benchmark['median_seconds_per_step']:.6g}")
            print(f"    mean_site_steps_per_second: {benchmark['mean_site_steps_per_second']:.6g}")
            print(f"    median_site_steps_per_second: {benchmark['median_site_steps_per_second']:.6g}")
            if benchmark["extended_norm_drift"] is not None:
                print(f"    extended_norm_drift: {benchmark['extended_norm_drift']:.6g}")
            print("    compiled_memory:")
            for key, value in benchmark["compiled_memory"].items():
                print(f"      {key}: {value}")
        print("    memory:")
        for key, value in benchmark["memory"].items():
            print(f"      {key}: {value}")
        if "memory_budget_fit" in benchmark:
            print("    memory_budget_fit:")
            for key, value in benchmark["memory_budget_fit"].items():
                print(f"      {key}: {value}")
        if "compiled_memory_budget_fit" in benchmark:
            print("    compiled_memory_budget_fit:")
            for key, value in benchmark["compiled_memory_budget_fit"].items():
                print(f"      {key}: {value}")
    if payload["memory_budget"] is not None:
        print("  memory_budget:")
        for key, value in payload["memory_budget"].items():
            print(f"    {key}: {value}")
    for comparison in payload["mode_comparisons"]:
        print(f"  mode_comparison: lattice_shape={tuple(comparison['lattice_shape'])}")
        for key, value in comparison.items():
            if key != "lattice_shape":
                print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
