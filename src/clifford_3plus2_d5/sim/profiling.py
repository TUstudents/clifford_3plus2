"""Generic lightweight profiling helpers for simulator callables."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.backend import jax_default_device


@dataclass(frozen=True)
class CallProfile:
    """JSON-safe timing and output metadata for one callable profile."""

    label: str
    device: str
    python_seconds: float
    output_summary: dict[str, Any]
    all_finite: bool
    metadata: dict[str, Any]
    jit_compile_seconds: float | None = None
    jit_run_seconds: float | None = None

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary payload."""

        return asdict(self)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, jnp.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_safe(item) for item in value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _profile_leaves(value: Any) -> tuple[Any, ...]:
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            leaf
            for field in fields(value)
            for leaf in _profile_leaves(getattr(value, field.name))
        )
    if isinstance(value, Mapping):
        return tuple(leaf for item in value.values() for leaf in _profile_leaves(item))
    if isinstance(value, tuple | list):
        return tuple(leaf for item in value for leaf in _profile_leaves(item))
    return tuple(jax.tree_util.tree_leaves(value))


def _block_until_ready(value: Any) -> None:
    for leaf in _profile_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _array_summary(value: Any) -> dict[str, Any] | None:
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return None
    return {
        "shape": tuple(int(size) for size in array.shape),
        "dtype": str(array.dtype),
    }


def _default_output_summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "observations") and hasattr(value, "final_fields"):
        observations = getattr(value, "observations")
        final_fields = getattr(value, "final_fields")
        return {
            "type": type(value).__name__,
            "observation_keys": tuple(sorted(str(key) for key in observations)),
            "final_field_shapes": {
                "state": tuple(int(size) for size in final_fields.state.shape),
                "links": tuple(int(size) for size in final_fields.links.shape),
                "momenta": tuple(int(size) for size in final_fields.momenta.shape),
                "phi": tuple(int(size) for size in final_fields.phi.shape),
                "higgs_momentum": tuple(int(size) for size in final_fields.higgs_momentum.shape),
                "higgs_links": tuple(int(size) for size in final_fields.higgs_links.shape),
            },
        }

    if isinstance(value, Mapping):
        return {
            "type": type(value).__name__,
            "keys": tuple(sorted(str(key) for key in value)),
        }

    summary = _array_summary(value)
    if summary is not None:
        summary["type"] = "array"
        return summary

    return {"type": type(value).__name__}


def _default_all_finite(value: Any) -> bool:
    if hasattr(value, "all_finite"):
        return bool(np.asarray(getattr(value, "all_finite")))

    leaves = _profile_leaves(value)
    finite_values: list[bool] = []
    for leaf in leaves:
        try:
            array = jnp.asarray(leaf)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind in {"b", "i", "u", "f", "c"}:
            finite_values.append(bool(np.asarray(jnp.all(jnp.isfinite(array)))))
    return all(finite_values) if finite_values else True


def profile_callable(
    label: str,
    fn: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_jit: bool = False,
    output_summary_fn: Callable[[Any], dict[str, Any]] | None = None,
    finite_fn: Callable[[Any], bool] | None = None,
) -> CallProfile:
    """Profile one callable with optional external ``jax.jit`` timings.

    Timings are informational and machine-dependent.  The helper forces JAX
    work to finish before recording elapsed wall time.
    """

    if kwargs is None:
        kwargs = {}

    start = perf_counter()
    output = fn(*args, **kwargs)
    _block_until_ready(output)
    python_seconds = perf_counter() - start

    jit_compile_seconds: float | None = None
    jit_run_seconds: float | None = None
    if include_jit:
        jitted = jax.jit(fn)
        start = perf_counter()
        compiled_output = jitted(*args, **kwargs)
        _block_until_ready(compiled_output)
        jit_compile_seconds = perf_counter() - start

        start = perf_counter()
        run_output = jitted(*args, **kwargs)
        _block_until_ready(run_output)
        jit_run_seconds = perf_counter() - start

    summarize = output_summary_fn or _default_output_summary
    finite = finite_fn or _default_all_finite
    return CallProfile(
        label=label,
        device=jax_default_device(),
        python_seconds=python_seconds,
        output_summary=_json_safe(summarize(output)),
        all_finite=finite(output),
        metadata=_json_safe(dict(metadata or {})),
        jit_compile_seconds=jit_compile_seconds,
        jit_run_seconds=jit_run_seconds,
    )
