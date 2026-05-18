"""Shared JAX backend utilities for simulation sidecars."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


DEFAULT_COMPLEX_DTYPE = jnp.complex64
DEFAULT_REAL_DTYPE = jnp.float32


@dataclass(frozen=True)
class JaxTiming:
    """Compile/run timing for a JAX callable."""

    compile_seconds: float
    run_seconds: float


def jax_default_device() -> str:
    """Return the default JAX device as a compact string."""

    return str(jax.devices()[0])


def as_jax(value: Any, *, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Convert ``value`` to a JAX array with an explicit dtype."""

    return jnp.asarray(value, dtype=dtype)


def as_numpy(value: Any) -> np.ndarray:
    """Convert a JAX/NumPy-like value to a NumPy array."""

    return np.asarray(value)


def time_jitted_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> JaxTiming:
    """Return separate compile and cached-run timing for ``jax.jit(fn)``.

    The function output is forced with ``block_until_ready`` when available.
    This helper is intended for local benchmarks and smoke tests, not strict
    performance assertions.
    """

    jitted = jax.jit(fn)
    start = perf_counter()
    compiled_result = jitted(*args, **kwargs)
    if hasattr(compiled_result, "block_until_ready"):
        compiled_result.block_until_ready()
    compile_seconds = perf_counter() - start

    start = perf_counter()
    run_result = jitted(*args, **kwargs)
    if hasattr(run_result, "block_until_ready"):
        run_result.block_until_ready()
    run_seconds = perf_counter() - start

    return JaxTiming(compile_seconds=compile_seconds, run_seconds=run_seconds)
