"""Lightweight benchmark helpers for JAX simulation kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from clifford_3plus2_d5.sim.backend import JaxTiming, time_jitted_call


def benchmark_jitted_kernel(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> JaxTiming:
    """Return compile and cached-run timing for a JAX kernel.

    This wrapper exists so sidecars can depend on a stable simulator benchmark
    API without importing backend internals.
    """

    return time_jitted_call(fn, *args, **kwargs)
