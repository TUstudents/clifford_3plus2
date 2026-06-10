"""Finite-stencil locality audit for the production map.

Stage 35 records the finite support envelope behind the current
physical-right production tick.  The certificate is intentionally static: it
checks the local dependency radius implied by the BCC hops, Higgs-gradient
terms, plaquette/streaming-current forces, and their composition.

This is a finite-speed bookkeeping layer for the implemented discrete map.  It
does not replace a future large-lattice spatial echo measurement and does not
claim a continuum causal-cone theorem.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS
from clifford_3plus2_d5.qca_smv0.sm_gauge import BCC_PLAQUETTE_PAIRS

Displacement = tuple[int, int, int]

LOCAL_STENCIL: tuple[Displacement, ...] = ((0, 0, 0),)


class PhysicalRightProductionStencilDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 35 finite-stencil locality."""

    bcc_hop_count: jnp.ndarray
    plaquette_pair_count: jnp.ndarray
    bcc_transport_radius: jnp.ndarray
    higgs_force_radius: jnp.ndarray
    two_hop_force_radius: jnp.ndarray
    production_tick_radius: jnp.ndarray
    inverse_one_tick_radius: jnp.ndarray
    inverse_two_tick_radius: jnp.ndarray
    one_tick_support_count: jnp.ndarray
    two_tick_support_count: jnp.ndarray
    radius_growth_per_tick: jnp.ndarray
    bcc_inverse_closure_residual: jnp.ndarray
    origin_in_tick_stencil: jnp.ndarray


def _sorted_stencil(stencil: set[Displacement]) -> tuple[Displacement, ...]:
    return tuple(sorted(stencil))


def _add(left: Displacement, right: Displacement) -> Displacement:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _compose(left: tuple[Displacement, ...], right: tuple[Displacement, ...]) -> tuple[Displacement, ...]:
    return _sorted_stencil({_add(a, b) for a in left for b in right})


def _stencil_power(stencil: tuple[Displacement, ...], steps: int) -> tuple[Displacement, ...]:
    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    result = LOCAL_STENCIL
    for _ in range(steps):
        result = _compose(result, stencil)
    return result


def _radius(stencil: tuple[Displacement, ...]) -> int:
    return max(max(abs(component) for component in displacement) for displacement in stencil)


def _contains_origin(stencil: tuple[Displacement, ...]) -> bool:
    return (0, 0, 0) in set(stencil)


def bcc_transport_stencil() -> tuple[Displacement, ...]:
    """Return the one-tick BCC transport support."""

    return _sorted_stencil(set(BCC_DISPLACEMENTS))


def higgs_force_stencil() -> tuple[Displacement, ...]:
    """Return a conservative one-hop Higgs-gradient support."""

    return _sorted_stencil(set(LOCAL_STENCIL) | set(BCC_DISPLACEMENTS))


def two_hop_force_stencil() -> tuple[Displacement, ...]:
    """Return a conservative two-hop force/staple support."""

    one_hop = higgs_force_stencil()
    return _compose(one_hop, one_hop)


def production_tick_stencil() -> tuple[Displacement, ...]:
    """Return the conservative spatial envelope of one production tick."""

    return _sorted_stencil(
        set(LOCAL_STENCIL)
        | set(bcc_transport_stencil())
        | set(higgs_force_stencil())
        | set(two_hop_force_stencil()),
    )


def sm_physical_right_production_stencil_diagnostics(
    *,
    echo_steps: int = 2,
) -> PhysicalRightProductionStencilDiagnostics:
    """Return Stage 35 finite-stencil locality diagnostics."""

    if echo_steps < 1:
        raise ValueError(f"echo_steps must be positive, got {echo_steps}")

    transport = bcc_transport_stencil()
    higgs = higgs_force_stencil()
    two_hop = two_hop_force_stencil()
    tick = production_tick_stencil()
    one_tick_inverse = _stencil_power(tick, 1)
    echo_inverse = _stencil_power(tick, echo_steps)
    inverse_closure = set(transport) ^ {tuple(-component for component in displacement) for displacement in transport}

    return PhysicalRightProductionStencilDiagnostics(
        bcc_hop_count=jnp.asarray(len(transport), dtype=jnp.int32),
        plaquette_pair_count=jnp.asarray(len(BCC_PLAQUETTE_PAIRS), dtype=jnp.int32),
        bcc_transport_radius=jnp.asarray(_radius(transport), dtype=jnp.int32),
        higgs_force_radius=jnp.asarray(_radius(higgs), dtype=jnp.int32),
        two_hop_force_radius=jnp.asarray(_radius(two_hop), dtype=jnp.int32),
        production_tick_radius=jnp.asarray(_radius(tick), dtype=jnp.int32),
        inverse_one_tick_radius=jnp.asarray(_radius(one_tick_inverse), dtype=jnp.int32),
        inverse_two_tick_radius=jnp.asarray(_radius(echo_inverse), dtype=jnp.int32),
        one_tick_support_count=jnp.asarray(len(one_tick_inverse), dtype=jnp.int32),
        two_tick_support_count=jnp.asarray(len(echo_inverse), dtype=jnp.int32),
        radius_growth_per_tick=jnp.asarray(_radius(echo_inverse) / echo_steps, dtype=jnp.float32),
        bcc_inverse_closure_residual=jnp.asarray(len(inverse_closure), dtype=jnp.int32),
        origin_in_tick_stencil=jnp.asarray(_contains_origin(tick)),
    )
