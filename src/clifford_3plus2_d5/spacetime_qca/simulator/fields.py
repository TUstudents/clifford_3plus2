"""JAX-pytree field bundles for the spacetime-QCA simulator."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.spacetime_qca.jax_scaling import ScalingInitialState


class SpacetimeFields(NamedTuple):
    """Coupled fermion/gauge/Higgs fields carried by the main simulator."""

    state: jnp.ndarray
    links: jnp.ndarray
    momenta: jnp.ndarray
    phi: jnp.ndarray
    higgs_momentum: jnp.ndarray
    higgs_links: jnp.ndarray


def fields_from_scaling_state(fields: ScalingInitialState) -> SpacetimeFields:
    """Convert existing scaling fields to a JAX-pytree simulator bundle."""

    return SpacetimeFields(
        state=fields.state,
        links=fields.links,
        momenta=fields.momenta,
        phi=fields.phi,
        higgs_momentum=fields.higgs_momentum,
        higgs_links=fields.higgs_links,
    )


def fields_to_scaling_state(fields: SpacetimeFields) -> ScalingInitialState:
    """Convert simulator fields to the existing coupled-step field bundle."""

    return ScalingInitialState(
        state=fields.state,
        links=fields.links,
        momenta=fields.momenta,
        phi=fields.phi,
        higgs_momentum=fields.higgs_momentum,
        higgs_links=fields.higgs_links,
    )
