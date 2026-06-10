"""Tests for the Stage 37 local Wilson-force replacement."""

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import (
    sm_finite_difference_left_wilson_force,
    sm_left_wilson_force,
    sm_local_wilson_force,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    deterministic_sm_link_theta,
    sm_link_field_from_algebra,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_force import (
    sm_local_wilson_force_ad_coordinate_residual,
    sm_physical_right_production_local_force_diagnostics,
)


def test_physical_right_production_local_force_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_local_force_diagnostics()

    assert diagnostics.identity_force_norm < 1e-7
    assert diagnostics.pure_gauge_force_norm < 1e-6
    assert diagnostics.nonflat_force_norm > 5e-5
    assert diagnostics.production_local_alias_residual < 1e-12
    assert diagnostics.covariance_residual < 5e-8
    assert diagnostics.ad_coordinate_residual < 2e-8
    assert diagnostics.legacy_fd_relative_residual < 0.08
    assert diagnostics.legacy_force_plaquette_holonomies == 4_608
    assert diagnostics.local_force_plaquette_holonomies == 12
    assert diagnostics.finite_difference_to_local_work_ratio == 384.0


def test_sm_left_wilson_force_is_local_force_alias() -> None:
    links = sm_link_field_from_algebra(deterministic_sm_link_theta((2, 1, 1), scale=1.0))

    assert jnp.linalg.norm(sm_left_wilson_force(links, epsilon=1e-3) - sm_local_wilson_force(links)) < 1e-12


def test_sm_local_wilson_force_matches_ad_left_update_coordinates() -> None:
    links = sm_link_field_from_algebra(deterministic_sm_link_theta((2, 1, 1), scale=1.0))

    assert sm_local_wilson_force_ad_coordinate_residual(links) < 2e-8


def test_legacy_finite_difference_force_remains_available_as_oracle() -> None:
    links = sm_link_field_from_algebra(deterministic_sm_link_theta((1, 1, 1), scale=2.0))
    local = sm_local_wilson_force(links)
    legacy = sm_finite_difference_left_wilson_force(links, epsilon=1e-2)

    assert jnp.linalg.norm(local - legacy) / jnp.linalg.norm(local) < 0.02
