"""Tests for the finite-stencil production locality audit."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_stencil import (
    bcc_transport_stencil,
    higgs_force_stencil,
    production_tick_stencil,
    sm_physical_right_production_stencil_diagnostics,
    two_hop_force_stencil,
)


def test_physical_right_production_stencil_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_stencil_diagnostics()

    assert diagnostics.bcc_hop_count == 8
    assert diagnostics.plaquette_pair_count == 6
    assert diagnostics.bcc_transport_radius == 1
    assert diagnostics.higgs_force_radius == 1
    assert diagnostics.two_hop_force_radius == 2
    assert diagnostics.production_tick_radius == 2
    assert diagnostics.inverse_one_tick_radius == 2
    assert diagnostics.inverse_two_tick_radius == 4
    assert diagnostics.radius_growth_per_tick <= 2.0
    assert diagnostics.bcc_inverse_closure_residual == 0
    assert diagnostics.origin_in_tick_stencil
    assert diagnostics.one_tick_support_count < diagnostics.two_tick_support_count


def test_physical_right_production_stencils_are_finite_and_nested() -> None:
    transport = set(bcc_transport_stencil())
    higgs = set(higgs_force_stencil())
    two_hop = set(two_hop_force_stencil())
    tick = set(production_tick_stencil())

    assert transport < higgs
    assert higgs < two_hop
    assert two_hop == tick
    assert (0, 0, 0) in tick


def test_physical_right_production_stencil_validates_inputs() -> None:
    with pytest.raises(ValueError, match="echo_steps must be positive"):
        sm_physical_right_production_stencil_diagnostics(echo_steps=0)
