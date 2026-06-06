"""Tests for the bath-reduction taxonomy."""

from clifford_3plus2_d5.universal_bath.reduction import (
    ReductionKind,
    reduction_taxonomy_payload,
    sector_reductions,
)


def test_reduction_taxonomy_assigns_three_distinct_tools() -> None:
    payload = reduction_taxonomy_payload()
    assert payload.final_verdict == "BATH_REDUCTION_TAXONOMY_PASS"
    assert "neutrino" in payload.positive_sectors
    assert "down_quark" in payload.indefinite_sectors
    assert "charged_lepton" in payload.cmv_sectors
    assert "up_quark" in payload.cmv_sectors


def test_chiral_phase_sectors_are_not_scalar_jacobi() -> None:
    assignments = {item.sector: item.reduction for item in sector_reductions()}
    assert assignments["charged_lepton"] == ReductionKind.CMV_OPUC
    assert assignments["up_quark"] == ReductionKind.CMV_OPUC
    assert assignments["ckm_phase"] == ReductionKind.CMV_OPUC

