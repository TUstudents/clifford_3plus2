"""Tests for U3 — couplings are SM quantum-number projections."""

from __future__ import annotations

from clifford_3plus2_d5.flavor_a_track.u3_coupling_catalog import (
    color_split_is_quark_triplet_lepton_singlet,
    coupling_catalog,
    coupling_catalog_audit_payload,
    coupling_catalog_verdict,
    hypercharges_match_table,
    multiplicities_factor_as_color_times_weak,
)


def test_multiplicities_factor_as_color_times_weak() -> None:
    assert multiplicities_factor_as_color_times_weak()


def test_hypercharges_match_lepton_table() -> None:
    assert hypercharges_match_table()


def test_quark_triplet_lepton_singlet() -> None:
    assert color_split_is_quark_triplet_lepton_singlet()


def test_catalog_has_six_fields_with_factored_multiplicities() -> None:
    catalog = coupling_catalog()
    assert set(catalog) == {"Q", "u^c", "d^c", "L", "e^c", "nu^c"}
    for entry in catalog.values():
        assert int(entry["chiral16_multiplicity"]) == int(entry["color"]) * int(entry["weak"])


def test_payload_pass() -> None:
    payload = coupling_catalog_audit_payload()
    assert payload.final_verdict == "COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS"
    assert payload.multiplicities_factor
    assert payload.hypercharges_match
    assert payload.color_split_correct
    assert payload.explicit_chiral16_projectors_deferred


def test_verdict_kills_when_any_check_fails() -> None:
    # Decisive negative control: any failing check forces the KILL string.
    assert (
        coupling_catalog_verdict(True, True, True)
        == "COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS"
    )
    kill = "COUPLINGS_NOT_QUANTUM_NUMBER_DETERMINED_KILL"
    assert coupling_catalog_verdict(False, True, True) == kill
    assert coupling_catalog_verdict(True, False, True) == kill
    assert coupling_catalog_verdict(True, True, False) == kill


def test_projector_deferral_tracks_pass_state() -> None:
    # The deferral is now conditional on the gate passing (a promissory note that
    # only holds when the structural claim survives), not an unconditional True.
    payload = coupling_catalog_audit_payload()
    passed = payload.final_verdict == "COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS"
    assert payload.explicit_chiral16_projectors_deferred == passed
