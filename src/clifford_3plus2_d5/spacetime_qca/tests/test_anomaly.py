"""Session 41 exact anomaly and hypercharge-normalization tests."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.sm_hypercharge import EXPECTED_HYPERCHARGE_SPECTRUM
from clifford_3plus2_d5.spacetime_qca.anomaly import (
    field_table_matches_session19b,
    hypercharge_spectrum_from_fields,
    matrix_charge_trace_diagnostics,
    perturbative_anomalies_cancel,
    sm_anomaly_audit_payload,
    sm_anomaly_sums,
    sm_chiral_field_table,
)


def test_field_table_matches_session19b_hypercharge_table() -> None:
    fields = sm_chiral_field_table()

    assert sum(field.complex_multiplicity for field in fields) == 16
    assert field_table_matches_session19b()
    assert hypercharge_spectrum_from_fields() == EXPECTED_HYPERCHARGE_SPECTRUM


def test_one_generation_perturbative_anomalies_cancel_exactly() -> None:
    sums = sm_anomaly_sums()

    assert sums["gravitational_u1_y"] == 0
    assert sums["u1_y_cubed"] == 0
    assert sums["su3_squared_u1_y"] == 0
    assert sums["su2_squared_u1_y"] == 0
    assert sums["su3_cubed"] == 0
    assert perturbative_anomalies_cancel()


def test_su2_witten_doublet_count_is_even() -> None:
    sums = sm_anomaly_sums()

    assert sums["su2_witten_doublet_count"] == 4
    assert sums["su2_witten_doublet_count_even"] is True


def test_matrix_charge_traces_match_table_anomalies() -> None:
    traces = matrix_charge_trace_diagnostics()

    assert traces["complex_trace_y"] == sp.Integer(0)
    assert traces["complex_trace_y_cubed"] == sp.Integer(0)


def test_anomaly_audit_payload_is_stable() -> None:
    payload = sm_anomaly_audit_payload()

    assert payload["field_table_matches_session19b"] is True
    assert payload["matches_physical_hypercharge_spectrum"] is True
    assert payload["perturbative_anomalies_cancel"] is True
    assert payload["global_su2_witten_anomaly_absent"] is True
    assert payload["matrix_traces_match_anomaly_sums"] is True
    assert payload["anomaly_sums"]["su3_squared_u1_y"] == 0
    assert payload["anomaly_sums"]["su2_squared_u1_y"] == 0
