"""Exact SM anomaly diagnostics for the spacetime QCA charge conventions.

Session 41 keeps the dynamical layer honest by checking that the
one-generation charge table used by the spacetime adapters is the physical
Session 19b table, not the raw Pati-Salam hypercharge generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.lepton.sm_hypercharge import (
    EXPECTED_HYPERCHARGE_SPECTRUM,
    hypercharge_observable,
    sm_field_multiplicity_table,
)


@dataclass(frozen=True)
class SMChiralField:
    """One left-handed Weyl multiplet in the one-generation SM table."""

    name: str
    hypercharge: sp.Expr
    color_dimension: int
    weak_dimension: int
    color_cubic_index: int
    color_dynkin_index: sp.Expr
    weak_dynkin_index: sp.Expr

    @property
    def complex_multiplicity(self) -> int:
        return self.color_dimension * self.weak_dimension

    @property
    def weak_doublet_count(self) -> int:
        return self.color_dimension if self.weak_dynkin_index != 0 else 0


def sm_chiral_field_table() -> tuple[SMChiralField, ...]:
    """Return the canonical one-generation left-handed SM Weyl multiplets."""

    half = sp.Rational(1, 2)
    return (
        SMChiralField("Q", sp.Rational(1, 6), 3, 2, +1, half, half),
        SMChiralField("u^c", sp.Rational(-2, 3), 3, 1, -1, half, sp.Integer(0)),
        SMChiralField("d^c", sp.Rational(1, 3), 3, 1, -1, half, sp.Integer(0)),
        SMChiralField("L", sp.Rational(-1, 2), 1, 2, 0, sp.Integer(0), half),
        SMChiralField("e^c", sp.Integer(1), 1, 1, 0, sp.Integer(0), sp.Integer(0)),
        SMChiralField("nu^c", sp.Integer(0), 1, 1, 0, sp.Integer(0), sp.Integer(0)),
    )


def field_table_matches_session19b() -> bool:
    """Return whether the local table agrees with the exact Session 19b table."""

    session19 = sm_field_multiplicity_table()
    for field in sm_chiral_field_table():
        row = session19[field.name]
        if row["Y"] != field.hypercharge:
            return False
        if row["complex_multiplicity"] != field.complex_multiplicity:
            return False
    return True


def hypercharge_spectrum_from_fields() -> dict[sp.Expr, int]:
    """Return the physical hypercharge spectrum implied by the field table."""

    spectrum: dict[sp.Expr, int] = {}
    for field in sm_chiral_field_table():
        spectrum[field.hypercharge] = spectrum.get(field.hypercharge, 0) + field.complex_multiplicity
    return spectrum


def sm_anomaly_sums() -> dict[str, sp.Expr | int | bool]:
    """Return exact one-generation anomaly sums in standard normalization.

    The nonabelian Dynkin index convention is ``T(fundamental)=1/2`` for
    ``SU(3)`` and ``SU(2)``.  The ``SU(3)^3`` sum uses cubic index ``+1`` for
    fundamentals and ``-1`` for antifundamentals.
    """

    fields = sm_chiral_field_table()
    weak_doublet_count = sum(field.weak_doublet_count for field in fields)
    return {
        "gravitational_u1_y": sp.simplify(
            sum(field.complex_multiplicity * field.hypercharge for field in fields)
        ),
        "u1_y_cubed": sp.simplify(
            sum(field.complex_multiplicity * field.hypercharge**3 for field in fields)
        ),
        "su3_squared_u1_y": sp.simplify(
            sum(field.weak_dimension * field.color_dynkin_index * field.hypercharge for field in fields)
        ),
        "su2_squared_u1_y": sp.simplify(
            sum(field.color_dimension * field.weak_dynkin_index * field.hypercharge for field in fields)
        ),
        "su3_cubed": sp.simplify(
            sum(field.weak_dimension * field.color_cubic_index for field in fields)
        ),
        "su2_witten_doublet_count": weak_doublet_count,
        "su2_witten_doublet_count_even": weak_doublet_count % 2 == 0,
    }


def matrix_charge_trace_diagnostics() -> dict[str, sp.Expr]:
    """Return trace checks from the real 32-dimensional charge observable."""

    observable = hypercharge_observable()
    return {
        "complex_trace_y": sp.simplify(sp.trace(observable) / 2),
        "complex_trace_y_cubed": sp.simplify(sp.trace(observable**3) / 2),
    }


def perturbative_anomalies_cancel() -> bool:
    """Return whether all exact perturbative anomaly sums vanish."""

    sums = sm_anomaly_sums()
    return all(
        sums[name] == 0
        for name in (
            "gravitational_u1_y",
            "u1_y_cubed",
            "su3_squared_u1_y",
            "su2_squared_u1_y",
            "su3_cubed",
        )
    )


def sm_anomaly_audit_payload() -> dict[str, object]:
    """Return a compact exact audit payload for Session 41 reports/tests."""

    spectrum = hypercharge_spectrum_from_fields()
    anomaly_sums = sm_anomaly_sums()
    trace_diagnostics = matrix_charge_trace_diagnostics()
    return {
        "field_table_matches_session19b": field_table_matches_session19b(),
        "hypercharge_spectrum": spectrum,
        "matches_physical_hypercharge_spectrum": spectrum == EXPECTED_HYPERCHARGE_SPECTRUM,
        "anomaly_sums": anomaly_sums,
        "perturbative_anomalies_cancel": perturbative_anomalies_cancel(),
        "global_su2_witten_anomaly_absent": anomaly_sums["su2_witten_doublet_count_even"],
        "matrix_trace_diagnostics": trace_diagnostics,
        "matrix_traces_match_anomaly_sums": (
            trace_diagnostics["complex_trace_y"] == anomaly_sums["gravitational_u1_y"]
            and trace_diagnostics["complex_trace_y_cubed"] == anomaly_sums["u1_y_cubed"]
        ),
    }
