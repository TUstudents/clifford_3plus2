from __future__ import annotations

import subprocess
import sys

from clifford_3plus2_d5.obstruction_r10.gauge_equivalence import route1_gauge_equivalence_certificate


def test_route1_gauge_equivalence_requires_strict_standard() -> None:
    certificate = route1_gauge_equivalence_certificate()

    assert certificate.standard_scope == "fixed_su3_su2_u1_hypercharge_no_charge_conjugation"
    assert certificate.compatible_j_count == 4
    assert certificate.global_pm_orbit_count == 2
    assert certificate.intrinsic_branching_tables_match
    assert not certificate.fixed_sm_branching_tables_match_mod_global_pm
    assert not certificate.rule_generated_normalizer_orbit_certified
    assert not certificate.charge_conjugation_orbit_checked
    assert not certificate.relaxed_standard_supported
    assert certificate.strict_standard_required
    assert certificate.verdict == "strict_standard_required"
    assert {pattern.global_pm_class for pattern in certificate.patterns} == {
        (1, 1),
        (1, -1),
    }


def test_gauge_equivalence_check_script_reports_strict_standard() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/gauge_equivalence_check.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "compatible_j_count: 4" in result.stdout
    assert "standard_scope: fixed_su3_su2_u1_hypercharge_no_charge_conjugation" in result.stdout
    assert "global_pm_orbit_count: 2" in result.stdout
    assert "fixed_sm_branching_tables_match_mod_global_pm: false" in result.stdout
    assert "charge_conjugation_orbit_checked: false" in result.stdout
    assert "verdict: strict_standard_required" in result.stdout
