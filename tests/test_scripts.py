from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_branching_check_script_reports_non_load_bearing_result() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/branching_check.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "This verifies standard Spin(10) branching only." in result.stdout
    assert "It does not prove the QCA supplies the split." in result.stdout
    assert "branching_check_passed: true" in result.stdout
    assert "load_bearing_qca_bridge: false" in result.stdout


def test_qca_split_audit_script_defaults_to_notation_only_without_data() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/qca_split_audit.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "No qca_data.json, no bridge claim." in result.stdout
    assert "qca_split_audit_verdict: notation_only" in result.stdout
    assert "qca_supplies_structural_3plus2_split: false" in result.stdout
    assert "load_bearing_qca_bridge: false" in result.stdout


def test_qca_split_audit_script_can_emit_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/qca_split_audit.py",
            "--json",
            "--expect-verdict",
            "notation_only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["verdict"] == "notation_only"
    assert payload["load_bearing_qca_bridge"] is False


def test_qca_split_audit_script_expect_verdict_fails_on_mismatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/qca_split_audit.py",
            "--expect-verdict",
            "structural_bridge",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1


def test_real_carrier_check_script_reports_non_load_bearing_result() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/real_carrier_check.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "This verifies the exact real carrier ansatz only." in result.stdout
    assert "It does not prove QCA dynamics force J." in result.stdout
    assert "phase_1_real_carrier_check_passed: true" in result.stdout
    assert "qca_forces_j: false" in result.stdout
    assert "load_bearing_qca_bridge: false" in result.stdout


def test_real_carrier_check_script_can_emit_json() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/real_carrier_check.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["phase_1_real_carrier_check_passed"] is True
    assert payload["dimension"] == 10
    assert payload["projector_3_rank"] == 6
    assert payload["projector_2_rank"] == 4
    assert payload["qca_forces_j"] is False


def test_forced_j_check_script_reports_candidate_only_result() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/forced_j_check.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "This checks whether declared exact gate words can produce J." in result.stdout
    assert "It does not prove microscopic QCA rule data force J." in result.stdout
    assert "generated_by_gate_word: true" in result.stdout
    assert "forced_j_check_passed: true" in result.stdout
    assert "qca_forces_j: false" in result.stdout
    assert "forced_j_verdict: candidate_only" in result.stdout
    assert "load_bearing_qca_bridge: false" in result.stdout


def test_forced_j_check_script_can_emit_json() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/forced_j_check.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["generated_by_gate_word"] is True
    assert payload["forced_j_verdict"] == "candidate_only"
    assert payload["qca_forces_j"] is False


def test_forced_j_check_script_can_report_addressability_falsifier() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/forced_j_check.py",
            "--include-addressable-rank-one",
            "--expect-verdict",
            "falsified",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rank_one_pair_rotations_addressable: true" in result.stdout
    assert "forced_j_verdict: falsified" in result.stdout


def test_forced_j_check_expect_verdict_fails_on_mismatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/forced_j_check.py",
            "--expect-verdict",
            "forced_j",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1


def test_structural_split_check_script_reports_candidate_only_result() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/structural_split_check.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "This verifies the exact structural split candidate only." in result.stdout
    assert "It does not prove QCA rule data force P_3/P_2." in result.stdout
    assert "projector_identities_passed: true" in result.stdout
    assert "projectors_commute_with_J: true" in result.stdout
    assert "addressability_algebra_safe: true" in result.stdout
    assert "qca_supplies_structural_3plus2_split: false" in result.stdout
    assert "structural_split_verdict: candidate_only" in result.stdout
    assert "load_bearing_qca_bridge: false" in result.stdout


def test_structural_split_check_script_can_emit_json() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/structural_split_check.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["projector_identities_passed"] is True
    assert payload["projectors_commute_with_j"] is True
    assert payload["structural_split_verdict"] == "candidate_only"
    assert payload["qca_supplies_structural_3plus2_split"] is False


def test_structural_split_check_script_can_report_color_falsifier() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/structural_split_check.py",
            "--include-rank-one-color",
            "--expect-verdict",
            "falsified",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rank_one_color_projectors_addressable: true" in result.stdout
    assert "structural_split_verdict: falsified" in result.stdout


def test_structural_split_check_script_can_report_weak_falsifier() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/structural_split_check.py",
            "--include-rank-one-weak",
            "--expect-verdict",
            "falsified",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rank_one_weak_projectors_addressable: true" in result.stdout
    assert "structural_split_verdict: falsified" in result.stdout


def test_structural_split_check_expect_verdict_fails_on_mismatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/structural_split_check.py",
            "--expect-verdict",
            "structural_split",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1


def test_gate_classification_check_script_reports_oracle_result() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/gate_classification_check.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "This verifies the exact SM commutant gate classifier only." in result.stdout
    assert "It does not prove QCA rule data supply only safe geometric gates." in result.stdout
    assert "P_3: safe_sm_commutant" in result.stdout
    assert "block_mixer: block_mixing_fail" in result.stdout
    assert "rank_one_color_projector: color_breaking_fail" in result.stdout
    assert "rank_one_weak_projector: weak_breaking_fail" in result.stdout
    assert "real_conjugation: antilinear_fail" in result.stdout
    assert "commutant_basis_matches_expected: true" in result.stdout
    assert "safe_algebra_closure_passed: true" in result.stdout
    assert "gate_classification_check_passed: true" in result.stdout
    assert "qca_geometric_gate_algebra_safe: false" in result.stdout
    assert "load_bearing_qca_bridge: false" in result.stdout


def test_gate_classification_check_script_can_emit_json() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/gate_classification_check.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["gate_classification_check_passed"] is True
    assert payload["commutant_basis_matches_expected"] is True
    assert payload["safe_algebra_closure_passed"] is True
    assert payload["qca_geometric_gate_algebra_safe"] is False
    assert payload["load_bearing_qca_bridge"] is False


def test_qca_update_check_script_reports_candidate_only_result() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/qca_update_check.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "This verifies the finite-depth QCA update candidate only." in result.stdout
    assert "It does not prove microscopic QCA rule data force this update." in result.stdout
    assert "finite_depth: true" in result.stdout
    assert "period_four_check_passed: true" in result.stdout
    assert "quarter_period_is_j: true" in result.stdout
    assert "half_period_is_minus_identity: true" in result.stdout
    assert "full_period_is_identity: true" in result.stdout
    assert "all_internal_actions_safe: true" in result.stdout
    assert "qca_rule_forces_update: false" in result.stdout
    assert "finite_depth_qca_verdict: candidate_only" in result.stdout
    assert "load_bearing_qca_bridge: false" in result.stdout


def test_qca_update_check_script_can_emit_json() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/qca_update_check.py", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["finite_depth"] is True
    assert payload["period_four_check_passed"] is True
    assert payload["finite_depth_qca_verdict"] == "candidate_only"
    assert payload["load_bearing_qca_bridge"] is False


def test_qca_update_check_script_can_report_color_shift_falsifier() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/qca_update_check.py",
            "--include-rank-one-color-shift",
            "--expect-verdict",
            "falsified",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "all_internal_actions_safe: false" in result.stdout
    assert "unsafe_gate_witnesses: rank_one_color_shift" in result.stdout
    assert "finite_depth_qca_verdict: falsified" in result.stdout


def test_qca_update_check_script_can_report_weak_shift_falsifier() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/qca_update_check.py",
            "--include-rank-one-weak-shift",
            "--expect-verdict",
            "falsified",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "all_internal_actions_safe: false" in result.stdout
    assert "unsafe_gate_witnesses: rank_one_weak_shift" in result.stdout
    assert "finite_depth_qca_verdict: falsified" in result.stdout


def test_qca_update_check_expect_verdict_fails_on_mismatch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/qca_update_check.py",
            "--expect-verdict",
            "finite_depth_candidate",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
