from __future__ import annotations

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
