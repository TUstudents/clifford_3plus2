from __future__ import annotations

import json
from pathlib import Path


# After the workspace reorganization, the trunk lives at
# src/clifford_3plus2_d5/obstruction_r10/. parents[1] from this test file
# resolves to that module directory, and the trunk's docs and data live
# directly under it (no docs/ subdirectory).
ROOT = Path(__file__).resolve().parents[1]


def test_qca_data_schema_exists_and_requires_handover_keys() -> None:
    schema_path = ROOT / "data" / "qca_data.schema.json"
    schema = json.loads(schema_path.read_text())

    assert schema["required"] == [
        "candidate_generators",
        "candidate_complex_structure",
        "allowed_gate_generators",
        "split_projectors",
    ]
    assert schema["$defs"]["rational"]["pattern"]
    assert "by_hand" in schema["$defs"]["complex_structure"]["properties"]["origin"]["enum"]
    assert "unknown" in schema["properties"]["structural_origin"]["enum"]


def test_handover_compliance_doc_exists() -> None:
    path = ROOT / "handover_compliance.md"

    assert path.exists()
