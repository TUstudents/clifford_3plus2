from __future__ import annotations

import json
from pathlib import Path

import pytest

from clifford_3plus2_d5.clifford_audit import audit_qca_payload, audit_qca_split, audit_to_dict
from clifford_3plus2_d5.status import QCASplitAudit


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "qca"


def _audit_fixture(name: str) -> QCASplitAudit:
    return audit_qca_split(FIXTURES / name)


def test_default_audit_values_do_not_pass() -> None:
    audit = QCASplitAudit()

    assert audit.candidate_generators == ()
    assert audit.anticommutation_matrix == ()
    assert audit.signature == "unknown"
    assert audit.structural_origin == "unknown"
    assert audit.complex_structure_operator is None
    assert audit.complex_structure_origin == "unknown"
    assert not audit.complex_structure_squares_to_minus_one
    assert not audit.complex_structure_preserves_3plus2_split
    assert not audit.complex_structure_in_allowed_gate_algebra
    assert not audit.complex_structure_compatible_with_3plus2_split
    assert not audit.off_block_gate_generators_present
    assert not audit.block_diagonal_gate_algebra
    assert not audit.sm_commutant_gate_algebra
    assert not audit.qca_supplies_structural_3plus2_split
    assert audit.verdict == "notation_only"


def test_missing_qca_data_defaults_to_notation_only(tmp_path: Path) -> None:
    audit = audit_qca_split(tmp_path / "missing.json")

    assert audit.verdict == "notation_only"
    assert audit.complex_structure_origin == "unknown"
    assert audit.complex_structure_operator is None
    assert not audit.qca_supplies_structural_3plus2_split
    assert not audit.sm_commutant_gate_algebra


def test_incomplete_qca_data_defaults_to_notation_only(tmp_path: Path) -> None:
    data_path = tmp_path / "qca_data.json"
    data_path.write_text(json.dumps({"candidate_generators": []}))

    audit = audit_qca_split(data_path)

    assert audit.verdict == "notation_only"
    assert audit.complex_structure_operator is None


def test_incomplete_fixture_defaults_to_notation_only() -> None:
    audit = _audit_fixture("incomplete.json")

    assert audit.verdict == "notation_only"
    assert audit.complex_structure_operator is None


def test_float_matrix_entries_are_rejected() -> None:
    audit = audit_qca_payload(
        {
            "candidate_generators": [
                {"name": "e1", "matrix": [[1.0]], "block": "V3"},
                {"name": "e2", "matrix": [["0"]], "block": "V3"},
                {"name": "e3", "matrix": [["0"]], "block": "V3"},
                {"name": "m1", "matrix": [["0"]], "block": "V2"},
                {"name": "m2", "matrix": [["0"]], "block": "V2"},
            ],
            "candidate_complex_structure": {
                "name": "J",
                "origin": "unknown",
                "matrix": [["0"]],
            },
            "allowed_gate_generators": [{"name": "G0", "matrix": [["0"]]}],
            "split_projectors": {"P3": [["1"]], "P2": [["0"]]},
        }
    )

    assert audit.verdict == "notation_only"
    assert audit.complex_structure_operator is None


def test_float_matrix_fixture_is_rejected() -> None:
    audit = _audit_fixture("float_matrix_entry.json")

    assert audit.verdict == "notation_only"
    assert audit.complex_structure_operator is None


def test_by_hand_complex_structure_is_explicitly_falsifying() -> None:
    audit = audit_qca_payload(_minimal_payload_with_j_origin("by_hand"))

    assert audit.complex_structure_origin == "by_hand"
    assert audit.complex_structure_operator is not None
    assert audit.complex_structure_squares_to_minus_one
    assert audit.verdict == "falsified"
    assert not audit.complex_structure_compatible_with_3plus2_split


def test_by_hand_complex_structure_fixture_is_falsifying() -> None:
    audit = _audit_fixture("hand_chosen_j.json")

    assert audit.complex_structure_origin == "by_hand"
    assert audit.complex_structure_squares_to_minus_one
    assert audit.verdict == "falsified"


def test_unknown_complex_structure_origin_cannot_pass() -> None:
    audit = audit_qca_payload(_minimal_payload_with_j_origin("unknown"))

    assert audit.complex_structure_origin == "unknown"
    assert audit.complex_structure_squares_to_minus_one
    assert audit.verdict == "notation_only"
    assert not audit.complex_structure_compatible_with_3plus2_split


@pytest.mark.parametrize(
    ("fixture_name", "field_name"),
    [
        ("unknown_j_origin.json", "complex_structure_origin"),
        ("j_squared_failure.json", "complex_structure_squares_to_minus_one"),
        ("j_not_preserving_split.json", "complex_structure_preserves_3plus2_split"),
        ("j_missing_from_allowed_gates.json", "complex_structure_in_allowed_gate_algebra"),
    ],
)
def test_complex_structure_fixtures_cannot_pass(fixture_name: str, field_name: str) -> None:
    audit = _audit_fixture(fixture_name)

    assert audit.verdict == "notation_only"
    assert not audit.complex_structure_compatible_with_3plus2_split
    if field_name == "complex_structure_origin":
        assert audit.complex_structure_origin == "unknown"
    else:
        assert not getattr(audit, field_name)


def test_arbitrary_structural_origin_fixture_is_falsifying() -> None:
    audit = _audit_fixture("arbitrary_structural_origin.json")

    assert audit.structural_origin == "arbitrary"
    assert audit.verdict == "falsified"


def test_off_block_gate_fixture_is_falsifying() -> None:
    audit = _audit_fixture("off_block_gate.json")

    assert audit.off_block_gate_generators_present
    assert not audit.block_diagonal_gate_algebra
    assert not audit.sm_commutant_gate_algebra
    assert audit.verdict == "falsified"


def test_color_projector_gate_fixture_is_falsifying() -> None:
    audit = _audit_fixture("color_projector_gate.json")

    assert not audit.off_block_gate_generators_present
    assert audit.block_diagonal_gate_algebra
    assert not audit.sm_commutant_gate_algebra
    assert audit.verdict == "falsified"


def test_audit_serialization_preserves_fraction_strings() -> None:
    audit = _audit_fixture("hand_chosen_j.json")
    payload = audit_to_dict(audit)

    assert payload["verdict"] == "falsified"
    assert payload["complex_structure_operator"] == [["0", "-1"], ["1", "0"]]
    assert payload["load_bearing_qca_bridge"] is False


def test_no_real_qca_data_file_is_checked_in() -> None:
    assert not (ROOT / "data" / "qca_data.json").exists()


def _minimal_payload_with_j_origin(origin: str) -> dict[str, object]:
    zero = [["0", "0"], ["0", "0"]]
    j = [["0", "-1"], ["1", "0"]]
    return {
        "structural_origin": "unknown",
        "candidate_generators": [
            {"name": "e1", "matrix": zero, "block": "V3"},
            {"name": "e2", "matrix": zero, "block": "V3"},
            {"name": "e3", "matrix": zero, "block": "V3"},
            {"name": "m1", "matrix": zero, "block": "V2"},
            {"name": "m2", "matrix": zero, "block": "V2"},
        ],
        "candidate_complex_structure": {"name": "J", "origin": origin, "matrix": j},
        "allowed_gate_generators": [{"name": "J", "matrix": j}],
        "split_projectors": {
            "P3": [["1", "0"], ["0", "1"]],
            "P2": zero,
        },
    }
