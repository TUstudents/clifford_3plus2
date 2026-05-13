"""Default-failing audit for whether QCA data supplies the 3+2 split."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from clifford_3plus2_d5.gate_algebra import (
    Matrix,
    NamedMatrix,
    audit_one_particle_gate_algebra,
    commutator,
    identity_matrix,
    is_zero_matrix,
    matrix_add,
    matrix_multiply,
    parse_fraction_matrix,
)
from clifford_3plus2_d5.status import (
    BridgeVerdict,
    ComplexStructureOrigin,
    QCASplitAudit,
    Signature,
    StructuralOrigin,
)


DEFAULT_QCA_DATA_PATH = Path("data/qca_data.json")

STRUCTURAL_SPLIT_ORIGINS: frozenset[StructuralOrigin] = frozenset(
    {"wall_normal_plus_floquet_time", "wall_normal_plus_floquet_phase"}
)
ALLOWED_COMPLEX_STRUCTURE_ORIGINS: frozenset[ComplexStructureOrigin] = frozenset(
    {"qca_chirality", "floquet_phase", "wick_rotation"}
)


@dataclass(frozen=True)
class CandidateGenerator:
    name: str
    matrix: Matrix
    block: str


def audit_qca_split(path: Path | str = DEFAULT_QCA_DATA_PATH) -> QCASplitAudit:
    """Audit a qca_data.json file, returning notation_only on missing or invalid input."""

    data_path = Path(path)
    if not data_path.exists():
        return QCASplitAudit()

    try:
        payload = json.loads(data_path.read_text())
        return audit_qca_payload(payload)
    except (OSError, TypeError, ValueError, KeyError):
        return QCASplitAudit()


def audit_qca_payload(payload: object) -> QCASplitAudit:
    if not isinstance(payload, Mapping):
        return QCASplitAudit()

    try:
        generators = _parse_candidate_generators(payload["candidate_generators"])
        complex_structure = _as_mapping(payload["candidate_complex_structure"])
        complex_structure_origin = _parse_complex_structure_origin(
            complex_structure.get("origin", "unknown")
        )
        complex_structure_operator = parse_fraction_matrix(_as_sequence(complex_structure["matrix"]))
        allowed_gates = _parse_allowed_gates(payload["allowed_gate_generators"])
        projectors = _as_mapping(payload["split_projectors"])
        p3 = parse_fraction_matrix(_as_sequence(projectors["P3"]))
        p2 = parse_fraction_matrix(_as_sequence(projectors["P2"]))
    except (TypeError, ValueError, KeyError):
        return QCASplitAudit()

    structural_origin = _parse_structural_origin(payload.get("structural_origin", "unknown"))
    try:
        anticommutation_matrix = _anticommutation_matrix(tuple(generator.matrix for generator in generators))
    except ValueError:
        return QCASplitAudit()
    signature = _infer_signature(anticommutation_matrix)
    generator_names = tuple(generator.name for generator in generators)

    j_squares_to_minus_one = _squares_to_minus_one(complex_structure_operator)
    j_preserves_split = _preserves_split(complex_structure_operator, p3, p2)
    j_in_allowed_gate_algebra = any(gate.matrix == complex_structure_operator for gate in allowed_gates)

    gate_audit = _audit_5d_gate_algebra(allowed_gates)
    off_block_gate_generators_present = (
        gate_audit.off_block_generators_present if gate_audit is not None else False
    )
    block_diagonal_gate_algebra = (
        gate_audit.block_diagonal_gate_algebra if gate_audit is not None else False
    )
    sm_commutant_gate_algebra = (
        gate_audit.sm_commutant_gate_algebra if gate_audit is not None else False
    )

    complex_structure_compatible = (
        complex_structure_origin in ALLOWED_COMPLEX_STRUCTURE_ORIGINS
        and j_squares_to_minus_one
        and j_preserves_split
        and j_in_allowed_gate_algebra
    )
    qca_supplies_split = (
        _has_3plus2_generator_blocks(generators)
        and _has_clifford_anticommutation(anticommutation_matrix)
        and _signature_compatible(signature, complex_structure_origin)
        and structural_origin in STRUCTURAL_SPLIT_ORIGINS
    )
    verdict = _verdict(
        qca_supplies_split=qca_supplies_split,
        complex_structure_compatible=complex_structure_compatible,
        structural_origin=structural_origin,
        complex_structure_origin=complex_structure_origin,
        off_block_gate_generators_present=off_block_gate_generators_present,
        explicit_sm_commutant_failure=(
            gate_audit is not None and not gate_audit.sm_commutant_gate_algebra
        ),
        sm_commutant_gate_algebra=sm_commutant_gate_algebra,
    )

    return QCASplitAudit(
        candidate_generators=generator_names,
        anticommutation_matrix=anticommutation_matrix,
        signature=signature,
        structural_origin=structural_origin,
        complex_structure_operator=complex_structure_operator,
        complex_structure_origin=complex_structure_origin,
        complex_structure_squares_to_minus_one=j_squares_to_minus_one,
        complex_structure_preserves_3plus2_split=j_preserves_split,
        complex_structure_in_allowed_gate_algebra=j_in_allowed_gate_algebra,
        complex_structure_compatible_with_3plus2_split=complex_structure_compatible,
        off_block_gate_generators_present=off_block_gate_generators_present,
        block_diagonal_gate_algebra=block_diagonal_gate_algebra,
        sm_commutant_gate_algebra=sm_commutant_gate_algebra,
        qca_supplies_structural_3plus2_split=qca_supplies_split,
        verdict=verdict,
    )


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected object")
    return value


def _as_sequence(value: object) -> Sequence[Sequence[object]]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("expected matrix sequence")
    return value  # type: ignore[return-value]


def _parse_candidate_generators(value: object) -> tuple[CandidateGenerator, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("candidate_generators must be a sequence")

    generators: list[CandidateGenerator] = []
    for item in value:
        item_mapping = _as_mapping(item)
        name = item_mapping["name"]
        block = item_mapping["block"]
        if not isinstance(name, str) or block not in {"V3", "V2"}:
            raise ValueError("candidate generators need string names and V3/V2 blocks")
        generators.append(
            CandidateGenerator(
                name=name,
                block=block,
                matrix=parse_fraction_matrix(_as_sequence(item_mapping["matrix"])),
            )
        )
    return tuple(generators)


def _parse_allowed_gates(value: object) -> tuple[NamedMatrix, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("allowed_gate_generators must be a sequence")

    gates: list[NamedMatrix] = []
    for item in value:
        item_mapping = _as_mapping(item)
        name = item_mapping["name"]
        if not isinstance(name, str):
            raise ValueError("gate generators need string names")
        gates.append(
            NamedMatrix(
                name=name,
                matrix=parse_fraction_matrix(_as_sequence(item_mapping["matrix"])),
            )
        )
    return tuple(gates)


def _audit_5d_gate_algebra(allowed_gates: tuple[NamedMatrix, ...]):
    if not allowed_gates:
        return None
    if any(len(gate.matrix) != 5 or len(gate.matrix[0]) != 5 for gate in allowed_gates):
        return None
    return audit_one_particle_gate_algebra(allowed_gates)


def _parse_complex_structure_origin(value: object) -> ComplexStructureOrigin:
    if value in get_args(ComplexStructureOrigin):
        return value  # type: ignore[return-value]
    return "unknown"


def _parse_structural_origin(value: object) -> StructuralOrigin:
    if value in get_args(StructuralOrigin):
        return value  # type: ignore[return-value]
    return "unknown"


def _anticommutation_matrix(generators: tuple[Matrix, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            _scalar_identity_integer(
                matrix_add(matrix_multiply(left, right), matrix_multiply(right, left))
            )
            for right in generators
        )
        for left in generators
    )


def _scalar_identity_integer(matrix: Matrix) -> int:
    try:
        dimension = len(matrix)
        scalar = matrix[0][0]
        if matrix != identity_matrix(dimension, scale=scalar):
            return 0
        if scalar.denominator != 1:
            return 0
        return scalar.numerator
    except (IndexError, ValueError):
        return 0


def _infer_signature(anticommutation_matrix: tuple[tuple[int, ...], ...]) -> Signature:
    if not anticommutation_matrix:
        return "unknown"

    diagonal = [row[index] for index, row in enumerate(anticommutation_matrix)]
    if any(value == 0 for value in diagonal):
        return "unknown"

    positive = sum(1 for value in diagonal if value > 0)
    negative = sum(1 for value in diagonal if value < 0)
    if positive == 5 and negative == 0:
        return "euclidean_5"
    if positive == 4 and negative == 1:
        return "lorentzian_1_4"
    if positive == 1 and negative == 4:
        return "lorentzian_4_1"
    return "mixed"


def _squares_to_minus_one(matrix: Matrix) -> bool:
    try:
        return matrix_multiply(matrix, matrix) == identity_matrix(len(matrix), scale=-1)
    except ValueError:
        return False


def _preserves_split(matrix: Matrix, p3: Matrix, p2: Matrix) -> bool:
    try:
        return is_zero_matrix(commutator(matrix, p3)) and is_zero_matrix(commutator(matrix, p2))
    except ValueError:
        return False


def _has_3plus2_generator_blocks(generators: tuple[CandidateGenerator, ...]) -> bool:
    return (
        len(generators) == 5
        and sum(1 for generator in generators if generator.block == "V3") == 3
        and sum(1 for generator in generators if generator.block == "V2") == 2
    )


def _has_clifford_anticommutation(anticommutation_matrix: tuple[tuple[int, ...], ...]) -> bool:
    if len(anticommutation_matrix) != 5 or any(len(row) != 5 for row in anticommutation_matrix):
        return False
    return all(
        (
            abs(anticommutation_matrix[row][column]) == 2
            if row == column
            else anticommutation_matrix[row][column] == 0
        )
        for row in range(5)
        for column in range(5)
    )


def _signature_compatible(
    signature: Signature, complex_structure_origin: ComplexStructureOrigin
) -> bool:
    return signature == "euclidean_5" or (
        signature in {"lorentzian_1_4", "lorentzian_4_1"}
        and complex_structure_origin == "wick_rotation"
    )


def _verdict(
    *,
    qca_supplies_split: bool,
    complex_structure_compatible: bool,
    structural_origin: StructuralOrigin,
    complex_structure_origin: ComplexStructureOrigin,
    off_block_gate_generators_present: bool,
    explicit_sm_commutant_failure: bool,
    sm_commutant_gate_algebra: bool,
) -> BridgeVerdict:
    if qca_supplies_split and complex_structure_compatible and sm_commutant_gate_algebra:
        return "structural_bridge"
    if (
        structural_origin == "arbitrary"
        or complex_structure_origin == "by_hand"
        or off_block_gate_generators_present
        or explicit_sm_commutant_failure
    ):
        return "falsified"
    return "notation_only"
