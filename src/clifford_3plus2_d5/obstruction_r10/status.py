"""Status objects for the QCA 3+2 split audit."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal


ComplexStructureOrigin = Literal[
    "qca_chirality",
    "floquet_phase",
    "wick_rotation",
    "by_hand",
    "unknown",
]

BridgeVerdict = Literal[
    "structural_bridge",
    "conditional_bridge",
    "notation_only",
    "falsified",
]

Signature = Literal[
    "euclidean_5",
    "lorentzian_1_4",
    "lorentzian_4_1",
    "mixed",
    "unknown",
]

StructuralOrigin = Literal[
    "wall_normal_plus_floquet_time",
    "wall_normal_plus_floquet_phase",
    "coin_tau_x_tau_y",
    "arbitrary",
    "unknown",
]


@dataclass(frozen=True)
class QCASplitAudit:
    candidate_generators: tuple[str, ...] = ()
    anticommutation_matrix: tuple[tuple[int, ...], ...] = ()
    signature: Signature = "unknown"
    structural_origin: StructuralOrigin = "unknown"
    complex_structure_operator: tuple[tuple[Fraction, ...], ...] | None = None
    complex_structure_origin: ComplexStructureOrigin = "unknown"
    complex_structure_squares_to_minus_one: bool = False
    complex_structure_preserves_3plus2_split: bool = False
    complex_structure_in_allowed_gate_algebra: bool = False
    complex_structure_compatible_with_3plus2_split: bool = False
    off_block_gate_generators_present: bool = False
    block_diagonal_gate_algebra: bool = False
    sm_commutant_gate_algebra: bool = False
    qca_supplies_structural_3plus2_split: bool = False
    verdict: BridgeVerdict = "notation_only"
