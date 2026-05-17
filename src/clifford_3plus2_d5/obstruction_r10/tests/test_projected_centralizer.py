import sympy as sp

from clifford_3plus2_d5.obstruction_r10.qca.projected_centralizer import (
    projected_centralizer_pair_diagnostics,
)
from clifford_3plus2_d5.obstruction_r10.qca.rule_verdict import CentralIdempotent


def _idempotent(matrix: sp.Matrix, rank: int) -> CentralIdempotent:
    return CentralIdempotent(expression=(), matrix=matrix, rank=rank)


def _coarse_idempotents() -> tuple[CentralIdempotent, ...]:
    zero = sp.zeros(10)
    one = sp.eye(10)
    p6 = sp.diag(1, 1, 1, 1, 1, 1, 0, 0, 0, 0)
    p4 = one - p6
    return (
        _idempotent(zero, 0),
        _idempotent(p4, 4),
        _idempotent(p6, 6),
        _idempotent(one, 10),
    )


def test_projected_centralizer_classifies_split_real_blocks() -> None:
    e1 = sp.diag(1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    e2 = sp.diag(0, 0, 1, 1, 0, 0, 0, 0, 0, 0)
    diagnostics = projected_centralizer_pair_diagnostics(
        (sp.eye(10), e1, e2),
        _coarse_idempotents(),
    )

    assert len(diagnostics) == 1
    rank6, rank4 = diagnostics[0].blocks
    assert rank6.classification == "split_real"
    assert rank6.primitive_component_types == ("R", "R", "R")
    assert rank4.classification == "split_real"
    assert rank4.primitive_component_types == ("R",)


def test_projected_centralizer_detects_complex_factor() -> None:
    j4 = sp.zeros(10)
    j4[6, 7] = -1
    j4[7, 6] = 1
    j4[8, 9] = -1
    j4[9, 8] = 1
    diagnostics = projected_centralizer_pair_diagnostics(
        (sp.eye(10), j4),
        _coarse_idempotents(),
    )

    assert len(diagnostics) == 1
    rank4 = diagnostics[0].blocks[1]
    assert rank4.classification == "contains_complex_factor"
    assert rank4.primitive_component_types == ("C",)
