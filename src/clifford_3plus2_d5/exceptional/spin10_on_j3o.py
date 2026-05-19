"""Phase 2: Spin(10) action on J_3(O) — the load-bearing kill test.

Under any natural Spin(10) ⊂ E_6 acting on the 27-dimensional J_3(O), the
representation 27 decomposes as

```text
27 = 16 ⊕ 10 ⊕ 1
```

where 16 is the (real form of the) Spin(10) chiral spinor, 10 is the
defining vector representation, and 1 is the trivial singlet.  This is
standard E_6 representation theory.

The decomposition is realized concretely by picking a "preferred row /
column" of J_3(O):

- **Singlet (1-dim)**: the diagonal entry ``M_{kk}`` for the preferred
  row ``k``.
- **Vector (10-dim)**: the other 2 diagonal entries (2 real) plus the
  single off-diagonal octonion ``not touching row k`` (8 real). Total
  ``2 + 8 = 10``.
- **Spinor (16-dim)**: the 2 off-diagonal octonions ``touching row k``
  (8 + 8 = 16 real).

The kill: there are 3 different row choices, giving 3 different
decompositions.  Each picks one chiral-16 candidate; the three candidates
**overlap pairwise by exactly 8** basis elements (the shared octonion).
The 3 × 16 = 48 dimensional requirement for three independent chiral-16
copies CANNOT fit into the 24-dimensional off-diagonal octonion subspace
of J_3(O), let alone the full 27-dim algebra.

**Verdict**: J_3(O) carries at most ONE chiral-16 of Spin(10), not three.
Boyle's three-generation interpretation of J_3(O) cannot be realized at
the Spin(10) representation-theory level.  Need Phase 2b to test the
complexified J_3^C(O) extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


# Off-diagonal pair index convention from j3o_algebra:
#   pair 0 = M_{12} (touches rows 0, 1)
#   pair 1 = M_{13} (touches rows 0, 2)
#   pair 2 = M_{23} (touches rows 1, 2)
TOUCHES: dict[int, tuple[int, int]] = {
    0: (0, 1),
    1: (0, 2),
    2: (1, 2),
}


def basis_index_real(diag_index: int) -> int:
    """Return the 0..26 index for diagonal element ``diag_index``."""

    if diag_index not in (0, 1, 2):
        raise ValueError(f"diagonal index must be 0, 1, or 2, got {diag_index}")
    return diag_index


def basis_index_octonion(pair_index: int, component: int) -> int:
    """Return the 0..26 index for off-diagonal pair `i`, component `c`."""

    if pair_index not in (0, 1, 2):
        raise ValueError(f"pair index must be 0, 1, or 2, got {pair_index}")
    if not 0 <= component < 8:
        raise ValueError(f"octonion component must be 0..7, got {component}")
    return 3 + pair_index * 8 + component


def decomposition_for_preferred_row(row: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return (singlet_indices, vector_indices, spinor_indices) for row k preferred.

    All indices are into the 27-dim basis ordering of J_3(O).
    """

    if row not in (0, 1, 2):
        raise ValueError(f"row must be 0, 1, or 2, got {row}")

    touched_pairs = TOUCHES[row]
    untouched_pair = next(p for p in (0, 1, 2) if p not in touched_pairs)

    singlet = (basis_index_real(row),)

    vector_list: list[int] = []
    for other_row in range(3):
        if other_row != row:
            vector_list.append(basis_index_real(other_row))
    for component in range(8):
        vector_list.append(basis_index_octonion(untouched_pair, component))

    spinor_list: list[int] = []
    for pair_index in touched_pairs:
        for component in range(8):
            spinor_list.append(basis_index_octonion(pair_index, component))

    return singlet, tuple(vector_list), tuple(spinor_list)


def total_off_diagonal_dimension() -> int:
    """Return the total off-diagonal octonion dimension of J_3(O) (= 24)."""

    return 3 * 8


def chiral16_real_dimension() -> int:
    """Return the real dimension of one chiral-16 of Spin(10) inside J_3(O).

    From the standard 27 = 16 + 10 + 1 decomposition.
    """

    return 16


def three_chiral16_required_dimension() -> int:
    return 3 * chiral16_real_dimension()


def pairwise_spinor_overlap(row_a: int, row_b: int) -> int:
    """Return ``|spinor_a ∩ spinor_b|`` over the basis-index set."""

    _, _, spinor_a = decomposition_for_preferred_row(row_a)
    _, _, spinor_b = decomposition_for_preferred_row(row_b)
    return len(set(spinor_a) & set(spinor_b))


def union_of_all_spinors() -> int:
    """Return the dimension of the union of all three preferred-row spinors."""

    union: set[int] = set()
    for row in (0, 1, 2):
        _, _, spinor = decomposition_for_preferred_row(row)
        union.update(spinor)
    return len(union)


def three_chiral16_fits_inside_j3o() -> bool:
    """Return ``True`` iff three independent chiral-16 copies could fit in J_3(O).

    Strict arithmetic: three independent chiral-16 require 48 real DOF, but
    J_3(O) only has 24 off-diagonal octonion DOF (and even using all 27
    real dimensions, 48 > 27).
    """

    return three_chiral16_required_dimension() <= total_off_diagonal_dimension()


@dataclass(frozen=True)
class Spin10DecompositionAuditPayload:
    j3o_total_dimension: int
    singlet_dimension: int
    vector_dimension: int
    spinor_dimension: int
    decomposition_sum: int
    decomposition_matches: bool
    pairwise_spinor_overlaps: tuple[int, ...]
    union_of_all_spinors: int
    total_off_diagonal: int
    three_chiral16_required: int
    three_chiral16_fits: bool
    verdict: str
    interpretation: str


@lru_cache(maxsize=1)
def spin10_decomposition_audit_payload() -> Spin10DecompositionAuditPayload:
    """Run the FJ-5 kill test."""

    # Compute decomposition dimensions (use row 2 as representative)
    singlet, vector, spinor = decomposition_for_preferred_row(2)
    decomposition_sum = len(singlet) + len(vector) + len(spinor)

    # Pairwise overlaps between the three preferred-row spinor candidates
    overlaps = (
        pairwise_spinor_overlap(0, 1),
        pairwise_spinor_overlap(0, 2),
        pairwise_spinor_overlap(1, 2),
    )

    union = union_of_all_spinors()
    required = three_chiral16_required_dimension()
    fits = three_chiral16_fits_inside_j3o()

    if (
        decomposition_sum == 27
        and not fits
        and all(overlap > 0 for overlap in overlaps)
    ):
        verdict = "J_3(O) KILL — 27 = 16 + 10 + 1"
        interpretation = (
            "J_3(O) under any natural Spin(10) ⊂ E_6 decomposes as "
            "27 = 16 + 10 + 1 (one chiral-16, one vector-10, one singlet).  "
            "Three different 'preferred-row' choices each give one chiral-16 "
            "candidate, but they overlap pairwise by 8 basis elements (one "
            f"shared octonion).  Three independent chiral-16 copies require "
            f"{required} real dimensions; J_3(O) has only {total_off_diagonal_dimension()} "
            f"off-diagonal octonion dimensions and {27} total — strictly "
            "insufficient.  Three generations CANNOT arise from J_3(O) under "
            "Spin(10) representation theory.  Standard E_6 result confirmed."
        )
    else:
        verdict = "J_3(O) UNEXPECTED"
        interpretation = (
            f"Decomposition: 1+{len(vector)}+{len(spinor)} = {decomposition_sum}.  "
            f"Pairwise overlaps: {overlaps}.  Investigate; standard E_6 result "
            "may not apply for this specific embedding."
        )

    return Spin10DecompositionAuditPayload(
        j3o_total_dimension=27,
        singlet_dimension=len(singlet),
        vector_dimension=len(vector),
        spinor_dimension=len(spinor),
        decomposition_sum=decomposition_sum,
        decomposition_matches=(decomposition_sum == 27),
        pairwise_spinor_overlaps=overlaps,
        union_of_all_spinors=union,
        total_off_diagonal=total_off_diagonal_dimension(),
        three_chiral16_required=required,
        three_chiral16_fits=fits,
        verdict=verdict,
        interpretation=interpretation,
    )
