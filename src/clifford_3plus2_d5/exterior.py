"""Exterior-algebra bookkeeping for the fixed 3+2 split."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True, order=True)
class ExteriorBasisElement:
    """A basis wedge in Lambda(C^3 oplus C^2)."""

    subset: tuple[int, ...]
    n3: int
    n2: int

    @property
    def degree(self) -> int:
        return len(self.subset)

    @property
    def sector(self) -> tuple[int, int]:
        return (self.n3, self.n2)


def split_counts(subset: tuple[int, ...], *, v3_dimension: int = 3) -> tuple[int, int]:
    """Count selected basis directions in the C^3 and C^2 blocks."""

    n3 = sum(1 for index in subset if index <= v3_dimension)
    return n3, len(subset) - n3


def even_subsets(dimension: int = 5) -> tuple[tuple[int, ...], ...]:
    """Enumerate even subsets of {1, ..., dimension}."""

    basis = tuple(range(1, dimension + 1))
    return tuple(
        subset
        for degree in range(0, dimension + 1, 2)
        for subset in combinations(basis, degree)
    )


def even_basis_3plus2() -> tuple[ExteriorBasisElement, ...]:
    """Enumerate the 16 even wedges for C^3 oplus C^2."""

    return tuple(
        ExteriorBasisElement(subset=subset, n3=n3, n2=n2)
        for subset in even_subsets(5)
        for n3, n2 in (split_counts(subset, v3_dimension=3),)
    )


def sector_multiplicities() -> dict[tuple[int, int], int]:
    """Return multiplicities grouped by (N_3, N_2)."""

    counts: dict[tuple[int, int], int] = {}
    for basis_element in even_basis_3plus2():
        counts[basis_element.sector] = counts.get(basis_element.sector, 0) + 1
    return counts
