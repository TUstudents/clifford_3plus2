"""Finite periodic lattice helpers for real-space BCC steps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TypeAlias


Site: TypeAlias = tuple[int, int, int]
Displacement: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class PeriodicLattice3D:
    shape: Site

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(size <= 0 for size in self.shape):
            raise ValueError("PeriodicLattice3D shape must contain three positive sizes")

    @property
    def volume(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def wrap(self, site: Site) -> Site:
        return tuple(coord % size for coord, size in zip(site, self.shape, strict=True))  # type: ignore[return-value]

    def translate(self, site: Site, displacement: Displacement) -> Site:
        return self.wrap(
            tuple(coord + step for coord, step in zip(site, displacement, strict=True)),  # type: ignore[arg-type]
        )

    def sites(self) -> tuple[Site, ...]:
        nx, ny, nz = self.shape
        return tuple((x, y, z) for x, y, z in product(range(nx), range(ny), range(nz)))
