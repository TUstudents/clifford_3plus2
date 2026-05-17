"""Lab A sampled Bloch candidates on the R^4 clock-plane carrier."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.lepton.primitives import (
    LAB_A_DIMENSION,
    LAB_A_MODE_COUNT,
    LabACandidate,
    mode_swap_matrix,
    signed_twist_matrix,
)
from clifford_3plus2_d5.obstruction_r10.qca.rule_verdict import RuleLayerInput


@dataclass(frozen=True)
class LabABlochCandidate(LabACandidate):
    """Joint sampled Bloch algebra candidate for the Lab A carrier."""

    period: int = 1
    windings: tuple[int, ...] = ()
    onsite_kind: str = "identity"

    @property
    def metadata(self) -> tuple[tuple[str, str], ...]:
        return (
            ("period", str(self.period)),
            ("windings", ",".join(str(item) for item in self.windings)),
            ("onsite_kind", self.onsite_kind),
            ("sample_count", str(len(self.layers))),
        )


def _clock_phase_block(angle: sp.Expr) -> sp.Matrix:
    cosine = sp.simplify(sp.cos(angle))
    sine = sp.simplify(sp.sin(angle))
    return sp.Matrix([[cosine, -sine], [sine, cosine]])


def bloch_phase_matrix(
    *,
    period: int,
    sample: int,
    windings: Sequence[int],
) -> sp.Matrix:
    """Return the real R^4 phase matrix for one sampled Bloch root."""

    if period <= 0:
        raise ValueError("period must be positive")
    if not 0 <= sample < period:
        raise ValueError("sample must lie in [0, period)")
    if len(windings) != LAB_A_MODE_COUNT:
        raise ValueError("Lab A Bloch windings require two mode entries")

    matrix = sp.zeros(LAB_A_DIMENSION)
    for mode, winding in enumerate(windings):
        angle = 2 * sp.pi * sp.Rational(sample * winding, period)
        block = _clock_phase_block(angle)
        x_index = mode
        y_index = LAB_A_MODE_COUNT + mode
        matrix[x_index, x_index] = block[0, 0]
        matrix[x_index, y_index] = block[0, 1]
        matrix[y_index, x_index] = block[1, 0]
        matrix[y_index, y_index] = block[1, 1]
    return matrix.applyfunc(sp.simplify)


def _onsite_matrix(kind: str) -> sp.Matrix:
    if kind == "identity":
        return identity(LAB_A_DIMENSION)
    if kind == "mode_swap":
        return mode_swap_matrix()
    if kind == "signed_twist_first":
        return signed_twist_matrix((-1, 1))
    if kind == "signed_twist_second":
        return signed_twist_matrix((1, -1))
    raise ValueError(f"unknown Lab A Bloch onsite kind: {kind}")


def bloch_sample_layer(
    *,
    name: str,
    period: int,
    sample: int,
    windings: Sequence[int],
    onsite_kind: str,
) -> RuleLayerInput:
    """Return one real sampled Bloch symbol as a verdict layer."""

    phase = bloch_phase_matrix(period=period, sample=sample, windings=windings)
    matrix = (_onsite_matrix(onsite_kind) * phase).applyfunc(sp.simplify)
    return RuleLayerInput(
        name=f"{name}_sample_{sample}",
        matrix=matrix,
        support=(0,),
        locality_radius=0,
    )


def lab_a_bloch_candidate(
    *,
    period: int,
    windings: Sequence[int],
    onsite_kind: str = "identity",
) -> LabABlochCandidate:
    """Build the joint sampled algebra candidate over all roots in one period."""

    winding_tuple = tuple(int(item) for item in windings)
    name = (
        f"lab_a_bloch_p{period}_w"
        f"{'_'.join(str(item) for item in winding_tuple)}_{onsite_kind}"
    )
    layers = tuple(
        bloch_sample_layer(
            name=name,
            period=period,
            sample=sample,
            windings=winding_tuple,
            onsite_kind=onsite_kind,
        )
        for sample in range(period)
    )
    return LabABlochCandidate(
        name=name,
        layers=layers,
        period=period,
        windings=winding_tuple,
        onsite_kind=onsite_kind,
    )


def iter_lab_a_bloch_candidates(
    *,
    periods: Sequence[int] = (3, 4, 6),
    winding_values: Sequence[int] = (-1, 1, 2),
    onsite_kinds: Sequence[str] = ("identity", "mode_swap"),
    max_candidates: int | None = None,
) -> Iterable[LabABlochCandidate]:
    """Yield bounded Lab A sampled Bloch candidates.

    The sampled symbols are represented as real clock-plane rotations on R^4.
    This keeps the first Lab A Bloch scan inside the same v2 verdict contract as
    the on-site scan, while still recording the joint sampled algebra generated
    by all period roots.
    """

    yielded = 0
    for period in periods:
        if period <= 0:
            raise ValueError("periods must be positive")
        for windings in product(winding_values, repeat=LAB_A_MODE_COUNT):
            normalized = tuple(item % period for item in windings)
            if all(item == 0 for item in normalized):
                continue
            for onsite_kind in onsite_kinds:
                yield lab_a_bloch_candidate(
                    period=period,
                    windings=windings,
                    onsite_kind=onsite_kind,
                )
                yielded += 1
                if max_candidates is not None and yielded >= max_candidates:
                    return
