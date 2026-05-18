"""K1/K2 kill test for the triality sidecar.

Per ``PLAN.md``, the load-bearing question is whether the explicit Spin(8)
triality automorphism preserves the SM-inside-Spin(8) subgroup and produces
three equivalent generation carriers.  The SM-inside-Spin(8) is necessarily
weaker than the full Pati-Salam SM:

- ``SU(3)_c`` survives entirely (Cl(0,6) ⊂ Cl(0,8));
- ``SU(2)_L`` does not survive (each generator mixes indices ``{6, 7}``
  inside Spin(8) with indices ``{8, 9}`` outside);
- ``U(1)_Y`` survives only as the projection ``Y'`` of the physical
  hypercharge onto the Spin(8) Cartan span.

The kill test focuses on the Cartan of this Spin(8)-restricted SM:

``Cartan(SU(3)_c) ⊕ U(1)_{Y'}``  (3-dimensional inside ``Cartan(so(8))``).

**K1 — Cartan necessary condition**.  Apply the triality Cartan matrix
``T`` to each generator of the 3-dimensional SM Cartan.  If any image
falls outside the 3-dimensional subspace, K1 fails and triality cannot
preserve the full SM-inside-Spin(8).

**K2 — Y' spectrum on the chiral-16**.  Report the eigenvalue multiset of
the Y' charge observable on the chiral-16.  This is an SM-restricted
audit (not a triality audit per se), but it provides context for the
interpretation of the K1 result.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

import sympy as sp

from clifford_3plus2_d5.triality.reuse import (
    charge_observable,
    physical_hypercharge_generator,
    su3_c_generators_from_su4,
)
from clifford_3plus2_d5.triality.spin8_triality import (
    apply_triality_to_cartan_vector,
    cartan_coordinates,
    spin8_cartan_on_chiral16,
)

MatrixTuple: TypeAlias = tuple[sp.Matrix, ...]
ChargeSpectrum: TypeAlias = dict[sp.Expr, int]


def _flatten_to_column(matrix: sp.Matrix) -> sp.Matrix:
    entries = [matrix[row, col] for row in range(matrix.rows) for col in range(matrix.cols)]
    return sp.Matrix(matrix.rows * matrix.cols, 1, entries)


def _spin8_cartan_basis_matrix() -> sp.Matrix:
    return sp.Matrix.hstack(
        *(_flatten_to_column(item) for item in spin8_cartan_on_chiral16()),
    )


def restricted_hypercharge_cartan_coords() -> sp.Matrix:
    """Return ``Y'`` as a 4x1 vector in the ``(H_0, H_1, H_2, H_3)`` basis.

    Computed by least-squares projection of ``physical_hypercharge_generator``
    onto the Spin(8) Cartan span.  The residual is the part of ``Y`` that
    lives outside Spin(8) (specifically the ``γ_8 γ_9`` direction).
    """

    basis = _spin8_cartan_basis_matrix()
    target = _flatten_to_column(physical_hypercharge_generator())
    coords = basis.solve_least_squares(target).applyfunc(sp.simplify)
    return coords


def restricted_hypercharge_generator() -> sp.Matrix:
    """Return ``Y'`` as a 32x32 skew matrix on the chiral-16."""

    coords = restricted_hypercharge_cartan_coords()
    cartan = spin8_cartan_on_chiral16()
    result = sp.zeros(32)
    for index in range(4):
        result = (result + coords[index, 0] * cartan[index]).applyfunc(sp.simplify)
    return result


def restricted_hypercharge_residual_norm_squared() -> sp.Expr:
    """Return the squared Frobenius norm of ``Y - Y'``.

    This is the part of physical hypercharge that lives outside Spin(8).
    Nonzero confirms Y has out-of-Spin(8) content.
    """

    full = physical_hypercharge_generator()
    restricted = restricted_hypercharge_generator()
    residual = (full - restricted).applyfunc(sp.simplify)
    return sp.simplify(sum(residual[r, c] ** 2 for r in range(32) for c in range(32)))


def su3_c_cartan_indices() -> tuple[int, ...]:
    """Indices into ``su3_c_generators_from_su4`` that lie in the Spin(8) Cartan."""

    indices: list[int] = []
    for index, generator in enumerate(su3_c_generators_from_su4()):
        coords = cartan_coordinates(generator)
        if coords is None:
            continue
        if any(coords[row, 0] != 0 for row in range(4)):
            indices.append(index)
    return tuple(indices)


def su3_c_cartan_coords() -> tuple[sp.Matrix, ...]:
    """Cartan coords for the SU(3)_c generators that lie in Spin(8) Cartan."""

    pieces: list[sp.Matrix] = []
    for index in su3_c_cartan_indices():
        coords = cartan_coordinates(su3_c_generators_from_su4()[index])
        if coords is None:
            raise RuntimeError(
                f"SU(3)_c generator {index} unexpectedly not in Cartan span",
            )
        pieces.append(coords)
    return tuple(pieces)


def g_sm_8_cartan_basis_coords() -> tuple[sp.Matrix, ...]:
    """SM-in-Spin(8) Cartan basis: SU(3)_c Cartan plus ``Y'``."""

    return (*su3_c_cartan_coords(), restricted_hypercharge_cartan_coords())


def _span_matrix(vectors: tuple[sp.Matrix, ...]) -> sp.Matrix:
    return sp.Matrix.hstack(*vectors)


def _is_in_column_span(span: sp.Matrix, vector: sp.Matrix) -> bool:
    augmented = span.row_join(vector)
    return augmented.rank() == span.rank()


def k1_failure_witnesses() -> tuple[tuple[sp.Matrix, sp.Matrix], ...]:
    """Return ``(v, T v)`` pairs for SM Cartan generators that triality maps out.

    A non-empty result means K1 fails.
    """

    basis = g_sm_8_cartan_basis_coords()
    span = _span_matrix(basis)
    witnesses: list[tuple[sp.Matrix, sp.Matrix]] = []
    for vector in basis:
        image = apply_triality_to_cartan_vector(vector)
        if not _is_in_column_span(span, image):
            witnesses.append((vector, image))
    return tuple(witnesses)


def k1_passes() -> bool:
    """Return ``True`` iff the SM-inside-Spin(8) Cartan is closed under triality."""

    return not k1_failure_witnesses()


@lru_cache(maxsize=1)
def y_prime_observable() -> sp.Matrix:
    """Return the Y' charge observable on the chiral-16 (32x32 real symmetric)."""

    return charge_observable(restricted_hypercharge_generator())


@lru_cache(maxsize=1)
def y_prime_real_spectrum() -> ChargeSpectrum:
    """Return the real-multiplicity eigenvalue spectrum of Y' on chiral-16."""

    return {sp.simplify(key): value for key, value in y_prime_observable().eigenvals().items()}


def y_prime_complex_spectrum() -> ChargeSpectrum:
    """Return complex multiplicities (real // 2) for ``Y'`` eigenvalues."""

    spectrum = y_prime_real_spectrum()
    if any(multiplicity % 2 != 0 for multiplicity in spectrum.values()):
        raise RuntimeError("Y' real-spectrum multiplicities must all be even")
    return {charge: multiplicity // 2 for charge, multiplicity in spectrum.items()}


@dataclass(frozen=True)
class TrialityKillTestAudit:
    """Result payload for the K1/K2 triality kill test."""

    # Setup facts
    spin8_embedding: str
    su3_c_cartan_indices: tuple[int, ...]
    su3_c_cartan_dimension: int
    g_sm_8_cartan_dimension: int
    restricted_hypercharge_cartan_coords: tuple[sp.Expr, ...]
    hypercharge_residual_norm_squared: sp.Expr

    # K1: Cartan-level test
    k1_passes: bool
    k1_failure_witness_count: int

    # K2: spectrum diagnostic
    y_prime_eigenvalue_count: int
    y_prime_total_complex_multiplicity: int
    y_prime_spectrum_real: ChargeSpectrum
    y_prime_spectrum_complex: ChargeSpectrum

    # Verdict
    verdict: str
    interpretation: str


def kill_test_audit_payload() -> TrialityKillTestAudit:
    """Run the K1/K2 kill test and return the audit payload."""

    coords = restricted_hypercharge_cartan_coords()
    residual_norm_sq = restricted_hypercharge_residual_norm_squared()
    su3_indices = su3_c_cartan_indices()
    g_sm_8_dim = len(su3_indices) + 1
    k1_pass = k1_passes()
    witnesses = k1_failure_witnesses()
    real_spectrum = y_prime_real_spectrum()
    complex_spectrum = y_prime_complex_spectrum()
    total_complex = sum(complex_spectrum.values())

    if k1_pass:
        verdict = "K1 PASS"
        interpretation = (
            "Triality preserves the SM-inside-Spin(8) Cartan as a subspace. "
            "The sidecar earns the right to grow into a full-K1 audit on all "
            "su3_c root vectors and into K2 generation-equivalence work."
        )
    else:
        verdict = "K1 FAIL"
        interpretation = (
            f"Triality maps {len(witnesses)} of {g_sm_8_dim} SM-inside-Spin(8) "
            "Cartan generators outside the SM Cartan subspace.  Triality does "
            "not preserve the SM-inside-Spin(8) subalgebra as a subgroup.  "
            "The three triality-rotated chiral-16 carriers carry inequivalent "
            "SM-content positions, so they cannot represent three equivalent "
            "generations.  Program dies cleanly at K1."
        )

    return TrialityKillTestAudit(
        spin8_embedding="indices {0..7} of Cl(0,10), Cartan H_k = γ_{2k} γ_{2k+1} / 2",
        su3_c_cartan_indices=su3_indices,
        su3_c_cartan_dimension=len(su3_indices),
        g_sm_8_cartan_dimension=g_sm_8_dim,
        restricted_hypercharge_cartan_coords=tuple(coords[i, 0] for i in range(4)),
        hypercharge_residual_norm_squared=residual_norm_sq,
        k1_passes=k1_pass,
        k1_failure_witness_count=len(witnesses),
        y_prime_eigenvalue_count=len(real_spectrum),
        y_prime_total_complex_multiplicity=total_complex,
        y_prime_spectrum_real=real_spectrum,
        y_prime_spectrum_complex=complex_spectrum,
        verdict=verdict,
        interpretation=interpretation,
    )
