"""Phase A-1 (FA-1, FA-2): identify the SME sector that H^(1) maps to.

H^(1) is the BCC Bialynicki-Birula walk's O(ε) correction to the free-
Dirac Hamiltonian.  cp/continuum_cp.py already verified that H^(1) is

- Hermitian,
- chirality-preserving (block-diagonal in chiral basis),
- 100% CP-odd,
- entirely in the T_{2g} cubic-harmonic irrep,
- bilinear in spatial momentum (degree 2 in k).

This module re-verifies each of these symmetry classes directly from
the cp/ outputs (no new lattice computation) and packages them into a
single dataclass, then maps to the natural SME-sector target:

> Dim-5 non-minimal SME (Kostelecky-Mewes, arXiv:1102.4068), fermion
> sector, in a CP-odd subclass of the spin-tensor dim-5 coefficients.

The natural targets, given the chirality-preserving + spin-Σ^i
structure of H^(1) in chiral basis, are subsets of the dim-5
coefficients ``m^(5)``, ``g^(5)``, and ``H^(5)`` (where the latter
two carry the rotation-anisotropic spin index).  The precise tensor-
index pattern is fixed by Phase A-2; this module only labels the
sector.

Pre-named failure mode F-sme-5 (H^(1) maps to a field-redefinition-
trivial direction) is checked at the audit level in Phase A-5; here
we only declare the sector class.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.sme.reuse import (
    cp_irrep_norm_table,
    effective_hamiltonian_first_correction,
    symbolic_momentum,
)


def h1_is_hermitian() -> bool:
    """Return whether H^(1) - H^(1)† = 0 symbolically."""

    h1 = effective_hamiltonian_first_correction()
    residual = (h1 - h1.H).applyfunc(sp.simplify)
    return residual == sp.zeros(h1.rows, h1.cols)


def h1_is_chirality_preserving() -> bool:
    """Return whether H^(1) is block-diagonal in the 4-spinor chiral basis.

    In chiral basis the upper-left and lower-right 2×2 blocks are the
    left- and right-handed Weyl-spinor parts.  Chirality-preserving
    means the upper-right and lower-left 2×2 blocks vanish.
    """

    h1 = effective_hamiltonian_first_correction()
    upper_right = h1[:2, 2:].applyfunc(sp.simplify)
    lower_left = h1[2:, :2].applyfunc(sp.simplify)
    return upper_right == sp.zeros(2, 2) and lower_left == sp.zeros(2, 2)


def h1_is_momentum_bilinear() -> bool:
    """Return whether every entry of H^(1) is a polynomial of total degree 2 in k.

    Verified by Taylor-expanding each entry in (kx, ky, kz) and checking
    that only the second-order terms are non-zero.
    """

    _, kx, ky, kz = symbolic_momentum()
    h1 = effective_hamiltonian_first_correction()
    for row in range(h1.rows):
        for col in range(h1.cols):
            entry = sp.expand(h1[row, col])
            if entry == 0:
                continue
            poly = sp.Poly(entry, kx, ky, kz)
            for monomial in poly.monoms():
                if sum(monomial) != 2:
                    return False
    return True


def h1_cp_odd_fraction() -> sp.Expr:
    """Return ``||CP-odd part||² / ||H^(1)||²`` from the cp/ norm table."""

    table = cp_irrep_norm_table()
    cp_odd_total = sum(value for key, value in table.items() if key[0] == "CP-odd")
    total = sum(table.values())
    if sp.simplify(total) == 0:
        return sp.Integer(0)
    return sp.simplify(cp_odd_total / total)


def h1_t2g_fraction() -> sp.Expr:
    """Return ``||T_{2g} part||² / ||H^(1)||²`` from the cp/ norm table."""

    table = cp_irrep_norm_table()
    t2g_total = sum(
        value for key, value in table.items() if key[1] == "T2g"
    )
    total = sum(table.values())
    if sp.simplify(total) == 0:
        return sp.Integer(0)
    return sp.simplify(t2g_total / total)


def h1_lives_entirely_in_cp_odd_t2g() -> bool:
    """Return whether 100% of ||H^(1)||² is concentrated in (CP-odd, T_{2g})."""

    table = cp_irrep_norm_table()
    cp_odd_t2g = sp.simplify(table[("CP-odd", "T2g")])
    total = sp.simplify(sum(table.values()))
    if total == 0:
        return False
    return sp.simplify(cp_odd_t2g / total - 1) == 0


def dim5_sme_sector_label() -> str:
    """Return the natural SME-sector label for H^(1).

    Based on the FA-1 / FA-2 analysis: H^(1) is bilinear in k and acts
    as a spin-tensor on the per-chirality 2-spinor, identical on both
    chiralities.  This places it in the dim-5 non-minimal SME fermion
    sector with CPT- and CP-properties to be pinned down in Phase A-2.
    """

    return "dim-5 non-minimal SME, fermion sector, CP-odd spin-tensor"


@dataclass(frozen=True)
class H1SymmetryClass:
    """The five load-bearing symmetry properties of H^(1)."""

    hermitian: bool
    chirality_preserving: bool
    momentum_bilinear: bool
    cp_odd_fraction: sp.Expr
    t2g_fraction: sp.Expr
    lives_entirely_in_cp_odd_t2g: bool
    sme_sector_label: str


def h1_symmetry_class() -> H1SymmetryClass:
    """Return the load-bearing symmetry class of H^(1)."""

    return H1SymmetryClass(
        hermitian=h1_is_hermitian(),
        chirality_preserving=h1_is_chirality_preserving(),
        momentum_bilinear=h1_is_momentum_bilinear(),
        cp_odd_fraction=h1_cp_odd_fraction(),
        t2g_fraction=h1_t2g_fraction(),
        lives_entirely_in_cp_odd_t2g=h1_lives_entirely_in_cp_odd_t2g(),
        sme_sector_label=dim5_sme_sector_label(),
    )


@dataclass(frozen=True)
class FrameworkIdentificationPayload:
    symmetry_class: H1SymmetryClass
    sme_sector_label: str
    all_classes_consistent: bool
    verdict: str
    interpretation: str


def framework_identification_payload() -> FrameworkIdentificationPayload:
    """Run the Phase A-1 audit."""

    sym = h1_symmetry_class()
    sector = dim5_sme_sector_label()

    all_consistent = (
        sym.hermitian
        and sym.chirality_preserving
        and sym.momentum_bilinear
        and sym.lives_entirely_in_cp_odd_t2g
        and sp.simplify(sym.cp_odd_fraction - 1) == 0
        and sp.simplify(sym.t2g_fraction - 1) == 0
    )

    if all_consistent:
        verdict = "FRAMEWORK IDENTIFIED — dim-5 non-minimal SME fermion sector"
        interpretation = (
            "H^(1) satisfies all five symmetry classes (Hermitian, "
            "chirality-preserving, momentum-bilinear, 100% CP-odd, "
            "100% T_{2g}).  Maps to the dim-5 non-minimal SME fermion "
            "sector with rotation-anisotropic spin-tensor coefficients.  "
            "Phase A-2 pins down the specific Kostelecky-Mewes tensor "
            "indices that carry non-zero coefficients."
        )
    else:
        verdict = "FRAMEWORK MISMATCH"
        interpretation = (
            f"Symmetry class inconsistency.  hermitian={sym.hermitian}, "
            f"chirality-preserving={sym.chirality_preserving}, "
            f"momentum-bilinear={sym.momentum_bilinear}, "
            f"cp-odd-fraction={sym.cp_odd_fraction}, "
            f"t2g-fraction={sym.t2g_fraction}.  "
            "Investigate before proceeding to Phase A-2."
        )

    return FrameworkIdentificationPayload(
        symmetry_class=sym,
        sme_sector_label=sector,
        all_classes_consistent=all_consistent,
        verdict=verdict,
        interpretation=interpretation,
    )
