"""Phase A-2 (FA-3, FA-4): map H^(1) to dim-5 SME tensor components.

The (CP-odd, T_{2g}) cell of H^(1) is the 4×4 chiral-basis Hamiltonian
correction.  It is block-diagonal with identical 2×2 blocks per
chirality.  Per-chirality, the block can be written as

    H^(1)_chir(k) = T^{a ij} σ^a k_i k_j

where σ^a are the Pauli matrices (a = x, y, z) and T^{aij} is the
sparse symbolic tensor.

This module:

1. Extracts the (CP-odd, T_{2g}) cell from cp/.
2. Projects each per-chirality block onto the Pauli basis
   ``{I, σ^x, σ^y, σ^z}`` to obtain four momentum-polynomial
   coefficients.
3. Reads off the non-zero ``T^{aij}`` entries.
4. Maps the resulting structure to the dim-5 non-minimal SME fermion
   sector:

       L_correction ⊃ ε · T^{aij} · ψ̄ γ^a γ^5 ∂_i ∂_j ψ

   This is the **axial-vector × two-derivative** non-minimal SME
   coefficient.  In the Kostelecky-Mewes conventions of arXiv:1102.4068
   and arXiv:1306.6088 this is the ``d^{(5)}_{αβγ}`` coefficient
   (CPT-even, with CP-properties determined by the index pattern).
   H^(1)'s CP-odd character combined with the lattice walk's
   CPT-invariance pins the class to ``d^{(5)}`` (CPT-even, CP-odd).

The mapping below is in the ψ̄-bilinear convention of Kostelecky-Mewes;
the precise Hamiltonian-form normalization is also recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.sme.reuse import (
    cp_irrep_decomposition,
    symbolic_momentum,
)


_PAULI_NAMES = ("I", "x", "y", "z")


def _pauli_matrix(label: str) -> sp.Matrix:
    """Return the 2×2 Pauli matrix labelled by ``"I"``, ``"x"``, ``"y"``, or ``"z"``."""

    if label == "I":
        return sp.eye(2)
    if label == "x":
        return sp.Matrix([[0, 1], [1, 0]])
    if label == "y":
        return sp.Matrix([[0, -sp.I], [sp.I, 0]])
    if label == "z":
        return sp.Matrix([[1, 0], [0, -1]])
    raise ValueError(f"unknown Pauli label: {label}")


def h1_per_chirality_block() -> sp.Matrix:
    """Return the 2×2 upper-left block of the (CP-odd, T_{2g}) cell of H^(1)."""

    decomp = cp_irrep_decomposition()
    cell = decomp[("CP-odd", "T2g")]
    block = cell[:2, :2]
    return block.applyfunc(sp.expand)


def lower_chirality_block_equals_upper() -> bool:
    """Return whether the lower-right 2×2 block equals the upper-left."""

    decomp = cp_irrep_decomposition()
    cell = decomp[("CP-odd", "T2g")]
    upper = cell[:2, :2].applyfunc(sp.expand)
    lower = cell[2:, 2:].applyfunc(sp.expand)
    return (upper - lower).applyfunc(sp.simplify) == sp.zeros(2, 2)


def pauli_decomposition_per_chirality() -> dict[str, sp.Expr]:
    """Return ``{"I", "x", "y", "z"} → polynomial in (kx, ky, kz)``.

    Pauli projection: ``c_a = (1/2) tr(σ^a · M)``.  Returns the four
    momentum-polynomial coefficients of the per-chirality block.
    """

    block = h1_per_chirality_block()
    coefficients: dict[str, sp.Expr] = {}
    for label in _PAULI_NAMES:
        sigma = _pauli_matrix(label)
        projected = (sigma * block).trace() / 2
        coefficients[label] = sp.expand(projected)
    return coefficients


@dataclass(frozen=True)
class T2gTensorEntry:
    """A non-zero entry of the T^{aij} tensor.

    Fields:
        pauli_index: 'x', 'y', or 'z' (the spin axis).
        momentum_pair: (i, j) with i ≤ j; the T_{2g} basis element.
        coefficient: the symbolic SymPy coefficient (typically ±1).
    """

    pauli_index: str
    momentum_pair: tuple[str, str]
    coefficient: sp.Expr


def t2g_tensor_entries() -> tuple[T2gTensorEntry, ...]:
    """Return the non-zero T^{aij} tensor entries extracted from H^(1).

    Each entry corresponds to one term in the per-chirality
    decomposition

        H^(1)_chir(k) = Σ_entry  coefficient · σ^{pauli} · k_i k_j .
    """

    _, kx, ky, kz = symbolic_momentum()
    coefficients = pauli_decomposition_per_chirality()

    # T_{2g} momentum basis (i ≤ j ordering)
    t2g_basis: tuple[tuple[str, str, sp.Symbol], ...] = (
        ("y", "z", ky * kz),
        ("x", "z", kx * kz),
        ("x", "y", kx * ky),
    )

    entries: list[T2gTensorEntry] = []
    for pauli_label in ("x", "y", "z"):
        polynomial = sp.expand(coefficients[pauli_label])
        for label_i, label_j, monomial in t2g_basis:
            # Extract the coefficient of the monomial.
            coeff = polynomial.coeff(monomial)
            coeff = sp.expand(coeff)
            if coeff != 0:
                entries.append(
                    T2gTensorEntry(
                        pauli_index=pauli_label,
                        momentum_pair=(label_i, label_j),
                        coefficient=coeff,
                    )
                )
    return tuple(entries)


def identity_pauli_polynomial_is_zero() -> bool:
    """Return whether the Pauli-identity coefficient of H^(1)_chir vanishes.

    A non-zero identity component would indicate a chirality-blind
    scalar correction, which is NOT what H^(1)'s spin-tensor structure
    predicts.
    """

    coefficients = pauli_decomposition_per_chirality()
    return sp.expand(coefficients["I"]) == 0


def dim5_sme_target_label() -> str:
    """Return the dim-5 non-minimal SME coefficient label for H^(1).

    Per Kostelecky-Mewes 2011 (arXiv:1102.4068) and 2013 (arXiv:1306.6088)
    conventions, an axial-vector bilinear (γ^μ γ^5) with two derivatives,
    CPT-even and CP-odd, is the ``d^{(5)}_{αβγ}`` coefficient.
    """

    return "d^{(5)}_{αβγ}  (axial-vector × 2 derivatives; CPT-even, CP-odd)"


def expected_cpt_class() -> str:
    """Return ``"CPT-even"``.

    The BCC Bialynicki-Birula walk is CPT-invariant (standard for any
    unitary walk derivable from a Dirac-like update rule).  CP-odd +
    CPT-conserved implies T-odd, which is internally consistent.  The
    dim-5 SME target must therefore be CPT-even, picking ``d^{(5)}``
    over ``b^{(5)}`` (which is CPT-odd).
    """

    return "CPT-even"


@dataclass(frozen=True)
class SMETensorMappingPayload:
    """Result of the Phase A-2 mapping."""

    pauli_coefficients: dict[str, sp.Expr]
    identity_coefficient_vanishes: bool
    lower_block_equals_upper: bool
    tensor_entries: tuple[T2gTensorEntry, ...]
    sme_target_label: str
    cpt_class: str
    nonzero_component_count: int
    verdict: str
    interpretation: str


def mapping_audit_payload() -> SMETensorMappingPayload:
    """Run the Phase A-2 mapping audit."""

    coefficients = pauli_decomposition_per_chirality()
    identity_vanishes = identity_pauli_polynomial_is_zero()
    lower_equals_upper = lower_chirality_block_equals_upper()
    entries = t2g_tensor_entries()
    label = dim5_sme_target_label()
    cpt = expected_cpt_class()

    consistent = (
        identity_vanishes
        and lower_equals_upper
        and len(entries) == 3  # T_{2g} has dimension 3
    )

    if consistent:
        verdict = "MAPPING IDENTIFIED — d^{(5)} axial-vector dim-5 SME"
        interpretation = (
            f"H^(1) per chirality decomposes onto {len(entries)} non-zero "
            f"Pauli × T_{{2g}}-momentum components.  The identity-Pauli "
            f"coefficient vanishes, ruling out a chirality-blind scalar "
            f"correction.  Upper and lower chirality blocks are equal, "
            f"confirming the natural ``ψ̄ γ^a γ^5 ∂_i ∂_j ψ`` Lagrangian "
            f"structure — the dim-5 non-minimal SME ``{label}`` coefficient.  "
            f"CPT class: {cpt}.  Phase A-3 pulls experimental bounds on "
            f"the three non-zero d^{{(5)}} components."
        )
    else:
        verdict = "MAPPING INCONSISTENT"
        interpretation = (
            f"identity_vanishes={identity_vanishes}, "
            f"lower_equals_upper={lower_equals_upper}, "
            f"entries={len(entries)}.  Investigate before Phase A-3."
        )

    return SMETensorMappingPayload(
        pauli_coefficients=coefficients,
        identity_coefficient_vanishes=identity_vanishes,
        lower_block_equals_upper=lower_equals_upper,
        tensor_entries=entries,
        sme_target_label=label,
        cpt_class=cpt,
        nonzero_component_count=len(entries),
        verdict=verdict,
        interpretation=interpretation,
    )
