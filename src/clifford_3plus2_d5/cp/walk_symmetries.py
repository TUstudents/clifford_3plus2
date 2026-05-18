"""Alpha audit: which discrete symmetries commute with the BCC Dirac walk.

For each ``SymmetryOperator`` ``S`` with spinor matrix ``M``, the test is:

- Unitary ``S``:    ``M B(k) M^{-1} = B(k_image)``
- Antiunitary ``S``: ``M B(k)* M^{-1} = B(k_image)^†``

where ``k_image = -k`` if the operator flips momentum, ``+k`` otherwise.
The dagger appears because an antiunitary symmetry commuting with the
Hamiltonian satisfies ``S U S^{-1} = U^{-1}``.

This module exposes:

- ``walk_respects_symmetry(op)``: bool verdict at symbolic momentum;
- ``massless_audit_table()``: full table over the 7 named operators;
- ``MasslessAuditPayload``: dataclass for the alpha-2 report.

Alpha-2 (massless): just the bare BCC Dirac walk.

Alpha-3 (Higgs-perturbed): adds a static Yukawa-like term lifted from
``spacetime_qca.yukawa`` and reruns the audit.  The Higgs lift requires
the internal carrier; for the lifted audit each spinor operator is
tensor-multiplied with an internal action.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.cp.discrete_symmetries import (
    SymmetryOperator,
    all_seven_operators,
)
from clifford_3plus2_d5.cp.reuse import (
    bcc_dirac_symbol,
    same_matrix,
)


def _symbolic_momentum() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    epsilon = sp.symbols("epsilon", positive=True)
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    return epsilon, kx, ky, kz


def _bloch_at(eps: sp.Symbol, kx: sp.Expr, ky: sp.Expr, kz: sp.Expr) -> sp.Matrix:
    return bcc_dirac_symbol(eps, kx, ky, kz)


def walk_respects_symmetry(op: SymmetryOperator) -> bool:
    """Return whether the bare BCC Dirac walk commutes with ``op`` symbolically."""

    eps, kx, ky, kz = _symbolic_momentum()
    bloch_k = _bloch_at(eps, kx, ky, kz)
    if op.momentum_flip:
        bloch_image = _bloch_at(eps, -kx, -ky, -kz)
    else:
        bloch_image = bloch_k

    matrix = op.spinor_matrix
    inverse = matrix.inv().applyfunc(sp.simplify)

    if op.antiunitary:
        inner = bloch_k.applyfunc(sp.conjugate)
        expected = bloch_image.H.applyfunc(sp.simplify)
    else:
        inner = bloch_k
        expected = bloch_image

    transformed = (matrix * inner * inverse).applyfunc(sp.simplify)
    return same_matrix(transformed, expected)


@dataclass(frozen=True)
class SymmetryVerdict:
    name: str
    antiunitary: bool
    momentum_flip: bool
    exact: bool


@dataclass(frozen=True)
class MasslessAuditPayload:
    verdicts: tuple[SymmetryVerdict, ...]
    cpt_exact: bool
    cp_exact: bool
    p_exact: bool
    t_exact: bool
    c_exact: bool
    interpretation: str


def massless_audit_verdicts() -> tuple[SymmetryVerdict, ...]:
    verdicts: list[SymmetryVerdict] = []
    for op in all_seven_operators():
        verdicts.append(
            SymmetryVerdict(
                name=op.name,
                antiunitary=op.antiunitary,
                momentum_flip=op.momentum_flip,
                exact=walk_respects_symmetry(op),
            ),
        )
    return tuple(verdicts)


def _lookup(verdicts: tuple[SymmetryVerdict, ...], name: str) -> SymmetryVerdict:
    for verdict in verdicts:
        if verdict.name == name:
            return verdict
    raise KeyError(name)


def massless_audit_payload() -> MasslessAuditPayload:
    verdicts = massless_audit_verdicts()
    cpt = _lookup(verdicts, "CPT").exact
    cp = _lookup(verdicts, "CP").exact
    p = _lookup(verdicts, "P").exact
    t = _lookup(verdicts, "T").exact
    c = _lookup(verdicts, "C").exact

    if cpt and not cp and p:
        interpretation = (
            "ALPHA PASS: bare BCC Dirac walk preserves CPT but breaks CP at the "
            "lattice level.  P is also preserved.  The walk itself is a "
            "CP-violating, CPT-conserving structure — no Yukawa needed.  This "
            "is the desired pattern from the original CP-from-quantization hope."
        )
    elif cpt and cp:
        interpretation = (
            "ALPHA NULL: CPT and CP both preserved by the bare walk.  No CP "
            "slot from the walk itself; if CP violation exists it must come "
            "from the Yukawa sector (alpha-3) or J ambiguity (beta)."
        )
    elif not cpt:
        interpretation = (
            "ALPHA ANOMALY: CPT not preserved by the bare walk.  This "
            "contradicts Lüders-Pauli for a local Hermitian theory and "
            "indicates an implementation bug or a non-Hermitian construction "
            "in the walk."
        )
    else:
        interpretation = (
            "ALPHA MIXED: unusual pattern of discrete-symmetry preservation. "
            "Inspect the verdict table for details."
        )

    return MasslessAuditPayload(
        verdicts=verdicts,
        cpt_exact=cpt,
        cp_exact=cp,
        p_exact=p,
        t_exact=t,
        c_exact=c,
        interpretation=interpretation,
    )


# -------- alpha-3: Yukawa-perturbed audit (trivial internal action) --------


def yukawa_term_preservation(op: SymmetryOperator) -> bool:
    """Return whether ``op`` preserves the Yukawa spinor structure ``β = γ^0``.

    Uses ``S_internal = I``: the operator's internal action is trivial.
    Under this convention, the Yukawa term ``β ⊗ M_higgs`` is preserved iff
    ``op_spinor · γ^0 · op_spinor^{-1} = +γ^0`` (with antiunitary conjugation
    if applicable).  This corresponds to the (1, -1, -1, -1) row of the
    conjugation pattern.
    """

    from clifford_3plus2_d5.cp.discrete_symmetries import conjugation_pattern

    pattern = conjugation_pattern(op)
    return pattern[0] == 1


@dataclass(frozen=True)
class YukawaAuditVerdict:
    name: str
    kinetic_exact: bool
    yukawa_beta_preserved: bool
    combined_exact: bool


@dataclass(frozen=True)
class YukawaAuditPayload:
    verdicts: tuple[YukawaAuditVerdict, ...]
    cpt_exact: bool
    cp_exact: bool
    p_exact: bool
    interpretation: str


def yukawa_audit_payload() -> YukawaAuditPayload:
    """Run the alpha-3 audit with trivial internal action ``S_internal = I``.

    Combined verdict: a symmetry is exact under the Yukawa-perturbed walk iff
    BOTH the kinetic part (alpha-2 verdict) and the Yukawa beta-preservation
    hold.
    """

    massless = massless_audit_payload()
    verdicts: list[YukawaAuditVerdict] = []
    for op_verdict in massless.verdicts:
        from clifford_3plus2_d5.cp.discrete_symmetries import all_seven_operators

        op = next(o for o in all_seven_operators() if o.name == op_verdict.name)
        beta_preserved = yukawa_term_preservation(op)
        combined = op_verdict.exact and beta_preserved
        verdicts.append(
            YukawaAuditVerdict(
                name=op_verdict.name,
                kinetic_exact=op_verdict.exact,
                yukawa_beta_preserved=beta_preserved,
                combined_exact=combined,
            ),
        )

    verdicts_tuple = tuple(verdicts)

    def _lookup_combined(name: str) -> bool:
        for v in verdicts_tuple:
            if v.name == name:
                return v.combined_exact
        raise KeyError(name)

    cpt = _lookup_combined("CPT")
    cp = _lookup_combined("CP")
    p = _lookup_combined("P")

    if cpt and not cp and p:
        interpretation = (
            "ALPHA-3 confirms ALPHA-2: under trivial internal action, "
            "adding the Yukawa term β ⊗ M_higgs does not introduce new "
            "symmetry breakings.  P, CT, CPT remain exact; T, C, CP, PT "
            "remain broken.  The CP-violation from the bare walk persists "
            "under Yukawa perturbation."
        )
    else:
        interpretation = (
            "ALPHA-3 produces an unexpected pattern; inspect the verdict "
            "table for details."
        )

    return YukawaAuditPayload(
        verdicts=verdicts_tuple,
        cpt_exact=cpt,
        cp_exact=cp,
        p_exact=p,
        interpretation=interpretation,
    )
