"""Phase KO-2: BCC body-diagonal Z₃ acting on the σ^a-indexed flavor 3-vector.

The BCC body-diagonal Z₃ rotation R = [[0,1,0],[0,0,1],[1,0,0]]
cyclically permutes the three spatial axes (x, y, z).  Per cp/'s
H^(1) T_{2g} structure, the same R simultaneously permutes the three
Pauli-axis labels σ^x, σ^y, σ^z (cf. ``H^(1)_chir = σ^x k_y k_z −
σ^y k_x k_z + σ^z k_x k_y``, which transforms as the T_{2g} irrep of
O_h with σ^a co-cycling with k_b k_c).

We identify (m_e, m_μ, m_τ) ↔ (m_{σx}, m_{σy}, m_{σz}) — pinning a
convention.  Under this identification:

- The (1, 1, 1) direction in flavor-space corresponds to the
  Z₃-trivial irrep of the BCC body-diagonal action.
- The orthogonal 2D plane corresponds to the Z₃-non-trivial irrep
  (complex eigenvalues ω, ω̄ where ω = exp(2πi/3)).
- Koide's 45° cone condition becomes: the σ^a-indexed mass-vector
  has equal magnitude in the Z₃-trivial and Z₃-non-trivial sectors.

This module verifies the Z₃-irrep decomposition is consistent with
KO-1's equipartition form for the PDG mass-vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.koide.koide_geometry import (
    pdg_sqrt_mass_vector,
    trace_projector,
    traceless_projector,
)
from clifford_3plus2_d5.koide.reuse import body_diagonal_rotation_matrix


def z3_rotation_3d() -> sp.Matrix:
    """Return the BCC body-diagonal Z₃ rotation R as a 3×3 matrix.

    Cyclic permutation (x, y, z) → (y, z, x); equivalently
    ``R = [[0,1,0],[0,0,1],[1,0,0]]``.
    """

    return body_diagonal_rotation_matrix()


def z3_rotation_fixes_diagonal_direction() -> bool:
    """Return whether R · (1,1,1) = (1,1,1) (the trace direction is the +1 eigenvector)."""

    R = z3_rotation_3d()
    v = sp.Matrix([1, 1, 1])
    return (R * v - v).applyfunc(sp.simplify) == sp.zeros(3, 1)


def z3_rotation_order_three() -> bool:
    """Return whether R³ = I."""

    R = z3_rotation_3d()
    return (R**3 - sp.eye(3)).applyfunc(sp.simplify) == sp.zeros(3, 3)


def z3_rotation_orthogonal_det_one() -> bool:
    """Return whether R^T R = I and det R = +1."""

    R = z3_rotation_3d()
    orth = (R.T * R - sp.eye(3)).applyfunc(sp.simplify) == sp.zeros(3, 3)
    det_ok = R.det() == 1
    return orth and det_ok


def trace_projector_commutes_with_z3() -> bool:
    """Return whether P_trace commutes with R (i.e., P_trace is Z₃-invariant)."""

    R = z3_rotation_3d()
    P_t = trace_projector()
    commutator = (R * P_t - P_t * R).applyfunc(sp.simplify)
    return commutator == sp.zeros(3, 3)


def traceless_projector_commutes_with_z3() -> bool:
    """Return whether P_traceless commutes with R."""

    R = z3_rotation_3d()
    P_o = traceless_projector()
    commutator = (R * P_o - P_o * R).applyfunc(sp.simplify)
    return commutator == sp.zeros(3, 3)


def sigma_axis_to_generation_label(axis: str) -> str:
    """Return the generation label identified with each σ^a axis.

    The pinned convention:
        σ^x  ↔  e   (electron)
        σ^y  ↔  μ   (muon)
        σ^z  ↔  τ   (tau)

    Other cyclic conventions (x↔μ, x↔τ) differ by a Z₃ phase but
    leave the Koide structure invariant (Koide is K under any
    permutation of the three masses).
    """

    table = {"x": "e", "y": "μ", "z": "τ"}
    if axis not in table:
        raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}")
    return table[axis]


def generation_to_sigma_axis(generation: str) -> str:
    """Inverse of ``sigma_axis_to_generation_label``."""

    inverse = {"e": "x", "μ": "y", "τ": "z"}
    if generation not in inverse:
        raise ValueError(
            f"generation must be one of 'e', 'μ', 'τ'; got {generation!r}"
        )
    return inverse[generation]


def koide_condition_on_pdg_vector(tolerance: float = 1e-4) -> bool:
    """Return whether the PDG σ^a-indexed mass-vector satisfies Koide.

    Per the σ^a ↔ generation identification, the PDG vector is
    ``v = (v_e, v_μ, v_τ) = (v_{σx}, v_{σy}, v_{σz})``.  Koide's
    equipartition form (|P_trace v|² = |P_traceless v|²) must hold.
    """

    from clifford_3plus2_d5.koide.koide_geometry import equipartition_ratio

    ratio = float(equipartition_ratio(pdg_sqrt_mass_vector()))
    return abs(ratio - 1.0) < tolerance


@dataclass(frozen=True)
class BCCZ3OnFlavorPayload:
    """Result of the Phase KO-2 audit."""

    z3_fixes_diagonal: bool
    z3_is_order_three: bool
    z3_orthogonal_det_one: bool
    trace_projector_z3_invariant: bool
    traceless_projector_z3_invariant: bool
    sigma_to_generation_convention: dict[str, str]
    pdg_koide_condition_holds: bool
    verdict: str
    interpretation: str


def bcc_z3_on_flavor_payload() -> BCCZ3OnFlavorPayload:
    """Run the Phase KO-2 audit."""

    fixes = z3_rotation_fixes_diagonal_direction()
    order3 = z3_rotation_order_three()
    orth = z3_rotation_orthogonal_det_one()
    tp_inv = trace_projector_commutes_with_z3()
    op_inv = traceless_projector_commutes_with_z3()
    convention = {
        sigma_axis_to_generation_label("x"): "x",
        sigma_axis_to_generation_label("y"): "y",
        sigma_axis_to_generation_label("z"): "z",
    }
    koide_ok = koide_condition_on_pdg_vector()

    all_ok = (
        fixes and order3 and orth and tp_inv and op_inv and koide_ok
    )

    if all_ok:
        verdict = "BCC Z₃ STRUCTURE CONSISTENT WITH KOIDE EQUIPARTITION"
        interpretation = (
            "BCC body-diagonal Z₃ rotation R = [[0,1,0],[0,0,1],[1,0,0]] "
            "fixes the diagonal direction (1,1,1)/√3 and cyclically "
            "permutes the orthogonal 2D plane.  Trace and traceless "
            "projectors commute with R — they are the Z₃-irrep "
            "projectors (trivial + 2D non-trivial).  Under the "
            "σ^x↔e, σ^y↔μ, σ^z↔τ identification, the PDG mass-vector "
            "satisfies Koide's equipartition form |v_trace|² = "
            "|v_traceless|² to ~10⁻⁵.  Z₃-trivial sector lies along "
            "the BCC body-diagonal; Z₃-non-trivial 2D sector is the "
            "transverse plane.  Phase KO-3 constructs the program's "
            "natural Yukawa locus from this structure."
        )
    else:
        verdict = "BCC Z₃ STRUCTURE INCONSISTENCY"
        interpretation = (
            f"fixes={fixes}, order3={order3}, orth={orth}, "
            f"trace_inv={tp_inv}, traceless_inv={op_inv}, "
            f"koide={koide_ok}.  Investigate before KO-3."
        )

    return BCCZ3OnFlavorPayload(
        z3_fixes_diagonal=fixes,
        z3_is_order_three=order3,
        z3_orthogonal_det_one=orth,
        trace_projector_z3_invariant=tp_inv,
        traceless_projector_z3_invariant=op_inv,
        sigma_to_generation_convention=convention,
        pdg_koide_condition_holds=koide_ok,
        verdict=verdict,
        interpretation=interpretation,
    )
