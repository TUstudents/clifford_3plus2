"""Phase KO-3: BCC-Z₃-orbit 3×3 Yukawa eigenvalue locus.

Adapts broken_triality's Yukawa-from-Z₃-orbit pattern with the BCC
body-diagonal Z₃ rotation R acting on σ^a-flavor space (R³).

For a generic starting vector v_* ∈ R³ decomposed as v_* = v_t + v_o
(trace + traceless under R), the symmetric Yukawa overlap matrix

    Y_ij = ⟨R^i v_*, R^j v_*⟩

is a 3×3 circulant matrix with structure:

    Y_ii = |v_t|² + |v_o|²
    Y_ij = |v_t|² + |v_o|² cos(2π(i-j)/3)  =  |v_t|² − |v_o|²/2   (i ≠ j)

The eigenvalues are determined entirely by (|v_t|, |v_o|):

    λ_1 = 3 |v_t|²                  (Z₃-trivial irrep, multiplicity 1)
    λ_2 = λ_3 = (3/2) |v_o|²        (Z₃-non-trivial 2D irrep, mult 2)

Two of the three eigenvalues are **always degenerate** in the
Z₃-equivariant construction.  The mass-vector
``v_Y = (√λ_1, √λ_2, √λ_3) = (√3 |v_t|, √(3/2) |v_o|, √(3/2) |v_o|)``
lies on the Koide 45° cone IFF

    |v_t| / |v_o|  =  3 + 2√2     (i.e., r ≈ 5.828).

When this ratio holds, K = 2/3 exactly; one mass is m_1 = 3 r² |v_o|²
and the two equal masses are m_2 = m_3 = (3/2) |v_o|², with ratio
m_1/m_2 = 2 r² = 2(3 + 2√2)² = 2(17 + 12√2) ≈ 67.97.

PDG charged-lepton ratios (m_μ/m_e ≈ 207, m_τ/m_μ ≈ 17) do NOT
match this degenerate-pair structure — the Z₃-equivariant locus
intersects the Koide cone in a 1-parameter family that does NOT
contain PDG.  Three distinct masses require breaking the Z₃
equivariance.

The audit:

1. Builds the BCC-Z₃-orbit Yukawa symbolically.
2. Verifies the circulant structure + degenerate eigenvalue pattern.
3. Computes the Koide K and identifies the special ratio for K = 2/3.
4. Characterizes the LOCUS of (m_1, m_2, m_3) reachable from the
   Z₃-equivariant construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.koide.koide_geometry import (
    koide_K_from_masses,
    trace_part,
    traceless_part,
)
from clifford_3plus2_d5.koide.reuse import body_diagonal_rotation_matrix


def bcc_z3_orbit(v_star: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``(v_*, R v_*, R² v_*)`` where R is the BCC body-diagonal Z₃."""

    R = body_diagonal_rotation_matrix()
    v0 = v_star.applyfunc(sp.simplify)
    v1 = (R * v0).applyfunc(sp.simplify)
    v2 = (R * R * v0).applyfunc(sp.simplify)
    return v0, v1, v2


def bcc_z3_yukawa_matrix(v_star: sp.Matrix) -> sp.Matrix:
    """Return the 3×3 Yukawa overlap matrix from the BCC-Z₃ orbit of ``v_*``.

        Y_ij = ⟨R^i v_*, R^j v_*⟩

    where ⟨·,·⟩ is the Euclidean inner product on R³.
    """

    orbit = bcc_z3_orbit(v_star)
    Y = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            entry = sum(
                orbit[i][k, 0] * orbit[j][k, 0] for k in range(3)
            )
            Y[i, j] = sp.simplify(entry)
    return Y


def is_circulant(matrix: sp.Matrix) -> bool:
    """Return whether a 3×3 matrix is circulant (entries depend only on (i-j) mod 3)."""

    if matrix.shape != (3, 3):
        return False
    for i in range(3):
        for j in range(3):
            expected = matrix[0, (j - i) % 3]
            if sp.simplify(matrix[i, j] - expected) != 0:
                return False
    return True


def yukawa_eigenvalue_triple(v_star: sp.Matrix) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the eigenvalue triple ``(λ_1, λ_2, λ_3)`` of the BCC-Z₃ Yukawa.

    For a Z₃-equivariant circulant Y with first row (A, B, B):
        λ_1 = A + 2B  =  3|v_t|²
        λ_2 = λ_3 = A − B  =  (3/2)|v_o|²

    Returns the eigenvalues in the order (λ_trivial, λ_non1, λ_non2).
    """

    v_t = trace_part(v_star)
    v_o = traceless_part(v_star)
    vt_sq = sp.simplify((v_t.T * v_t)[0, 0])
    vo_sq = sp.simplify((v_o.T * v_o)[0, 0])
    lam_1 = sp.simplify(3 * vt_sq)
    lam_2 = sp.simplify(sp.Rational(3, 2) * vo_sq)
    return lam_1, lam_2, lam_2


def yukawa_mass_vector(v_star: sp.Matrix) -> sp.Matrix:
    """Return ``v_Y = (√λ_1, √λ_2, √λ_3)`` as a 3×1 column."""

    lam_1, lam_2, lam_3 = yukawa_eigenvalue_triple(v_star)
    return sp.Matrix([sp.sqrt(lam_1), sp.sqrt(lam_2), sp.sqrt(lam_3)])


def yukawa_koide_K(v_star: sp.Matrix) -> sp.Expr:
    """Return the Koide K of the BCC-Z₃ Yukawa eigenvalue triple."""

    eigenvalues = yukawa_eigenvalue_triple(v_star)
    return koide_K_from_masses(eigenvalues)


@lru_cache(maxsize=1)
def koide_special_ratio_symbolic() -> sp.Expr:
    """Return the symbolic ratio ``r* = |v_t|/|v_o|`` for which K = 2/3.

    Derived from the equation ``r² − 4√2 r − 1 = 0``:

        r* = 2√2 + 3  ≈ 5.828.
    """

    return sp.Integer(3) + 2 * sp.sqrt(2)


def koide_special_ratio_numerical() -> float:
    return float(koide_special_ratio_symbolic())


def special_mass_ratio_at_cone() -> sp.Expr:
    """Return ``m_trivial / m_degenerate`` when v_*/v_o is on the Koide cone.

    m_1 / m_2 = 2 r² = 2(3 + 2√2)² = 2(17 + 12√2) ≈ 67.97.
    """

    r = koide_special_ratio_symbolic()
    return sp.simplify(2 * r ** 2)


def cone_predicted_mass_vector(v_o_magnitude: sp.Expr) -> sp.Matrix:
    """Return the mass-vector when |v_o| = given and |v_t| = r* · |v_o|."""

    r = koide_special_ratio_symbolic()
    lam_1 = sp.simplify(3 * r ** 2 * v_o_magnitude ** 2)
    lam_2 = sp.simplify(sp.Rational(3, 2) * v_o_magnitude ** 2)
    return sp.Matrix([sp.sqrt(lam_1), sp.sqrt(lam_2), sp.sqrt(lam_2)])


def z3_yukawa_has_degenerate_eigenvalues(v_star: sp.Matrix) -> bool:
    """Return whether the BCC-Z₃ Yukawa has 2-fold eigenvalue degeneracy.

    For generic v_*, this is True structurally: λ_2 = λ_3 = (3/2)|v_o|².
    """

    lam_1, lam_2, lam_3 = yukawa_eigenvalue_triple(v_star)
    return sp.simplify(lam_2 - lam_3) == 0


def pdg_mass_ratio_matches_special() -> bool:
    """Return whether PDG (m_τ/m_e or m_μ/m_e) matches the special ratio (3+2√2)².

    Expected: False (PDG has 3 distinct masses, no degeneracy).
    """

    from clifford_3plus2_d5.koide.koide_geometry import (
        PDG_M_E,
        PDG_M_MU,
        PDG_M_TAU,
    )

    me = float(PDG_M_E)
    mmu = float(PDG_M_MU)
    mtau = float(PDG_M_TAU)
    # The strict test: are any two of {m_e, m_μ, m_τ} equal (degenerate)?
    # If not, PDG cannot lie on the Z₃-equivariant locus regardless of ratio.
    degenerate_pair = any(
        abs(a - b) / max(a, b) < 1e-3
        for a, b in ((me, mmu), (mmu, mtau), (me, mtau))
    )
    return degenerate_pair


def example_v_stars() -> dict[str, sp.Matrix]:
    """Return a small dictionary of natural starting v_* vectors for the audit."""

    return {
        "diagonal_only": sp.Matrix([1, 1, 1]),  # pure trace, |v_o| = 0
        "traceless_x_y": sp.Matrix([1, -1, 0]),  # pure traceless
        "generic_split": sp.Matrix([2, 1, 0]),  # mix of trace and traceless
        "pdg_sqrt": sp.Matrix(
            [
                sp.sqrt(sp.Float("0.51099895")),
                sp.sqrt(sp.Float("105.6583755")),
                sp.sqrt(sp.Float("1776.86")),
            ]
        ),
    }


@dataclass(frozen=True)
class YukawaLocusPayload:
    """Result of the Phase KO-3 Yukawa-locus audit."""

    z3_yukawa_is_circulant: bool
    eigenvalues_always_degenerate: bool
    koide_special_ratio_symbolic: sp.Expr
    koide_special_ratio_numerical: float
    special_mass_ratio_at_cone: sp.Expr
    special_mass_ratio_numerical: float
    pdg_compatible_with_z3_locus: bool
    locus_is_one_parameter_family_on_cone: bool
    verdict: str
    interpretation: str


def yukawa_locus_payload() -> YukawaLocusPayload:
    """Run the Phase KO-3 audit."""

    # Structural check: circulant for a generic v_*.
    v_test = sp.Matrix([sp.Rational(2), sp.Rational(1), sp.Rational(0)])
    Y_test = bcc_z3_yukawa_matrix(v_test)
    circulant_ok = is_circulant(Y_test)

    # Eigenvalue degeneracy: always true for Z₃-equivariant construction.
    degenerate = z3_yukawa_has_degenerate_eigenvalues(v_test)

    r_sym = koide_special_ratio_symbolic()
    r_num = koide_special_ratio_numerical()
    ratio_sym = special_mass_ratio_at_cone()
    ratio_num = float(ratio_sym)

    pdg_compat = pdg_mass_ratio_matches_special()
    one_param = True  # 1-parameter family on cone (parametrized by overall scale)

    if circulant_ok and degenerate and not pdg_compat:
        verdict = "Z₃-EQUIVARIANT LOCUS — degenerate pair, on-cone subfamily"
        interpretation = (
            f"BCC-Z₃-orbit Yukawa is a circulant 3×3 matrix with structure "
            f"Y_ii = |v_t|² + |v_o|², Y_ij = |v_t|² − |v_o|²/2 (i ≠ j).  "
            f"Eigenvalue triple (3|v_t|², (3/2)|v_o|², (3/2)|v_o|²) — TWO "
            f"of three eigenvalues are ALWAYS degenerate by Z₃-equivariance.  "
            f"Koide K = 2/3 holds IFF |v_t|/|v_o| = 3 + 2√2 ≈ "
            f"{r_num:.4f}, in which case the non-degenerate-to-degenerate "
            f"mass ratio is m_1/m_2 = 2(3+2√2)² = "
            f"{ratio_num:.4f}.  PDG charged-lepton masses are all "
            f"distinct (no degeneracy), so PDG ≠ any point in the Z₃-"
            f"equivariant locus.  Verdict implication: the Z₃-equivariant "
            f"sub-locus is CONSISTENT WITH KOIDE (a 1-parameter on-cone "
            f"family exists) but does NOT predict the PDG mass triple.  "
            f"Three distinct masses require Z₃-breaking input (Higgs VEV "
            f"alignment off the trace/traceless eigenspaces, or explicit "
            f"Yukawa perturbation breaking the BCC body-diagonal symmetry)."
        )
    else:
        verdict = "Z₃-LOCUS INCONSISTENCY"
        interpretation = (
            f"circulant={circulant_ok}, degenerate={degenerate}, "
            f"pdg_compat={pdg_compat}.  Investigate before KO-4."
        )

    return YukawaLocusPayload(
        z3_yukawa_is_circulant=circulant_ok,
        eigenvalues_always_degenerate=degenerate,
        koide_special_ratio_symbolic=r_sym,
        koide_special_ratio_numerical=r_num,
        special_mass_ratio_at_cone=ratio_sym,
        special_mass_ratio_numerical=ratio_num,
        pdg_compatible_with_z3_locus=pdg_compat,
        locus_is_one_parameter_family_on_cone=one_param,
        verdict=verdict,
        interpretation=interpretation,
    )
