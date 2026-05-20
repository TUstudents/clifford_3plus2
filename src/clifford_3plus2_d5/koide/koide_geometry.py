"""Phase KO-1: empirical Koide + 45° cone geometry.

Verifies the Koide formula

    K = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²  =  2/3

empirically from PDG charged-lepton masses, then establishes two
equivalent geometric forms:

1. **Angle form**: the vector v⃗ = (√m_e, √m_μ, √m_τ) makes an angle
   of 45° with n̂ = (1, 1, 1)/√3.  Equivalently, the projection
   ratio satisfies (v⃗ · n̂)² / |v⃗|² = 1/2.

2. **Equipartition form**: decomposing v⃗ = v_trace + v_traceless
   under the Z₃-trivial / Z₃-non-trivial irrep split (trace
   projector along n̂; complement P_traceless = I − P_trace),
   Koide ⇔ |v_trace|² = |v_traceless|².

Both forms are checked numerically against PDG and shown to agree
with K = 2/3 to ~10⁻⁵ precision.

The (1, 1, 1)/√3 direction is exactly the BCC body-diagonal axis
audited in topology/SC-2; the geometric setup here is the bridge to
Phase KO-2's structural analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


# PDG 2024 charged-lepton masses (in MeV).  Sources:
#   m_e  = 0.51099895000(15) MeV  — PDG 2024 (CODATA).
#   m_μ  = 105.6583755(23) MeV   — PDG 2024.
#   m_τ  = 1776.86(12) MeV       — PDG 2024.
PDG_M_E: sp.Expr = sp.Float("0.51099895")
PDG_M_MU: sp.Expr = sp.Float("105.6583755")
PDG_M_TAU: sp.Expr = sp.Float("1776.86")

KOIDE_K_TARGET: sp.Expr = sp.Rational(2, 3)


def pdg_charged_lepton_masses() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``(m_e, m_μ, m_τ)`` in MeV from PDG 2024."""

    return (PDG_M_E, PDG_M_MU, PDG_M_TAU)


def pdg_sqrt_mass_vector() -> sp.Matrix:
    """Return v⃗ = ``(√m_e, √m_μ, √m_τ)`` as a 3×1 SymPy column."""

    me, mmu, mtau = pdg_charged_lepton_masses()
    return sp.Matrix([sp.sqrt(me), sp.sqrt(mmu), sp.sqrt(mtau)])


def koide_K_from_masses(
    masses: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> sp.Expr:
    """Return ``K = Σm_i / (Σ√m_i)²`` for a generic mass triple."""

    m1, m2, m3 = masses
    sqrt_sum = sp.sqrt(m1) + sp.sqrt(m2) + sp.sqrt(m3)
    return sp.simplify((m1 + m2 + m3) / (sqrt_sum ** 2))


def pdg_koide_K() -> sp.Expr:
    """Return K computed from PDG charged-lepton masses."""

    return koide_K_from_masses(pdg_charged_lepton_masses())


def pdg_koide_deviation_from_two_thirds() -> sp.Expr:
    """Return ``K_PDG − 2/3``.  Expected magnitude ~10⁻⁵."""

    return sp.simplify(pdg_koide_K() - KOIDE_K_TARGET)


def trace_direction() -> sp.Matrix:
    """Return n̂ = (1, 1, 1)/√3 as a 3×1 SymPy column."""

    return sp.Matrix([1, 1, 1]) / sp.sqrt(3)


def angle_form_projection_ratio(vector: sp.Matrix) -> sp.Expr:
    """Return ``(v · n̂)² / |v|²``.  Koide ⇔ this = 1/2."""

    n_hat = trace_direction()
    dot = (vector.T * n_hat)[0, 0]
    norm_sq = (vector.T * vector)[0, 0]
    return sp.simplify(dot ** 2 / norm_sq)


def trace_projector() -> sp.Matrix:
    """Return P_trace = n̂ n̂^T projecting onto the (1,1,1)/√3 direction."""

    n_hat = trace_direction()
    return n_hat * n_hat.T


def traceless_projector() -> sp.Matrix:
    """Return P_traceless = I_3 − P_trace, the Z₃-non-trivial 2D irrep projector."""

    return sp.eye(3) - trace_projector()


def trace_part(vector: sp.Matrix) -> sp.Matrix:
    """Return P_trace · v (the component of v along (1,1,1)/√3)."""

    return trace_projector() * vector


def traceless_part(vector: sp.Matrix) -> sp.Matrix:
    """Return P_traceless · v (the component orthogonal to (1,1,1)/√3)."""

    return traceless_projector() * vector


def equipartition_ratio(vector: sp.Matrix) -> sp.Expr:
    """Return ``|v_trace|² / |v_traceless|²``.  Koide ⇔ this = 1."""

    v_t = trace_part(vector)
    v_o = traceless_part(vector)
    num = sp.simplify((v_t.T * v_t)[0, 0])
    den = sp.simplify((v_o.T * v_o)[0, 0])
    return sp.simplify(num / den)


def angle_form_holds_for_pdg(tolerance: float = 1e-4) -> bool:
    """Return whether |projection_ratio - 1/2| < tolerance for PDG values."""

    ratio = float(angle_form_projection_ratio(pdg_sqrt_mass_vector()))
    return abs(ratio - 0.5) < tolerance


def equipartition_holds_for_pdg(tolerance: float = 1e-4) -> bool:
    """Return whether |equipartition_ratio - 1| < tolerance for PDG values."""

    ratio = float(equipartition_ratio(pdg_sqrt_mass_vector()))
    return abs(ratio - 1.0) < tolerance


def koide_holds_empirically(tolerance: float = 1e-4) -> bool:
    """Return whether |K_PDG - 2/3| < tolerance."""

    deviation = float(pdg_koide_deviation_from_two_thirds())
    return abs(deviation) < tolerance


def cone_parametrization_radius(vector: sp.Matrix) -> sp.Expr:
    """Return ``|v|`` — the radial parameter of the cone parametrization."""

    return sp.sqrt((vector.T * vector)[0, 0])


def cone_parametrization_azimuth_unit(vector: sp.Matrix) -> sp.Matrix:
    """Return the unit vector ê(φ) in the orthogonal plane (the azimuthal direction).

    The 45° cone is parametrized as v = r·(cos(45°)·n̂ + sin(45°)·ê(φ)).
    Given v, ê is the normalized traceless part.
    """

    v_o = traceless_part(vector)
    norm = sp.sqrt((v_o.T * v_o)[0, 0])
    return sp.simplify(v_o / norm)


@dataclass(frozen=True)
class KoideGeometryPayload:
    """Result of the Phase KO-1 empirical + geometric audit."""

    pdg_masses: tuple[sp.Expr, sp.Expr, sp.Expr]
    pdg_K: sp.Expr
    K_deviation_from_2_3: sp.Expr
    pdg_angle_form_projection_ratio: sp.Expr
    pdg_equipartition_ratio: sp.Expr
    koide_holds_empirically: bool
    angle_form_holds: bool
    equipartition_form_holds: bool
    verdict: str
    interpretation: str


def koide_geometry_payload() -> KoideGeometryPayload:
    """Run the Phase KO-1 audit."""

    v_pdg = pdg_sqrt_mass_vector()
    K_pdg = pdg_koide_K()
    deviation = pdg_koide_deviation_from_two_thirds()
    angle_ratio = angle_form_projection_ratio(v_pdg)
    equip_ratio = equipartition_ratio(v_pdg)
    K_ok = koide_holds_empirically()
    angle_ok = angle_form_holds_for_pdg()
    equip_ok = equipartition_holds_for_pdg()

    if K_ok and angle_ok and equip_ok:
        verdict = "KOIDE GEOMETRY VERIFIED — three equivalent forms agree on PDG"
        interpretation = (
            f"PDG K = {float(K_pdg):.6f} vs 2/3 = {float(KOIDE_K_TARGET):.6f}; "
            f"deviation {float(deviation):.2e}.  The angle form "
            f"((v·n̂)²/|v|² = 1/2) gives ratio = {float(angle_ratio):.6f}, "
            f"matching the cone condition.  The equipartition form "
            f"(|v_trace|² = |v_traceless|²) gives ratio = "
            f"{float(equip_ratio):.6f}, matching unity.  All three forms "
            f"of Koide agree with PDG charged-lepton masses to ~10⁻⁵.  "
            f"The geometric setup is consistent; Phase KO-2 maps the "
            f"(1,1,1) direction to the BCC body-diagonal axis."
        )
    else:
        verdict = "KOIDE GEOMETRY INCONSISTENCY"
        interpretation = (
            f"K = {float(K_pdg):.6f}; angle = {float(angle_ratio):.6f}; "
            f"equipartition = {float(equip_ratio):.6f}.  Investigate "
            f"PDG values or arithmetic before proceeding."
        )

    return KoideGeometryPayload(
        pdg_masses=pdg_charged_lepton_masses(),
        pdg_K=K_pdg,
        K_deviation_from_2_3=deviation,
        pdg_angle_form_projection_ratio=angle_ratio,
        pdg_equipartition_ratio=equip_ratio,
        koide_holds_empirically=K_ok,
        angle_form_holds=angle_ok,
        equipartition_form_holds=equip_ok,
        verdict=verdict,
        interpretation=interpretation,
    )
