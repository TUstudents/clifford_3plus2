"""W1 — coefficient-Walsh decomposition of the BCC Weyl hop shell.

For the eight 2x2 hop matrices ``H_v`` over the cube directions ``v in {+-1}^3``,
compute the Walsh coefficients

    Hhat_S = (1/8) sum_v chi_S(v) H_v,   chi_S(v) = prod_{i in S} v_i,

and assign them to O_h irreps by Walsh degree |S|:

    A1g  |S|=0  Hhat_{}                          depth 0   parity +
    T1u  |S|=1  (Hhat_x, Hhat_y, Hhat_z)          depth 2   parity -
    T2g  |S|=2  (Hhat_xy, Hhat_yz, Hhat_zx)       depth 4   parity +
    A2u  |S|=3  Hhat_xyz                          depth 6   parity -

Parity is automatic: chi_S(-v) = (-1)^|S| chi_S(v).

CAVEAT (coefficient-Walsh vs covariant): this classifies only the directional (v)
dependence of the matrix-valued coefficients. It is NOT the full covariant O_h
irreducibility of the spinor-valued source (which would also conjugate,
H_v -> U_R H_v U_R^dagger). ``is_c3_covariant`` checks whether the source is
C3-equivariant about [111], so the coefficient classification lifts to the
covariant one on the [111] sector.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_hop_walsh.reuse import (
    apply_rotation_to_direction,
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    dirac_spinor_lift,
    opposite_helicity_hops,
)

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
WALSH_SUBSETS: tuple[str, ...] = ("", "x", "y", "z", "xy", "yz", "zx", "xyz")
IRREP_BY_SUBSET: dict[str, str] = {
    "": "A1g",
    "x": "T1u", "y": "T1u", "z": "T1u",
    "xy": "T2g", "yz": "T2g", "zx": "T2g",
    "xyz": "A2u",
}


def hops_for_helicity(helicity: str) -> tuple[sp.Matrix, ...]:
    """Return the eight hop matrices for the given helicity."""

    if helicity == "right":
        return bialynicki_birula_hops()
    if helicity == "left":
        return opposite_helicity_hops()
    raise ValueError("helicity must be 'right' or 'left'")


def _direction_tuples() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(d[0]), int(d[1]), int(d[2])) for d in bialynicki_birula_directions()
    )


def _chi(subset: str, v: tuple[int, int, int]) -> int:
    product = 1
    for axis in subset:
        product *= v[AXIS_INDEX[axis]]
    return product


def walsh_coefficient(subset: str, helicity: str = "right") -> sp.Matrix:
    """Return Hhat_S = (1/8) sum_v chi_S(v) H_v as a 2x2 matrix."""

    directions = _direction_tuples()
    hops = hops_for_helicity(helicity)
    total = sp.zeros(2, 2)
    for v, hop in zip(directions, hops, strict=True):
        total += _chi(subset, v) * hop
    return sp.simplify(total / 8)


def walsh_coefficients(helicity: str = "right") -> dict[str, sp.Matrix]:
    """Return all eight Walsh coefficients keyed by subset string."""

    return {subset: walsh_coefficient(subset, helicity) for subset in WALSH_SUBSETS}


# --- [111]-singlet projections (the family-relevant combinations) ---

def a1g_baseline(helicity: str = "right") -> sp.Matrix:
    """Return the A1g baseline Hhat_{} (the trivial even scalar)."""

    return walsh_coefficient("", helicity)


def t1u_singlet(helicity: str = "right") -> sp.Matrix:
    """Return the [111] T1u singlet (Hhat_x + Hhat_y + Hhat_z)/sqrt(3)."""

    coefficients = walsh_coefficients(helicity)
    return sp.simplify((coefficients["x"] + coefficients["y"] + coefficients["z"]) / sp.sqrt(3))


def t2g_singlet(helicity: str = "right") -> sp.Matrix:
    """Return the [111] T2g singlet (Hhat_xy + Hhat_yz + Hhat_zx)/sqrt(3)."""

    coefficients = walsh_coefficients(helicity)
    return sp.simplify((coefficients["xy"] + coefficients["yz"] + coefficients["zx"]) / sp.sqrt(3))


def a2u_component(helicity: str = "right") -> sp.Matrix:
    """Return the A2u component Hhat_xyz (the parity-odd pseudoscalar)."""

    return walsh_coefficient("xyz", helicity)


# --- two-tier "nonzero" (Refinement 3) ---

def is_zero_symbolic(matrix: sp.Matrix) -> bool:
    """Return true when every entry simplifies to 0 (the preferred criterion)."""

    return all(sp.simplify(entry) == 0 for entry in matrix)


def frobenius_norm_squared(matrix: sp.Matrix) -> sp.Expr:
    """Return Tr(M^dagger M), the squared Frobenius norm (numeric fallback)."""

    return sp.simplify((matrix.conjugate().T * matrix).trace())


# --- C3-about-[111] covariance (Refinement 2) ---

def is_c3_covariant(helicity: str = "right") -> bool:
    """Return whether the hop source is C3-equivariant about [111].

    Tests U_R H_v U_R^{-1} == H_{R v} for all directions (either rotation sense),
    with U_R the spinor lift and R the body-diagonal cyclic rotation. If true, the
    coefficient-Walsh classification lifts to the covariant O_h one.
    """

    directions = bialynicki_birula_directions()
    hops = hops_for_helicity(helicity)
    hop_by_direction = {
        tuple(int(c) for c in d): hop for d, hop in zip(directions, hops, strict=True)
    }
    u = dirac_spinor_lift()
    u_inv = u.inv()

    def _conjugation_matches(left: sp.Matrix, right: sp.Matrix) -> bool:
        rotated_index = {
            tuple(int(c) for c in d): tuple(
                int(c) for c in apply_rotation_to_direction(d)
            )
            for d in directions
        }
        for d in directions:
            v = tuple(int(c) for c in d)
            conjugated = sp.simplify(left * hop_by_direction[v] * right)
            target = hop_by_direction[rotated_index[v]]
            if not all(sp.simplify(entry) == 0 for entry in (conjugated - target)):
                return False
        return True

    # Accept either rotation sense (U H U^-1 = H_{Rv} or U^-1 H U = H_{Rv}).
    return _conjugation_matches(u, u_inv) or _conjugation_matches(u_inv, u)


@dataclass(frozen=True)
class HopWalshDecompositionPayload:
    """W1 payload: the Walsh coefficients and their O_h/parity assignment."""

    helicity: str
    coefficients: dict[str, sp.Matrix]
    irrep_by_subset: dict[str, str]
    a1g_baseline: sp.Matrix
    t1u_singlet: sp.Matrix
    t2g_singlet: sp.Matrix
    a2u_component: sp.Matrix
    covariance_check: bool


def hop_walsh_decomposition_payload(helicity: str = "right") -> HopWalshDecompositionPayload:
    """Return the W1 Walsh decomposition for one helicity."""

    return HopWalshDecompositionPayload(
        helicity=helicity,
        coefficients=walsh_coefficients(helicity),
        irrep_by_subset=dict(IRREP_BY_SUBSET),
        a1g_baseline=a1g_baseline(helicity),
        t1u_singlet=t1u_singlet(helicity),
        t2g_singlet=t2g_singlet(helicity),
        a2u_component=a2u_component(helicity),
        covariance_check=is_c3_covariant(helicity),
    )
