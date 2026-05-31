"""W4 — full covariant O-decomposition of the matrix-valued hop source.

The escape hatch (raised in review): the coefficient-Walsh transform (W1/W2)
classifies only the directional (v) dependence of the matrix-valued coefficients,
ignoring that under a cubic rotation the Pauli matrices ALSO transform. The full
covariant object is

    (rho(g) H)_v = U_g H_{R_g^{-1} v} U_g^dagger,   g in O (octahedral rotations),

with U_g the spin-1/2 lift of the rotation R_g (paired consistently, R_g = Ad(U_g)).
This module decomposes the source into the genuine covariant O-irreps
{A1, A2, E, T1, T2} via the character projectors

    P_Gamma = (dim_Gamma / |O|) sum_g chi_Gamma(g) rho(g),

and reports the squared Frobenius norm in each irrep.

Mechanism mapping (O has no inversion; parity g/u is recovered from helicity):
ALLOWED family irreps A1g, T1u, A2u -> A1, T1, A2; FORBIDDEN even quadrupoles
Eg, T2g -> E, T2.

Result for the genuine Bialynicki-Birula source: A1=1, A2=1/3, E=2/3, T1=T2=0.
The coefficient-Walsh "T2g" reassembles to covariant T2=0, but a forbidden E
quadrupole is present and the T1 vector is absent — so Claim A is killed covariantly
too. The escape hatch is closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.depth_hop_walsh.hop_walsh_decomposition import hops_for_helicity
from clifford_3plus2_d5.depth_hop_walsh.reuse import (
    bialynicki_birula_directions,
    pauli_matrices,
)

IRREP_NAMES: tuple[str, ...] = ("A1", "A2", "E", "T1", "T2")
IRREP_DIM: dict[str, int] = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}
# O character table by conjugacy class {E, 8C3, 3C2(=C4^2), 6C4, 6C2'}.
O_CHARACTER: dict[str, dict[str, int]] = {
    "A1": {"E": 1, "C3": 1, "C2": 1, "C4": 1, "C2p": 1},
    "A2": {"E": 1, "C3": 1, "C2": 1, "C4": -1, "C2p": -1},
    "E": {"E": 2, "C3": -1, "C2": 2, "C4": 0, "C2p": 0},
    "T1": {"E": 3, "C3": 0, "C2": -1, "C4": 1, "C2p": -1},
    "T2": {"E": 3, "C3": 0, "C2": -1, "C4": -1, "C2p": 1},
}
ALLOWED_IRREPS = ("A1", "A2", "T1")  # images of A1g, A2u, T1u
FORBIDDEN_IRREPS = ("E", "T2")  # images of Eg, T2g (even quadrupoles)


def _spin(axis: tuple[int, int, int], theta: sp.Expr) -> sp.Matrix:
    """Return the SU(2) lift exp(-i theta (n.sigma)/2) for a unit-axis rotation."""

    sx, sy, sz = pauli_matrices()
    nx, ny, nz = axis
    norm = sp.sqrt(nx * nx + ny * ny + nz * nz)
    n_dot_sigma = (nx * sx + ny * sy + nz * sz) / norm
    return sp.simplify(
        sp.cos(theta / 2) * sp.eye(2) - sp.I * sp.sin(theta / 2) * n_dot_sigma
    )


def _adjoint_rotation(u: sp.Matrix) -> sp.Matrix:
    """Return the 3x3 rotation R with U sigma_i U^dagger = sum_j R_ji sigma_j."""

    sx, sy, sz = pauli_matrices()
    u_dag = u.conjugate().T
    columns = []
    for sigma in (sx, sy, sz):
        rotated = u * sigma * u_dag
        columns.append(
            (
                sp.simplify(sp.trace(rotated * sx) / 2),
                sp.simplify(sp.trace(rotated * sy) / 2),
                sp.simplify(sp.trace(rotated * sz) / 2),
            )
        )
    return sp.Matrix(
        [
            [columns[0][0], columns[1][0], columns[2][0]],
            [columns[0][1], columns[1][1], columns[2][1]],
            [columns[0][2], columns[1][2], columns[2][2]],
        ]
    )


@lru_cache(maxsize=1)
def octahedral_group() -> tuple[tuple[sp.Matrix, sp.Matrix], ...]:
    """Return the 24 octahedral rotations as (R, U) pairs with R = Ad(U).

    Built by closure from C3[111] and C4[z]; each U is paired with the rotation it
    actually implements, so R-products and U-products share the same order.
    """

    u1 = _spin((1, 1, 1), sp.Rational(2, 3) * sp.pi)
    u2 = _spin((0, 0, 1), sp.pi / 2)
    generators = ((_adjoint_rotation(u1), u1), (_adjoint_rotation(u2), u2))

    def key(matrix: sp.Matrix) -> tuple[int, ...]:
        return tuple(int(entry) for entry in matrix)

    elements: dict[tuple[int, ...], tuple[sp.Matrix, sp.Matrix]] = {
        key(sp.eye(3)): (sp.eye(3), sp.eye(2))
    }
    frontier = [(sp.eye(3), sp.eye(2))]
    while frontier:
        rotation, lift = frontier.pop()
        for gen_rotation, gen_lift in generators:
            new_rotation = gen_rotation * rotation
            new_lift = sp.simplify(gen_lift * lift)
            element_key = key(new_rotation)
            if element_key not in elements:
                elements[element_key] = (new_rotation, new_lift)
                frontier.append((new_rotation, new_lift))
    return tuple(elements.values())


def _rotation_order(rotation: sp.Matrix) -> int:
    power = rotation
    for order in range(1, 7):
        if power == sp.eye(3):
            return order
        power = rotation * power
    return -1


def _conjugacy_class(rotation: sp.Matrix) -> str:
    trace = int(rotation.trace())
    if trace == 3:
        return "E"
    order = _rotation_order(rotation)
    if order == 3:
        return "C3"
    if order == 4:
        return "C4"
    # order 2: face axis (diagonal) is 3C2; edge-diagonal axis is 6C2'.
    is_diagonal = all(
        rotation[i, j] == 0 for i in range(3) for j in range(3) if i != j
    )
    return "C2" if is_diagonal else "C2p"


def _directions() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(d[0]), int(d[1]), int(d[2])) for d in bialynicki_birula_directions()
    )


def _source(helicity: str) -> dict[tuple[int, int, int], sp.Matrix]:
    return dict(zip(_directions(), hops_for_helicity(helicity), strict=True))


def _apply(
    element: tuple[sp.Matrix, sp.Matrix],
    source: dict[tuple[int, int, int], sp.Matrix],
) -> dict[tuple[int, int, int], sp.Matrix]:
    rotation, lift = element
    inverse = rotation.inv()
    lift_dag = lift.conjugate().T
    out: dict[tuple[int, int, int], sp.Matrix] = {}
    for v in _directions():
        preimage = inverse * sp.Matrix(v)
        key = (int(preimage[0]), int(preimage[1]), int(preimage[2]))
        out[v] = lift * source[key] * lift_dag
    return out


def project_irrep(
    irrep: str, source: dict[tuple[int, int, int], sp.Matrix]
) -> dict[tuple[int, int, int], sp.Matrix]:
    """Return P_irrep applied to the source (a matrix-valued cube function)."""

    accumulator = {v: sp.zeros(2, 2) for v in _directions()}
    for rotation, lift in octahedral_group():
        character = O_CHARACTER[irrep][_conjugacy_class(rotation)]
        if character == 0:
            continue
        transformed = _apply((rotation, lift), source)
        for v in _directions():
            accumulator[v] = accumulator[v] + character * transformed[v]
    factor = sp.Rational(IRREP_DIM[irrep], len(octahedral_group()))
    return {v: sp.simplify(factor * accumulator[v]) for v in _directions()}


def _norm_squared(function: dict[tuple[int, int, int], sp.Matrix]) -> sp.Expr:
    return sp.simplify(
        sum((function[v].conjugate().T * function[v]).trace() for v in _directions())
    )


@lru_cache(maxsize=2)
def covariant_irrep_norms(helicity: str = "right") -> tuple[tuple[str, sp.Expr], ...]:
    """Return ((irrep, squared-Frobenius-norm), ...) for the source decomposition."""

    source = _source(helicity)
    return tuple((irrep, _norm_squared(project_irrep(irrep, source))) for irrep in IRREP_NAMES)


def reconstruction_is_exact(helicity: str = "right") -> bool:
    """Validation: the irrep norms sum to the source norm (projectors complete)."""

    source = _source(helicity)
    total = _norm_squared(source)
    summed = sum((norm for _, norm in covariant_irrep_norms(helicity)), sp.Integer(0))
    return sp.simplify(total - summed) == 0


def group_is_octahedral() -> bool:
    """Validation: 24 elements and R = Ad(U) for every element."""

    elements = octahedral_group()
    if len(elements) != 24:
        return False
    return all(sp.simplify(_adjoint_rotation(u) - r) == sp.zeros(3, 3) for r, u in elements)


def covariant_support_verdict(norms: dict[str, sp.Expr]) -> str:
    """Return the covariant verdict. Pure decision (KILL-testable).

    The clean tower needs A1, A2, T1 present and E, T2 absent. A present forbidden
    quadrupole (E or T2) or a missing allowed mode kills Claim A covariantly.
    """

    def present(name: str) -> bool:
        return norms[name] != 0

    if present("E") or present("T2"):
        return "COVARIANT_KILL_FORBIDDEN_QUADRUPOLE"
    if not present("T1"):
        return "COVARIANT_KILL_MISSING_VECTOR"
    if not present("A2"):
        return "COVARIANT_KILL_MISSING_PSEUDOSCALAR"
    if not present("A1"):
        return "COVARIANT_KILL_MISSING_BASELINE"
    return "COVARIANT_SUPPORT_PASS"


@dataclass(frozen=True)
class CovariantODecompositionPayload:
    """W4 payload: covariant O-irrep norms per helicity + escape-hatch resolution."""

    final_verdict: str
    right_norms: dict[str, sp.Expr]
    left_norms: dict[str, sp.Expr]
    right_verdict: str
    left_verdict: str
    reconstruction_exact: bool
    group_valid: bool
    escape_hatch_closed: bool
    interpretation: str


def covariant_o_decomposition_payload() -> CovariantODecompositionPayload:
    """Return the W4 covariant decomposition (the escape-hatch resolution)."""

    right = {name: norm for name, norm in covariant_irrep_norms("right")}
    left = {name: norm for name, norm in covariant_irrep_norms("left")}
    right_verdict = covariant_support_verdict(right)
    left_verdict = covariant_support_verdict(left)
    final = right_verdict if right_verdict == left_verdict else "COVARIANT_HELICITY_SPLIT"
    escape_hatch_closed = final != "COVARIANT_SUPPORT_PASS"

    interpretation = (
        f"Covariant O-decomposition (right): {{{', '.join(f'{k}={v}' for k, v in right.items())}}}. "
        "The coefficient-Walsh T2g reassembles to covariant T2=0 (the escape hatch's "
        "mechanism is real), BUT a forbidden E quadrupole is present and the T1 "
        "vector is absent — so Claim A is killed covariantly. The escape hatch is "
        f"{'closed' if escape_hatch_closed else 'open'}: the source is A1 (+) A2 (+) E, "
        "not the clean A1+T1+A2 tower; and 'depth = 2*Walsh-degree' is not even a "
        "covariant label."
    )

    return CovariantODecompositionPayload(
        final_verdict=final,
        right_norms=right,
        left_norms=left,
        right_verdict=right_verdict,
        left_verdict=left_verdict,
        reconstruction_exact=reconstruction_is_exact("right") and reconstruction_is_exact("left"),
        group_valid=group_is_octahedral(),
        escape_hatch_closed=escape_hatch_closed,
        interpretation=interpretation,
    )
