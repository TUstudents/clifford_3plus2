"""W5 — the S3/Schur obstruction (the core reason {0,2,6} is not derivable).

The residual three-port family space decomposes under S3 as

    3_ports = 1 (trivial) + 2 (standard).

By Schur's lemma, any S3-INVARIANT depth operator commutes with the representation
and therefore acts as a scalar on each isotypic block:

    [D, S3] = 0  =>  D = alpha P_1 + beta P_2,  spectrum {alpha, beta, beta}.

So an unbroken residual symmetry can give {0, d, d} but never the three DISTINCT
values {0, 2, 6}. Concretely the K3 Laplacian (the residual graph that supplies
epsilon) has spectrum {0, 3, 3} -> doubled {0, 6, 6} != {0, 2, 6}. The depth
operator diag(0,2,6) has three distinct eigenvalues, so it is necessarily an
S3-breaking spurion: its S3-invariant part is (8/3) I and its traceless doublet
spurion diag(-8/3, -2/3, 10/3) ~ (-4, -1, 5) is nonzero.

Therefore deriving {0,2,6} is equivalent to deriving the family-symmetry-breaking
spurion — the same closed-negative generation problem
(triality/broken_triality/exceptional). The depth hierarchy is not a separable
topological consequence of BCC; it is the generation problem in transfer-depth
language.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

DEPTHS = (0, 2, 6)


def s3_permutation_generators() -> tuple[sp.Matrix, sp.Matrix]:
    """Return the S3 generators on the 3 ports: a transposition and a 3-cycle."""

    transposition = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # (1 2)
    three_cycle = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # (1 2 3)
    return transposition, three_cycle


def s3_group_elements() -> tuple[sp.Matrix, ...]:
    """Return all 6 permutation matrices of S3 (closure of the generators)."""

    transposition, three_cycle = s3_permutation_generators()
    elements = {tuple(sp.eye(3)): sp.eye(3)}
    frontier = [sp.eye(3)]
    while frontier:
        current = frontier.pop()
        for generator in (transposition, three_cycle):
            product = generator * current
            key = tuple(product)
            if key not in elements:
                elements[key] = product
                frontier.append(product)
    return tuple(elements.values())


def trivial_projector() -> sp.Matrix:
    """P_1: projector onto the S3 trivial (1,1,1) port-singlet."""

    return sp.ones(3, 3) / 3


def standard_projector() -> sp.Matrix:
    """P_2: projector onto the 2-dim S3 standard (doublet) sector."""

    return sp.eye(3) - trivial_projector()


def commutant_dimension() -> int:
    """Return dim of the S3 commutant on the 3 ports (Schur: should be 2)."""

    transposition, three_cycle = s3_permutation_generators()
    entries = sp.symbols("m0:9")
    matrix = sp.Matrix(3, 3, entries)
    equations: list[sp.Expr] = []
    for generator in (transposition, three_cycle):
        equations.extend(matrix * generator - generator * matrix)
    coefficients, _ = sp.linear_eq_to_matrix(equations, entries)
    return len(entries) - coefficients.rank()


def schur_spectrum(alpha: sp.Expr, beta: sp.Expr) -> list[sp.Expr]:
    """Return the spectrum of D = alpha P_1 + beta P_2 (must be {alpha, beta, beta})."""

    operator = alpha * trivial_projector() + beta * standard_projector()
    return sorted(operator.eigenvals().items(), key=lambda kv: sp.default_sort_key(kv[0]))


def invariant_ops_have_at_most_two_distinct_eigenvalues() -> bool:
    """Schur: a generic S3-invariant operator has spectrum {alpha, beta, beta}."""

    alpha, beta = sp.symbols("alpha beta")
    eigenvalues = (alpha * trivial_projector() + beta * standard_projector()).eigenvals()
    # multiplicities: one eigenvalue x1, one eigenvalue x2 (multiplicity 2).
    multiplicities = sorted(eigenvalues.values())
    return multiplicities == [1, 2]


def k3_laplacian() -> sp.Matrix:
    """Return the residual K3 graph Laplacian L = 3I - J (degree 2, all-ones off-diag)."""

    return 3 * sp.eye(3) - sp.ones(3, 3)


def k3_laplacian_spectrum() -> dict[sp.Expr, int]:
    """Return the K3 Laplacian eigenvalues with multiplicities ({0:1, 3:2})."""

    return k3_laplacian().eigenvals()


def doubled_k3_spectrum() -> set[sp.Expr]:
    """Return the spectrum of 2 * L(K3) = {0, 6}."""

    return set((2 * k3_laplacian()).eigenvals())


def depth_operator() -> sp.Matrix:
    """Return D = diag(0, 2, 6) in the port basis."""

    return sp.diag(*DEPTHS)


def s3_invariant_part(operator: sp.Matrix) -> sp.Matrix:
    """Return the S3-average (1/6) sum_g g D g^{-1} (the invariant projection)."""

    elements = s3_group_elements()
    total = sp.zeros(3, 3)
    for g in elements:
        total += g * operator * g.inv()
    return sp.simplify(total / len(elements))


def depth_is_s3_invariant() -> bool:
    """Return whether diag(0,2,6) is S3-invariant (it is not)."""

    operator = depth_operator()
    return sp.simplify(operator - s3_invariant_part(operator)) == sp.zeros(3, 3)


def depth_breaking_spurion() -> sp.Matrix:
    """Return the S3-breaking (traceless) part of diag(0,2,6): diag(-8/3,-2/3,10/3)."""

    operator = depth_operator()
    return sp.simplify(operator - s3_invariant_part(operator))


def depth_distinct_eigenvalue_count() -> int:
    """Return the number of distinct eigenvalues of diag(0,2,6) (= 3)."""

    return len(depth_operator().eigenvals())


@dataclass(frozen=True)
class S3SchurObstructionPayload:
    """W5 payload: the representation-theory obstruction to deriving {0,2,6}."""

    final_verdict: str
    commutant_dimension: int
    invariant_spectrum_at_most_two_distinct: bool
    k3_laplacian_spectrum: dict[sp.Expr, int]
    doubled_k3_spectrum: set[sp.Expr]
    depth_distinct_eigenvalues: int
    depth_is_s3_invariant: bool
    breaking_spurion: sp.Matrix
    interpretation: str


def s3_schur_obstruction_payload() -> S3SchurObstructionPayload:
    """Return the W5 obstruction verdict."""

    dim = commutant_dimension()
    at_most_two = invariant_ops_have_at_most_two_distinct_eigenvalues()
    doubled = doubled_k3_spectrum()
    distinct = depth_distinct_eigenvalue_count()
    invariant = depth_is_s3_invariant()

    # The obstruction holds when: the commutant is 2-dim (Schur), invariant ops
    # have <=2 distinct eigenvalues, the doubled K3 spectrum is not {0,2,6}, and
    # {0,2,6} (3 distinct) is therefore not S3-invariant.
    obstruction = (
        dim == 2
        and at_most_two
        and doubled != set(DEPTHS)
        and distinct == 3
        and not invariant
    )
    final_verdict = (
        "DEPTH_HIERARCHY_REQUIRES_S3_BREAKING" if obstruction else "S3_OBSTRUCTION_ABSENT"
    )
    interpretation = (
        "Schur on the residual 3 = 1 + 2: any S3-invariant depth operator has "
        f"spectrum {{alpha, beta, beta}} (commutant dim {dim}, <=2 distinct "
        "eigenvalues). The unbroken K3 Laplacian gives {0, 3, 3} -> doubled "
        f"{sorted(doubled)} != {{0, 2, 6}}. But diag(0,2,6) has {distinct} distinct "
        "eigenvalues, so it is NOT S3-invariant: invariant part (8/3)I, breaking "
        "spurion diag(-8/3,-2/3,10/3) ~ (-4,-1,5) in the doublet sector. Hence "
        "deriving {0,2,6} == deriving the family-symmetry-breaking spurion, the "
        "same closed-negative generation problem. The depth hierarchy is the "
        "generation problem in transfer-depth language, not a separable BCC "
        "consequence."
    )

    return S3SchurObstructionPayload(
        final_verdict=final_verdict,
        commutant_dimension=dim,
        invariant_spectrum_at_most_two_distinct=at_most_two,
        k3_laplacian_spectrum=k3_laplacian_spectrum(),
        doubled_k3_spectrum=doubled,
        depth_distinct_eigenvalues=distinct,
        depth_is_s3_invariant=invariant,
        breaking_spurion=depth_breaking_spurion(),
        interpretation=interpretation,
    )
