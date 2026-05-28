"""V16 primitive ergodicity theorem.

V15 proves that the CKM phase is equivalent to the flat even/odd ratio
``r = 1`` in the isotropic quark coin.  V16 proves which symmetry can and
cannot force that flatness.

Parity-preserving shell symmetry fixes the even channel and acts transitively
on the five odd channels.  Its invariant amplitude space is two-dimensional:

    span{ |even>, |odd_1> + ... + |odd_5> }.

So it can make the five odd amplitudes equal, but it cannot determine their
relative amplitude against the even channel.  Full transitive symmetry on all
six primitive channels collapses the invariant space to the flat vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    isotropic_quark_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)

SHELL_DIMENSION = 6
ODD_START = 1


def primitive_even_vector() -> sp.Matrix:
    """Return the primitive even channel basis vector."""

    return sp.Matrix([1, 0, 0, 0, 0, 0])


def primitive_odd_sum_vector() -> sp.Matrix:
    """Return the unnormalized odd-shell invariant vector."""

    return sp.Matrix([0, 1, 1, 1, 1, 1])


def full_flat_vector() -> sp.Matrix:
    """Return the unnormalized six-channel flat vector."""

    return sp.Matrix([1, 1, 1, 1, 1, 1])


def swap_matrix(left: int, right: int, *, dimension: int = SHELL_DIMENSION) -> sp.Matrix:
    """Return the permutation matrix swapping two primitive channels."""

    matrix = sp.eye(dimension)
    matrix[left, left] = 0
    matrix[right, right] = 0
    matrix[left, right] = 1
    matrix[right, left] = 1
    return matrix


def parity_preserving_generators() -> tuple[sp.Matrix, ...]:
    """Return adjacent transpositions generating the odd-shell ``S_5``."""

    return tuple(
        swap_matrix(index, index + 1)
        for index in range(ODD_START, SHELL_DIMENSION - 1)
    )


def full_transitive_generators() -> tuple[sp.Matrix, ...]:
    """Return adjacent transpositions generating transitive ``S_6``."""

    return tuple(
        swap_matrix(index, index + 1)
        for index in range(SHELL_DIMENSION - 1)
    )


def invariant_subspace_basis(generators: tuple[sp.Matrix, ...]) -> tuple[sp.Matrix, ...]:
    """Return exact invariant vectors for a permutation-generator set."""

    identity = sp.eye(SHELL_DIMENSION)
    constraints = sp.Matrix.vstack(*(generator - identity for generator in generators))
    return tuple(constraints.nullspace())


def parity_preserving_invariant_basis() -> tuple[sp.Matrix, ...]:
    """Return the exact invariant basis for the parity-preserving shell symmetry."""

    return invariant_subspace_basis(parity_preserving_generators())


def full_transitive_invariant_basis() -> tuple[sp.Matrix, ...]:
    """Return the exact invariant basis for full six-channel transitivity."""

    return invariant_subspace_basis(full_transitive_generators())


def primitive_shell_amplitude(ratio: sp.Expr) -> sp.Matrix:
    """Return the normalized shell amplitude matching the V15 ratio convention."""

    r = sp.sympify(ratio)
    return sp.simplify((primitive_even_vector() + r * primitive_odd_sum_vector()) / sp.sqrt(1 + 5 * r**2))


def vector_is_invariant(vector: sp.Matrix, generators: tuple[sp.Matrix, ...]) -> bool:
    """Return true when ``vector`` is fixed by every supplied generator."""

    return all(
        all(sp.simplify(entry) == 0 for entry in generator * vector - vector)
        for generator in generators
    )


def odd_components_are_equal(vector: sp.Matrix) -> bool:
    """Return true when all five odd-shell components are equal."""

    first_odd = vector[ODD_START, 0]
    return all(
        sp.simplify(vector[index, 0] - first_odd) == 0
        for index in range(ODD_START + 1, SHELL_DIMENSION)
    )


def parity_preserving_symmetry_forces_flatness() -> bool:
    """Return false: odd-shell symmetry leaves the even/odd ratio free."""

    ratios = (sp.Rational(1, 2), sp.Integer(1), sp.Integer(2))
    return not all(
        vector_is_invariant(primitive_shell_amplitude(ratio), parity_preserving_generators())
        for ratio in ratios
    )


def full_transitive_symmetry_forces_flatness() -> bool:
    """Return true when full six-channel transitivity fixes the flat vector."""

    full_generators = full_transitive_generators()
    flat_invariant = vector_is_invariant(primitive_shell_amplitude(1), full_generators)
    nonflat_rejected = all(
        not vector_is_invariant(primitive_shell_amplitude(ratio), full_generators)
        for ratio in (sp.Rational(1, 2), sp.Integer(2))
    )
    return flat_invariant and nonflat_rejected


@dataclass(frozen=True)
class PrimitiveErgodicityAuditPayload:
    """Verdict payload for the V16 primitive ergodicity theorem."""

    final_verdict: str
    parity_invariant_dimension: int
    full_transitive_invariant_dimension: int
    odd_shell_components_equal: bool
    parity_preserving_symmetry_forces_flatness: bool
    full_transitive_symmetry_forces_flatness: bool
    nonflat_parity_controls_invariant: bool
    nonflat_full_controls_rejected: bool
    ckm_phase_requires_extra_ergodicity_principle: bool
    interpretation: str


def primitive_ergodicity_audit_payload() -> PrimitiveErgodicityAuditPayload:
    """Return the V16 primitive ergodicity theorem verdict."""

    parity_basis = parity_preserving_invariant_basis()
    full_basis = full_transitive_invariant_basis()
    parity_generators = parity_preserving_generators()
    full_generators = full_transitive_generators()
    nonflat_ratios = (sp.Rational(1, 2), sp.Integer(2))

    odd_equal = all(odd_components_are_equal(vector) for vector in parity_basis)
    nonflat_parity_controls_invariant = all(
        vector_is_invariant(primitive_shell_amplitude(ratio), parity_generators)
        for ratio in nonflat_ratios
    )
    nonflat_full_controls_rejected = all(
        not vector_is_invariant(primitive_shell_amplitude(ratio), full_generators)
        for ratio in nonflat_ratios
    )
    only_flat_matches_ckm_phase = (
        sp.simplify(isotropic_quark_phase_angle(1) - quark_boundary_phase_angle()) == 0
        and all(
            sp.simplify(isotropic_quark_phase_angle(ratio) - quark_boundary_phase_angle()) != 0
            for ratio in nonflat_ratios
        )
    )

    checks_pass = (
        len(parity_basis) == 2
        and len(full_basis) == 1
        and odd_equal
        and not parity_preserving_symmetry_forces_flatness()
        and full_transitive_symmetry_forces_flatness()
        and nonflat_parity_controls_invariant
        and nonflat_full_controls_rejected
        and only_flat_matches_ckm_phase
    )

    if checks_pass:
        final_verdict = "PRIMITIVE_ERGODICITY_NO_GO_PASS"
        interpretation = (
            "Parity-preserving primitive-shell symmetry fixes equality within "
            "the five odd channels but leaves the even/odd ratio free. The CKM "
            "phase therefore requires an extra flat primitive ergodicity "
            "principle, or a microscopic dynamics that effectively mixes all "
            "six primitive channels transitively."
        )
    else:
        final_verdict = "PRIMITIVE_ERGODICITY_NO_GO_KILL"
        interpretation = (
            "The invariant-subspace dimensions, flatness controls, or CKM "
            "phase comparison failed."
        )

    return PrimitiveErgodicityAuditPayload(
        final_verdict=final_verdict,
        parity_invariant_dimension=len(parity_basis),
        full_transitive_invariant_dimension=len(full_basis),
        odd_shell_components_equal=odd_equal,
        parity_preserving_symmetry_forces_flatness=parity_preserving_symmetry_forces_flatness(),
        full_transitive_symmetry_forces_flatness=full_transitive_symmetry_forces_flatness(),
        nonflat_parity_controls_invariant=nonflat_parity_controls_invariant,
        nonflat_full_controls_rejected=nonflat_full_controls_rejected,
        ckm_phase_requires_extra_ergodicity_principle=True,
        interpretation=interpretation,
    )
