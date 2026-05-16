"""Profile factories for the leptonic bridge laboratory."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.lepton.carrier import (
    lab_b_complex_structure,
    lab_b_physical_wall_complex_structure,
    lab_b_singlet_doublet_projectors,
)
from clifford_3plus2_d5.lepton.gauge import su2_l_u1_y_generators_r6
from clifford_3plus2_d5.lepton.predicates import (
    central_j_candidates_default,
    domain_wall_trivial_center_policy,
    j_centralizes_gauge_all_candidates,
    lab_a_idempotent_policy,
    no_commutant_check,
    side_local_gauge_with_wall_transition,
    strict_split_idempotent_policy,
)
from clifford_3plus2_d5.lepton.verdict import V2Verdict, VerdictProfile


def lepton_singlet_doublet_projectors() -> tuple[sp.Matrix, sp.Matrix]:
    """Return the R^6 singlet/doublet projectors with real ranks 2 and 4."""

    return lab_b_singlet_doublet_projectors()


def lab_b_split_center_basis() -> tuple[sp.Matrix, ...]:
    singlet, doublet = lab_b_singlet_doublet_projectors()
    j = lab_b_complex_structure()
    return (
        singlet,
        (j * singlet).applyfunc(sp.simplify),
        doublet,
        (j * doublet).applyfunc(sp.simplify),
    )


def lab_a_profile() -> VerdictProfile:
    return VerdictProfile(
        name="lab_a_r4_clock_plane_primitive_closure",
        dimension=4,
        central_j_candidates=central_j_candidates_default,
        idempotent_policy=lab_a_idempotent_policy,
        commutant_policy=no_commutant_check,
        require_noncommutative=True,
        minimum_algebra_dimension=8,
        expected_center_dimension=2,
        expected_block_dimensions=(8,),
        expected_block_commutativity=(False,),
        max_algebra_dimension=16,
        max_center_dimension=4,
        max_compatible_basis_dimension=8,
        positive_verdict=V2Verdict.CLOCK_PLANE_CLOSURE_CANDIDATE,
    )


def lab_b_structural_profile() -> VerdictProfile:
    singlet, doublet = lepton_singlet_doublet_projectors()
    return VerdictProfile(
        name="lab_b_r6_structural_strict",
        dimension=6,
        target_projectors=(singlet, doublet),
        central_j_candidates=central_j_candidates_default,
        idempotent_policy=strict_split_idempotent_policy,
        commutant_policy=j_centralizes_gauge_all_candidates,
        require_noncommutative=True,
        minimum_algebra_dimension=10,
        expected_center_dimension=4,
        expected_block_dimensions=(2, 8),
        expected_block_commutativity=(True, False),
        max_algebra_dimension=32,
        max_center_dimension=8,
        gauge_generators=su2_l_u1_y_generators_r6(),
        known_center_basis=lab_b_split_center_basis(),
        use_target_projector_idempotents=True,
    )


def lab_b_structural_wall_profile() -> VerdictProfile:
    base = lab_b_structural_profile()
    return VerdictProfile(
        **{
            **base.__dict__,
            "name": "lab_b_r6_structural_wall",
            "wall_context_required": True,
        }
    )


def lab_b_domain_wall_profile() -> VerdictProfile:
    return VerdictProfile(
        name="lab_b_r6_domain_wall",
        dimension=6,
        central_j_candidates=central_j_candidates_default,
        idempotent_policy=domain_wall_trivial_center_policy,
        commutant_policy=side_local_gauge_with_wall_transition,
        require_noncommutative=True,
        minimum_algebra_dimension=4,
        expected_center_dimension=2,
        expected_block_dimensions=None,
        expected_block_commutativity=None,
        max_algebra_dimension=32,
        max_center_dimension=4,
        wall_context_required=True,
        known_center_basis=(identity(6), lab_b_complex_structure()),
        verify_known_center_basis_exact=True,
        positive_verdict=V2Verdict.DOMAIN_WALL_CANDIDATE,
    )


def lab_b_physical_domain_wall_profile() -> VerdictProfile:
    return VerdictProfile(
        name="lab_b_r12_physical_domain_wall",
        dimension=12,
        central_j_candidates=central_j_candidates_default,
        idempotent_policy=domain_wall_trivial_center_policy,
        commutant_policy=side_local_gauge_with_wall_transition,
        require_noncommutative=True,
        minimum_algebra_dimension=4,
        expected_center_dimension=2,
        expected_block_dimensions=None,
        expected_block_commutativity=None,
        max_algebra_dimension=32,
        max_center_dimension=4,
        wall_context_required=True,
        known_center_basis=(identity(12), lab_b_physical_wall_complex_structure()),
        verify_known_center_basis_exact=False,
        positive_verdict=V2Verdict.DOMAIN_WALL_CANDIDATE,
    )
