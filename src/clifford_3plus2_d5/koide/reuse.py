"""Single import surface for the koide/ sidecar.

Imports from cp/ (Higgs map basis, J-decomposition, T_{2g} projector),
topology/ (BCC body-diagonal rotation), broken_triality/ (3×3 Yukawa
from Z₃ orbit pattern), and spacetime_qca (BCC walk pieces).  No new
octonion / Clifford code.
"""

from __future__ import annotations

from clifford_3plus2_d5.broken_triality.reuse import (
    apply_triality_to_cartan_vector,
    g_sm_8_cartan_basis_coords,
    restricted_hypercharge_cartan_coords,
    triality_cartan_matrix,
)
from clifford_3plus2_d5.broken_triality.yukawa_overlaps import (
    project_onto_sm_cartan,
    sm_cartan_span_matrix,
    yukawa_overlap_matrix,
)
from clifford_3plus2_d5.cp.continuum_cp import (
    cp_irrep_decomposition,
    effective_hamiltonian_first_correction,
    symbolic_momentum,
)
from clifford_3plus2_d5.cp.cubic_harmonics import projector_T2g
from clifford_3plus2_d5.cp.j_misalignment import (
    color_singlet_charge_shift_basis,
    higgs_like_charge_shift_candidate,
    j_anticommuting_fraction,
    j_decomposition,
    viable_j_candidates,
)
from clifford_3plus2_d5.topology.bcc_z3_rotation import (
    bcc_direction_permutation,
    body_diagonal_rotation_matrix,
    dirac_spinor_lift,
    dirac_spinor_lift_4d,
)

__all__ = [
    "apply_triality_to_cartan_vector",
    "bcc_direction_permutation",
    "body_diagonal_rotation_matrix",
    "color_singlet_charge_shift_basis",
    "cp_irrep_decomposition",
    "j_anticommuting_fraction",
    "dirac_spinor_lift",
    "dirac_spinor_lift_4d",
    "effective_hamiltonian_first_correction",
    "g_sm_8_cartan_basis_coords",
    "higgs_like_charge_shift_candidate",
    "j_decomposition",
    "project_onto_sm_cartan",
    "projector_T2g",
    "sm_cartan_span_matrix",
    "restricted_hypercharge_cartan_coords",
    "symbolic_momentum",
    "triality_cartan_matrix",
    "viable_j_candidates",
    "yukawa_overlap_matrix",
]
