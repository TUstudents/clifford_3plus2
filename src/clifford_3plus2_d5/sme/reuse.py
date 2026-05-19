"""Single import surface for the SME sidecar.

Imports from ``cp/`` (H^(1), CP irrep decomposition, T_{2g} projector)
and ``spacetime_qca.dirac`` (chiral-basis γ matrices for the bilinear
projection in Phase A-2).  No new octonion / Clifford code.
"""

from __future__ import annotations

from clifford_3plus2_d5.cp.continuum_cp import (
    cp_irrep_decomposition,
    cp_irrep_norm_table,
    effective_hamiltonian_first_correction,
    symbolic_momentum,
)
from clifford_3plus2_d5.cp.cubic_harmonics import (
    decompose_matrix_of_polynomials,
    monomial_basis,
    projector_T2g,
)
from clifford_3plus2_d5.spacetime_qca import (
    gamma0,
    gamma5,
    gamma_matrices,
    gamma_spatial_matrices,
    same_matrix,
)

__all__ = [
    "cp_irrep_decomposition",
    "cp_irrep_norm_table",
    "decompose_matrix_of_polynomials",
    "effective_hamiltonian_first_correction",
    "gamma0",
    "gamma5",
    "gamma_matrices",
    "gamma_spatial_matrices",
    "monomial_basis",
    "projector_T2g",
    "same_matrix",
    "symbolic_momentum",
]
