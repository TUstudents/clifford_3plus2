"""Single import surface for the strongcp sidecar.

Imports from cp/ (cubic harmonics degree-2 baseline, H^(1), BCH
machinery, P/T/C operators) and spacetime_qca/ (BCC geometry, BB
walk, Wilson plaquettes).  No new octonion / Clifford code.
"""

from __future__ import annotations

from clifford_3plus2_d5.cp.continuum_cp import (
    bloch_order_two,
    effective_hamiltonian_first_correction,
    symbolic_momentum,
)
from clifford_3plus2_d5.cp.cubic_harmonics import (
    decompose_matrix_of_polynomials,
    monomial_basis,
    polynomial_to_coefficient_vector,
    projector_A1g,
    projector_Eg,
    projector_T2g,
)
from clifford_3plus2_d5.cp.discrete_symmetries import (
    charge_conjugation_spinor,
    parity_spinor,
    time_reversal_spinor,
)
from clifford_3plus2_d5.cp.reuse import (
    bcc_dirac_symbol,
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    gamma0,
    gamma5,
    gamma_matrices,
    gamma_spatial_matrices,
    same_matrix,
)
from clifford_3plus2_d5.spacetime_qca.continuum import (
    nth_order_in_epsilon,
)

__all__ = [
    "bcc_dirac_symbol",
    "bialynicki_birula_directions",
    "bialynicki_birula_hops",
    "bloch_order_two",
    "charge_conjugation_spinor",
    "decompose_matrix_of_polynomials",
    "effective_hamiltonian_first_correction",
    "gamma0",
    "gamma5",
    "gamma_matrices",
    "gamma_spatial_matrices",
    "monomial_basis",
    "nth_order_in_epsilon",
    "parity_spinor",
    "polynomial_to_coefficient_vector",
    "projector_A1g",
    "projector_Eg",
    "projector_T2g",
    "same_matrix",
    "symbolic_momentum",
    "time_reversal_spinor",
]
