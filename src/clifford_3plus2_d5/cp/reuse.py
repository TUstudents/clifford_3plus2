"""Single import surface for the CP sidecar.

Imports from ``spacetime_qca`` (BCC Dirac walk, Higgs maps),
``lepton.clifford_patisalam`` (J candidates), and the existing chiral
``spacetime_qca.dirac`` matrices.  No new octonion / Clifford code.
"""

from __future__ import annotations

from clifford_3plus2_d5.lepton.clifford_patisalam import (
    cl04_chosen_commutant_j,
    cl04_commutant_complex_structures,
    patisalam_chiral16_block_matrix,
    patisalam_chosen_complex_structure,
)
from clifford_3plus2_d5.spacetime_qca import (
    alpha_matrices,
    bcc_dirac_symbol,
    bcc_weyl_symbol,
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    block_diag,
    dirac_hamiltonian,
    gamma0,
    gamma5,
    gamma_matrices,
    gamma_spatial_matrices,
    opposite_helicity_hops,
    same_matrix,
    sigma_x,
    sigma_y,
    sigma_z,
)
from clifford_3plus2_d5.spacetime_qca.yukawa import (
    beta_is_off_diagonal_between_chiralities,
    color_singlet_charge_shift_basis,
    higgs_like_charge_shift_candidate,
    higgs_like_charge_shift_pair,
)

__all__ = [
    "alpha_matrices",
    "bcc_dirac_symbol",
    "bcc_weyl_symbol",
    "beta_is_off_diagonal_between_chiralities",
    "bialynicki_birula_directions",
    "bialynicki_birula_hops",
    "block_diag",
    "cl04_chosen_commutant_j",
    "cl04_commutant_complex_structures",
    "color_singlet_charge_shift_basis",
    "dirac_hamiltonian",
    "gamma0",
    "gamma5",
    "gamma_matrices",
    "gamma_spatial_matrices",
    "higgs_like_charge_shift_candidate",
    "higgs_like_charge_shift_pair",
    "opposite_helicity_hops",
    "patisalam_chiral16_block_matrix",
    "patisalam_chosen_complex_structure",
    "same_matrix",
    "sigma_x",
    "sigma_y",
    "sigma_z",
]
