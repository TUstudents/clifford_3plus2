"""Single import surface for the topology/ sidecar."""

from __future__ import annotations

from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_chiral16_basis_matrix,
    patisalam_chiral16_block_matrix,
    patisalam_cl010_gamma_matrices,
)
from clifford_3plus2_d5.lepton.patisalam_sm import (
    b_minus_l_generator_from_su4,
    sm_gauge_generators,
    su3_c_generators_from_su4,
)
from clifford_3plus2_d5.lepton.sm_hypercharge import (
    physical_hypercharge_generator,
)
from clifford_3plus2_d5.spacetime_qca.bcc_weyl import (
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    opposite_helicity_hops,
)
from clifford_3plus2_d5.spacetime_qca.dirac import (
    alpha_matrices,
    block_diag,
    gamma0,
    gamma5,
    gamma_matrices,
    gamma_spatial_matrices,
)
from clifford_3plus2_d5.spacetime_qca.pauli import (
    pauli_matrices,
    same_matrix,
    sigma_x,
    sigma_y,
    sigma_z,
)

__all__ = [
    "alpha_matrices",
    "b_minus_l_generator_from_su4",
    "bialynicki_birula_directions",
    "bialynicki_birula_hops",
    "block_diag",
    "gamma0",
    "gamma5",
    "gamma_matrices",
    "gamma_spatial_matrices",
    "opposite_helicity_hops",
    "patisalam_chiral16_basis_matrix",
    "patisalam_chiral16_block_matrix",
    "patisalam_cl010_gamma_matrices",
    "pauli_matrices",
    "physical_hypercharge_generator",
    "same_matrix",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "sm_gauge_generators",
    "su3_c_generators_from_su4",
]
