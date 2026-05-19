"""Single import surface for the exceptional/ sidecar.

Imports octonion / Clifford / Pati-Salam / SM helpers from ``lepton`` and
CP machinery from ``cp``.  No duplication.
"""

from __future__ import annotations

from clifford_3plus2_d5.lepton.clifford_octonion import (
    cl08_gamma_matrices,
    octonion_derivation_basis,
    octonion_fano_triples,
    octonion_left_multiplication,
    octonion_multiply,
    octonion_right_multiplication,
    octonion_structure_constants,
    su3_stabilizer_basis,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_chiral16_basis_matrix,
    patisalam_chiral16_block_matrix,
    patisalam_chosen_complex_structure,
    patisalam_cl010_gamma_matrices,
    spin04_generators_chiral16,
    spin06_generators_chiral16,
    su2_l_generators_from_spin04,
    su2_r_generators_from_spin04,
    su4_generators_from_spin06,
)
from clifford_3plus2_d5.lepton.patisalam_sm import (
    b_minus_l_generator_from_su4,
    hypercharge_generator,
    sm_gauge_generators,
    su3_c_generators_from_su4,
)
from clifford_3plus2_d5.lepton.sm_hypercharge import (
    EXPECTED_HYPERCHARGE_SPECTRUM,
    EXPECTED_JOINT_Y_T3L_TABLE,
    hypercharge_observable,
    normalized_t3_l_observable,
    physical_hypercharge_generator,
)

__all__ = [
    "EXPECTED_HYPERCHARGE_SPECTRUM",
    "EXPECTED_JOINT_Y_T3L_TABLE",
    "b_minus_l_generator_from_su4",
    "cl08_gamma_matrices",
    "hypercharge_generator",
    "hypercharge_observable",
    "normalized_t3_l_observable",
    "octonion_derivation_basis",
    "octonion_fano_triples",
    "octonion_left_multiplication",
    "octonion_multiply",
    "octonion_right_multiplication",
    "octonion_structure_constants",
    "patisalam_chiral16_basis_matrix",
    "patisalam_chiral16_block_matrix",
    "patisalam_chosen_complex_structure",
    "patisalam_cl010_gamma_matrices",
    "physical_hypercharge_generator",
    "sm_gauge_generators",
    "spin04_generators_chiral16",
    "spin06_generators_chiral16",
    "su2_l_generators_from_spin04",
    "su2_r_generators_from_spin04",
    "su3_c_generators_from_su4",
    "su3_stabilizer_basis",
    "su4_generators_from_spin06",
]
