"""Single import surface from ``triality/`` and (via that) ``lepton/``.

This sidecar adds no new Clifford / octonion / Pati-Salam code.  Every
algebraic helper comes from the exact-triality sidecar, which in turn
imports from lepton.
"""

from __future__ import annotations

from clifford_3plus2_d5.triality.reuse import (
    EXPECTED_HYPERCHARGE_SPECTRUM,
    EXPECTED_JOINT_Y_T3L_TABLE,
    charge_observable,
    hypercharge_observable,
    normalized_t3_l_observable,
    patisalam_chiral16_block_matrix,
    patisalam_chosen_complex_structure,
    patisalam_cl010_gamma_matrices,
    physical_hypercharge_generator,
    su3_c_generators_from_su4,
)
from clifford_3plus2_d5.triality.sm_restriction import (
    g_sm_8_cartan_basis_coords,
    restricted_hypercharge_cartan_coords,
    restricted_hypercharge_generator,
    su3_c_cartan_coords,
    su3_c_cartan_indices,
)
from clifford_3plus2_d5.triality.spin8_triality import (
    apply_triality_to_cartan_vector,
    cartan_coordinates,
    spin8_cartan_on_chiral16,
    spin8_generator_on_chiral16,
    spin8_generators_on_chiral16,
    triality_cartan_matrix,
)

__all__ = [
    "EXPECTED_HYPERCHARGE_SPECTRUM",
    "EXPECTED_JOINT_Y_T3L_TABLE",
    "apply_triality_to_cartan_vector",
    "cartan_coordinates",
    "charge_observable",
    "g_sm_8_cartan_basis_coords",
    "hypercharge_observable",
    "normalized_t3_l_observable",
    "patisalam_chiral16_block_matrix",
    "patisalam_chosen_complex_structure",
    "patisalam_cl010_gamma_matrices",
    "physical_hypercharge_generator",
    "restricted_hypercharge_cartan_coords",
    "restricted_hypercharge_generator",
    "spin8_cartan_on_chiral16",
    "spin8_generator_on_chiral16",
    "spin8_generators_on_chiral16",
    "su3_c_cartan_coords",
    "su3_c_cartan_indices",
    "su3_c_generators_from_su4",
    "triality_cartan_matrix",
]
