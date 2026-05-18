"""Re-exports of lepton infrastructure used by the triality sidecar.

This module is the single import surface between the triality experiment and
``clifford_3plus2_d5.lepton``.  Per the sidecar review:

> The sidecar should not start with a new octonion.py; it should wrap/import
> the lepton infrastructure.

The triality sidecar uses these helpers and nothing else from ``lepton``.
"""

from __future__ import annotations

from clifford_3plus2_d5.lepton.clifford_octonion import (
    chirality_projectors,
    cl08_even_generators,
    cl08_gamma_matrices,
    octonion_derivation_basis,
    octonion_left_multiplication,
    octonion_right_multiplication,
    su3_stabilizer_basis,
    volume_element,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_chiral16_basis_matrix,
    patisalam_chiral16_block_matrix,
    patisalam_chosen_complex_structure,
    patisalam_cl010_gamma_matrices,
    su2_l_generators_from_spin04,
    su2_r_generators_from_spin04,
    su4_generators_from_spin06,
)
from clifford_3plus2_d5.lepton.sm_hypercharge import (
    charge_observable as charge_observable_from_lepton,
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


def charge_observable(generator):
    """Re-export of ``lepton.sm_hypercharge.charge_observable``."""

    return charge_observable_from_lepton(generator)

__all__ = [
    "EXPECTED_HYPERCHARGE_SPECTRUM",
    "EXPECTED_JOINT_Y_T3L_TABLE",
    "b_minus_l_generator_from_su4",
    "charge_observable",
    "chirality_projectors",
    "cl08_even_generators",
    "cl08_gamma_matrices",
    "hypercharge_generator",
    "hypercharge_observable",
    "normalized_t3_l_observable",
    "octonion_derivation_basis",
    "octonion_left_multiplication",
    "octonion_right_multiplication",
    "patisalam_chiral16_basis_matrix",
    "patisalam_chiral16_block_matrix",
    "patisalam_chosen_complex_structure",
    "patisalam_cl010_gamma_matrices",
    "physical_hypercharge_generator",
    "sm_gauge_generators",
    "su2_l_generators_from_spin04",
    "su2_r_generators_from_spin04",
    "su3_c_generators_from_su4",
    "su3_stabilizer_basis",
    "su4_generators_from_spin06",
    "volume_element",
]
