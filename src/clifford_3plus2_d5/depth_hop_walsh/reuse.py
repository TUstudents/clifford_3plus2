"""External import surface for the depth-hop-Walsh probe.

Borrows the genuine BCC Weyl hop source, the body-diagonal C3 rotation + spinor
lift, the [111] direction, and (diagnostic only) the strong-CP effective-Hamiltonian
results. No new physics is defined here.
"""

from __future__ import annotations

from clifford_3plus2_d5.koide.koide_geometry import trace_direction
from clifford_3plus2_d5.spacetime_qca.bcc_weyl import (
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    bialynicki_birula_s_matrices,
    opposite_helicity_hops,
)
from clifford_3plus2_d5.spacetime_qca.pauli import pauli_matrices, same_matrix
from clifford_3plus2_d5.strongcp.higher_order_parity import h2_a2u_component_is_zero
from clifford_3plus2_d5.strongcp.reuse import effective_hamiltonian_first_correction
from clifford_3plus2_d5.topology.bcc_z3_rotation import (
    apply_rotation_to_direction,
    body_diagonal_rotation_matrix,
    dirac_spinor_lift,
)

__all__ = [
    "apply_rotation_to_direction",
    "bialynicki_birula_directions",
    "bialynicki_birula_hops",
    "bialynicki_birula_s_matrices",
    "body_diagonal_rotation_matrix",
    "dirac_spinor_lift",
    "effective_hamiltonian_first_correction",
    "h2_a2u_component_is_zero",
    "opposite_helicity_hops",
    "pauli_matrices",
    "same_matrix",
    "trace_direction",
]
