"""Spacetime QCA experiments.

This package is intentionally separate from ``lepton``.  The lepton package
contains the internal Cl(0,10) / Pati-Salam / SM gauge-content audits.  This
package is for spacetime dynamics: BCC Weyl/Dirac walks, STA-compatible carrier
choices, lattice-doubling controls, and later tensor lifts against the internal
carrier.
"""

from clifford_3plus2_d5.spacetime_qca.audit import (
    SpacetimeQCAInfrastructureAudit,
    infrastructure_audit_payload,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import (
    Vector3,
    bcc_body_diagonal_directions,
    hypercube_bz_corners,
    origin3,
    squared_norm,
    vector_dot,
)
from clifford_3plus2_d5.spacetime_qca.bcc_weyl import (
    validate_weyl_hops,
    weyl_bloch_symbol_from_hops,
)
from clifford_3plus2_d5.spacetime_qca.continuum import (
    effective_generator_from_floquet,
    first_order_in_epsilon,
    hamiltonian_from_generator,
    matrix_zero,
)
from clifford_3plus2_d5.spacetime_qca.dirac import (
    alpha_matrices,
    block_diag,
    dirac_hamiltonian,
    gamma0,
    gamma5,
    gamma_matrices,
    gamma_spatial_matrices,
    tensor_internal,
)
from clifford_3plus2_d5.spacetime_qca.gauge_lift import (
    background_gauge_hamiltonian,
    lift_internal_operator,
    lift_spacetime_operator,
)
from clifford_3plus2_d5.spacetime_qca.hypercube_control import (
    hypercube_continuum_target,
    hypercube_corner_eigenvalues,
    hypercube_gapless_corners,
    hypercube_is_gapless_at,
    naive_hypercube_hamiltonian,
)
from clifford_3plus2_d5.spacetime_qca.pauli import (
    identity2,
    is_hermitian,
    is_unitary,
    pauli_matrices,
    same_matrix,
    sigma_x,
    sigma_y,
    sigma_z,
)

__all__ = [
    "SpacetimeQCAInfrastructureAudit",
    "Vector3",
    "alpha_matrices",
    "background_gauge_hamiltonian",
    "bcc_body_diagonal_directions",
    "block_diag",
    "dirac_hamiltonian",
    "effective_generator_from_floquet",
    "first_order_in_epsilon",
    "gamma0",
    "gamma5",
    "gamma_matrices",
    "gamma_spatial_matrices",
    "hamiltonian_from_generator",
    "hypercube_bz_corners",
    "hypercube_continuum_target",
    "hypercube_corner_eigenvalues",
    "hypercube_gapless_corners",
    "hypercube_is_gapless_at",
    "identity2",
    "infrastructure_audit_payload",
    "is_hermitian",
    "is_unitary",
    "lift_internal_operator",
    "lift_spacetime_operator",
    "matrix_zero",
    "naive_hypercube_hamiltonian",
    "origin3",
    "pauli_matrices",
    "same_matrix",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "squared_norm",
    "tensor_internal",
    "validate_weyl_hops",
    "vector_dot",
    "weyl_bloch_symbol_from_hops",
]
