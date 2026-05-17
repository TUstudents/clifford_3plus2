"""Mass-layer helpers for the BCC Dirac spacetime QCA.

Session 21 tests the controlled Dirac mass layer

``H_m(k) = alpha.k x I_internal + beta x M_internal``

with ``beta = gamma^0`` in the chiral basis.  The scalar control
``M_internal = m I`` verifies the spacetime Dirac mass mechanism.  It is not a
Standard-Model Yukawa construction: realistic fermion masses require Higgs /
Yukawa structure because left- and right-handed SM fields transform
differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.dirac import alpha_matrices, dirac_hamiltonian, gamma0
from clifford_3plus2_d5.spacetime_qca.gauge_lift import lift_spacetime_operator


def _zero(rows: int, cols: int | None = None) -> sp.Matrix:
    return sp.zeros(rows, rows if cols is None else cols)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == _zero(left.rows, left.cols)


def commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return (left * right - right * left).applyfunc(sp.simplify)


def beta_matrix() -> sp.Matrix:
    """Return ``beta = gamma^0`` in chiral basis."""

    return gamma0()


def beta_anticommutes_with_alpha() -> bool:
    beta = beta_matrix()
    return all(_same_matrix(alpha * beta + beta * alpha, _zero(4)) for alpha in alpha_matrices())


def scalar_internal_mass(internal_dim: int, mass: sp.Expr) -> sp.Matrix:
    return (mass * sp.eye(internal_dim)).applyfunc(sp.simplify)


def mass_hamiltonian(internal_mass: sp.Matrix) -> sp.Matrix:
    """Return ``beta x M_internal``."""

    return sp.kronecker_product(beta_matrix(), internal_mass).applyfunc(sp.simplify)


def massive_dirac_hamiltonian(
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    internal_mass: sp.Matrix,
) -> sp.Matrix:
    """Return ``alpha.k x I + beta x M``."""

    return (
        lift_spacetime_operator(dirac_hamiltonian(kx, ky, kz), internal_mass.rows)
        + mass_hamiltonian(internal_mass)
    ).applyfunc(sp.simplify)


def scalar_mass_squared_residual(
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    mass: sp.Expr,
    *,
    internal_dim: int = 1,
) -> sp.Matrix:
    """Return ``H_m^2 - (|k|^2 + m^2) I`` for scalar mass control."""

    internal_mass = scalar_internal_mass(internal_dim, mass)
    hamiltonian = massive_dirac_hamiltonian(kx, ky, kz, internal_mass)
    expected = (kx**2 + ky**2 + kz**2 + mass**2) * sp.eye(4 * internal_dim)
    return (hamiltonian * hamiltonian - expected).applyfunc(sp.simplify)


def scalar_mass_dispersion_valid(
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    mass: sp.Expr,
    *,
    internal_dim: int = 1,
) -> bool:
    residual = scalar_mass_squared_residual(kx, ky, kz, mass, internal_dim=internal_dim)
    return _same_matrix(residual, _zero(residual.rows, residual.cols))


def k0_scalar_mass_spectrum(mass: sp.Expr, *, internal_dim: int = 1) -> dict[sp.Expr, int]:
    hamiltonian = massive_dirac_hamiltonian(
        sp.Integer(0),
        sp.Integer(0),
        sp.Integer(0),
        scalar_internal_mass(internal_dim, mass),
    )
    return hamiltonian.eigenvals()


def commutes_with_all(matrix: sp.Matrix, generators: Sequence[sp.Matrix]) -> bool:
    return all(_same_matrix(commutator(matrix, generator), _zero(matrix.rows)) for generator in generators)


def mass_preserves_gauge(internal_mass: sp.Matrix, generators: Sequence[sp.Matrix]) -> bool:
    return commutes_with_all(internal_mass, generators)


def projector_control_mass(internal_dim: int, rank: int, mass: sp.Expr = sp.Integer(1)) -> sp.Matrix:
    """Return a non-scalar diagonal projector mass used as a symmetry-breaking control."""

    if not 0 < rank < internal_dim:
        raise ValueError("rank must be strictly between 0 and internal_dim")
    diagonal = [mass if index < rank else 0 for index in range(internal_dim)]
    return sp.diag(*diagonal)


@dataclass(frozen=True)
class MassCompatibilityAudit:
    scalar_mass_dirac_valid: bool
    scalar_mass_patisalam_preserving: bool
    scalar_mass_sm_preserving: bool
    nonscalar_control_breaks_sm: bool
    interpretation: str


def mass_compatibility_audit_payload(
    *,
    patisalam_generators: Sequence[sp.Matrix],
    sm_generators: Sequence[sp.Matrix],
) -> MassCompatibilityAudit:
    """Return the Session 21 mass-layer audit payload."""

    if not patisalam_generators or not sm_generators:
        raise ValueError("mass audit requires nonempty gauge generator lists")
    internal_dim = patisalam_generators[0].rows
    mass = sp.symbols("m")
    scalar_mass = scalar_internal_mass(internal_dim, mass)
    nonscalar = projector_control_mass(internal_dim, internal_dim // 2)
    kx, ky, kz = sp.symbols("kx ky kz")
    return MassCompatibilityAudit(
        scalar_mass_dirac_valid=scalar_mass_dispersion_valid(kx, ky, kz, mass),
        scalar_mass_patisalam_preserving=mass_preserves_gauge(scalar_mass, patisalam_generators),
        scalar_mass_sm_preserving=mass_preserves_gauge(scalar_mass, sm_generators),
        nonscalar_control_breaks_sm=not mass_preserves_gauge(nonscalar, sm_generators),
        interpretation=(
            "Scalar mass validates the Dirac mass layer but is a universal control, "
            "not an SM Yukawa/Higgs mass spectrum."
        ),
    )
