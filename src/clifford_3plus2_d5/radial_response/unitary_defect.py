"""Minimal exact unitary S3-defect Floquet model."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    S3Element,
    left_regular_matrix,
)


def uniform_defect_vector() -> sp.Matrix:
    """Return the normalized uniform vector in the S3 regular shell."""

    return sp.Matrix([1 / sp.sqrt(6) for _ in range(6)])


def basis_defect_vector(index: int = 0) -> sp.Matrix:
    """Return one basis vector in the S3 regular shell."""

    if not 0 <= index < 6:
        raise ValueError("index must be in [0, 5]")
    vector = sp.zeros(6, 1)
    vector[index, 0] = 1
    return vector


def full_s3_shift(element: S3Element = (1, 0, 2)) -> sp.Matrix:
    """Return the resolved-plus-regular-shell shift block."""

    return sp.diag(sp.Integer(1), left_regular_matrix(element))


def normalized_full_defect_vector(q_vector: sp.Matrix) -> sp.Matrix:
    """Embed a normalized Q-shell defect vector into the full P+Q space."""

    norm_squared = sp.simplify((q_vector.T * q_vector)[0, 0])
    if norm_squared != 1:
        raise ValueError("q_vector must be normalized")
    return sp.Matrix.vstack(sp.zeros(1, 1), q_vector)


def givens_defect_coin(
    q_vector: sp.Matrix | None = None,
    cos_value: sp.Expr = 1 / sp.sqrt(2),
    sin_value: sp.Expr = 1 / sp.sqrt(2),
) -> sp.Matrix:
    """Return a Givens coin rotating resolved P into one defect vector."""

    q_shell = uniform_defect_vector() if q_vector is None else q_vector
    q_full = normalized_full_defect_vector(q_shell)
    p_full = sp.Matrix.vstack(sp.ones(1, 1), sp.zeros(6, 1))
    cos_value = sp.sympify(cos_value)
    sin_value = sp.sympify(sin_value)
    if sp.simplify(cos_value**2 + sin_value**2 - 1) != 0:
        raise ValueError("cos_value and sin_value must satisfy c^2+s^2=1")
    identity = sp.eye(7)
    p_projector = p_full * p_full.T
    q_projector = q_full * q_full.T
    rotation = q_full * p_full.T - p_full * q_full.T
    return (
        identity
        + (cos_value - 1) * (p_projector + q_projector)
        + sin_value * rotation
    ).applyfunc(sp.simplify)


def minimal_unitary_s3_defect(
    q_vector: sp.Matrix | None = None,
    cos_value: sp.Expr = 1 / sp.sqrt(2),
    sin_value: sp.Expr = 1 / sp.sqrt(2),
    shift_element: S3Element = (1, 0, 2),
) -> sp.Matrix:
    """Return the exact Floquet toy ``U = S C``."""

    shift = full_s3_shift(shift_element)
    coin = givens_defect_coin(q_vector, cos_value, sin_value)
    return (shift * coin).applyfunc(sp.simplify)


def is_exact_unitary(matrix: sp.Matrix) -> bool:
    """Return true when a real matrix is exactly orthogonal/unitary."""

    return sp.simplify(matrix.T * matrix - sp.eye(matrix.rows)) == sp.zeros(matrix.rows, matrix.cols)


def unitary_self_energy(z: sp.Expr, u_matrix: sp.Matrix, p_dim: int = 1) -> sp.Matrix:
    """Return unitary/Floquet self-energy ``U_PQ(zI-U_QQ)^-1U_QP``."""

    z = sp.sympify(z)
    u_pq = u_matrix[:p_dim, p_dim:]
    u_qp = u_matrix[p_dim:, :p_dim]
    u_qq = u_matrix[p_dim:, p_dim:]
    return (u_pq * (z * sp.eye(u_qq.rows) - u_qq).inv() * u_qp).applyfunc(sp.simplify)


def unitary_effective_resolvent(z: sp.Expr, u_matrix: sp.Matrix, p_dim: int = 1) -> sp.Matrix:
    """Return the P-block effective resolvent for a unitary block matrix."""

    z = sp.sympify(z)
    u_pp = u_matrix[:p_dim, :p_dim]
    sigma = unitary_self_energy(z, u_matrix, p_dim)
    return (z * sp.eye(p_dim) - u_pp - sigma).inv().applyfunc(sp.simplify)


def full_unitary_resolvent_p_block(z: sp.Expr, u_matrix: sp.Matrix, p_dim: int = 1) -> sp.Matrix:
    """Return the P block of ``(zI-U)^-1``."""

    z = sp.sympify(z)
    full = (z * sp.eye(u_matrix.rows) - u_matrix).inv().applyfunc(sp.simplify)
    return full[:p_dim, :p_dim].applyfunc(sp.simplify)


def self_energy_at_z_two(
    q_vector: sp.Matrix | None = None,
    cos_value: sp.Expr = 1 / sp.sqrt(2),
    sin_value: sp.Expr = 1 / sp.sqrt(2),
) -> sp.Matrix:
    """Return default self-energy at ``z=2`` for exact comparisons."""

    u_matrix = minimal_unitary_s3_defect(q_vector, cos_value, sin_value)
    return unitary_self_energy(sp.Integer(2), u_matrix)


@dataclass(frozen=True)
class MinimalUnitaryDefectPayload:
    """Payload for the minimal unitary S3-defect gate."""

    final_verdict: str
    unitary_passes: bool
    schur_matches_full_resolvent: bool
    coin_angle_changes_self_energy: bool
    defect_vector_changes_self_energy: bool
    phase_and_radial_values_forced_by_form: bool
    interpretation: str


def minimal_unitary_defect_payload() -> MinimalUnitaryDefectPayload:
    """Return the minimal unitary S3-defect form verdict."""

    z = sp.Integer(2)
    u_default = minimal_unitary_s3_defect()
    schur = unitary_effective_resolvent(z, u_default)
    full = full_unitary_resolvent_p_block(z, u_default)
    schur_matches = sp.simplify(schur - full) == sp.zeros(1, 1)
    default_sigma = self_energy_at_z_two()
    angle_control_sigma = self_energy_at_z_two(
        cos_value=sp.Rational(3, 5),
        sin_value=sp.Rational(4, 5),
    )
    vector_control_sigma = self_energy_at_z_two(q_vector=basis_defect_vector())
    angle_changes = sp.simplify(default_sigma - angle_control_sigma) != sp.zeros(1, 1)
    vector_changes = sp.simplify(default_sigma - vector_control_sigma) != sp.zeros(1, 1)
    checks_pass = (
        is_exact_unitary(u_default)
        and schur_matches
        and angle_changes
        and vector_changes
    )

    if checks_pass:
        final_verdict = "MINIMAL_UNITARY_S3_DEFECT_FORM_PASS"
        interpretation = (
            "The exact Floquet toy U=S C is unitary and has a valid unitary "
            "Schur self-energy. However, the self-energy changes under exact "
            "coin-angle and defect-vector controls, so the form does not by "
            "itself force radial pole values or continuous phases."
        )
    else:
        final_verdict = "MINIMAL_UNITARY_S3_DEFECT_FORM_KILL"
        interpretation = "The minimal unitary S3 defect failed unitarity or Schur controls."

    return MinimalUnitaryDefectPayload(
        final_verdict=final_verdict,
        unitary_passes=is_exact_unitary(u_default),
        schur_matches_full_resolvent=schur_matches,
        coin_angle_changes_self_energy=angle_changes,
        defect_vector_changes_self_energy=vector_changes,
        phase_and_radial_values_forced_by_form=False,
        interpretation=interpretation,
    )
