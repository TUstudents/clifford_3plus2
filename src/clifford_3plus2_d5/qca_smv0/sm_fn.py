"""Froggatt-Nielsen recirculation module for QCA_SMv0.

Stage 5 implements FN powers as finite hidden recirculation paths.  The module
does not claim to derive the charges or the attenuation parameter: both are
explicit simulator inputs.  What is implemented here is the constructive local
rule that a path of length ``n`` contributes the amplitude ``lambda**n`` through
a product of two-state unitary beam-splitter gates.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

SM_FAMILY_DIM = 3
FN_LAMBDA_WOLFENSTEIN = 0.225
FN_LAMBDA_SHEAR = float(jnp.sqrt(jnp.asarray(1.5, dtype=jnp.float32)) - 1.0)
FN_DEFAULT_Q_CHARGES = (3, 2, 0)
FN_DEFAULT_U_CHARGES = (5, 2, 0)
FN_DEFAULT_D_CHARGES = (1, 0, 0)
FN_EXPECTED_UP_DIAGONAL_EXPONENTS = (8, 4, 0)
FN_EXPECTED_DOWN_DIAGONAL_EXPONENTS = (4, 2, 0)


class FNChargeAssignment(NamedTuple):
    """Integer FN charges for a left/right Yukawa sector."""

    left: tuple[int, int, int]
    right: tuple[int, int, int]


class FNQuarkCharges(NamedTuple):
    """Default quark FN charge assignment used by the simulator rule."""

    q: tuple[int, int, int] = FN_DEFAULT_Q_CHARGES
    u: tuple[int, int, int] = FN_DEFAULT_U_CHARGES
    d: tuple[int, int, int] = FN_DEFAULT_D_CHARGES


class FNQuarkYukawas(NamedTuple):
    """Generated up/down Yukawa matrices."""

    up: jnp.ndarray
    down: jnp.ndarray


class FNRecirculationNetwork(NamedTuple):
    """Direct-sum unitary hidden path network for one FN Yukawa sector."""

    unitary: jnp.ndarray
    source_indices: jnp.ndarray
    sink_indices: jnp.ndarray
    path_lengths: jnp.ndarray


class FNRecirculationDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 5 FN recirculation paths."""

    beam_splitter_unitarity_residual: jnp.ndarray
    path_transfer_residual: jnp.ndarray
    up_network_unitarity_residual: jnp.ndarray
    down_network_unitarity_residual: jnp.ndarray
    up_network_transfer_residual: jnp.ndarray
    down_network_transfer_residual: jnp.ndarray
    up_exponent_residual: jnp.ndarray
    down_exponent_residual: jnp.ndarray
    up_diagonal_scaling_residual: jnp.ndarray
    down_diagonal_scaling_residual: jnp.ndarray
    wolfenstein_scaling_residual: jnp.ndarray
    ckm_unitarity_residual: jnp.ndarray
    jit_delta: jnp.ndarray


DEFAULT_FN_QUARK_CHARGES = FNQuarkCharges()


def _charge_array(charges: tuple[int, int, int] | jnp.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(charges, dtype=jnp.int32)
    if arr.shape != (SM_FAMILY_DIM,):
        raise ValueError("FN charges must have shape (3,)")
    return arr


def _validate_family_matrix(matrix: jnp.ndarray, name: str) -> jnp.ndarray:
    arr = jnp.asarray(matrix)
    if arr.shape != (SM_FAMILY_DIM, SM_FAMILY_DIM):
        raise ValueError(f"{name} must have shape (3,3)")
    return arr


def fn_beam_splitter(lambda_rec: float | jnp.ndarray) -> jnp.ndarray:
    """Return the two-state unitary advance gate for one recirculation step."""

    lam = jnp.asarray(lambda_rec, dtype=jnp.float32)
    stay = jnp.sqrt(1.0 - lam * lam)
    return jnp.asarray([[stay, -lam], [lam, stay]], dtype=jnp.float32)


def fn_beam_splitter_unitarity_residual(lambda_rec: float | jnp.ndarray) -> jnp.ndarray:
    """Return ``max_abs(B^T B-I)`` for the FN beam splitter."""

    splitter = fn_beam_splitter(lambda_rec)
    return jnp.max(jnp.abs(splitter.T @ splitter - jnp.eye(2, dtype=splitter.dtype)))


def fn_path_unitary(lambda_rec: float, path_length: int) -> jnp.ndarray:
    """Return the finite unitary chain whose end-transfer is ``lambda**n``."""

    if path_length < 0:
        raise ValueError("path_length must be non-negative")
    dim = path_length + 1
    unitary = jnp.eye(dim, dtype=jnp.float32)
    splitter = fn_beam_splitter(lambda_rec)
    for edge in range(path_length):
        gate = jnp.eye(dim, dtype=jnp.float32)
        gate = gate.at[edge : edge + 2, edge : edge + 2].set(splitter)
        unitary = gate @ unitary
    return unitary


def fn_path_transfer_amplitude(lambda_rec: float, path_length: int) -> jnp.ndarray:
    """Return the amplitude from the first to the last site of a path."""

    unitary = fn_path_unitary(lambda_rec, path_length)
    return unitary[path_length, 0]


def fn_path_transfer_residual(lambda_rec: float, *, max_path_length: int = 8) -> jnp.ndarray:
    """Return max residual between unitary-path transfer and ``lambda**n``."""

    residual = jnp.asarray(0.0, dtype=jnp.float32)
    lam = jnp.asarray(lambda_rec, dtype=jnp.float32)
    for path_length in range(max_path_length + 1):
        actual = fn_path_transfer_amplitude(float(lambda_rec), path_length)
        expected = lam**path_length
        residual = jnp.maximum(residual, jnp.abs(actual - expected))
    return residual


def fn_recirculation_network(
    lambda_rec: float | jnp.ndarray,
    left_charges: tuple[int, int, int],
    right_charges: tuple[int, int, int],
) -> FNRecirculationNetwork:
    """Return the explicit direct-sum hidden unitary for all FN paths.

    Each entry ``(i,j)`` gets an independent path of length
    ``n_ij=Q_i+R_j``.  The full hidden evolution is the direct sum of the local
    two-state beam-splitter chains, so the end-to-end transfer read from the
    block is ``lambda**n_ij`` without inserting that power directly.
    """

    path_lengths = fn_charge_exponents(left_charges, right_charges)
    left_py = tuple(int(charge) for charge in left_charges)
    right_py = tuple(int(charge) for charge in right_charges)
    path_lengths_py = tuple(left + right for left in left_py for right in right_py)
    total_dim = sum(length + 1 for length in path_lengths_py)
    unitary = jnp.zeros((total_dim, total_dim), dtype=jnp.float32)
    source_indices = []
    sink_indices = []
    offset = 0
    for length in path_lengths_py:
        path = fn_path_unitary(lambda_rec, length)
        dim = length + 1
        unitary = unitary.at[offset : offset + dim, offset : offset + dim].set(path)
        source_indices.append(offset)
        sink_indices.append(offset + length)
        offset += dim
    return FNRecirculationNetwork(
        unitary=unitary,
        source_indices=jnp.asarray(source_indices, dtype=jnp.int32).reshape((SM_FAMILY_DIM, SM_FAMILY_DIM)),
        sink_indices=jnp.asarray(sink_indices, dtype=jnp.int32).reshape((SM_FAMILY_DIM, SM_FAMILY_DIM)),
        path_lengths=path_lengths,
    )


def fn_recirculation_network_unitarity_residual(network: FNRecirculationNetwork) -> jnp.ndarray:
    """Return ``max_abs(U^T U-I)`` for a hidden FN recirculation network."""

    unitary = jnp.asarray(network.unitary)
    return jnp.max(jnp.abs(unitary.T @ unitary - jnp.eye(unitary.shape[0], dtype=unitary.dtype)))


def fn_recirculation_transfer_matrix(network: FNRecirculationNetwork) -> jnp.ndarray:
    """Read the effective family transfer matrix from a hidden FN network."""

    return network.unitary[network.sink_indices, network.source_indices]


def fn_recirculation_power_matrix(
    lambda_rec: float | jnp.ndarray,
    left_charges: tuple[int, int, int],
    right_charges: tuple[int, int, int],
) -> jnp.ndarray:
    """Return FN powers by measuring the explicit hidden path network."""

    return fn_recirculation_transfer_matrix(fn_recirculation_network(lambda_rec, left_charges, right_charges))


def fn_charge_exponents(left_charges: tuple[int, int, int], right_charges: tuple[int, int, int]) -> jnp.ndarray:
    """Return the FN path-length matrix ``n_ij=Q_i+R_j``."""

    left = _charge_array(left_charges)
    right = _charge_array(right_charges)
    return left[:, None] + right[None, :]


def fn_wolfenstein_scaling(left_charges: tuple[int, int, int], lambda_rec: float | jnp.ndarray) -> jnp.ndarray:
    """Return the left-frame mixing scale ``lambda**abs(Q_i-Q_j)``."""

    left = _charge_array(left_charges)
    exponents = jnp.abs(left[:, None] - left[None, :])
    return jnp.asarray(lambda_rec, dtype=jnp.float32) ** exponents


def fn_default_coefficients(kind: str = "up") -> jnp.ndarray:
    """Return deterministic order-one coefficients for simulator diagnostics."""

    if kind == "up":
        return jnp.asarray(
            [
                [1.0, 0.9, 1.1],
                [1.2, 1.0, 0.8],
                [0.7, 1.1, 1.0],
            ],
            dtype=jnp.complex64,
        )
    if kind == "down":
        return jnp.asarray(
            [
                [1.0, 1.1, 0.85],
                [0.9, 1.0, 1.15],
                [1.05, 0.95, 1.0],
            ],
            dtype=jnp.complex64,
        )
    raise ValueError("kind must be 'up' or 'down'")


def fn_effective_yukawa(
    lambda_rec: float | jnp.ndarray,
    left_charges: tuple[int, int, int],
    right_charges: tuple[int, int, int],
    *,
    coefficients: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return ``c_ij lambda**(Q_i+R_j)`` from FN recirculation path lengths."""

    coeffs = jnp.ones((SM_FAMILY_DIM, SM_FAMILY_DIM), dtype=jnp.complex64)
    if coefficients is not None:
        coeffs = _validate_family_matrix(coefficients, "coefficients").astype(jnp.complex64)
    powers = fn_recirculation_power_matrix(lambda_rec, left_charges, right_charges)
    return coeffs * powers.astype(jnp.complex64)


def fn_quark_yukawa_matrices(
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    up_coefficients: jnp.ndarray | None = None,
    down_coefficients: jnp.ndarray | None = None,
) -> FNQuarkYukawas:
    """Return the generated quark FN Yukawa matrices."""

    up_coeffs = fn_default_coefficients("up") if up_coefficients is None else up_coefficients
    down_coeffs = fn_default_coefficients("down") if down_coefficients is None else down_coefficients
    return FNQuarkYukawas(
        up=fn_effective_yukawa(lambda_rec, charges.q, charges.u, coefficients=up_coeffs),
        down=fn_effective_yukawa(lambda_rec, charges.q, charges.d, coefficients=down_coeffs),
    )


def fn_singular_masses(yukawa: jnp.ndarray) -> jnp.ndarray:
    """Return singular values of a generated Yukawa matrix, descending."""

    matrix = _validate_family_matrix(yukawa, "yukawa")
    return jnp.linalg.svd(matrix, compute_uv=False)


def fn_left_singular_frame(yukawa: jnp.ndarray) -> jnp.ndarray:
    """Return the left singular frame of a Yukawa matrix."""

    matrix = _validate_family_matrix(yukawa, "yukawa")
    left, _, _ = jnp.linalg.svd(matrix, full_matrices=True)
    return left


def fn_ckm_from_yukawas(up_yukawa: jnp.ndarray, down_yukawa: jnp.ndarray) -> jnp.ndarray:
    """Return the CKM-like left-frame mismatch generated by two Yukawas."""

    up_left = fn_left_singular_frame(up_yukawa)
    down_left = fn_left_singular_frame(down_yukawa)
    return jnp.swapaxes(jnp.conj(up_left), -1, -2) @ down_left


def fn_unitarity_residual(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return ``max_abs(U^dag U-I)`` for a square matrix."""

    arr = _validate_family_matrix(matrix, "matrix")
    return jnp.max(jnp.abs(jnp.swapaxes(jnp.conj(arr), -1, -2) @ arr - jnp.eye(SM_FAMILY_DIM, dtype=arr.dtype)))


def fn_recirculation_diagnostics(lambda_rec: float = FN_LAMBDA_WOLFENSTEIN) -> FNRecirculationDiagnostics:
    """Return focused Stage 5 FN recirculation diagnostics."""

    charges = DEFAULT_FN_QUARK_CHARGES
    up_exponents = fn_charge_exponents(charges.q, charges.u)
    down_exponents = fn_charge_exponents(charges.q, charges.d)
    expected_up = jnp.asarray(
        [
            [8, 5, 3],
            [7, 4, 2],
            [5, 2, 0],
        ],
        dtype=jnp.int32,
    )
    expected_down = jnp.asarray(
        [
            [4, 3, 3],
            [3, 2, 2],
            [1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    lambda_value = jnp.asarray(lambda_rec, dtype=jnp.float32)
    yukawas = fn_quark_yukawa_matrices(lambda_rec=lambda_rec, charges=charges)
    up_network = fn_recirculation_network(lambda_rec, charges.q, charges.u)
    down_network = fn_recirculation_network(lambda_rec, charges.q, charges.d)
    up_network_transfers = fn_recirculation_transfer_matrix(up_network)
    down_network_transfers = fn_recirculation_transfer_matrix(down_network)
    lambda_powers_up = lambda_value ** up_exponents
    lambda_powers_down = lambda_value ** down_exponents
    ckm = fn_ckm_from_yukawas(yukawas.up, yukawas.down)
    jit_yukawa = jax.jit(fn_effective_yukawa, static_argnames=("left_charges", "right_charges"))
    eager = fn_effective_yukawa(lambda_rec, charges.q, charges.u, coefficients=fn_default_coefficients("up"))
    jitted = jit_yukawa(lambda_rec, charges.q, charges.u, coefficients=fn_default_coefficients("up"))
    wolfenstein = fn_wolfenstein_scaling(charges.q, lambda_rec)
    unit_coefficients = jnp.eye(SM_FAMILY_DIM, dtype=jnp.complex64)
    diagonal_up = fn_effective_yukawa(lambda_rec, charges.q, charges.u, coefficients=unit_coefficients)
    diagonal_down = fn_effective_yukawa(lambda_rec, charges.q, charges.d, coefficients=unit_coefficients)
    expected_wolfenstein = jnp.asarray(lambda_rec, dtype=jnp.float32) ** jnp.asarray(
        [
            [0, 1, 3],
            [1, 0, 2],
            [3, 2, 0],
        ],
        dtype=jnp.int32,
    )

    return FNRecirculationDiagnostics(
        beam_splitter_unitarity_residual=fn_beam_splitter_unitarity_residual(lambda_rec),
        path_transfer_residual=fn_path_transfer_residual(lambda_rec),
        up_network_unitarity_residual=fn_recirculation_network_unitarity_residual(up_network),
        down_network_unitarity_residual=fn_recirculation_network_unitarity_residual(down_network),
        up_network_transfer_residual=jnp.max(jnp.abs(up_network_transfers - lambda_powers_up)),
        down_network_transfer_residual=jnp.max(jnp.abs(down_network_transfers - lambda_powers_down)),
        up_exponent_residual=jnp.max(jnp.abs(up_exponents - expected_up)),
        down_exponent_residual=jnp.max(jnp.abs(down_exponents - expected_down)),
        up_diagonal_scaling_residual=jnp.max(
            jnp.abs(jnp.diag(diagonal_up).real - lambda_value ** jnp.array(FN_EXPECTED_UP_DIAGONAL_EXPONENTS)),
        ),
        down_diagonal_scaling_residual=jnp.max(
            jnp.abs(jnp.diag(diagonal_down).real - lambda_value ** jnp.array(FN_EXPECTED_DOWN_DIAGONAL_EXPONENTS)),
        ),
        wolfenstein_scaling_residual=jnp.max(jnp.abs(wolfenstein - expected_wolfenstein)),
        ckm_unitarity_residual=fn_unitarity_residual(ckm),
        jit_delta=jnp.max(jnp.abs(jitted - eager)),
    )
