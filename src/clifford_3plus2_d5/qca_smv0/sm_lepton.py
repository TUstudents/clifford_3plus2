"""Lepton-sector helpers for QCA_SMv0.

The simulator convention here is deliberately production-facing:

* charged leptons are diagonal by default;
* direct neutrino mode uses ``Y_nu = U_PMNS diag(m_nu)``;
* Schur/seesaw mode builds a heavy sterile block whose Schur complement has
  the same low-energy neutrino spectrum.

This module does not claim to derive PMNS or the seesaw data.  It provides the
explicit local matrices that the QCA production collision can consume and the
checks that those matrices encode the requested lepton physics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


DEFAULT_PMNS_ANGLES = (
    0.5836381018669037,  # theta_12 = 33.44 deg
    0.8587019919812102,  # theta_23 = 49.20 deg
    0.14957417646704585,  # theta_13 = 8.57 deg
    3.4033920413889422,  # delta = 195 deg
)
DEFAULT_STERILE_MAJORANA_MASSES = (1.0, 1.0, 1.0)


class LeptonSeesawSchurData(NamedTuple):
    """Concrete type-I seesaw data and its low-energy Schur complement."""

    dirac_matrix: jnp.ndarray
    sterile_majorana: jnp.ndarray
    effective_majorana: jnp.ndarray
    target_masses: jnp.ndarray
    recovered_masses: jnp.ndarray
    spectrum_residual: jnp.ndarray


def sm_pmns_matrix(
    theta12: float,
    theta23: float,
    theta13: float,
    delta_cp: float,
    *,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Return the PDG Dirac-phase PMNS matrix with zero Majorana phases."""

    c12 = jnp.asarray(jnp.cos(theta12), dtype=dtype)
    s12 = jnp.asarray(jnp.sin(theta12), dtype=dtype)
    c23 = jnp.asarray(jnp.cos(theta23), dtype=dtype)
    s23 = jnp.asarray(jnp.sin(theta23), dtype=dtype)
    c13 = jnp.asarray(jnp.cos(theta13), dtype=dtype)
    s13 = jnp.asarray(jnp.sin(theta13), dtype=dtype)
    phase_pos = jnp.exp(1j * jnp.asarray(delta_cp, dtype=dtype))
    phase_neg = jnp.conj(phase_pos)
    return jnp.asarray(
        [
            [c12 * c13, s12 * c13, s13 * phase_neg],
            [-s12 * c23 - c12 * s23 * s13 * phase_pos, c12 * c23 - s12 * s23 * s13 * phase_pos, s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * phase_pos, -c12 * s23 - s12 * c23 * s13 * phase_pos, c23 * c13],
        ],
        dtype=dtype,
    )


def sm_default_pmns_matrix(*, dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return the benchmark nontrivial PMNS matrix used by canonical probes."""

    return sm_pmns_matrix(*DEFAULT_PMNS_ANGLES, dtype=dtype)


def sm_pmns_unitarity_residual(pmns: jnp.ndarray) -> jnp.ndarray:
    """Return ``max(abs(U^dagger U - I))`` for a PMNS candidate."""

    matrix = jnp.asarray(pmns, dtype=jnp.complex64)
    return jnp.max(jnp.abs(matrix.conj().T @ matrix - jnp.eye(3, dtype=jnp.complex64)))


def sm_lepton_direct_pmns_neutrino_yukawa(
    neutrino_masses: tuple[float, float, float] | jnp.ndarray,
    pmns: jnp.ndarray | None = None,
    *,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Return the direct effective neutrino Yukawa matrix ``U_PMNS diag(m)``."""

    masses = jnp.asarray(neutrino_masses, dtype=dtype)
    matrix = sm_default_pmns_matrix(dtype=dtype) if pmns is None else jnp.asarray(pmns, dtype=dtype)
    return matrix @ jnp.diag(masses)


def sm_lepton_pmns_expected_transfer_weights(
    neutrino_masses: tuple[float, float, float] | jnp.ndarray,
    pmns: jnp.ndarray | None = None,
    *,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Return row-normalized ``|U_PMNS diag(m)|^2`` transfer weights."""

    yukawa = sm_lepton_direct_pmns_neutrino_yukawa(neutrino_masses, pmns, dtype=dtype)
    weights = jnp.abs(yukawa) ** 2
    row_sums = jnp.sum(weights, axis=1, keepdims=True)
    return jnp.where(row_sums > 0.0, weights / row_sums, weights)


def sm_lepton_seesaw_schur_from_pmns(
    neutrino_masses: tuple[float, float, float] | jnp.ndarray,
    pmns: jnp.ndarray | None = None,
    *,
    sterile_majorana_masses: tuple[float, float, float] | jnp.ndarray = DEFAULT_STERILE_MAJORANA_MASSES,
    dtype: jnp.dtype = jnp.complex64,
) -> LeptonSeesawSchurData:
    """Build a type-I seesaw whose Schur complement has the target spectrum.

    With diagonal heavy sterile block ``M_R`` and
    ``m_D = U_PMNS.conj() diag(sqrt(m_i M_i))``, the low-energy Schur
    complement is ``-m_D M_R^-1 m_D^T``.  Its Takagi/singular spectrum is the
    target neutrino mass vector, including an exact zero if ``m_1=0``.
    """

    masses = jnp.asarray(neutrino_masses, dtype=jnp.float32)
    heavy = jnp.asarray(sterile_majorana_masses, dtype=jnp.float32)
    if masses.shape != (3,) or heavy.shape != (3,):
        raise ValueError("neutrino and sterile Majorana masses must be length-3 vectors")
    matrix = sm_default_pmns_matrix(dtype=dtype) if pmns is None else jnp.asarray(pmns, dtype=dtype)
    sqrt_scales = jnp.sqrt(jnp.maximum(masses * heavy, 0.0)).astype(dtype)
    dirac = matrix.conj() @ jnp.diag(sqrt_scales)
    sterile = jnp.diag(heavy.astype(dtype))
    effective = -(dirac @ jnp.diag((1.0 / heavy).astype(dtype)) @ dirac.T)
    recovered = jnp.sort(jnp.linalg.svd(effective, compute_uv=False).real)
    target = jnp.sort(masses)
    return LeptonSeesawSchurData(
        dirac_matrix=dirac,
        sterile_majorana=sterile,
        effective_majorana=effective,
        target_masses=target,
        recovered_masses=recovered,
        spectrum_residual=jnp.max(jnp.abs(recovered - target)),
    )
