"""Session 21 mass-layer tests for the spacetime QCA package."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_patisalam import (
    su2_l_generators_from_spin04,
    su2_r_generators_from_spin04,
    su4_generators_from_spin06,
)
from clifford_3plus2_d5.lepton.patisalam_sm import su3_c_generators_from_su4
from clifford_3plus2_d5.lepton.sm_hypercharge import physical_hypercharge_generator
from clifford_3plus2_d5.spacetime_qca import (
    beta_anticommutes_with_alpha,
    beta_matrix,
    k0_scalar_mass_spectrum,
    mass_compatibility_audit_payload,
    mass_hamiltonian,
    mass_preserves_gauge,
    massive_dirac_hamiltonian,
    projector_control_mass,
    same_matrix,
    scalar_internal_mass,
    scalar_mass_dispersion_valid,
    scalar_mass_squared_residual,
)


def _patisalam_generators() -> tuple[sp.Matrix, ...]:
    return (
        *su4_generators_from_spin06(),
        *su2_l_generators_from_spin04(),
        *su2_r_generators_from_spin04(),
    )


def _sm_generators() -> tuple[sp.Matrix, ...]:
    return (
        *su3_c_generators_from_su4(),
        *su2_l_generators_from_spin04(),
        physical_hypercharge_generator(),
    )


def test_beta_is_valid_mass_matrix_for_dirac_hamiltonian() -> None:
    beta = beta_matrix()
    assert same_matrix(beta * beta, sp.eye(4))
    assert beta_anticommutes_with_alpha()


def test_mass_hamiltonian_has_expected_shape() -> None:
    mass = sp.symbols("m")
    internal_mass = scalar_internal_mass(3, mass)
    assert mass_hamiltonian(internal_mass).shape == (12, 12)


def test_scalar_mass_squared_is_relativistic_dispersion() -> None:
    kx, ky, kz, mass = sp.symbols("kx ky kz m")
    residual = scalar_mass_squared_residual(kx, ky, kz, mass, internal_dim=2)
    assert residual == sp.zeros(8)
    assert scalar_mass_dispersion_valid(kx, ky, kz, mass, internal_dim=2)


def test_k0_scalar_mass_spectrum_has_expected_multiplicity() -> None:
    mass = sp.symbols("m")
    assert k0_scalar_mass_spectrum(mass, internal_dim=2) == {mass: 4, -mass: 4}


def test_massive_dirac_hamiltonian_lifts_internal_dimension() -> None:
    k, mass = sp.symbols("k m")
    hamiltonian = massive_dirac_hamiltonian(k, 0, 0, scalar_internal_mass(32, mass))
    assert hamiltonian.shape == (128, 128)


def test_scalar_mass_preserves_patisalam_and_sm_gauge_generators() -> None:
    mass = sp.symbols("m")
    internal_mass = scalar_internal_mass(32, mass)
    assert mass_preserves_gauge(internal_mass, _patisalam_generators())
    assert mass_preserves_gauge(internal_mass, _sm_generators())


def test_nonscalar_projector_control_breaks_sm_gauge_generators() -> None:
    control = projector_control_mass(32, 16)
    assert not mass_preserves_gauge(control, _sm_generators())


def test_mass_compatibility_payload_records_yukawa_boundary() -> None:
    payload = mass_compatibility_audit_payload(
        patisalam_generators=_patisalam_generators(),
        sm_generators=_sm_generators(),
    )
    assert payload.scalar_mass_dirac_valid
    assert payload.scalar_mass_patisalam_preserving
    assert payload.scalar_mass_sm_preserving
    assert payload.nonscalar_control_breaks_sm
    assert "not an SM Yukawa" in payload.interpretation
