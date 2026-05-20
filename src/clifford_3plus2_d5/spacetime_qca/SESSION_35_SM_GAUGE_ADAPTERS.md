# Session 35 - Pati-Salam and SM Gauge-Sector Adapters

Status: implemented as sector-generic chiral16 gauge adapters over the Session
34 basis-based compact Lie API.

## Question

Session 34 closed the Pati-Salam SU(4) force-projection gap for the exact
chiral16 carrier.  Session 35 asks whether the same memory-safe compact-link
machinery can address the full set of derived Pati-Salam and Standard Model
subsectors:

```text
Pati-Salam: SU(4), SU(2)_L, SU(2)_R, SU(4) x SU(2)_L x SU(2)_R
SM:         SU(3)_c, SU(2)_L, U(1)_Y, SU(3)_c x SU(2)_L x U(1)_Y
```

This is still a gauge-field infrastructure session.  It does not introduce
fermion backreaction, Gauss-law constraints, dynamical Higgs fields, or
physical coupling normalization.

## Implementation

Extended `jax_patisalam.py` with a sector-generic adapter layer:

- `jax_patisalam_generators_chiral16(sector, dtype=...)`.
- `jax_patisalam_algebra_matrix(theta, sector=...)`.
- `jax_patisalam_project_to_coordinates(matrix, sector=...)`.
- `jax_patisalam_link_from_algebra(theta, sector=...)`.
- `jax_patisalam_link_field_from_algebra(theta, sector=...)`.
- `jax_patisalam_site_field_from_algebra(site_theta, sector=...)`.
- `jax_patisalam_pure_gauge_links_from_site_algebra(site_theta, sector=...)`.
- `jax_patisalam_left_force(..., sector=...)`.
- `jax_patisalam_action_descent_step(..., sector=...)`.
- `jax_patisalam_transform_momentum_field(..., sector=...)`.
- `jax_patisalam_momentum_kinetic_energy_density(..., sector=...)`.
- `jax_patisalam_gauge_hamiltonian_density(..., sector=...)`.
- `jax_patisalam_apply_momentum_update(..., sector=...)`.
- `jax_patisalam_leapfrog_step(..., sector=...)`.

Supported sectors:

```text
su4         dimension 15
su2_l       dimension 3
su2_r       dimension 3
pati_salam  dimension 21
su3_c       dimension 8
u1_y        dimension 1   # physical Session 19b hypercharge after Session 41
u1_y_raw    dimension 1   # unnormalized Pati-Salam regression alias
sm          dimension 12  # SU(3)_c + SU(2)_L + physical Y
sm_raw      dimension 12  # old raw-Y regression alias
```

Existing `jax_patisalam_su4_*` APIs remain stable and now delegate to the
sector-generic implementation with `sector="su4"`.

## Verified

- Every sector has shape `(dim, 32, 32)`, anti-Hermitian generators, and a
  full-rank Gram matrix.
- Gram-matrix projection recovers deterministic coordinates for every sector.
- Compact exponentials preserve unitarity for every sector.
- Independent SM factor links commute for `SU(3)_c`, `SU(2)_L`, and `U(1)_Y`.
- Pure gauges have zero Wilson action for `SU(2)_L`, `SU(2)_R`, `SU(3)_c`,
  `U(1)_Y`, and the combined `SM` basis.
- Finite-difference left force is finite and lowers a non-flat Wilson action
  for representative sectors `SU(2)_L`, `SU(3)_c`, and `U(1)_Y`.
- Target-site adjoint momentum transforms preserve Gram-metric kinetic energy
  for `SU(2)_L`, `SU(3)_c`, `U(1)_Y`, and `SM`.
- Identity links with zero momenta are fixed by leapfrog for `SU(2)_L`,
  `SU(3)_c`, and `U(1)_Y`.
- Compact momentum updates preserve link unitarity for the combined `SM`
  basis.

## Interpretation

Session 35 gives `spacetime_qca` a uniform compact gauge adapter for the full
Pati-Salam and Standard Model gauge-content layer already derived in `lepton`.
The implementation stays representation-aware: every sector uses the explicit
chiral16 generator basis and the Session 34 Gram-metric projection rather than
assuming a fundamental SU(N) representation.

The combined `pati_salam` and `sm` sectors are representation-level convenience
bases.  Separate physical coupling constants, Gauss-law projection, and
fermion-coupled gauge dynamics remain future sessions.
