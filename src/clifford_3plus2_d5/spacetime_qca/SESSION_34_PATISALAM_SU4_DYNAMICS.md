# Session 34 - Basis-Based Pati-Salam SU(4) Compact Dynamics

Status: implemented as a public basis-based compact Lie dynamics layer plus a
chiral16 SU(4) Pati-Salam adapter.

## Question

Sessions 29-33 built compact Wilson-force and leapfrog controls for SO(2),
fundamental SU(2), and fundamental SU(3).  Session 34 asks whether the same
compact-link convention can be lifted to the Pati-Salam color factor that the
`lepton` package derives from the exact chiral16 carrier:

```text
Spin(0,6) ~= SU(4)
su4 basis dimension = 15
representation matrix size = 32 x 32 real-skew / anti-Hermitian
```

This is not fundamental SU(4).  Treating it as generic `SU(N)` would be
wrong: `N = 32` for the matrix representation, but the algebra has dimension
15.  The correct API is therefore basis-based.

## Implementation

Extended `jax_gauge_force.py` with generic compact Lie helpers:

- `jax_compact_lie_algebra_matrix(theta, generators)`.
- `jax_compact_lie_project_to_coordinates(matrix, generators)`.
- `jax_compact_lie_link_from_algebra(theta, generators)`.
- `jax_compact_lie_link_field_from_algebra(theta, generators)`.
- `jax_compact_lie_site_field_from_algebra(site_theta, generators)`.
- `jax_compact_lie_pure_gauge_links_from_site_algebra(site_theta, generators)`.
- `jax_compact_lie_left_force(..., method="autodiff" | "finite_difference")`.
- `jax_compact_lie_left_force_from_algebra(...)`.
- `jax_compact_lie_apply_left_update(...)`.
- `jax_compact_lie_action_descent_step(...)`.

The projection uses the real Hilbert-Schmidt Gram matrix:

```text
G_ab = Re Tr(T_a^dagger T_b)
G theta = Re Tr(T_a^dagger A)
```

so non-fundamental and non-unit-normalized bases are handled correctly.

Extended `jax_gauge_dynamics.py` with generic basis-coordinate dynamics:

- `jax_compact_lie_transform_momentum_field(...)`.
- `jax_compact_lie_momentum_kinetic_energy_density(...)`.
- `jax_compact_lie_gauge_hamiltonian_density(...)`.
- `jax_compact_lie_apply_momentum_update(...)`.
- `jax_compact_lie_leapfrog_step(...)`.

The generic momentum kinetic energy uses the same Gram metric, not a naive
coordinate Euclidean norm.  This matters for non-fundamental representations.

Added `jax_patisalam.py` as a thin adapter over `lepton`:

- `jax_patisalam_su4_generators_chiral16()`.
- `jax_patisalam_su4_*` wrappers for algebra, projection, links, pure gauges,
  left force, descent, momentum transforms, Hamiltonian density, momentum
  update, and leapfrog.

The SU(4) adapter defaults to `method="finite_difference"` for left forces.
This is deliberate.  It avoids the memory failure mode caused by reverse-mode
differentiation through many `32 x 32` matrix exponentials.

## Verified

- Generic projection recovers coordinates for non-unit generator norms.
- Generic link exponentials preserve compactness.
- Generic pure gauges have zero Wilson action and zero left force in the SU(2)
  control.
- Generic compact descent lowers a non-flat Wilson action in the SU(2) control.
- Generic target-site adjoint momentum transforms preserve Gram-metric kinetic
  energy.
- Chiral16 SU(4) generators have shape `(15, 32, 32)`, are anti-Hermitian, and
  have a non-degenerate Gram matrix.
- Chiral16 SU(4) projection recovers all 15 basis coordinates.
- Chiral16 SU(4) exponentials preserve unitarity.
- Chiral16 SU(4) pure-gauge fields have zero Wilson action.
- Chiral16 SU(4) finite-difference left force is finite, non-zero on a
  non-flat test field, and compact descent lowers Wilson action.
- Chiral16 SU(4) target-site adjoint momentum transforms preserve
  Gram-metric kinetic energy.
- Chiral16 SU(4) compact momentum update and leapfrog preserve compact links.
- Identity chiral16 SU(4) links with zero momenta are fixed by leapfrog.
- Chiral16 SU(4) link and momentum transforms use the same pull-link
  convention as earlier sessions.

## Interpretation

Session 34 closes the Pati-Salam SU(4) force-projection gap without making the
JAX stack pretend that every compact group is a fundamental SU(N).  The public
generic layer now accepts an explicit generator basis, and the Pati-Salam
adapter reuses the exact chiral16 SU(4) basis from `lepton`.

This is still a compact gauge-dynamics control, not a full Standard Model
simulation.  The remaining hard pieces are:

- SU(2)_L and U(1)_Y / full SM adapters on the same basis API.
- Fermion-coupled gauge-link evolution and Gauss-law constraints.
- Dynamical Higgs/Yukawa fields.
- Long-time numerical stability and performance benchmarks.

The key engineering result is that large-representation compact dynamics now
has a memory-safe finite-difference force path and a reusable basis API.
