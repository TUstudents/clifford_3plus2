# Session 36 — No-Backreaction Fermion/Gauge Coupling

## Goal

Connect the finite BCC Dirac fermion step to the Pati-Salam/SM compact gauge
link dynamics from Sessions 34-35.

This session intentionally implements the conservative v1 coupling:

1. Evolve gauge links and link momenta with the existing pure-gauge leapfrog.
2. Transport the fermion state through the updated links.
3. Do not feed a fermion current back into the gauge momentum update.

The result is a no-backreaction background-gauge simulation wrapper, not a full
fermion-coupled Yang-Mills QCA.

## Implemented API

- `jax_patisalam_dirac_step(state, links)`
  - Validates the chiral16 Pati-Salam layout.
  - Applies the BCC Dirac pull step through `(nx, ny, nz, 8, 32, 32)` links.

- `jax_transform_patisalam_dirac_state(state, site_gauge)`
  - Applies site-local internal gauge transforms to
    `(nx, ny, nz, 4, 32)` Dirac/internal states.

- `jax_patisalam_fermion_gauge_step(...)`
  - Runs one pure-gauge Pati-Salam/SM leapfrog update.
  - Then runs one fermion Dirac step through the updated links.
  - Returns `(updated_state, updated_links, updated_momenta)`.

- `jax_patisalam_fermion_gauge_energy_density(...)`
  - Reports fermion norm plus the pure-gauge Hamiltonian density.

## Convention

The coupled step uses this ordering:

```text
(links, momenta) -> pure gauge leapfrog
state            -> BCC Dirac step through updated links
```

The fermion state is a spectator for gauge dynamics.  This makes the update
useful for background-gauge transport and stability experiments while keeping
Gauss-law/backreaction work explicitly out of scope.

## Tests

The Session 36 tests verify:

- identity links and zero momenta reduce the wrapper to the ordinary Dirac
  step;
- a zero fermion state leaves gauge evolution identical to pure-gauge
  leapfrog;
- the Pati-Salam Dirac step is site-local gauge covariant;
- the zero-time coupled wrapper is gauge covariant;
- diagnostics report finite fermion norm and gauge Hamiltonian density.

The covariance tests use the exact pull-link convention:

```text
U[x,h] -> G[x] U[x,h] G[x+h]^dagger
psi[x] -> G[x] psi[x]
P[x,h] -> G[x] P[x,h] G[x]^dagger
```

## Interpretation

This is the first executable bridge between the fermion transport backend and
the evolving Pati-Salam/SM link backend.  It closes the immediate integration
gap after Session 35.

It does not yet implement:

- fermion current backreaction;
- Gauss-law constraints;
- fermion Hamiltonian energy diagnostics;
- dynamical Higgs/Yukawa coupling in the time step;
- production-scale vectorization or stability benchmarking.

Those remain the next dynamics-level tasks.
