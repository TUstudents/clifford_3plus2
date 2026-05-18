# Session 32 - SU(2) Reversible Gauge Dynamics Prototype

Status: implemented as a compact SU(2) leapfrog-control audit.

## Question

Sessions 29-31 built the gauge-force ladder:

```text
Wilson plaquette -> Wilson action -> SU(2) left force -> compact descent
```

Session 32 asks whether the same force convention can support a reversible
Hamiltonian-style gauge update with compact links and Lie-algebra momenta:

```text
H_density(U, P) = 1/2 mean_link ||P[x,h]||^2 + beta S_W_density(U)
U[x,h] in SU(2)
P[x,h] in su(2) coordinates
```

This is still a simulation-control prototype, not a QCA-native Yang-Mills
time evolution.

## Implementation

Added `jax_gauge_dynamics.py`:

- `jax_su2_algebra_matrix(theta)`.
- `jax_su2_transform_momentum_field(momenta, site_gauge)`.
- `jax_su2_momentum_kinetic_energy_density(momenta)`.
- `jax_su2_gauge_hamiltonian_density(links, momenta, beta=..., shapes=...)`.
- `jax_su2_apply_momentum_update(links, momenta, step_size=...)`.
- `jax_su2_leapfrog_step(links, momenta, step_size=..., beta=..., shapes=...)`.

The leapfrog convention is:

```text
P_half = P - dt/2 * beta * F(U)
U_next = exp(dt * P_half) U
P_next = P_half - dt/2 * beta * F(U_next)
```

where `F(U)` is the Session 31 left-trivialized Wilson-action-density force.

## Verified

- SU(2) momentum coordinates lift to anti-Hermitian traceless matrices.
- Finite target-site adjoint gauge transforms preserve momentum kinetic
  energy density.
- Identity links with zero momenta are fixed by leapfrog.
- Pure-gauge links with zero momenta are fixed by leapfrog.
- Momentum updates and leapfrog updates preserve link unitarity and
  determinant `1`.
- One forward leapfrog step followed by a negative-time step recovers the
  original links and momenta within numerical tolerance.
- Hamiltonian-density drift is finite and smaller at smaller step size.
- Leapfrog is covariant under finite site-local gauge transforms of links and
  target-site adjoint transforms of momenta.
- The leapfrog helper is `jax.jit` compilable and shape-preserving.

## Interpretation

This is the first genuine dynamical gauge-field prototype in `spacetime_qca`:
links and their conjugate momenta evolve together, links stay compact, and the
update is reversible.  It is not yet the final physical gauge-field layer.

The next hard choices are:

- vectorized/staple SU(2) force for performance;
- SU(3), SU(4), or Pati-Salam force projection;
- Gauss-law constraints and projection;
- coupling evolving gauge links to the BCC Dirac fermion step;
- dynamical Higgs/Yukawa fields.
