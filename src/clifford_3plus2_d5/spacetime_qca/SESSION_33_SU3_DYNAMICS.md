# Session 33 - SU(3) Compact Force and Reversible Dynamics

Status: implemented as a compact SU(3) force and leapfrog-control audit.

## Question

Sessions 29-32 built the gauge-force ladder through SU(2):

```text
Wilson plaquette -> Wilson action -> SU(2) left force -> compact descent
-> reversible leapfrog
```

Session 33 asks whether the same compact-link convention works for fundamental
SU(3) color links:

```text
U[x,h] in SU(3)
P[x,h] in su(3) coordinates
H_density(U, P) = 1/2 mean_link ||P[x,h]||^2 + beta S_W_density(U)
```

This is a color-gauge simulation-control prototype.  It is not yet coupled to
fermions, Gauss-law projected, or generalized to SU(4) / Pati-Salam links.

## Implementation

Extended `jax_gauge_force.py` with SU(3) compact-link helpers:

- `jax_su3_generators()` using anti-Hermitian Gell-Mann generators
  `T_a = -i lambda_a / 2`.
- `jax_su3_project_antihermitian_to_algebra(matrix)`.
- `jax_su3_algebra_matrix(theta)`.
- `jax_su3_link_from_algebra(theta)` using batched matrix exponentials.
- `jax_su3_link_field_from_algebra(theta)`.
- `jax_su3_site_field_from_algebra(site_theta)`.
- `jax_su3_pure_gauge_links_from_site_algebra(site_theta)`.
- `jax_su3_wilson_action_density(theta, shapes=...)`.
- `jax_su3_left_force(links, shapes=...)`.
- `jax_su3_left_force_from_algebra(theta, shapes=...)`.
- `jax_su3_apply_left_update(links, force, step_size=...)`.
- `jax_su3_action_descent_step(links, step_size=..., shapes=...)`.

Extended `jax_gauge_dynamics.py` with SU(3) dynamics helpers:

- `jax_su3_algebra_matrix(theta)`.
- `jax_su3_transform_momentum_field(momenta, site_gauge)`.
- `jax_su3_momentum_kinetic_energy_density(momenta)`.
- `jax_su3_gauge_hamiltonian_density(links, momenta, beta=..., shapes=...)`.
- `jax_su3_apply_momentum_update(links, momenta, step_size=...)`.
- `jax_su3_leapfrog_step(links, momenta, step_size=..., beta=..., shapes=...)`.

The leapfrog convention mirrors Session 32:

```text
P_half = P - dt/2 * beta * F(U)
U_next = exp(dt * P_half) U
P_next = P_half - dt/2 * beta * F(U_next)
```

where `F(U)` is the SU(3) left-trivialized Wilson-action-density force.

## Verified

- SU(3) generators are anti-Hermitian, traceless, and normalized with
  `Tr(T_a^dagger T_b) = 1/2 delta_ab`.
- Projection from anti-Hermitian traceless matrices recovers generator
  coordinates.
- SU(3) exponentials preserve unitarity and determinant `1`.
- Zero fields have zero Wilson action and zero left force.
- Finite pure-gauge fields have zero Wilson action and zero left force.
- The SU(3) left force matches a centered compact left directional finite
  difference.
- Non-flat SU(3) fields have non-zero left force, and compact descent lowers
  Wilson action.
- SU(3) momentum coordinates lift to anti-Hermitian traceless matrices.
- Target-site adjoint momentum transforms preserve kinetic energy density.
- Identity and finite pure-gauge links with zero momenta are fixed by
  leapfrog.
- Momentum updates and leapfrog updates preserve compact SU(3) links.
- A forward leapfrog step followed by a negative-time step recovers the
  original links and momenta within numerical tolerance.
- Hamiltonian-density drift is finite and smaller at smaller step size.
- Leapfrog is covariant under finite site-local gauge transforms of links and
  target-site adjoint transforms of momenta.
- The SU(3) compact momentum update is `jax.jit` compilable and
  shape-preserving.

## Interpretation

This is the first compact SU(3) color-gauge dynamics prototype in
`spacetime_qca`.  It shows that the Wilson-force and reversible leapfrog
convention used for SU(2) survives the move to fundamental SU(3) links.

The current SU(3) force uses reverse-mode autodiff through compact
left-perturbations and batched `3 x 3` matrix exponentials.  This is correct
for an audit, but expensive.  Default tests intentionally avoid JIT-compiling
the full SU(3) leapfrog force graph; that should wait for a vectorized staple
force or a custom compact SU(3) exponential path.

The remaining hard physics layers are still open:

- Gauss-law constraints and projection.
- Coupling evolving color links to the BCC Dirac fermion step.
- SU(4) / Pati-Salam compact force projection.
- Dynamical Higgs/Yukawa fields.
- Long-time numerical stability and performance benchmarks.
