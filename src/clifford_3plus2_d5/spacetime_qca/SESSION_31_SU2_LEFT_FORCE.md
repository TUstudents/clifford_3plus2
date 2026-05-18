# Session 31 - Left-Trivialized SU(2) Force and Compact Descent

Status: implemented as a compact-link force convention audit.

## Question

Session 30 differentiated the Wilson action with respect to unconstrained
coordinates `theta` in links of the form `U = exp(theta_a T_a)`.  That is a
useful coordinate-gradient check, but compact gauge dynamics normally wants a
force in the tangent space at each current link:

```text
U[x,h] -> exp(omega_a[x,h] T_a) U[x,h]
```

Session 31 asks whether the BCC Wilson-action layer can expose that
left-trivialized force and use it for a compact action-descent step without
leaving `SU(2)`.

## Implementation

Added to `jax_gauge_force.py`:

- `jax_su2_project_antihermitian_to_algebra(matrix)`.
- `jax_su2_left_force(links, shapes=...)`.
- `jax_su2_left_force_from_algebra(theta, shapes=...)`.
- `jax_su2_apply_left_update(links, force, step_size=...)`.
- `jax_su2_action_descent_step(links, step_size=..., shapes=...)`.

The generator convention remains:

```text
T_a = -i sigma_a / 2
```

Projection uses:

```text
theta_a = 2 Re Tr(T_a^dagger A)
```

so `A = theta_a T_a` projects back to `theta`.

The left force is computed by differentiating the Wilson action with respect
to compact left perturbations.  A finite-difference directional test pins the
same convention.  This is intentionally still an audit path, not a production
HMC/staple-force kernel.

## Verified

- Projection recovers SU(2) generator coordinates.
- Zero links have zero left force.
- Finite pure-gauge SU(2) links have zero left force.
- The left force matches a centered finite difference along a deterministic
  left-perturbation direction.
- A deterministic non-flat SU(2) field has non-zero left force.
- Compact descent lowers the Wilson action on the deterministic non-flat
  field.
- Compact left updates preserve link unitarity and determinant `1` to
  numerical precision.
- The descended action remains invariant after finite site-local gauge
  transforms.
- The left-force helper is `jax.jit` compilable on the default JAX device.

## Interpretation

This closes the first compact nonabelian gauge-update audit:

```text
curvature observable -> Wilson action -> compact left force -> compact descent
```

The update is a deterministic steepest-descent control, not physical gauge
time evolution.  It establishes the group-valued tangent-space convention
needed before leapfrog/HMC, heatbath, or a QCA-native gauge-field update can be
designed.

## Still open

- Vectorized staple formula for the SU(2) force.
- Symplectic / reversible gauge-field update rather than action descent.
- SU(3), SU(4), and Pati-Salam force projections.
- Dynamical gauge links coupled to the fermion walk.
- Long-time numerical stability and performance benchmarks.
