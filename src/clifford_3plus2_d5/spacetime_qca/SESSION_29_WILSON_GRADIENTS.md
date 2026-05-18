# Session 29 — Wilson-Action Gradients

## Status

Complete.

This session adds the first differentiable gauge-force audit for the BCC
Wilson action.  It is deliberately abelian: every BCC link is an SO(2)
rotation parameterized by one real angle.  The goal is not a dynamical gauge
update yet; the goal is to verify that the JAX Wilson action can be
differentiated and that the resulting force has the expected flat-field,
pure-gauge, and finite-difference behavior.

## Implemented

- `jax_gauge_force.py`
  - `jax_so2_rotation(theta)`
  - `jax_so2_link_field_from_angles(theta)`
  - `jax_so2_pure_gauge_angles(site_angles)`
  - `jax_so2_wilson_action_density(theta, shapes=None)`
  - `jax_so2_wilson_action_gradient(theta, shapes=None)`
  - `jax_centered_finite_difference(fn, theta, direction, epsilon=...)`
- Public exports through `spacetime_qca.__init__`.
- Session tests in `tests/test_jax_gauge_force.py`.

## Convention

The link array stores pull-form BCC links:

```text
U[x, h] : source x+h -> target x
```

For site gauge angles `phi[x]`, the SO(2) pure-gauge link is:

```text
theta[x, h] = phi[x] - phi[x+h]
U[x, h] = R(theta[x, h])
```

This matches the existing exact and JAX link-field convention used by the
real-space BCC step and Wilson plaquette code.

## Results

- Zero SO(2) angle fields have zero Wilson action density and zero gradient.
- Pure-gauge SO(2) angle fields have zero Wilson action density and zero
  gradient.
- A deterministic non-flat SO(2) field has positive action density and
  non-zero gradient.
- The JAX gradient matches a centered directional finite difference.
- The action gradient is orthogonal to pure-gauge perturbation directions,
  verifying the infinitesimal gauge-invariance check in the abelian testbed.
- The gradient helper is `jax.jit` compilable and preserves the angle-array
  shape.

## Interpretation

This proves the numerical Wilson-action layer is differentiable in the compact
link variables and that the resulting force behaves correctly on the minimal
SO(2) gauge group.  It is a precondition for dynamical gauge-field updates,
but it is not itself a gauge-field time evolution rule.

## Still Open

- Nonabelian SU(N) link parameterization and force projection.
- A symplectic/leapfrog or other unitary gauge-field update.
- Gauge-field dynamics coupled to the BCC Dirac fermion step.
- Numerical performance benchmarks on larger lattices.
