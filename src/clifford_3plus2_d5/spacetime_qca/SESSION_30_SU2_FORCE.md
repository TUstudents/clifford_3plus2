# Session 30 — SU(2) Wilson-Force Audit

## Status

Complete.

This session extends the Session 29 SO(2) Wilson-gradient audit to the
smallest compact nonabelian group: fundamental SU(2).  The goal is to verify
that compact nonabelian link coordinates can feed the existing JAX Wilson
action, produce finite gradients, preserve finite gauge invariance, and pass
finite-difference checks.

This is still not a dynamical gauge update.  The gradient is with respect to
the chosen Lie-algebra coordinates in `exp(theta_a T_a)`, not yet a
left-trivialized HMC force or a leapfrog/heatbath evolution rule.

## Implemented

- Fundamental SU(2) generators `T_a = -i sigma_a / 2`.
- Closed-form compact SU(2) exponential from algebra coordinates.
- Stable small-angle branch using `||theta||^2`, avoiding NaN gradients at
  zero links.
- BCC SU(2) link fields from `(nx, ny, nz, 8, 3)` algebra coordinates.
- Site-local SU(2) fields from `(nx, ny, nz, 3)` algebra coordinates.
- Finite link-field gauge transform in the existing pull convention:
  `U[x,h] -> G[x] U[x,h] G[x+h]^dagger`.
- Pure-gauge SU(2) links generated from site-local gauges.
- SU(2) Wilson action density and coordinate gradient through `jax.grad`.

## Results

- SU(2) generators are anti-Hermitian, traceless, and close with the expected
  cyclic commutator convention.
- Zero algebra coordinates produce identity links.
- Representative SU(2) links are unitary with determinant `1`.
- The exponential linearizes correctly as `I + theta_a T_a` at small angle.
- Zero fields have zero action and zero gradient.
- Cartan-subgroup pure-gauge coordinates have zero action and zero gradient.
- Full finite pure-gauge links have zero action.
- A deterministic non-flat SU(2) field has positive action and non-zero
  gradient.
- The gradient matches a centered directional finite difference.
- Finite site-local SU(2) gauge transforms preserve the Wilson action density.
- The SU(2) gradient helper is `jax.jit` compilable.

## Interpretation

The Wilson-action layer now supports a compact nonabelian force audit.  The
important checkpoint is not the group choice itself, but the fact that the
existing BCC plaquette action differentiates cleanly through noncommuting
compact links and respects finite site-local gauge invariance.

The next dynamical step should not skip force geometry: full SU(3) or
Pati-Salam force projection still needs either a larger compact-link
parameterization or a matrix-group update convention.

## Still Open

- Left-trivialized nonabelian force projection.
- SU(3), SU(4), or Pati-Salam link parameterizations.
- Gauge-field time evolution, such as leapfrog/HMC or heatbath-style updates.
- Coupling a dynamical gauge update to the BCC Dirac step.
- Large-lattice performance work.
