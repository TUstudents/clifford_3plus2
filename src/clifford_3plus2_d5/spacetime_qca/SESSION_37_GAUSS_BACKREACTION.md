# Session 37 — Gauss Law And Fermion Backreaction Prototype

## Goal

Move beyond the Session 36 no-backreaction wrapper by adding the first
matter/gauge constraint layer:

- a lattice electric-divergence residual in the current BCC pull-link
  convention;
- a fermion charge density in Pati-Salam/SM generator coordinates;
- an explicit matter-current kick for link momenta;
- a sourced wrapper that reduces to Session 36 when the matter coupling is
  zero.

This remains a prototype.  It is not a full Gauss-law projector, not a
production HMC integrator, and not a quantum-current uniqueness claim.

## Implemented API

New module:

```text
jax_gauss.py
```

Public functions:

- `jax_patisalam_momentum_algebra(momenta, sector=...)`
  - Converts basis-coordinate momenta to anti-Hermitian `32 x 32` matrices.

- `jax_patisalam_electric_divergence(links, momenta, sector=...)`
  - Computes
    `sum_h (P[x,h] - U[x+h,-h]^dagger P[x+h,-h] U[x+h,-h])`
    and projects back to sector coordinates.

- `jax_patisalam_fermion_charge_density(state, sector=..., coordinate_mode=...)`
  - Computes site charge density from Hermitian observables `Q_a = i T_a`.
  - Supports `raw` moments and `gram_dual` coordinates.

- `jax_patisalam_gauss_residual(state, links, momenta, sector=..., matter_coupling=...)`
  - Returns `divE - g rho`.

- `jax_patisalam_fermion_link_current(state, links, sector=..., epsilon=...)`
  - Computes a finite-difference current from the left variation of
    `Re <psi, D_U psi>`.

- `jax_patisalam_apply_fermion_backreaction(momenta, current, step_size=..., matter_coupling=...)`
  - Applies the explicit source kick
    `P -> P + dt * g * J`.

- `jax_patisalam_fermion_gauge_step_with_backreaction(...)`
  - Computes current, kicks momenta, calls the existing pure-gauge leapfrog,
    then transports the fermion state through the updated links.

## Convention

The electric-divergence convention is target-site covariant:

```text
U[x,h] maps x+h -> x
P[x,h] transforms at x

divE[x] =
  sum_h ( P[x,h]
        - U[x+h,-h]^dagger P[x+h,-h] U[x+h,-h] )
```

The charge convention is:

```text
Q_a = i T_a
rho_raw_a[x] = sum_s psi[x,s]^dagger Q_a psi[x,s]
rho = Gram^-1 rho_raw
```

The default Gauss residual uses `gram_dual` coordinates so it matches the
basis-coordinate momentum convention from the compact gauge dynamics stack.

## Tests

Session 37 adds focused tests for:

- momentum algebra shape and anti-Hermiticity;
- zero-momentum electric divergence;
- gauge covariance of electric divergence;
- zero-state fermion charge;
- deterministic raw `U(1)_Y` charge formula;
- gauge covariance of nonabelian charge density;
- Gauss residual reductions:
  - zero fermion state gives `divE`;
  - zero momenta gives `-g rho`;
- gauge covariance of the Gauss residual;
- zero-state link current;
- finite `U(1)_Y`-invariant and nonabelian-covariant finite-difference link
  current controls;
- linear/no-op backreaction kick;
- zero-coupling sourced step exactly matching the Session 36 wrapper.

Focused run:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_gauss_backreaction.py -q
```

Result:

```text
14 passed
```

## Interpretation

The module now has the first executable constrained-coupling layer:

- gauge momenta can be inspected through a target-site Gauss residual;
- fermion states produce sector charge densities;
- fermion transport can source a link-momentum kick;
- the sourced wrapper has the expected zero-coupling regression to the
  no-backreaction Session 36 step.

This is enough to proceed to the Hermitian Yukawa operator in Session 38
without leaving the gauge layer completely unconstrained.

## Remaining Gaps

Still open:

- full Gauss-law projection / constraint solving;
- analytic matter current replacing the finite-difference prototype;
- physical coupling constants for independent gauge factors;
- long-time stability of sourced evolution;
- dynamical Higgs/Yukawa fields;
- anomaly/current diagnostics beyond the tiny-lattice covariance tests.
