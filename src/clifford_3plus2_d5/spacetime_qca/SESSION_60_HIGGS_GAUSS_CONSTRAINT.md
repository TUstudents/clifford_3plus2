# Session 60 — Higgs Charge in the Gauss Residual

Session 60 adds the site-local Higgs charge term to the coupled Gauss-law
diagnostic path.

## Implemented

- `jax_higgs_charge_density(Phi, Pi)` computes Higgs charge coordinates using
  the momentum-map convention:

  ```text
  rho_a = Re(Pi^\dagger T_a Phi)
  ```

  where `T_a` are the anti-Hermitian Higgs generators in the order
  `(su2_x, su2_y, su2_z, u1_y)`.

- `jax_higgs_charge_to_patisalam_sector` embeds Higgs charge into `u1_y`,
  `su2_l`, and `sm` sector coordinates.
- `jax_patisalam_fermion_higgs_gauss_residual` computes

  ```text
  divE - g_f rho_f - g_H rho_H
  ```

  while leaving the older fermion-only `jax_patisalam_gauss_residual`
  unchanged.

- `jax_patisalam_fermion_higgs_gauss_descent_step` is a tiny-lattice
  diagnostic control that takes a gradient-descent step on
  `0.5 ||Gauss||^2` with respect to gauge momenta.
- Coupled diagnostics now use the combined residual when `higgs_coupling` is
  nonzero.  The default `higgs_coupling=0.0` preserves earlier behavior.

## Still Open

- The descent step is a diagnostic control, not a production constraint
  projection.
- No simulator step automatically projects onto the Gauss surface.
- Analytic/vectorized Higgs charge/current formulas and long-time constrained
  stability remain future work.
