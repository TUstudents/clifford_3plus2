# Session 59 — Higgs Current Backreaction

Session 59 adds the first Higgs-current source term for gauge momenta in the
coupled spacetime-QCA prototype.

## Implemented

- `jax_higgs_link_current(Phi, U_H)` computes a finite-difference
  left-trivialized Higgs link current in Higgs generator coordinates
  `(su2_x, su2_y, su2_z, u1_y)`.
- `jax_higgs_current_to_patisalam_sector` embeds that current into the
  supported coupled sectors:
  `u1_y`, `su2_l`, and `sm`.
- `jax_patisalam_apply_higgs_backreaction` applies the explicit source kick
  `P <- P + dt * g_H * J_H`.
- `jax_patisalam_fermion_gauge_higgs_step` accepts
  `higgs_coupling` and `higgs_current_epsilon`.
  The default `higgs_coupling=0.0` preserves previous behavior.
- Scaling and simulator configs expose the same controls so runs can enable
  Higgs backreaction without changing the public runner shape.

## Convention

The BCC Higgs links use the existing pull convention:

```text
U[x,h] : x + h -> x
D_h Phi[x] = U[x,h] Phi[x+h] - Phi[x]
```

The finite-difference current is the negative derivative of total Higgs energy
under left link perturbations:

```text
U[x,h] -> exp(theta_a T_a) U[x,h]
J_H[x,h,a] = - d E_H / d theta_a
```

This sign makes the explicit momentum kick force-like, matching the existing
gauge-force convention.

## Still Open

- The current is finite-difference and audit-oriented, not an analytic
  production current.
- Gauss-law residuals still include the fermion charge only; adding Higgs
  charge density and constraint projection is future work.
- Long-time stability with nonzero Higgs backreaction is not established.
