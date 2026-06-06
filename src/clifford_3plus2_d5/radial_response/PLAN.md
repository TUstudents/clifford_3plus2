# radial_response Plan

## Purpose

Develop the mass-sector idea:

```text
masses = QCA boundary recirculation residues / pole shifts.
```

This sidecar is theory-first and symbolic. It does not run the BCC simulator.

## Implemented Gates

### R1 - Green-function form

Pass only if:

- the Schur complement gives `Sigma(z)=V^T(z-H_Q)^-1V`;
- the full resolvent P block matches the Schur P block;
- the series expansion is identifiable as repeated boundary returns.

### R2 - Up stacking law

Pass only if:

- exponential stacking gives invariant `C_u C_t / C_c^2 = 1/2`;
- geometric stacking gives invariant `C_u C_t / C_c^2 = 1`;
- the sidecar records the fork without deriving `x`.

### R3 - Literal nilpotent Yukawa kill

Pass only if:

- literal `exp(N)` at `x=1` has singular values `(2,1,1/2)`;
- its left metric is not diagonal;
- the verdict rejects `exp(xN)` as the family-space Yukawa matrix.

### R4 - Down dark-line framing

Pass only if:

- the scalar-clebsch S3 projector audit passes;
- rank 5 is available but not unique;
- the rank-2 standard copy remains defect-selected, not central-forced.

### R5 - Two-channel repair isometry

Pass only if:

- two equal no-leakage scalar repair successors force `1/sqrt(2)`;
- one-channel, three-channel, leakage, and asymmetric controls fail to force
  the target value;
- the sidecar records that the microscopic successor/no-leakage hypotheses are
  still the real burden.

### R6 - Minimal unitary S3 defect form

Pass only if:

- the exact toy has the Floquet form `U = S C`;
- `U` is exactly unitary;
- the P-block of the full resolvent agrees with the unitary Schur complement;
- coin-angle and defect-vector controls change the self-energy, proving that
  the form itself does not force the physical phase or radial values.

### R7 - Scalar successor/no-leakage certificate

Pass only if:

- the finite modeled candidate basis has exactly two allowed scalar repair
  successors;
- the allowed successors are a Z2-conjugate pair;
- same-state, wrong-height, two-tick, external-leakage, asymmetric-sector, and
  third-successor controls are vetoed;
- the payload records that full microscopic basis completeness is not derived.

### R8 - S3 scalar-shell completeness

Pass only if:

- all six S3 elements are classified;
- the identity is rejected as same-state/no repair;
- exactly the two non-identity A3 cycles are allowed;
- the three transpositions are rejected as Hermitian/Z2 repair-sector elements;
- transposition conjugation swaps the two triality successors;
- the payload records that full QCA boundary completeness is not derived.

### R9 - QCA-to-S3 scalar boundary reduction

Pass only if:

- all 24 tetrahedral exit automorphisms are classified;
- the selected-exit stabilizer has size 6;
- selected-exit-preserving automorphisms induce the full residual S3 shell;
- selected-exit-moving and non-automorphism controls are rejected;
- scalar holomorphic restriction leaves exactly the R8 triality pair;
- the payload records that the automorphism premise is not derived from the full
  BB/QCA update.

### R10 - Scalar automorphism premise bridge

Pass only if:

- declared one-tick scalar-local boundary maps with deterministic exit action
  land inside the R9 tetrahedral automorphism census;
- the only accepted maps induce `triality_plus` and `triality_minus`;
- selected-exit-moving, generic linear-mixture, non-automorphism, nonlocal,
  non-scalar, same-state, and Hermitian/Z2 controls are rejected;
- the payload records that this derives the R9 premise from the declared
  scalar-local map class, not from the full BB/QCA update.

### R11 - Silver transfer inheritance

Pass only if:

- the radial sidecar imports the already-derived transfer root
  `epsilon = sqrt(2)-1` from `boundary_response`;
- `eta = epsilon^2` and `r = epsilon^4` are exact inherited powers, not local
  fits or recalculations;
- the residual K3 graph source, sterile-chain Weyl source, flavor A-track shared
  transfer gate, and quark common-chain Schur gate agree;
- independent fitted `eta`, K2/K4 alternate roots, and the claim that the
  minimal `U=S C` toy alone forces pole values are rejected.

### R12 - Radial pole/residue rigidity no-go

Pass only if:

- two baths preserve the same inherited transfer data, scalar successor pair,
  and two-channel coupling norm;
- their Schur self-energies are different;
- their pole locations and residues are different;
- the payload records that finite S3/silver-transfer data do not force mass
  pole values without an additional spectral-density principle.

### R13 - Spectral-measure selection study

Pass only if:

- the target up, down-baseline, and down-candidate mass textures define positive
  finite Stieltjes measures with inherited `eta`;
- each target measure reconstructs a finite Jacobi bath and round-trips through
  boundary moments;
- R12 baths, the P3 repair Jacobi control, the constant silver-tail Jacobi
  control, and the minimal unitary S3 toy do not select the target measures;
- the payload records the result as inverse reconstruction only, not a forward
  QCA mass derivation.

## Next Gates

- R14: identify a forward spectral-density principle that selects the target
  measure: filled-band energy, constrained maximum entropy, or a QCA-local
  impurity bath derived from the scalar boundary update.
- R15: derive or kill the down dark-line selection rule.
- R16: connect the symbolic spectral measure to a small BCC/QCA simulator
  extraction of `J_f(omega)`.
