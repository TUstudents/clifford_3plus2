# QCA_SMv0 Plan

## Purpose

Build a focused simulator sidecar for the next Standard-Model QCA prototype.

The first rule is:

```text
infrastructure first, theory second
```

## Session 00 - Infrastructure

Pass only if:

- the sidecar package exists;
- `README.md`, `PLAN.md`, and `STATUS.md` exist;
- `scripts/` and `tests/` exist;
- the package exposes only sidecar metadata;
- the placeholder test verifies only infrastructure;
- the central repo docs link to the sidecar;
- no simulator theory, mass model, or hidden ledger is introduced.

Verdict:

```text
QCA_SMV0_INFRASTRUCTURE_READY
```

## Stage 1 - Free BCC Weyl/Dirac walk

Pass only if:

- the Weyl state has shape `(nx, ny, nz, 2)`;
- the eight BCC source offsets are exactly `{+-1}^3`;
- the Pauli projectors satisfy `P_j^- + P_j^+ = I` and are orthogonal;
- the hop matrices are `A_h=P_x^{h_x}P_y^{h_y}P_z^{h_z}`;
- `sum_h A_h^dagger A_h = I`;
- the position kernel uses the repository pull convention `psi[x+h]`;
- the Bloch symbol from the split product equals the summed hop symbol;
- the Bloch symbol is unitary;
- the periodic-lattice step preserves norm;
- the step is JIT-compatible;
- the small-momentum eigenphase has Weyl speed `1` near the origin;
- the directional phase-speed spread decreases under momentum halving;
- the four-component Dirac step is assembled from opposite-chirality Weyl
  blocks;
- Dirac hops satisfy `sum_h D_h^dagger D_h = I_4`;
- the Dirac symbol is unitary;
- the Dirac step preserves norm and is JIT-compatible;
- no boundary, gauge, Higgs, flavor, or recirculation rule is introduced.

Verdict:

```text
QCA_SMV0_STAGE1_FREE_BCC_PASS
```

## Stage 2 - Static SM Gauge Background

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the internal register has shape `32`, interpreted as one SM chiral-16 label
  set duplicated over the Dirac spin block;
- local anti-Hermitian `SU(3)_c x SU(2)_L x U(1)_Y` generators are explicit;
- finite static BCC links have shape `(nx, ny, nz, 8, 32, 32)`;
- link exponentiation produces unitary matrices;
- identity links reduce the gauged Dirac update to the Stage 1 free Dirac
  update applied independently to each internal component;
- the static-link update preserves norm on a periodic lattice;
- the state/link transformation law gives local gauge covariance:
  `U_G(psi^Omega, A^Omega) = Omega U_G(psi, A)`;
- identity and pure-gauge backgrounds have zero Wilson action, while a
  deterministic non-pure background has non-zero Wilson action;
- normalized Wilson traces are gauge-invariant;
- weak-link transport agrees with the first-order covariant-derivative
  expansion with quadratic residual scaling under `epsilon -> epsilon/2`;
- the gauged step is JIT-compatible;
- no dynamic gauge-field update, Higgs, Yukawa, flavor, boundary, or
  recirculation rule is introduced.

Verdict:

```text
QCA_SMV0_STAGE2_STATIC_SM_GAUGE_PASS
```

## Next Session Placeholder

The next session should specify whether Stage 3 is:

- dynamic gauge-field evolution and Gauss-law diagnostics;
- Higgs/Yukawa collision on static gauge backgrounds;
- or performance/layout work before adding new physics.

Until that is specified, this sidecar should remain at the static-background
transport layer.
