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

## Next Session Placeholder

The next session should specify:

- whether to add static gauge links or a minimal SM carrier next;
- the minimal upstream imports;
- the state layout;
- the JAX performance boundary;
- the focused test or audit that defines success.

Until that is specified, this sidecar should remain at the free bulk-walk
layer.
