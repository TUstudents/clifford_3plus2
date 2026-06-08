# QCA_SMv0 Status

## Verdict

```text
QCA_SMV0_STAGE2_STATIC_SM_GAUGE_PASS
```

## Current State

Stage 2 static Standard-Model gauge-background transport implemented on top of
the Stage 1 free BCC Weyl/Dirac walk.

Implemented:

- package scaffold;
- documentation scaffold;
- `scripts/` and `tests/` directories;
- package metadata constants and free-walk exports;
- two-component BCC Weyl state layout `(nx, ny, nz, 2)`;
- Pauli/projector helpers;
- eight BCC hop matrices;
- pull-convention free BCC Weyl step;
- split-product Bloch symbol and hop-sum Bloch symbol;
- norm, hop-completeness, symbol-unitarity, small-k, and anisotropy-scaling
  diagnostics;
- four-component massless Dirac assembly from opposite-chirality Weyl blocks;
- Dirac hop-completeness, symbol-unitarity, norm, dispersion, and JIT tests;
- Session 01 script;
- local 32-component SM internal carrier;
- anti-Hermitian `SU(3)_c x SU(2)_L x U(1)_Y` generator basis;
- static BCC SM link fields with shape `(nx, ny, nz, 8, 32, 32)`;
- local state/link gauge transformations;
- static gauge-covariant Dirac BCC transport;
- identity-link reduction to the free internal Dirac step;
- Wilson plaquette traces over selected BCC parallelograms;
- weak-link linearization check for the continuum covariant derivative;
- Session 02 script;
- focused tests for algebra, norm preservation, local gauge covariance, Wilson
  response, weak-link scaling, JIT compatibility, and small-momentum
  Weyl/Dirac behavior.

Not implemented:

- boundary rules;
- dynamic gauge fields;
- Higgs/Yukawa collision;
- flavor or FN recirculation;
- center-holonomy CP;
- performance profiles beyond the small Session 01/02 diagnostics.

## Working Boundary

This sidecar is meant to focus the next simulator prototype around recent
sidecar work while reusing `sim` and local `qca_smv0` kernels only.  It does
not import from `spacetime_qca` or theory sidecars.

Run only focused `qca_smv0` tests while developing this sidecar.
