# QCA_SMv0 Status

## Verdict

```text
QCA_SMV0_STAGE1_FREE_BCC_PASS
```

## Current State

Stage 1 free BCC Weyl/Dirac walk implemented.

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
- focused tests for algebra, norm preservation, JIT compatibility, and
  small-momentum Weyl/Dirac behavior.

Not implemented:

- boundary rules;
- gauge links or gauge dynamics;
- Standard Model carrier;
- Higgs/Yukawa collision;
- flavor or FN recirculation;
- center-holonomy CP;
- performance profiles beyond the small Session 01 diagnostic.

## Working Boundary

This sidecar is meant to focus the next simulator prototype around recent
sidecar work while reusing `sim` and `spacetime_qca` only through explicit,
session-scoped imports.

Run only focused `qca_smv0` tests while developing this sidecar.
