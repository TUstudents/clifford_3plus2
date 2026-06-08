# QCA_SMv0 Status

## Verdict

```text
QCA_SMV0_INFRASTRUCTURE_READY
```

## Current State

Infrastructure only.

Implemented:

- package scaffold;
- documentation scaffold;
- `scripts/` and `tests/` directories;
- package metadata constants;
- placeholder infrastructure test.

Not implemented:

- simulator kernels;
- state layouts;
- JAX stepping rules;
- physics ledgers;
- theory-specific scripts;
- performance profiles.

## Working Boundary

This sidecar is meant to focus the next simulator prototype around recent
sidecar work while reusing `sim` and `spacetime_qca` only through explicit,
session-scoped imports.

Run only focused `qca_smv0` tests while developing this sidecar.
