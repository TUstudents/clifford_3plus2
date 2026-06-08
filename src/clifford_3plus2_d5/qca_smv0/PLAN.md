# QCA_SMv0 Plan

## Purpose

Build a focused simulator sidecar for the next Standard-Model QCA prototype.

The first rule is:

```text
infrastructure first, theory second
```

This file intentionally contains no detailed mechanism yet.  The next session
should define the first simulator target before any kernel or ledger is added.

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

## Next Session Placeholder

The next session should specify:

- the physical simulator target;
- the minimal upstream imports;
- the state layout;
- the JAX performance boundary;
- the focused test or audit that defines success.

Until that is specified, this sidecar should remain scaffold-only.
