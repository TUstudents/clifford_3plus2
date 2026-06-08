# QCA_SMv0

Simulator sidecar for focused Standard-Model QCA experiments.

This sidecar is infrastructure only.  It is the place to build a smaller,
recent-sidecar-informed simulator without changing the shared `sim` package or
the broader `spacetime_qca` production prototype.

## Scope

`QCA_SMv0` should be used for:

- focused QCA simulator experiments tied to the recent boundary/flavor sidecars;
- small, JAX-native kernels that can later graduate into `spacetime_qca`;
- explicit session notes, local scripts, and focused tests.

It should not be used for:

- broad regression bookkeeping;
- importing theory claims before a session states them;
- replacing the shared `sim` infrastructure.

## Current State

Only the scaffold exists:

```text
qca_smv0/
  README.md
  PLAN.md
  STATUS.md
  __init__.py
  scripts/
  tests/
```

No physics kernels, ledgers, or simulator sessions are implemented yet.

## Reuse Boundary

Allowed upstream infrastructure:

- `clifford_3plus2_d5.sim` for generic JAX state, links, scan runners, profiling,
  and persistence;
- `clifford_3plus2_d5.spacetime_qca` for existing BCC/BB/QCA kernels when a
  session explicitly imports them.

Future sessions should keep imports narrow and run only the tests for the
current session.

## Run

```bash
uv run pytest src/clifford_3plus2_d5/qca_smv0/tests -q
```
