# Session 57 — Exact Yukawa Fast Path

## Goal

Session 56 showed that the realistic cold SM simulator profile was no longer
dominated by Wilson-force evaluation.  The first exact-unitary Yukawa half-kick
was the next apparent bottleneck.  Session 57 specializes that exact Yukawa
path for the selected Session 38 Higgs map while preserving an eigensolve oracle
for validation.

## Implementation

`jax_apply_site_local_yukawa_unitary` remains the public exact-unitary mode, but
it no longer diagonalizes the local `32 x 32` internal Hermitian matrix by
default.  For the selected two-complex Higgs slice,

```text
Y(Phi)^3 = lambda(Phi)^2 Y(Phi)
lambda(Phi)^2 = 256 ||Phi||^2
```

so the local matrix functions are evaluated as

```text
cos(dt Y) = I + ((cos(dt lambda) - 1) / lambda^2) Y^2
sin(dt Y) = (sin(dt lambda) / lambda) Y
```

with stable zero-field limits.  The local Dirac/internal update is then still

```text
exp(-i dt beta x Y) = I x cos(dt Y) - i beta x sin(dt Y).
```

The old diagonalization path is retained explicitly as
`yukawa_mode="unitary_eigh"` through
`jax_apply_site_local_yukawa_unitary_eigh`.  It is an oracle and profiling
comparison path, not the production default.

## Tests

Focused checks cover:

- the cubic identity `Y^3 = lambda^2 Y` on zero, neutral, and mixed Higgs
  samples;
- polynomial `cos/sin` against the eigensolve oracle;
- polynomial exact-unitary state update against the eigensolve oracle;
- CLI/config acceptance of `unitary_eigh`;
- step-breakdown cases for production polynomial and eigensolve-oracle Yukawa
  half-kicks.

The eigensolve-oracle comparisons are marked `slow` so the default fast lane
does not pay the 32-by-32 JAX eigensystem compile path.

## Profile Results

Cold first-call timings are still dominated by process-level JAX/XLA setup:

- polynomial `yukawa_half_kick_sm`, fresh process: `103.953 s`;
- eigensolve-oracle `yukawa_half_kick_eigh_sm`, fresh process: `102.836 s`.

This means the old Session 56 cold bottleneck should not be read as a pure
algorithmic eigensolve cost.  It is mostly first-touch setup in this CUDA/JAX
environment.

Warm timings show the local exact update is no longer the practical per-step
bottleneck:

- polynomial `yukawa_half_kick_sm`, one warmup then one timed run: `0.0127 s`;
- four-case same-process profile after the first cold case:
  `yukawa_final_half_kick_sm = 0.0125 s`,
  `yukawa_final_half_kick_eigh_sm = 0.0080 s`;
- warm whole-step `step_no_matter_sm` with analytic force:
  `0.3347 s`.

## Interpretation

Session 57 turns the exact-unitary Yukawa path into a cheap warm local kernel
and keeps the eigensolve implementation only as a validation oracle.  It does
not remove the large first-call JAX setup cost in a fresh process.  The next
performance work should treat cold setup separately from per-step execution:
prewarm kernels, cache static scan-backed field construction, and profile
multi-step trajectories where the one-time setup is amortized.

## Validation

Focused validation used:

```bash
uv run pytest \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_unitary_yukawa.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  -m "not slow" -q
```

Result: `31 passed, 9 deselected`.

Before marking the oracle checks slow, the same focused file set ran the new
eigensolve comparisons as well: `34 passed, 6 deselected`.

Lint:

```bash
uv run ruff check <touched spacetime_qca files>
```

Result: all checks passed.
