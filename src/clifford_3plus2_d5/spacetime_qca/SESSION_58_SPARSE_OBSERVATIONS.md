# Session 58 — Sparse Observation Scan Runner

## Goal

The scan-backed simulator already supports `record_every`, but the shared
`run_recorded_scan` implementation computed observables at every step and then
discarded unrecorded samples.  Session 58 makes `record_every` reduce actual
observable work.

This is a simulator optimization, not a profiling session and not a physics
kernel change.

## Implementation

- `sim.runner.run_recorded_scan` now advances the state between requested
  record points with `jax.lax.scan` and calls `observe_fn` only at recorded
  indices.
- Public result shape is unchanged:
  - step `0` is always recorded;
  - every `record_every` step is recorded;
  - the final step is recorded even when it is not divisible by
    `record_every`.
- `run_recorded_loop` is unchanged.
- `use_jit=True` is preserved by JIT-wrapping the scan chunk used to advance
  between recorded observations.

## Tests

Focused shared-runner tests now cover:

- loop/scan equivalence for toy states;
- sparse scan observation calls only at `(0, record_every, ..., final)`;
- zero-step scan observes only the initial state and never calls `step_fn`;
- JIT scan equivalence against the loop runner.

Focused validation:

```bash
uv run pytest src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py -q
uv run pytest \
  src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  -m "not slow" -q
```

Results: `16 passed` for the shared `sim` test file, and
`26 passed, 1 deselected` for the focused shared-runner plus simulator-runner
regression set.

## Interpretation

For long simulator runs with expensive diagnostics and `record_every > 1`, the
scan runner no longer pays observable extraction cost for discarded steps.
Physics kernels, force methods, Yukawa modes, and simulator output schema are
unchanged.
