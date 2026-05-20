# Session 53 — Batched Finite-Difference Force

Status: implemented.

Session 53 adds a batched finite-difference Wilson force path inside the
existing compact Lie gauge-force stack.  This is the first optimization pass
after Session 52 identified the SM left-force as the steady bottleneck.

## Implementation

- `jax_compact_lie_left_force(..., method="finite_difference_batched")` now
  evaluates centered finite differences in small batches.
- `chunk_size` limits the batch width.  The default is conservative and
  memory-aware for tiny SM probes.
- `jax_compact_lie_left_force_from_algebra`, action descent, compact leapfrog,
  and Pati-Salam force adapters forward the new method and chunk control.
- The no-backreaction, Gauss/backreaction, coupled Higgs, scaling, and main
  simulator config paths forward `force_method` and `force_chunk_size`.
- `profile_step_breakdown` exposes batched comparison cases:
  `left_force_batched_sm`, `first_left_force_batched_sm`,
  `second_left_force_batched_sm`, and `gauge_leapfrog_batched_sm`.
- The original scalar `method="finite_difference"` path remains the reference
  oracle.

## Local Spot Profile

One `(1, 1, 1)` SM left-force profile on the local default JAX device:

| Method | Chunk | Time |
| --- | ---: | ---: |
| `finite_difference` | — | 9.431 s |
| `finite_difference_batched` | 16 | 2.821 s |

This is a roughly 3.3x speedup for the first SM left-force probe without
changing the Wilson action or force convention.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_gauge_force.py \
  src/clifford_3plus2_d5/spacetime_qca/jax_gauge_dynamics.py \
  src/clifford_3plus2_d5/spacetime_qca/jax_patisalam.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_compact_lie.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_patisalam_subgroup_adapters.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_compact_lie.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_patisalam_subgroup_adapters.py \
  -m "not slow" -q
```

Result: `49 passed, 6 deselected`.

Additional threading checks:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  -m "not slow" -q

uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --case first_left_force_batched_sm --warmup-runs 0 --timed-runs 1
```

Result: the combined focused test run with force and simulator checks reported
`69 passed, 7 deselected`; the profiler smoke reached
`first_left_force_batched_sm` and produced a finite `(1, 1, 1, 8, 12)` force in
`2.890 s`.

The full `spacetime_qca` suite remains intentionally out of scope.

## Next Step

Use the new profiler cases to compare scalar and batched force paths on the
same machine, then decide whether Session 54 should keep tuning chunk size or
replace the finite-difference path with a staple-like analytic Wilson force.
