# Session 50 — Warm Kernel Profiling

Status: implemented.

Session 50 refines Session 49's simulator-level profile into a bounded
kernel-level profiler.  The aim is to separate setup, link construction,
observable extraction, coupled-step cost, and optional finite-difference matter
current before changing physics kernels.

## Implementation

- `sim.profiling` now exports `RepeatedCallProfile` and
  `profile_callable_repeated`, with `warmup_runs`, `timed_runs`, and
  `min/mean/max` timing payloads.
- `spacetime_qca.simulator.kernel_profile` adds named `(1, 1, 1)` probes for:
  initial-state construction, observable extraction, SM-sector link
  construction, no-matter coupled steps, and opt-in matter-current profiling.
- `spacetime_qca.simulator.scripts.profile_kernels` provides:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_kernels
```

Use `--case <name>` for focused probes, `--include-current` for the expensive
finite-difference current probe, and `--output profile.json` for full payloads.

## Local Spot Profile

Command:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_kernels \
  --case initial_state_u1_y \
  --case link_field_sm \
  --case step_no_matter_sm \
  --warmup-runs 0 \
  --timed-runs 1 \
  --output /tmp/spacetime_qca_session50_kernel_profile.json
```

Observed on the local default JAX device (`cuda:0`):

| Case | Mean wall time |
| --- | ---: |
| `initial_state_u1_y` | 3.102 s |
| `link_field_sm` | 0.306 s |
| `step_no_matter_sm` | 127.277 s |

This spot profile intentionally used one timed run and no warmup for the
expensive SM step to avoid a multi-minute repeated benchmark.  The profiler
supports warm/repeated runs; use them for cheaper focused cases.

## Bottleneck Verdict

First target: full-SM no-matter coupled step path.

SM link construction alone is not the dominant cost in this spot profile:
`link_field_sm` is small relative to `step_no_matter_sm`.  The next optimization
should decompose the no-matter coupled step, especially the SM Wilson
force/left-force path and diagnostics invoked inside `jax_scaling_snapshot`.

Matter current remains opt-in for profiling because the current implementation
uses explicit finite differences and is expected to be expensive.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/sim/profiling.py \
  src/clifford_3plus2_d5/sim/__init__.py \
  src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/kernel_profile.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_kernels.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/__init__.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_kernel_profiling.py

uv run pytest src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_profiling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_kernel_profiling.py \
  -m "not slow" -q
```

Result: `25 passed in 8.93s`.

The full `spacetime_qca` suite was intentionally not run.
