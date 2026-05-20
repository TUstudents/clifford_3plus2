# Session 49 — Simulator Profiling

Status: implemented.

Session 49 adds a bounded profiling path for the scan-backed
`spacetime_qca.simulator` runner.  The goal is evidence, not optimization: the
profiler records tiny `(1, 1, 1)` cases, emits JSON-safe timing payloads, and
names the first bottleneck to attack before increasing lattice size.

## Implementation

- `sim.profiling` adds a generic `profile_callable` helper and `CallProfile`
  payload with device, wall time, output summary, finite-result status, and
  optional external JIT timings.
- `spacetime_qca.simulator.profiling` defines fixed Session 49 cases and a
  deterministic bottleneck recommendation.
- `spacetime_qca.simulator.scripts.profile_sim` provides:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_sim
```

Use `--case <name>` for focused profiling, `--output profile.json` to write the
payload, and `--include-jit` to opt into the simulator's internal JIT path.

## Local Profile

Command:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_sim \
  --output /tmp/spacetime_qca_session49_profile.json
```

Observed on the local default JAX device (`cuda:0`):

| Case | Wall time |
| --- | ---: |
| `runner_zero_step` | 5.933 s |
| `scan_zero_step_4` | 0.995 s |
| `scan_zero_step_16` | 0.729 s |
| `physics_no_matter` | 111.045 s |
| `physics_with_matter` | 7.212 s |
| `sector_u1_y` | 4.044 s |
| `sector_su2_l` | 10.719 s |
| `sector_sm` | 34.362 s |

The first nonzero physics case paid a large cold-start/JAX compilation cost, so
the better steady-ish comparison is the later sector sweep:
`sector_sm / sector_u1_y ~= 8.5`.

## Bottleneck Verdict

First target: full-SM sector gauge force/link algebra.

The full SM-sector one-step cost is substantially higher than the physical
hypercharge and `SU(2)_L` cases even on a single BCC cell.  Before increasing
lattice size, profile and then replace the expensive SM-sector gauge-force/link
path with a more vectorized or cached implementation.

Second target: separate cold-start compile time from steady-state timings.  The
`physics_no_matter` outlier shows that future profiles should include warmup or
repeat timing when comparing physics kernels.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/sim/profiling.py \
  src/clifford_3plus2_d5/sim/__init__.py \
  src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/profiling.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_sim.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/__init__.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_profiling.py

uv run pytest src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_profiling.py \
  -m "not slow" -q
```

Result: `18 passed in 10.82s`.

The full `spacetime_qca` suite was intentionally not run.
