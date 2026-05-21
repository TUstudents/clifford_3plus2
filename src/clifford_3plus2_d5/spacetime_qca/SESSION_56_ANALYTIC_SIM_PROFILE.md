# Session 56 — Analytic-Force Whole-Step Profiling

Status: implemented.

Session 56 checks whether the Session 55 analytic staple-like Wilson force
changes the scan-backed simulator bottleneck.  It adds force-method overrides
to the bounded profiling CLIs and profiles the full SM no-matter path with
analytic force enabled.

## Implementation

- `profile_kernels` accepts `--force-method` and `--force-chunk-size`.
  Overrides apply only to `step_no_matter_*` cases.
- `profile_sim` accepts `--force-method` and `--force-chunk-size`.
  Overrides apply to the selected simulator cases.
- Payload metadata records force overrides and each case records the final
  config used.
- Defaults remain unchanged: simulator configs still default to scalar
  finite-difference force.
- Recommendation logic no longer compares a single SM-only profile against a
  missing U(1) baseline; it now points single-SM payloads to step breakdown.

## Local Profiles

Commands used one timed run and no warmup.  These are cold JAX timings on the
local default device (`cuda:0`); the CUDA driver version warning appeared but
all runs completed with finite outputs.

### Whole SM No-Matter Step

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_kernels \
  --case step_no_matter_sm \
  --force-method finite_difference_batched \
  --force-chunk-size 32 \
  --warmup-runs 0 --timed-runs 1

uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_kernels \
  --case step_no_matter_sm \
  --force-method analytic_staple \
  --warmup-runs 0 --timed-runs 1
```

| Case | Force method | Time |
| --- | --- | ---: |
| `step_no_matter_sm` | `finite_difference_batched`, chunk `32` | 108.384 s |
| `step_no_matter_sm` | `analytic_staple` | 106.655 s |

Analytic force does not materially change the cold whole-step timing.  The
force-only Session 55 profile remains valid, so this points to another
whole-step cost.

### Scan-Backed Simulator Smoke

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_sim \
  --case sector_sm \
  --force-method analytic_staple
```

Result: `sector_sm = 111.787 s`, all finite.

### Post-Force Step Breakdown

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --case yukawa_half_kick_sm \
  --case yukawa_final_half_kick_sm \
  --case gauge_leapfrog_analytic_sm \
  --case dirac_transport_sm \
  --case higgs_leapfrog_sm \
  --case diagnostics_sm \
  --warmup-runs 0 --timed-runs 1
```

| Case | Time |
| --- | ---: |
| `yukawa_half_kick_sm` | 104.159 s |
| `diagnostics_sm` | 2.245 s |
| `higgs_leapfrog_sm` | 1.834 s |
| `gauge_leapfrog_analytic_sm` | 0.740 s |
| `dirac_transport_sm` | 0.226 s |
| `yukawa_final_half_kick_sm` | 0.006 s |

The first exact-unitary Yukawa half-kick dominates the cold timing.  The final
half-kick is tiny after the first call, so the likely issue is JAX
compilation/eigensystem setup for the exact-unitary Yukawa path rather than
steady-state arithmetic alone.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/simulator/kernel_profile.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/profiling.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_kernels.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_sim.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_kernel_profiling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_profiling.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_kernel_profiling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_profiling.py \
  -m "not slow" -q
```

Result: `19 passed`.

The full `spacetime_qca` suite remains intentionally out of scope.

## Recommendation

The next optimization target is the exact-unitary Yukawa insertion path, not
the Wilson force.  Session 57 should profile and specialize the site-local
unitary Yukawa update:

- separate cold compile/setup from warm execution;
- cache or precompute static pieces of the Higgs/Yukawa eigensystem;
- preserve the first-order Yukawa path as a cheap oracle/fallback;
- keep `analytic_staple` as the force method for simulator profiles while
  optimizing Yukawa.
