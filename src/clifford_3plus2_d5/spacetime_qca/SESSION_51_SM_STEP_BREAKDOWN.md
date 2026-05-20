# Session 51 — SM Step Breakdown Profiling

Status: implemented.

Session 51 decomposes Session 50's expensive `step_no_matter_sm` profile into
bounded `(1, 1, 1)` sub-kernel probes.  This is diagnostic infrastructure only:
it does not optimize or rewrite the physics kernels.

## Implementation

- `spacetime_qca.simulator.step_breakdown_profile` adds named full-SM probes
  for:
  - first and final half Yukawa kicks;
  - no-backreaction fermion/gauge step;
  - Higgs leapfrog;
  - coupled diagnostics;
  - Gauss residual;
  - gauge Hamiltonian density;
  - finite-difference Wilson left-force.
- `spacetime_qca.simulator.scripts.profile_step_breakdown` emits JSON summaries:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --case higgs_leapfrog_sm \
  --warmup-runs 0 \
  --timed-runs 1
```

The CLI defaults to one timed run and no warmup so a broad full-SM breakdown is
not accidentally repeated several times.  The no-argument default runs only the
cheap `higgs_leapfrog_sm` probe; use focused `--case` runs first, or
`--all-cases` only when a full breakdown is intentional.

## Interpretation

This profiler is the next diagnostic layer after Session 50.  If
`left_force_sm` dominates, the first optimization target is the
finite-difference Wilson force.  If `fermion_gauge_no_matter_sm` dominates
while `left_force_sm` does not, the next breakdown should split pure gauge
leapfrog from Dirac transport.  If a Yukawa probe dominates, the local unitary
eigensolve should be cached or specialized.

No local full breakdown is recorded here because the full-SM no-matter path was
already observed to take minutes on this machine.  The supported workflow is to
run the new CLI with one or two named cases and append the timings to the next
optimization report.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/simulator/step_breakdown_profile.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/__init__.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py

uv run pytest src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_kernel_profiling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py \
  -m "not slow" -q
```

The full `spacetime_qca` suite remains intentionally out of scope for this
profiling session.
