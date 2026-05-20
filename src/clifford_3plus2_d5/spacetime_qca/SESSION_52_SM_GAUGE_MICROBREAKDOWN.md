# Session 52 — SM Gauge-Step Microbreakdown

Status: implemented.

Session 52 extends the Session 51 profiler so the expensive full-SM
no-matter path can be split into gauge leapfrog, Dirac transport, compact
momentum update, and first/second Wilson-force calls.  This remains diagnostic
infrastructure only: no force replacement or physics-kernel optimization is
implemented here.

## Implementation

- `profile_step_breakdown` now exposes additional full-SM cases:
  - `gauge_leapfrog_sm`;
  - `dirac_transport_sm`;
  - `momentum_update_sm`;
  - `first_left_force_sm`;
  - `second_left_force_sm`.
- The no-argument CLI default remains the cheap `higgs_leapfrog_sm` probe.
  Force-heavy cases must be selected explicitly with `--case` or `--all-cases`.
- Recommendation logic now distinguishes:
  - Wilson left-force bottlenecks;
  - broader gauge-leapfrog bottlenecks;
  - BCC Dirac transport bottlenecks;
  - compact momentum-update / matrix-exponential bottlenecks.

Example focused commands:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --case dirac_transport_sm \
  --warmup-runs 0 \
  --timed-runs 1

uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --case first_left_force_sm \
  --warmup-runs 0 \
  --timed-runs 1
```

## Interpretation

The next optimization should be chosen from focused Session 52 timings:

- If `first_left_force_sm` or `second_left_force_sm` dominates, Session 53
  should replace or vectorize the finite-difference Wilson force.
- If `gauge_leapfrog_sm` dominates but individual force probes do not, split
  the compact leapfrog internals further.
- If `momentum_update_sm` dominates, target chiral16 matrix exponentials in the
  compact link update.
- If `dirac_transport_sm` dominates, target BCC internal-link transport.

No broad local timing table is recorded here because force-heavy SM probes can
take minutes on the local backend.  Run one or two focused cases at a time.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/simulator/step_breakdown_profile.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py \
  -m "not slow" -q
```

The full `spacetime_qca` suite remains intentionally out of scope.
