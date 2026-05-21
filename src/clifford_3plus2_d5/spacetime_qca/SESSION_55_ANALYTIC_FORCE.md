# Session 55 — Analytic Staple-Like Wilson Force

Status: implemented.

Session 55 adds an opt-in analytic compact Wilson left-force for the current
BCC plaquette convention.  It keeps scalar and batched finite differences as
oracles while removing the per-coordinate action re-evaluation from the new
`analytic_staple` path.

## Implementation

- `jax_compact_lie_left_force(..., method="analytic_staple")` now contracts
  each plaquette occurrence directly.
- The convention matches the finite-difference oracle:
  `U[x,h] -> exp(t T_a) U[x,h]` and
  `d/dt action(exp(t T_a) U)|_{t=0}`.
- The derivative contribution for each occurrence is
  `-Re Tr(prefix * T_a * U * suffix) / (matrix_dim * plaquette_count)`.
- Pati-Salam adapters inherit the method through the existing
  `CompactLieForceMethod` type.
- Step-breakdown profiling now includes:
  `left_force_analytic_sm`, `first_left_force_analytic_sm`,
  `second_left_force_analytic_sm`, and `gauge_leapfrog_analytic_sm`.
- `profile_step_breakdown --force-method analytic_staple` and
  `run_sim --force-method analytic_staple` are accepted.
- A zero-step simulator CLI regression pins `--force-method analytic_staple`
  without executing the expensive force kernel.

The method is still opt-in.  Simulator defaults remain scalar finite
difference until the analytic path has more coverage.

## Local Profile

Command:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --case first_left_force_analytic_sm \
  --case second_left_force_analytic_sm \
  --warmup-runs 0 --timed-runs 1

uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --case gauge_leapfrog_analytic_sm \
  --warmup-runs 0 --timed-runs 1
```

Local default JAX device: `cuda:0`.  The CUDA driver version warning appeared
again, but both profiles completed with finite outputs.

| Case | Time |
| --- | ---: |
| `first_left_force_analytic_sm` | 0.680 s |
| `second_left_force_analytic_sm` | 0.099 s |
| `gauge_leapfrog_analytic_sm` | 0.860 s |

Compared with the Session 54 scalar baselines, analytic force gives roughly
`13.8x` on the first left-force probe and `86.2x` on the second.  Compared
with the best Session 54 batched chunk (`32`), it is roughly `3.4x` faster on
the first force and `5.0x` faster on the second.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_gauge_force.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/step_breakdown_profile.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/run_sim.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_compact_lie.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_patisalam_subgroup_adapters.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_compact_lie.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_patisalam_subgroup_adapters.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  -m "not slow" -q
```

Result: `79 passed, 7 deselected`.

The full `spacetime_qca` suite remains intentionally out of scope.

## Recommendation

Use `analytic_staple` for the next SM performance profiles and simulator smoke
runs.  Keep scalar finite difference and batched finite difference as oracle
paths until analytic force is covered on more sectors and larger lattices.
