# Session 44 - Performance And Longer Stability Runs

Status: implemented as a tiny-lattice multi-step stability and timing harness.

## Target

Session 43 added one-step coupled scaling diagnostics.  Session 45 added an
optional exact local unitary Yukawa insertion.  Session 44 connects those two
pieces: it records bounded multi-step trajectories and compares first-order
versus exact-unitary Yukawa behavior without promoting any large simulation
claim.

This is still a diagnostic layer, not production-scale lattice simulation.

## Added API

In `jax_scaling.py`:

- `ScalingTrajectorySample`
- `ScalingTrajectory`
- `YukawaModeComparison`
- `jax_coupled_scaling_trajectory`
- `jax_compare_yukawa_modes`
- `jax_scaling_timing_probe`

The defaults remain memory-safe:

```text
lattice_shape  (1, 1, 1)
sector         u1_y
steps          4
record_every   1
plaquettes     one canonical BCC shape
```

## Implementation

`jax_coupled_scaling_trajectory` records `ScalingSnapshot` samples at step 0,
each requested `record_every` interval, and the final step.  Each sample
reports drift from the initial sample for:

```text
fermion_norm
gauge_hamiltonian_density
higgs_total_energy
gauss_residual_norm
total_energy_proxy
```

The trajectory summary reports max drift across recorded samples and an
`all_finite` flag.  Zero-step-size trajectories short-circuit field evolution,
which keeps control tests fast and avoids calling expensive force paths when no
time evolution is requested.

`jax_compare_yukawa_modes` runs paired first-order and exact-unitary
trajectories from the same deterministic initial data and reports drift ratios.

`jax_scaling_timing_probe` uses the shared `sim` JAX benchmark helper on a tiny
jitted kernel.  Timing values are informational; tests only check that the
probe returns finite nonnegative values.

## Audit Results

- Zero-step trajectories keep drift diagnostics at zero.
- Sample recording respects `record_every` and always records the final step.
- Invalid step controls fail explicitly.
- Nonzero exact-unitary trajectories return finite bounded drift diagnostics
  on the tiny deterministic setup.
- Paired first-order/unitary comparisons show the unitary path does not
  increase fermion norm drift in the deterministic tiny-lattice check.
- The new public trajectory/timing API is covered by an export smoke test.
- Timing probes are available without adding hard performance thresholds.

## Validation

Scoped commands:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_scaling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling_trajectory.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling_trajectory.py -m "not slow" -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling_trajectory.py -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling_trajectory.py \
  -m "not slow" -q
```

Focused results:

```text
7 passed, 2 deselected
9 passed
12 passed, 5 deselected
```

## Still Open

- Production-scale performance work and vectorized Wilson-force paths.
- Long-time stability beyond tiny deterministic trajectories.
- Higgs current backreaction into gauge momenta.
- Full Gauss-law projection and analytic matter current.
- Continuum renormalization analysis.
