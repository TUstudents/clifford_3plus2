# Session 46 - Minimal Simulation Runner

Status: implemented as a deterministic tiny-lattice runner around the existing
coupled fermion/gauge/Higgs prototype.

## Target

Sessions 43-45 added one-step diagnostics, bounded multi-step stability
probes, and an exact local unitary Yukawa mode.  Session 46 adds the first
simulator-shaped entrypoint:

```text
run_simulation(config) -> SimulationResult
```

The runner is intentionally small.  It is a reproducible wrapper around the
existing coupled kernels, not a new physics rule and not a production-scale
`jax.lax.scan` simulator.

## Added API

In `jax_runner.py`:

- `SimulationRunConfig`
- `SimulationHistory`
- `SimulationResult`
- `run_simulation`
- `simulation_metadata`
- `simulation_summary`
- `save_simulation_result`

The runner advances through the public `jax_advance_scaling_fields` helper from
`jax_scaling.py`, so it does not depend on the scaling module's private
implementation helper.

The default run is memory-safe:

```text
lattice_shape  (1, 1, 1)
sector         u1_y
steps          4
record_every   1
step_size      0.0025
yukawa_mode    unitary
```

The module CLI is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.scripts.run_tiny_sim
```

It prints a JSON summary and optionally writes `run.npz` plus `run.json`.

## Implementation

`run_simulation` uses the deterministic Session 43 initial data and records
scalar observables at step 0, each requested `record_every` interval, and the
final step.  The recorded history includes:

```text
fermion_norm
gauge_hamiltonian_density
higgs_total_energy
higgs_energy_density_mean
gauss_residual_norm
yukawa_norm_drift
total_energy_proxy
```

`save_simulation_result` writes observable histories and final fields to
`.npz`, with a JSON sidecar containing metadata and a compact summary.

## Audit Results

- Default config is memory-safe.
- Zero-step-size runs record finite histories with no norm/energy drift.
- `record_every` preserves requested samples and always records final.
- Invalid runner controls fail explicitly.
- Saved `.npz` and `.json` files contain the expected arrays and metadata.
- The CLI works in summary-only and output-writing modes.
- A slow nonzero exact-unitary run produces finite final fields and history.

## Validation

Scoped commands:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/jax_scaling.py \
  src/clifford_3plus2_d5/spacetime_qca/scripts/run_tiny_sim.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling_trajectory.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_runner.py -m "not slow" -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_runner.py -q
```

Focused results:

```text
9 passed, 1 deselected
10 passed
```

## Still Open

- Replace Python stepping with a jittable `jax.lax.scan` runner.
- Add larger-lattice presets only after force/current paths are optimized.
- Add plotting or post-processing helpers for saved runs.
- Add Gauss projection and analytic current before making strong long-time
  claims.
