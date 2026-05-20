# Session 47 - Simulator Split

Status: implemented.  The simulation layer is now split into generic
infrastructure, a prototype lab runner, and a scan-backed main simulator.

## Result

- `sim.runner` adds physics-agnostic recorded Python-loop and `jax.lax.scan`
  runners over arbitrary JAX pytrees.
- `sim.observables` adds generic observable stacking, step selection, and
  finite-value checks.
- `sim.io` adds `.npz` plus JSON sidecar persistence.
- Session 46's deterministic runner moved to `spacetime_qca.lab.tiny_runner`.
- The old `spacetime_qca.jax_runner` module was removed intentionally.
- `spacetime_qca.simulator` adds:
  - `SpacetimeSimulationConfig`
  - `SpacetimeFields`
  - `run_spacetime_simulation`
  - `save_spacetime_simulation_result`
  - tiny presets and a stable `run_sim` CLI.
- The main simulator uses `jax.lax.scan` by default.  JIT wrapping is
  explicitly opt-in through `use_jit=True` or the CLI `--jit` flag until the
  physics kernels are profiled.

## Boundary

`sim` stays generic.  It has no BCC, Pati-Salam, Wilson-force, Higgs, Yukawa,
or Gauss-law policy.  Physics-specific kernels and observables remain in
`spacetime_qca`.

The lab runner remains the fast-prototyping path.  The main simulator is the
stable path for future performance and campaign work.

## Validation

```bash
uv run ruff check src/clifford_3plus2_d5/sim \
  src/clifford_3plus2_d5/spacetime_qca/lab \
  src/clifford_3plus2_d5/spacetime_qca/simulator \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_lab_tiny_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  src/clifford_3plus2_d5/sim/tests/test_sim_infrastructure.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py

uv run pytest src/clifford_3plus2_d5/sim/tests -q

uv run pytest src/clifford_3plus2_d5/sim/tests \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_lab_tiny_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  -m "not slow" -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -m "not slow" -q
```

Focused results:

```text
11 passed
26 passed, 2 deselected
7 passed
162 passed, 137 deselected
```

## Next

Session 48 should profile the scan-backed simulator and remove the most
expensive physics-kernel bottlenecks before increasing lattice sizes.
