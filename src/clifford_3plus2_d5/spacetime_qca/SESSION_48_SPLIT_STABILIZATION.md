# Session 48 - Simulator Split Stabilization

Status: implemented.  This session locks down the import and documentation
boundary introduced in Session 47.

## Result

- Added short package notes for `spacetime_qca.lab` and
  `spacetime_qca.simulator`.
- Added import-boundary tests proving:
  - `spacetime_qca.jax_runner` is not supported;
  - `spacetime_qca.scripts.run_tiny_sim` is not supported;
  - old runner symbols are not exported from top-level `spacetime_qca`;
  - `spacetime_qca.lab` exports the prototype runner API;
  - `spacetime_qca.simulator` exports the main scan-backed simulator API.
- Added simulator CLI output coverage, including JSON-sidecar metadata.

## API Boundary

Supported:

- `clifford_3plus2_d5.spacetime_qca.lab.tiny_runner`
- `clifford_3plus2_d5.spacetime_qca.simulator`
- `clifford_3plus2_d5.sim.runner`
- `clifford_3plus2_d5.sim.io`
- `clifford_3plus2_d5.sim.observables`

Unsupported:

- `clifford_3plus2_d5.spacetime_qca.jax_runner`
- `clifford_3plus2_d5.spacetime_qca.scripts.run_tiny_sim`

No compatibility shims are intentionally provided.

## Validation

```bash
uv run ruff check \
  src/clifford_3plus2_d5/sim \
  src/clifford_3plus2_d5/spacetime_qca/lab \
  src/clifford_3plus2_d5/spacetime_qca/simulator \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_lab_tiny_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_split_boundaries.py

uv run pytest \
  src/clifford_3plus2_d5/sim/tests \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_lab_tiny_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_runner.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_split_boundaries.py \
  -m "not slow" -q
```

Focused result:

```text
31 passed, 2 deselected
```

## Next

Session 49 should profile the scan-backed simulator and identify the first
physics-kernel bottleneck to remove before increasing lattice sizes.
