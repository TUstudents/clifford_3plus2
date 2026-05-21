# Session 54 — Force Chunk Tuning

Status: implemented.

Session 54 turns the Session 53 batched-force path into a reproducible
comparison workflow.  The goal is not to add a new simulator layer; it is to
measure whether batched finite differences are good enough before replacing the
Wilson force with an analytic staple-like formula.

## Implementation

- `profile_step_breakdown` now accepts force overrides:
  `--force-method` and `--force-chunk-size`.
- Force overrides apply only to force-related cases, leaving safe non-force
  probes unchanged.
- `--force-comparison` runs scalar baselines against batched chunks for
  `first_left_force` and `second_left_force`.
- Comparison payloads include per-candidate speedups versus the scalar
  finite-difference baseline.

## Local Force Comparison

Command:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown \
  --force-comparison --warmup-runs 0 --timed-runs 1
```

Local default JAX device: `cuda:0`.  The CUDA driver version warning appeared
again, but the profile completed with finite outputs.

### First left force

| Method | Chunk | Time | Speedup |
| --- | ---: | ---: | ---: |
| `finite_difference` | — | 9.411 s | — |
| `finite_difference_batched` | 4 | 5.564 s | 1.69x |
| `finite_difference_batched` | 8 | 3.062 s | 3.07x |
| `finite_difference_batched` | 16 | 2.870 s | 3.28x |
| `finite_difference_batched` | 32 | 2.312 s | 4.07x |

### Second left force

| Method | Chunk | Time | Speedup |
| --- | ---: | ---: | ---: |
| `finite_difference` | — | 8.536 s | — |
| `finite_difference_batched` | 4 | 3.807 s | 2.24x |
| `finite_difference_batched` | 8 | 1.924 s | 4.44x |
| `finite_difference_batched` | 16 | 0.966 s | 8.84x |
| `finite_difference_batched` | 32 | 0.497 s | 17.16x |

## Recommendation

Use `chunk_size=32` for current batched-force smoke/prototype runs when memory
allows.  The first force still costs `2.312 s` on the local `(1, 1, 1)` SM
probe, so the next optimization session should implement an analytic
staple-like compact Wilson force for the current BCC plaquette convention.

## Validation

Focused checks only:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/simulator/step_breakdown_profile.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/scripts/profile_step_breakdown.py \
  src/clifford_3plus2_d5/spacetime_qca/simulator/__init__.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_simulator_step_breakdown.py \
  -m "not slow" -q
```

Result: `17 passed`.

The full `spacetime_qca` suite remains intentionally out of scope.
