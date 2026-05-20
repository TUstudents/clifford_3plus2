# sim — Status

**Status**: shared infrastructure package.

`sim` contains generic JAX simulation helpers that can be reused by sidecars.
It intentionally does not define Pati-Salam, Spin(10), BCC Dirac dynamics,
Wilson plaquette normalization, or gauge-force physics policy.

## What Exists

- `backend.py` — dtype defaults, array conversion, default-device reporting,
  and JIT compile/run timing.
- `lattice.py` — 3D periodic pull-roll helpers and displacement validation.
- `state.py` — SymPy-to-NumPy conversion, JAX state allocation, flattening,
  and norm diagnostics.
- `links.py` — generic identity/constant pull-link fields and finite
  site-local gauge transforms.
- `diagnostics.py` — finite-value and state-transition metrics.
- `observables.py` — generic observable stacking, selection, and finite-value
  checks.
- `runner.py` — physics-agnostic Python-loop and `jax.lax.scan` recorded
  runners over arbitrary JAX pytrees.
- `io.py` — generic `.npz` plus JSON sidecar persistence.
- `benchmarks.py` — stable benchmark wrapper for JAX kernels.
- `profiling.py` — JSON-safe callable profiling payloads for simulator
  callables and bounded benchmark reports.

## Boundary

Physics-specific code remains in sidecars until a more general architecture is
confirmed.  In particular, these stay in `spacetime_qca`:

- BCC Weyl and Dirac walk kernels.
- BCC plaquette geometry and Wilson observables.
- SO(2)/SU(2) Wilson-force policies.
- Internal Pati-Salam / SM tensor lifts.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/sim/tests -q
```

Expected current result: `12 passed`.
