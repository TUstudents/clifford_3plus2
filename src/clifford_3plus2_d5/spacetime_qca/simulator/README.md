# spacetime_qca.simulator

Stable, production-facing spacetime-QCA simulator APIs live here.

Use this package for:

- scan-backed simulator runs;
- stable configs, field bundles, and result objects;
- simulation presets and campaign entrypoints;
- saved `.npz` plus JSON outputs.

The main entrypoint is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.run_sim
```

The bounded profiling entrypoint is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_sim
```

Use `--case runner_zero_step` or another named case for a fast focused probe;
the full bounded profile intentionally includes physics kernels and can take
minutes on the local JAX backend.

The simulator uses `jax.lax.scan` by default.  JIT wrapping is opt-in with the
`--jit` CLI flag or `SpacetimeSimulationConfig(use_jit=True)` until the physics
kernels are profiled for memory use.

Physics-specific kernels and observables stay in `spacetime_qca`; reusable
generic runner, observable, and persistence mechanics stay in `sim`.
