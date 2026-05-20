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

The bounded warm kernel profiling entrypoint is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_kernels
```

Use `--case initial_state_u1_y` or another named kernel case for focused
profiling.  The finite-difference matter-current probe is opt-in with
`--include-current`.

The full-SM coupled-step breakdown entrypoint is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown
```

Use `--case higgs_leapfrog_sm` or another named sub-kernel for focused
profiling.  The no-argument default runs only `higgs_leapfrog_sm`; use repeated
`--case` arguments or explicit `--all-cases` for the broader expensive
breakdown.  The CLI uses one timed run with no warmup by default.

The simulator uses `jax.lax.scan` by default.  JIT wrapping is opt-in with the
`--jit` CLI flag or `SpacetimeSimulationConfig(use_jit=True)` until the physics
kernels are profiled for memory use.

Physics-specific kernels and observables stay in `spacetime_qca`; reusable
generic runner, observable, and persistence mechanics stay in `sim`.
