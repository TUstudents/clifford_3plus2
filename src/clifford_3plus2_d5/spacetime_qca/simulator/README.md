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
Use `--force-method analytic_staple` to profile selected simulator cases with
the analytic compact Wilson-force path.

The bounded warm kernel profiling entrypoint is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_kernels
```

Use `--case initial_state_u1_y` or another named kernel case for focused
profiling.  The finite-difference matter-current probe is opt-in with
`--include-current`.
Use `--force-method analytic_staple` for `step_no_matter_*` kernel cases.

The full-SM coupled-step breakdown entrypoint is:

```bash
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.profile_step_breakdown
```

Use `--case higgs_leapfrog_sm` or another named sub-kernel for focused
profiling.  The no-argument default runs only `higgs_leapfrog_sm`; use repeated
`--case` arguments or explicit `--all-cases` for the broader expensive
breakdown.  The CLI uses one timed run with no warmup by default.
Session 52 adds microbreakdown cases such as `dirac_transport_sm`,
`momentum_update_sm`, `first_left_force_sm`, and `second_left_force_sm`; run
force-heavy cases one at a time.
Session 53 adds batched comparison cases such as `first_left_force_batched_sm`,
`second_left_force_batched_sm`, and `gauge_leapfrog_batched_sm`.
Session 54 adds `--force-comparison` to compare scalar force against batched
chunks `4, 8, 16, 32`.
Session 55 adds analytic cases such as `first_left_force_analytic_sm`,
`second_left_force_analytic_sm`, and `gauge_leapfrog_analytic_sm`.
Session 56 adds force-method overrides to `profile_sim` and `profile_kernels`
and identifies the current cold whole-step bottleneck as the exact-unitary
Yukawa insertion, not Wilson force.
Session 57 specializes the production exact-unitary Yukawa path with the
selected Higgs-map cubic polynomial.  Use `--yukawa-mode unitary` for the
polynomial exact path and `--yukawa-mode unitary_eigh` only as the explicit
eigensolve oracle.  Cold first-call profiles still include JAX/XLA setup; use
warmup runs when measuring steady-state simulator kernels.

The simulator uses `jax.lax.scan` by default.  JIT wrapping is opt-in with the
`--jit` CLI flag or `SpacetimeSimulationConfig(use_jit=True)` until the physics
kernels are profiled for memory use.
Session 58 changes the shared scan runner so `record_every` controls observable
work, not only saved-output density.  Sparse recordings now skip diagnostics on
discarded intermediate steps.
Use `--force-method analytic_staple` to select the analytic compact
Wilson-force path in scan-backed runs when the current BCC plaquette convention
is applicable.  Use
`--force-method finite_difference_batched --force-chunk-size 32` for the
batched finite-difference oracle path.
Session 61 adds off-by-default Gauss projection controls:
`--gauss-projection-steps` and `--gauss-projection-step-size`.  Keep them at
zero for baseline runs and use the Session 62 projection sweep before enabling
them in longer lab runs.

Physics-specific kernels and observables stay in `spacetime_qca`; reusable
generic runner, observable, and persistence mechanics stay in `sim`.
