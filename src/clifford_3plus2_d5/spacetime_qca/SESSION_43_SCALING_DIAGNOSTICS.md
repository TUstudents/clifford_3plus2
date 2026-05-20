# Session 43 - Scaling Diagnostics

Status: implemented as tiny-lattice stability and normalization diagnostics.

## Target

Sessions 37-42 built the first coupled fermion/gauge/Higgs prototype and the
first finite-spacing dispersion audit.  Session 43 adds the next control
layer: deterministic probes for how the prototype's diagnostics behave under
small step-size changes and tiny lattice-size changes.

This is not a continuum renormalization proof.  It is a bounded audit for
finite diagnostics, density normalization, and one-step drift behavior before
larger simulations or longer time runs.

## Added API

In `jax_scaling.py`:

- `ScalingRunConfig`
- `ScalingInitialState`
- `ScalingSnapshot`
- `ScalingTrial`
- `NeutralVacuumDensityProbe`
- `Session43ScalingAuditPayload`
- `jax_default_scaling_initial_state`
- `jax_scaling_snapshot`
- `jax_coupled_scaling_trial`
- `jax_step_size_scaling_sweep`
- `jax_neutral_vacuum_density_probe`
- `session43_scaling_audit_payload`

The default probe is intentionally memory-safe:

```text
sector        u1_y
lattice       (1, 1, 1)
step sizes    0.0, 0.0025, 0.005
plaquettes    one canonical BCC shape
```

## Diagnostics

`ScalingSnapshot` records scalarized quantities from the existing coupled
machinery:

```text
fermion_norm
gauge_hamiltonian_density
higgs_total_energy
higgs_energy_density_mean
gauss_residual_norm
yukawa_norm_drift
total_energy_proxy
```

The `total_energy_proxy` is:

```text
gauge_hamiltonian_density + mean(higgs_energy_density)
```

It intentionally excludes the fermion norm.  This is a stability proxy for the
current prototype, not a full interacting Hamiltonian.

`ScalingTrial` compares a before/after one-step run and reports absolute
drifts in the fermion norm, gauge energy, Higgs energy, Gauss residual, and
the total energy proxy.

## Audit Results

- Deterministic tiny-lattice initial data is produced for `u1_y`, `su2_l`, and
  `sm`-compatible sector shapes.
- Neutral vacuum density probes are zero on `(1, 1, 1)` and `(2, 1, 1)` with
  identity links and zero momenta.
- One-step coupled trials return finite scalar diagnostics.
- Step-size sweeps preserve ordering and bounded scalar payloads while keeping
  heavy coupled-step tests marked as `slow`.

## Interpretation

Session 43 moves the package from "all fields can be stepped once" to "the
one-step prototype has a reusable measurement harness."  This is the minimum
tooling needed before judging longer numerical runs.

The result does not claim continuum convergence.  It establishes a stable
place to measure that convergence later.

## Validation

Scoped commands:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_scaling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling.py -m "not slow" -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_scaling.py -q
```

Focused results:

```text
5 passed, 3 deselected
8 passed
```

## Still Open

- Full Gauss-law projection / constraint solving.
- Analytic matter current replacing finite-difference current controls.
- Long-time stability benchmarks.
- Scaling diagnostics over larger grids after vectorization work.
- A true continuum renormalization analysis.
