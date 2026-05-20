# spacetime_qca — Roadmap After Session 48

## Current Position

Sessions 20-48 have built the static and simulation-control stack:

- BCC Weyl/Dirac spacetime walk with the `alpha . k` continuum precursor.
- Finite periodic real-space BCC stepping.
- Position-dependent background gauge covariance.
- BCC plaquette holonomy and Wilson action observables.
- Compact Wilson forces and leapfrog controls for SU(2), SU(3), SU(4), and
  Pati-Salam/SM generator sectors.
- Static Higgs/Yukawa representation audits, including the Higgs-doublet
  charge pattern and neutral-VEV breaking pattern.
- A no-backreaction fermion/gauge wrapper: gauge links evolve by pure-gauge
  leapfrog, then fermions are transported through the updated links.
- A first Gauss-law/backreaction prototype: electric divergence, fermion
  charge density, finite-difference link current, and explicit momentum
  source kick.
- A static Hermitian Yukawa `Y(Phi)` layer on a deterministic two-complex
  Higgs-like map slice, with exact SymPy and JAX parity.
- A site-local Higgs field `Phi(x)` with finite `SU(2)_L x U(1)_Y` gauge
  transforms, BCC covariant differences, kinetic/potential diagnostics, and a
  sitewise bridge back to the static `Y(Phi)` layer.
- A first coupled fermion/gauge/Higgs prototype with Higgs conjugate momentum,
  fixed-link Higgs leapfrog, a first-order site-local Yukawa kick, and
  finite diagnostics for all field sectors.
- Physical `U(1)_Y` as the default spacetime-QCA sector convention, with exact
  one-generation anomaly cancellation diagnostics and raw Pati-Salam
  hypercharge kept only under explicit regression aliases.
- Exact finite-spacing free-dispersion diagnostics showing BCC Dirac anisotropy
  begins at `O(epsilon^4)` after chiral Weyl cubic-term cancellation, while
  the naive hypercube control has `O(epsilon^2)` anisotropy.
- Tiny-lattice scaling/stability diagnostics for the coupled prototype:
  neutral-vacuum density probes, scalarized snapshots, one-step drift trials,
  and bounded step-size sweeps.
- Optional exact local unitary Yukawa insertion for fixed `Phi`, while keeping
  the first-order kick as the default regression path.
- Bounded multi-step tiny-lattice trajectory diagnostics and timing probes for
  the coupled prototype.
- A deterministic tiny-lattice simulation runner with observable history,
  JSON summaries, and `.npz`/JSON output.
- A clean simulator split: prototype lab runner, reusable generic `sim`
  runner/io helpers, and a scan-backed main `spacetime_qca.simulator` API.
- Import-boundary tests and package notes that pin the lab/main/sim split.
- A bounded scan-backed simulator profiling CLI and first bottleneck report.
- Warm kernel profiling that separates setup, link construction, observable
  extraction, no-matter coupled steps, and optional matter current.

The module now has enough infrastructure to move from background-gauge
kinematics toward coupled field dynamics.  The remaining work is not one
feature; it is a sequence of increasingly physical constraints.

### Session 46 — Minimal Simulation Runner

Status: complete. Result report:
[SESSION_46_SIMULATION_RUNNER.md](SESSION_46_SIMULATION_RUNNER.md).

Delivered:

- `run_simulation(config) -> SimulationResult` for deterministic tiny-lattice
  coupled runs.
- Observable histories for fermion norm, gauge energy, Higgs energy, Gauss
  residual, Yukawa drift, and total energy proxy.
- `.npz` plus JSON output for saved runs.
- A small module CLI:
  `python -m clifford_3plus2_d5.spacetime_qca.lab.scripts.run_tiny_sim`.

Non-goal: the lab runner remains a Python-stepped prototype.

### Session 47 — Simulator Split And Generic Runner Infrastructure

Status: complete.

Delivered:

- Moved the Session 46 runner into `spacetime_qca.lab.tiny_runner`.
- Removed the old `spacetime_qca.jax_runner` import path.
- Added generic `sim.runner`, `sim.observables`, and `sim.io` helpers.
- Added `spacetime_qca.simulator` with a scan-backed main runner, field
  adapters, physics observables, presets, and a stable CLI:
  `python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.run_sim`.
- The main simulator is scan-backed by default; JIT is opt-in with `--jit`
  until the physics kernels are profiled for memory use.
- Added lab/main equivalence tests on zero-step and one-step tiny runs.

Next: profile and optimize the physics kernels used by the scan-backed
simulator, especially finite-difference current/force paths.

### Session 48 — Split Stabilization

Status: complete. Result report:
[SESSION_48_SPLIT_STABILIZATION.md](SESSION_48_SPLIT_STABILIZATION.md).

Delivered:

- Local usage notes in `spacetime_qca.lab` and `spacetime_qca.simulator`.
- Import-boundary tests for removed old runner paths and supported new paths.
- Stable simulator CLI output coverage with JSON-sidecar metadata checks.

Next: run focused profiling on the scan-backed simulator to identify the first
physics-kernel bottleneck.

### Session 49 — Simulator Profiling And Bottleneck Report

Status: complete. Result report:
[SESSION_49_SIMULATOR_PROFILING.md](SESSION_49_SIMULATOR_PROFILING.md).

Delivered:

- Generic `sim.profiling` callable profiles with JSON-safe payloads.
- `spacetime_qca.simulator.profiling` bounded `(1, 1, 1)` profile cases.
- `profile_sim` CLI for JSON profiling output.
- Local non-JIT bottleneck report identifying full-SM sector gauge force/link
  algebra as the first optimization target.

Next: add warm steady-state timing for physics kernels, then optimize the
full-SM sector force/link path before increasing lattice size.

### Session 50 — Warm Kernel Profiling

Status: complete. Result report:
[SESSION_50_WARM_KERNEL_PROFILING.md](SESSION_50_WARM_KERNEL_PROFILING.md).

Delivered:

- Repeated warm timing payloads in `sim.profiling`.
- Bounded kernel probes in `spacetime_qca.simulator.kernel_profile`.
- `profile_kernels` CLI for focused kernel timing.
- Local spot profile showing SM link construction is not the dominant cost by
  itself; the full SM no-matter coupled step is the next target.

Next: decompose or optimize the full-SM no-matter coupled step path, especially
Wilson force and diagnostics inside the SM sector.

## Immediate Priorities

### Session 37 — Gauss Law And Fermion Backreaction

Status: complete.  Result report:
[SESSION_37_GAUSS_BACKREACTION.md](SESSION_37_GAUSS_BACKREACTION.md).  Original
implementation plan:
[SESSION_37_GAUSS_BACKREACTION_PLAN.md](SESSION_37_GAUSS_BACKREACTION_PLAN.md).

Session 37 turned the Session 36 wrapper from "fermions in an evolving
background" into the first constrained gauge/fermion prototype.

Detailed plan: [SESSION_37_GAUSS_BACKREACTION_PLAN.md](SESSION_37_GAUSS_BACKREACTION_PLAN.md).

Deliverables:

- Define lattice Gauss-law residuals for compact Pati-Salam/SM links and
  link momenta in the current pull convention.
- Define the fermion color/weak/hypercharge density induced by
  `(nx, ny, nz, 4, 32)` Dirac/internal states and sector generators.
- Add a no-surprises source term audit: zero fermion state gives zero source;
  gauge-transformed states give covariantly transformed source.
- Add a first backreaction update prototype that modifies link momenta by a
  sector-current term.
- Verify the update is gauge-covariant on tiny lattices.

Still non-goals after Session 37:

- Production HMC.
- Quantum-current operator ordering.
- Dynamical Higgs coupling.
- Full constraint projection on large lattices.

Why it mattered: dynamical gauge fields were already represented by compact
link/momentum evolution, but a physically coupled theory needs Gauss-law
constraints and matter current.  Session 37 supplies the first residual/source
prototype while keeping full constraint projection as an open task.

### Session 38 — General Hermitian Yukawa Operator

Status: complete.  Result report:
[SESSION_38_HERMITIAN_YUKAWA.md](SESSION_38_HERMITIAN_YUKAWA.md).

Goal: promote the Session 23/25 Higgs-like map space into a general static
Hermitian Yukawa operator.

Target form:

```text
Y(Phi) = beta x (sum_a Phi_a M_a + conjugate(Phi_a) M_a^dagger)
```

where `M_a` spans a deterministic selected slice of the Higgs-like internal
map module and `Phi` is a static complex Higgs-doublet value.

Delivered:

- Exact SymPy API using explicit `(real, imag)` coordinates for
  `phi_plus` and `phi_zero`.
- Deterministic selected basis `(upper[0], upper[1], lower[0], lower[1])`
  for the two-complex `Phi` slice.
- `Y_internal(Phi) = A(Phi) + A(Phi)^T` and `beta x Y_internal(Phi)`.
- JAX mirrors using native complex scalar inputs.
- Tests for zero `Phi`, linearity, Hermiticity, neutral-VEV `Q_em`
  preservation, charged-component `Q_em` breaking, and JAX/SymPy parity.

Non-goals:

- Site-local Higgs field dynamics.
- Yukawa hierarchy fitting.
- Fermion family structure.

Why this comes before dynamical Higgs: the static Hermitian operator is the
algebraic bridge from "there is a Higgs-like slot" to "the fermion update can
carry a Yukawa term."  The dynamic field should not be introduced until this
operator is unambiguous.

### Session 39 — Site-Local Dynamical Higgs Field

Status: complete.  Result report:
[SESSION_39_DYNAMICAL_HIGGS.md](SESSION_39_DYNAMICAL_HIGGS.md).

Goal: introduce `Phi(x)` as a lattice field with gauge transformation,
covariant transport, kinetic energy, and potential diagnostics.

Delivered:

- Define a site-local Higgs field layout, initially in the static doublet basis
  from Session 38.
- Implement gauge transformation of `Phi(x)` under `SU(2)_L x U(1)_Y`.
- Define gauge-covariant finite differences using BCC links.
- Add Higgs kinetic energy and Mexican-hat potential diagnostics.
- Verify gauge invariance of Higgs energy on tiny lattices.
- Add sitewise `Y(Phi[x])` bridge helpers for the Session 38 Yukawa layer.

Non-goals:

- Full coupled Higgs time evolution.
- Continuum renormalization analysis.
- Nonperturbative phase diagram.

Why this is separate: a dynamical Higgs field is not just a Yukawa matrix.  It
needs its own site-local state, gauge law, kinetic term, and potential before
it can safely enter the QCA step.

### Session 40 — Coupled Fermion + Gauge + Higgs Step

Status: complete.  Result report:
[SESSION_40_COUPLED_HIGGS_STEP.md](SESSION_40_COUPLED_HIGGS_STEP.md).

Goal: combine the pieces into the first small-lattice coupled update.

Delivered:

- Fermion step with BCC kinetic transport plus static or site-local Yukawa
  insertion.
- Gauge leapfrog with matter-current source from Session 37.
- Higgs conjugate momentum and a fixed-link leapfrog update using the Session
  39 kinetic/potential controls.
- Diagnostics for norm drift, gauge Hamiltonian, Higgs energy, and Gauss-law
  residual before/after one step.
- Tiny-lattice smoke tests for `u1_y`, `su2_l`, and `sm` sectors.

Non-goals:

- Long-time stability claims.
- Production-scale performance.
- Continuum-limit renormalization.

Interpretation: this is the first "all fields present" prototype.  It is still
a research control, not a final Standard Model QCA.

## Hardening And Physics Audits

### Session 41 — Anomaly And Charge-Current Diagnostics

Status: complete. Result report:
[SESSION_41_ANOMALY_CURRENT.md](SESSION_41_ANOMALY_CURRENT.md).

Delivered:

- Corrected the JAX sector convention so `u1_y` and `sm` use physical
  Session 19b hypercharge.
- Kept the old unnormalized Pati-Salam convention as explicit `u1_y_raw` and
  `sm_raw` aliases.
- Added exact one-generation anomaly diagnostics for `Tr Y`, `Tr Y^3`,
  `SU(3)^2-U(1)_Y`, `SU(2)^2-U(1)_Y`, `SU(3)^3`, and the `SU(2)_L` Witten
  doublet parity.
- Added regression tests for physical hypercharge normalization and JAX
  charge-density expectations.

Non-goal: regulator-level anomaly analysis for the full interacting lattice
theory remains future work.

### Session 42 — Lorentz Recovery Beyond `alpha . k`

Status: complete. Result report:
[SESSION_42_LORENTZ_RECOVERY.md](SESSION_42_LORENTZ_RECOVERY.md).

Delivered:

- Added exact trace-cosine dispersion diagnostics for BCC Weyl and BCC Dirac
  symbols.
- Verified that opposite Weyl helicities have opposite cubic anisotropy.
- Verified that the Dirac pair cancels the cubic term and first deviates from
  continuum dispersion at `O(epsilon^4)`.
- Verified that the naive hypercube control has lower-order
  `O(epsilon^2)` anisotropy.

Non-goal: this is a free-dispersion audit, not a full interacting Lorentz
recovery proof.

### Session 43 — Scaling Diagnostics

Status: complete. Result report:
[SESSION_43_SCALING_DIAGNOSTICS.md](SESSION_43_SCALING_DIAGNOSTICS.md).

Delivered:

- Deterministic tiny-lattice initial data for the coupled
  fermion/gauge/Higgs prototype.
- Scalarized `ScalingSnapshot` diagnostics for fermion norm, gauge energy,
  Higgs energy, Gauss residual, Yukawa norm drift, and a total-energy proxy.
- `ScalingTrial` before/after one-step drift diagnostics.
- Neutral-vacuum density normalization probes across `(1, 1, 1)` and
  `(2, 1, 1)`.
- Step-size sweeps over bounded tiny-grid controls.

Non-goal: this is a stability/normalization harness, not a continuum
renormalization proof.

### Session 44 — Performance And Longer Stability Runs

Status: complete. Result report:
[SESSION_44_PERFORMANCE_STABILITY.md](SESSION_44_PERFORMANCE_STABILITY.md).

Delivered:

- Multi-step tiny-lattice trajectories with drift-from-initial summaries.
- First-order versus exact-unitary Yukawa trajectory comparisons.
- A tiny jitted timing probe through the shared `sim` benchmark helper.
- Memory-safe defaults and slow tagging for nonzero coupled trajectories.

Non-goal: this is a bounded stability/timing harness, not production-scale
simulation or a full vectorized Wilson-force rewrite.

### Session 45 — Exact Unitary Yukawa Insertion

Status: complete. Result report:
[SESSION_45_UNITARY_YUKAWA.md](SESSION_45_UNITARY_YUKAWA.md).

Delivered:

- Exact site-local `exp(-i dt y beta x Y(Phi[x]))` update using
  `I x cos(dt y Y) - i beta x sin(dt y Y)`.
- Eigensystem-based `32 x 32` internal matrix functions instead of full
  `128 x 128` matrix exponentials.
- `yukawa_mode="first_order" | "unitary"` in the coupled wrapper.
- `yukawa_mode` support in Session 43 scaling configs.
- Slow tests proving norm preservation and reduced norm drift in the tiny
  deterministic scaling setup.

Non-goal: this fixes the local Yukawa substep only.  Higgs current
backreaction, Gauss projection, and long-time interacting stability remain
future work.

### Parallel Track — Performance

- Vectorize Wilson action/force evaluation for larger lattices.
- Reduce finite-difference force calls in chiral16 sectors.
- Add benchmark scripts under `spacetime_qca/scripts/` only when stable.
- Keep general-purpose JAX array infrastructure in `sim`; keep BCC and
  physics-specific policy in `spacetime_qca`.

## How The Open Points Map To The Roadmap

### Dynamical Gauge Fields

Current status: partially implemented as compact link/momentum dynamics.

Done:

- Wilson action and force controls.
- Compact left-trivialized updates.
- Reversible leapfrog prototypes.
- Pati-Salam/SM sector adapters.
- No-backreaction coupling to fermion transport.
- Target-site electric divergence and Gauss residual.
- Fermion charge density in sector coordinates.
- Finite-difference matter current and explicit momentum source kick.

Still open:

- Gauss-law residuals and projection.
- Full Gauss-law projection / constraint solving.
- Analytic matter current and long-time stability.
- Physical coupling constants per gauge factor.
- Longer stability and scaling tests.

Roadmap owner: Session 40 and Session 41 for the next coupled-field audits.

### Dynamical Higgs

Current status: partially implemented as a site-local field and first coupled
prototype.

Done:

- Higgs-like map space identified.
- Static Higgs-doublet charge pattern verified.
- Neutral-VEV breaking pattern verified.
- Site-local Higgs field `Phi(x)` with `SU(2)_L x U(1)_Y` gauge law.
- BCC covariant finite difference, kinetic energy, and Mexican-hat potential.
- Higgs conjugate momentum and fixed-link leapfrog.
- First coupled fermion/gauge/Higgs smoke path.
- Exact local unitary Yukawa insertion for fixed `Phi`.

Still open:

- Higgs current backreaction into gauge momenta.
- Long-time stability and scaling diagnostics.

Roadmap owner: Sessions 39-40 implemented the v1 field/update path; future
hardening continues after Session 41.

### Hermitian Yukawa From The Dim-4 Space

Current status: complete for static controls.

Done:

- Charge-shift map basis exists.
- Static Hermitian rank-control probes exist.
- Neutral VEV preserves electromagnetism in the static audit.
- General `Y(Phi)` construction.
- Complex coordinate convention for `Phi`.
- Hermiticity proof for arbitrary `Phi`.
- Mass-gap diagnostics for representative Higgs backgrounds.

Still open:

- Yukawa hierarchy/family structure.

Roadmap owner: Session 38.

## Decision Points

After Session 37:

- Gauss-law/backreaction is coherent enough to proceed to the static Hermitian
  Yukawa operator.
- Full constraint projection and analytic current remain future hardening
  tasks.

After Session 38:

- If `Y(Phi)` is clean and Hermitian, promote `Phi` to a field.
- If the real-form/J-adapted convention is messy, add a compressed explicit
  `C^16` internal basis before going dynamic.

After Session 48:

- The next project goal should be numerical simulation scale-up and
  performance hardening, because the prototype lab and scan-backed main
  simulator split is now pinned by tests and reusable multi-step controls
  already exist.

After Session 49:

- The first optimization target is full-SM sector gauge force/link algebra.
- Future profiles need warmup/repeat timing so cold JAX setup does not distort
  physics-kernel comparisons.

After Session 50:

- The bottleneck is narrowed to the full-SM no-matter coupled step path.
- Link construction alone is not the main cost in the local spot profile.

After Session 51:

- Use `profile_step_breakdown` to time one or two full-SM sub-kernels at a
  time before optimizing.
- If `left_force_sm` dominates, replace or vectorize the finite-difference
  Wilson left-force path first.
- If `fermion_gauge_no_matter_sm` dominates without `left_force_sm`, split the
  gauge leapfrog and Dirac transport paths next.
