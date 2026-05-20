# spacetime_qca — Roadmap After Session 43

## Current Position

Sessions 20-43 have built the static and simulation-control stack:

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

The module now has enough infrastructure to move from background-gauge
kinematics toward coupled field dynamics.  The remaining work is not one
feature; it is a sequence of increasingly physical constraints.

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

- Vectorize the slowest Wilson/gauge-force paths where possible.
- Add benchmark scripts for tiny and small lattice runs.
- Use Session 43 snapshots to measure short-run drift over multiple steps.
- Keep memory-safe defaults and mark long runs as `slow` or scripts, not fast
  tests.

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
- Stability and scaling tests.

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

Still open:

- Higgs current backreaction into gauge momenta.
- Exact unitary Yukawa insertion.
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

- Exact unitary insertion in the coupled update.
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

After Session 43:

- The next project goal should be numerical simulation scale-up and
  performance hardening, because a reusable scaling harness now exists.
