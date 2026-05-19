# spacetime_qca — Roadmap After Session 36

## Current Position

Sessions 20-36 have built the static and simulation-control stack:

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

Goal: promote the Session 23/25 Higgs-like map space into a general static
Hermitian Yukawa operator.

Target form:

```text
Y(Phi) = beta x (sum_a Phi_a M_a + conjugate(Phi_a) M_a^dagger)
```

where `M_a` spans the Higgs-like internal map module and `Phi` is a static
complex Higgs-doublet value.

Deliverables:

- Choose an explicit complex coordinate convention for the 4-real-dimensional
  Higgs map space.
- Implement `Y(Phi)` in the current real-form carrier convention.
- Prove Hermiticity for arbitrary static `Phi`.
- Verify neutral-VEV `Phi` preserves `Q_em` and breaks `Y` / `T3_L`.
- Compute the induced `k = 0` mass-gap controls for representative `Phi`.

Non-goals:

- Site-local Higgs field dynamics.
- Yukawa hierarchy fitting.
- Fermion family structure.

Why this comes before dynamical Higgs: the static Hermitian operator is the
algebraic bridge from "there is a Higgs-like slot" to "the fermion update can
carry a Yukawa term."  The dynamic field should not be introduced until this
operator is unambiguous.

### Session 39 — Site-Local Dynamical Higgs Field

Goal: introduce `Phi(x)` as a lattice field with gauge transformation,
covariant transport, kinetic energy, and potential diagnostics.

Deliverables:

- Define a site-local Higgs field layout, initially in the static doublet basis
  from Session 38.
- Implement gauge transformation of `Phi(x)` under `SU(2)_L x U(1)_Y`.
- Define gauge-covariant finite differences using BCC links.
- Add Higgs kinetic energy and Mexican-hat potential diagnostics.
- Verify gauge invariance of Higgs energy on tiny lattices.

Non-goals:

- Full coupled Higgs time evolution.
- Continuum renormalization analysis.
- Nonperturbative phase diagram.

Why this is separate: a dynamical Higgs field is not just a Yukawa matrix.  It
needs its own site-local state, gauge law, kinetic term, and potential before
it can safely enter the QCA step.

### Session 40 — Coupled Fermion + Gauge + Higgs Step

Goal: combine the pieces into the first small-lattice coupled update.

Deliverables:

- Fermion step with BCC kinetic transport plus static or site-local Yukawa
  insertion.
- Gauge leapfrog with matter-current source from Session 37.
- Higgs update using the Session 39 kinetic/potential controls.
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

- Check that the one-generation chiral16 charge table used by the dynamics
  has the expected anomaly cancellations.
- Verify lattice current definitions respect the same cancellations in the
  continuum/small-field limit.
- Add regression tests for hypercharge normalization and sector traces.

### Session 42 — Lorentz Recovery Beyond `alpha . k`

- Quantify BCC anisotropy at finite lattice spacing.
- Show rotational/Lorentz symmetry recovery in the small-momentum limit.
- Compare BCC dispersion against the naive hypercube control beyond corner
  doubling.

### Session 43 — Renormalization / Scaling Diagnostics

- Define dimensionless lattice couplings and scaling observables.
- Track how Wilson, Higgs, and Yukawa diagnostics behave under lattice
  refinement on small numerical grids.
- Do not claim a continuum quantum field theory until these diagnostics exist.

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

Current status: not implemented as a field.

Done:

- Higgs-like map space identified.
- Static Higgs-doublet charge pattern verified.
- Neutral-VEV breaking pattern verified.

Still open:

- Site-local Higgs field `Phi(x)`.
- Gauge transformation and covariant finite difference.
- Kinetic energy and potential.
- Higgs time update.

Roadmap owner: Session 39, then Session 40.

### Hermitian Yukawa From The Dim-4 Space

Current status: static controls exist, but the general Hermitian operator is
not complete.

Done:

- Charge-shift map basis exists.
- Static Hermitian rank-control probes exist.
- Neutral VEV preserves electromagnetism in the static audit.

Still open:

- General `Y(Phi)` construction.
- Complex coordinate convention for `Phi`.
- Hermiticity proof for arbitrary `Phi`.
- Mass-gap diagnostics for representative Higgs backgrounds.

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

After Session 40:

- Decide whether the next project goal is physics validation
  (anomaly/Lorentz/renormalization) or numerical simulation scale-up.
