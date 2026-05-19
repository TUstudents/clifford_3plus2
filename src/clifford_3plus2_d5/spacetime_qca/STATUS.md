# spacetime_qca — Status

**Status**: in progress. Sessions 20-39 complete through finite real-space
BCC stepping, representation-level Higgs/Yukawa audit, position-dependent
background gauge covariance, BCC plaquette holonomy geometry, and a static
Higgs/Yukawa map-layer audit, a JAX numerical backend, Wilson plaquette
observables/action normalization, SO(2) Wilson-action gradients, and SU(2)
nonabelian Wilson-force controls with compact left-trivialized descent and
reversible leapfrog dynamics, plus compact SU(3) force/descent and reversible
leapfrog controls, and basis-based chiral16 Pati-Salam SU(4) compact
dynamics plus Pati-Salam/SM subgroup adapters, a no-backreaction
fermion/gauge coupling wrapper, a first Gauss-law/backreaction prototype, and
a static Hermitian Yukawa `Y(Phi)` layer with JAX parity, plus a site-local
Higgs field layer with finite `SU(2)_L x U(1)_Y` gauge covariance,
BCC covariant differences, kinetic/potential diagnostics, and a sitewise
Yukawa bridge.

This module builds the 3D spatial side of the QCA: a BCC Weyl walk
(Bialynicki-Birula 1994) and its chiral assembly into a 4D Dirac carrier,
with a constant-background tensor lift against `lepton`'s chiral-16 internal
carrier.

Carrier convention: the internal chiral-16 is represented in the exact real
`R^32` basis supplied by `lepton`, with its compatible `J` carrying the
`C^16` interpretation.  Current tensor-lift matrices therefore use
`C^4_Dirac x R^32_internal` and have size `128 x 128` over the Bloch complex
scalars.  A compressed explicit `C^16_internal` basis is not implemented yet.

## What exists

- `bcc_geometry.py` — 8 body-diagonal BCC vectors, Brillouin-zone sample points.
- `pauli.py` — 2×2 Pauli matrices.
- `bcc_weyl.py` — Bialynicki-Birula hop matrices and 2×2 Weyl Bloch symbol.
- `dirac.py` — chiral-basis Dirac assembly.
- `hypercube_control.py` — naive central-difference cubic control Hamiltonian.
- `continuum.py` — first-order expansion utilities.
- `gauge_lift.py` — constant-background internal gauge tensor lift.
- `links.py` — finite-lattice position-dependent internal link fields.
- `plaquette.py` — BCC elementary plaquette shapes and holonomy covariance.
- `mass.py` — Dirac mass-layer and gauge-compatibility helpers.
- `yukawa.py` — representation-level Higgs/Yukawa charge-shift, static
  controls, and Hermitian `Y(Phi)` audits.
- `jax_state.py`, `jax_links.py` — compatibility wrappers and exact-backend
  converters over shared `sim` JAX state/link helpers.
- `jax_step.py` — BCC-specific numerical Dirac step kernels using shared
  `sim` roll/link primitives.
- `wilson.py`, `jax_wilson.py` — exact and numerical Wilson plaquette
  observables and action densities.
- `jax_gauge_force.py` — SO(2), SU(2), SU(3), and generic basis-based
  compact-link JAX Wilson-action gradients/forces, left-trivialized force
  controls, and compact action descent.
- `jax_gauge_dynamics.py` — SU(2)/SU(3) and generic basis-coordinate momentum
  fields, Hamiltonian-density helpers, and compact leapfrog updates.
- `jax_patisalam.py` — chiral16 Pati-Salam and SM sector JAX adapters over
  the exact `lepton` generator bases.
- `jax_fermion_gauge.py` — no-backreaction coupling between the BCC Dirac
  fermion step and evolving Pati-Salam/SM links.
- `jax_gauss.py` — Gauss residual, fermion charge density, finite-difference
  matter current, and explicit momentum-source kick helpers.
- `jax_yukawa.py` — numerical mirrors of the static Hermitian `Y(Phi)` layer.
- `jax_higgs.py` — site-local Higgs field, electroweak gauge transforms,
  BCC covariant differences, Higgs energy diagnostics, and sitewise Yukawa
  bridge helpers.
- `lattice.py`, `state.py`, `step.py` — finite periodic real-space BCC step.
- `audit.py` — result payloads for the report.
- `SESSION_20_BCC_DIRAC.md` — Session 20 result report.
- `SESSION_21_MASS_LAYER.md` — Session 21 result report.
- `SESSION_22_REAL_SPACE_STEP.md` — Session 22 result report.
- `SESSION_23_YUKAWA_REPRESENTATION.md` — Session 23 result report.
- `SESSION_24_GAUGE_COVARIANCE.md` — Session 24 result report.
- `SESSION_24B_PLAQUETTE_HOLONOMY.md` — Session 24b result report.
- `SESSION_25_STATIC_YUKAWA.md` — Session 25 result report.
- `SESSION_26_JAX_BACKEND.md` — Session 26 result report.
- `SESSION_27_WILSON_OBSERVABLES.md` — Session 27 result report.
- `SESSION_28_WILSON_ACTION.md` — Session 28 result report.
- `SESSION_29_WILSON_GRADIENTS.md` — Session 29 result report.
- `SESSION_30_SU2_FORCE.md` — Session 30 result report.
- `SESSION_31_SU2_LEFT_FORCE.md` — Session 31 result report.
- `SESSION_32_SU2_LEAPFROG.md` — Session 32 result report.
- `SESSION_33_SU3_DYNAMICS.md` — Session 33 result report.
- `SESSION_34_PATISALAM_SU4_DYNAMICS.md` — Session 34 result report.
- `SESSION_35_SM_GAUGE_ADAPTERS.md` — Session 35 result report.
- `SESSION_36_FERMION_GAUGE_COUPLING.md` — Session 36 result report.
- `SESSION_37_GAUSS_BACKREACTION_PLAN.md` — Session 37 implementation plan.
- `SESSION_37_GAUSS_BACKREACTION.md` — Session 37 result report.
- `SESSION_38_HERMITIAN_YUKAWA.md` — Session 38 result report.
- `SESSION_39_DYNAMICAL_HIGGS.md` — Session 39 result report.
- `ROADMAP.md` — roadmap from no-backreaction coupling toward constrained
  gauge/fermion/Higgs dynamics.
- 221 passing tests; slow exact/JAX bridge tests are marked with `slow`.

## Session 20 result

- Bialynicki-Birula hop matrices are pinned to Phys. Rev. D 49, 6920
  (1994), Section II.
- Right Weyl block has first-order Hamiltonian `H_R(k) = sigma . k`.
- Left Weyl block has first-order Hamiltonian `H_L(k) = -sigma . k`.
- Chiral Dirac assembly has `H_D(k) = alpha . k`.
- Hypercube control has 8 literal cubic Brillouin-zone corner doublers.
- BCC cubic-corner gapless representatives are reciprocal-lattice origin
  equivalents for the body-diagonal lattice.
- BCC Bloch-symbol unitarity is sample-checked; the all-momentum unitarity
  claim is inherited from the pinned Bialynicki-Birula source convention.
- The hypercube control is Hamiltonian-form, not a unitary cubic walk; it is
  used as the exact naive-lattice doubling diagnostic.
- Constant background gauge lift gives
  `H_eff = alpha.k x I_internal + I_spacetime x iA`.
- The gauge lift is tested with a real 32×32 Pati-Salam `SU(2)_L` generator
  from `lepton`.

## Session 22 result

- Periodic 3D lattice wrapper implemented.
- Pull-form Weyl step implemented:
  `out[x] = sum_h W(h) psi[x+h]`.
- Dirac step implemented as blockwise right/left Weyl steps.
- Plane-wave states match the BCC Weyl/Dirac Bloch symbols exactly.
- Weyl and Dirac steps preserve norm on deterministic finite-lattice
  plane-wave states.
- Constant identity-link tensor step matches ungauged Dirac stepping.

## What is open

- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Full symbolic all-momentum BCC Bloch-symbol unitarity proof.
- Explicit `J`-adapted `C^16` internal basis, if we want code dimensions to
  match the compressed complex carrier rather than the current `R^32` real
  form.
- Vectorized Wilson action evaluation for large lattices.
- Site-local dynamical Higgs field with gauge-covariant kinetic/potential
  diagnostics.
- Dynamical gauge fields beyond the current SU(2)/SU(3)/SU(4) leapfrog
  prototypes.
- Full Gauss-law projection / constraint solving beyond the Session 37
  residual and source-kick prototype.
- Vectorized SU(2) staple force and Gauss-law constraints.
- Lorentz boost recovery beyond the `alpha . k` continuum precursor.
- Numerical performance benchmarks and long-time stability tests.

## Sessions ahead

- Session 20b: full symbolic BCC unitarity and no-doubling hardening.
- Session 39: site-local dynamical Higgs field with gauge covariance.
- Session 40: first coupled fermion + gauge + Higgs step on tiny lattices.
- Session 41+: anomaly/current diagnostics, Lorentz recovery, scaling, and
  performance work.

See [PLAN.md](PLAN.md) for the detailed Session 20 plan and
[SESSION_20_BCC_DIRAC.md](SESSION_20_BCC_DIRAC.md) for the initial running
report.  See [ROADMAP.md](ROADMAP.md) for the current post-Session-38 roadmap.

## Cross-module dependency

The Session 20 gauge-lift test and Session 21 mass-compatibility tests import
real Pati-Salam / SM generators from `lepton`.  The spacetime pieces remain
factored, and future coupling should continue through explicit tensor-lift
interfaces.

The generic JAX simulation infrastructure now lives in `sim`.  Physics-specific
BCC Dirac, BCC Wilson, and Wilson-force policy remains in `spacetime_qca`.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -m "not slow" -q
```

Expected fast-path result: `96 passed, 117 deselected` in about 45 seconds on
the current CPU environment.

Slow exact-symbolic and JAX dynamics/parity suites are marked with
`pytest.mark.slow`.
Run the full module suite, including slow tests, with:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
```

Expected full result: `213 passed`.  The full run currently takes several
minutes because it includes exact Higgs-map nullspace construction and JAX
gradient/leapfrog checks.

On memory-constrained machines, prefer running the JAX dynamics files in
smaller groups.  JAX compilation caches can accumulate across the full
`spacetime_qca` suite.  The SU(3) force path currently differentiates through
batched matrix exponentials; the chiral16 SU(4) path defaults to a slower but
memory-safe finite-difference force.

## Session 21 result

- `beta = gamma^0` satisfies `beta^2 = I`.
- `beta` anticommutes with all three `alpha_i`.
- Scalar mass control obeys
  `H_m(k)^2 = (k_x^2 + k_y^2 + k_z^2 + m^2) I`.
- At `k = 0`, the spectrum is `+m` and `-m` with the expected
  Dirac/internal multiplicities.
- Scalar internal mass commutes with Pati-Salam and SM gauge generators from
  `lepton`.
- A non-scalar projector mass breaks at least one SM generator, verifying the
  compatibility test is sensitive.

Interpretation: scalar mass validates the Dirac mass slot
`beta x M_internal`, but it is not an SM Yukawa/Higgs mass spectrum.

## Session 23 result

- `beta = gamma^0` is off-diagonal between spacetime chiralities.
- The existing lepton `(Y, T3_L)` joint table is used as the SM charge
  decomposition.
- Universal scalar mass preserves all SM sectors but is not Higgs-like.
- A non-scalar projector control breaks at least one SM sector.
- An exact color-singlet internal map exists with
  `Delta Y = +1/2` and `Delta T3_L = +1/2`.
- Its transpose gives the conjugate component with
  `Delta Y = -1/2` and `Delta T3_L = -1/2`.
- The Higgs-like charge-shift solution space has real dimension `4`.

Interpretation: the chiral-16 internal carrier contains a representation-level
Higgs-like map slot.  The transpose result is a real-linear charge-shift
conjugate, not a complex Hermitian-conjugate Higgs field.  The 4-real-dim
solution space has not yet been decomposed as an `SU(2)_L` Higgs doublet.
This is not yet a dynamical Higgs/Yukawa QCA layer.

## Session 25 result

- The upper Higgs-like map space has real dimension `4` and charge shift
  `(Delta Y, Delta T3_L) = (+1/2, +1/2)`.
- Acting with the non-Cartan `SU(2)_L` generators produces the lower
  component, also real dimension `4`, with
  `(Delta Y, Delta T3_L) = (+1/2, -1/2)`.
- The combined upper/lower map module has real dimension `8`.
- Static real-form controls built as `M + M.T` are real symmetric, and
  `beta x Y_static` is Hermitian.
- The neutral lower-component VEV control preserves color and
  `Q_em = Y + T3_L` while breaking `Y` and `T3_L` separately.
- The default static controls are rank `2` with nullity `30`, so they are
  low-rank background probes, not realistic mass matrices.

Interpretation: the module now has the exact static Higgs-doublet charge
structure and neutral-VEV breaking pattern
`SU(2)_L x U(1)_Y -> U(1)_em`.  It is still not a dynamical Higgs field or a
full Yukawa mass spectrum.

## Session 26 result

- JAX state layout implemented for flat Dirac states and
  `Dirac x internal` tensor states.
- JAX link-field layout implemented as `(nx, ny, nz, 8, d, d)` in the same
  BCC pull convention as the exact backend.
- Ungauged JAX BCC Dirac step matches exact SymPy `dirac_step` on a delta
  state.
- JAX identity-link tensor step matches exact `dirac_step_with_constant_link`.
- JAX position-dependent link step matches exact `dirac_step_with_link_field`.
- The constant-link shortcut matches an explicitly broadcast link field.
- The linked JAX step is `jax.jit` compilable and runs on the default JAX
  device.

Interpretation: `spacetime_qca` now has a numerical backend suitable for
finite-lattice simulation and later dynamical-field experiments.  The exact
SymPy backend remains the reference implementation.

## Session 27 result

- Exact Wilson-loop trace and normalized trace helpers implemented on top of
  BCC plaquette holonomy.
- Exact average normalized Wilson loop implemented over all sites and the six
  canonical BCC plaquette shapes.
- JAX plaquette holonomy, Wilson trace, normalized trace, and average helpers
  implemented for `(nx, ny, nz, 8, d, d)` link fields.
- Identity links give normalized Wilson loop `1`.
- Pure-gauge links give average normalized Wilson loop `1`.
- Exact and JAX Wilson traces are invariant under finite site-local gauge
  transforms.
- JAX Wilson observables match exact SymPy observables on tiny lattices.

Interpretation: the module now has the gauge-invariant curvature-observable
layer needed before a Wilson action or dynamical gauge update.

## Session 28 result

- Exact Wilson plaquette energy implemented as
  `1 - Re(Tr(H) / internal_dim)`.
- Exact average action density and total Wilson action implemented with
  configurable `beta`.
- JAX plaquette energy, average action density, and total action implemented
  with the same normalization.
- Identity and pure-gauge links have zero action density.
- A nontrivial sign/flip link field has nonzero, nonnegative action density.
- Exact and JAX action values are invariant under finite site-local gauge
  transforms.
- JAX action values match exact SymPy values on tiny lattices.

Interpretation: the module now has a fixed-background gauge-energy observable.
This is the last observable layer before gradient/dynamics work; it is still
not a dynamical gauge-field update.

## Session 29 result

- SO(2) BCC link fields are parameterized by compact link angles.
- Pure-gauge SO(2) fields use the existing pull-link convention
  `theta[x,h] = phi[x] - phi[x+h]`.
- JAX Wilson action gradients are implemented with `jax.grad`.
- Zero fields and pure gauges have zero action density and zero gradient.
- A deterministic non-flat field has positive action density and non-zero
  gradient.
- The analytic JAX gradient matches centered directional finite differences.
- The action gradient is orthogonal to pure-gauge perturbation directions.
- The gradient helper is `jax.jit` compilable.

Interpretation: the Wilson-action layer is now differentiable in compact
SO(2) link variables and has the expected abelian gauge-force controls.  This
is not yet a nonabelian force projection or a dynamical gauge update.

## Session 30 result

- Fundamental SU(2) generators `T_a = -i sigma_a / 2` are implemented in JAX.
- Compact SU(2) links are built from Lie-algebra coordinates using a closed
  form exponential with a stable small-angle branch.
- Site-local SU(2) gauge fields and finite link-field gauge transforms are
  implemented in the existing BCC pull convention.
- Pure-gauge SU(2) links can be generated from site-local gauges.
- SU(2) Wilson action density and coordinate gradients are implemented with
  `jax.grad`.
- Zero fields and Cartan-subgroup pure-gauge coordinates have zero action and
  zero gradient.
- Full finite pure-gauge links have zero action.
- A deterministic non-flat SU(2) field has positive action and non-zero
  gradient.
- The SU(2) gradient matches centered directional finite differences.
- Finite site-local SU(2) gauge transforms preserve the Wilson action density.
- The SU(2) gradient helper is `jax.jit` compilable.

Interpretation: the Wilson-action layer now supports a compact nonabelian
force audit.  This is still not a left-trivialized force projection or a
dynamical gauge-field update.

## Session 31 result

- Anti-Hermitian traceless `2 x 2` matrices can be projected back to SU(2)
  generator coordinates with the convention
  `theta_a = 2 Re Tr(T_a^dagger A)`.
- The SU(2) Wilson action now exposes a left-trivialized force for compact
  perturbations `U[x,h] -> exp(omega_a T_a) U[x,h]`.
- Zero links and finite pure-gauge SU(2) links have zero left force.
- The left force matches a centered finite difference along a deterministic
  left-perturbation direction.
- A deterministic non-flat SU(2) field has non-zero left force.
- Compact left updates
  `U -> exp(-eta force_a T_a) U` preserve unitarity and determinant `1`.
- One deterministic compact descent step lowers the Wilson action.
- The descended action is invariant under finite site-local gauge transforms.
- The left-force helper is `jax.jit` compilable.

Interpretation: the module now has the first compact nonabelian update control:
curvature observable, Wilson action, left-trivialized force, and compact
descent are all present.  This is still steepest-descent infrastructure, not a
physical gauge-field evolution rule.

## Session 32 result

- SU(2) momentum coordinates lift to anti-Hermitian traceless matrices.
- Target-site adjoint gauge transforms of momentum fields preserve kinetic
  energy density.
- The SU(2) gauge Hamiltonian-density helper combines
  `0.5 mean_link ||P||^2` with `beta * S_W_density`.
- Compact momentum updates apply `U -> exp(dt P_a T_a) U`.
- The leapfrog update uses half momentum, full compact link, half momentum
  updates against the Session 31 left force.
- Identity links and finite pure-gauge links with zero momenta are fixed by
  leapfrog.
- Momentum and leapfrog updates preserve link unitarity and determinant `1`.
- Forward then negative-time leapfrog recovers links and momenta within
  numerical tolerance.
- Hamiltonian-density drift is finite and smaller at smaller step size.
- Leapfrog is covariant under finite site-local gauge transforms.
- The leapfrog helper is `jax.jit` compilable.

Interpretation: the module now has a first reversible compact SU(2)
gauge-field dynamics prototype.  It is still not a full physical Yang-Mills
QCA: there is no Gauss-law projection, no fermion backreaction, and no SU(4) or
Pati-Salam force projection yet.

## Session 33 result

- Fundamental SU(3) generators `T_a = -i lambda_a / 2` are implemented in JAX.
- Anti-Hermitian traceless `3 x 3` matrices can be projected back to SU(3)
  generator coordinates with
  `theta_a = 2 Re Tr(T_a^dagger A)`.
- Compact SU(3) links are built from Lie-algebra coordinates with batched
  matrix exponentials.
- Site-local SU(3) gauge fields and finite pure-gauge link fields are
  implemented in the BCC pull convention.
- Zero links and finite pure-gauge SU(3) links have zero Wilson action and
  zero left force.
- A deterministic non-flat SU(3) field has non-zero left force.
- The left force matches a centered compact left directional finite
  difference.
- Compact left descent preserves SU(3) links and lowers the Wilson action.
- SU(3) momentum coordinates lift to anti-Hermitian traceless matrices.
- Target-site adjoint gauge transforms of momentum fields preserve kinetic
  energy density.
- The SU(3) gauge Hamiltonian-density helper combines
  `0.5 mean_link ||P||^2` with `beta * S_W_density`.
- Compact momentum updates and leapfrog updates preserve link unitarity and
  determinant `1`.
- Forward then negative-time leapfrog recovers links and momenta within
  numerical tolerance.
- Hamiltonian-density drift is finite and smaller at smaller step size.
- Leapfrog is covariant under finite site-local SU(3) gauge transforms.
- The compact SU(3) momentum update is `jax.jit` compilable.

Interpretation: the module now has a compact SU(3) color-gauge dynamics
prototype matching the SU(2) force/leapfrog convention.  It is still a gauge
simulation-control layer: no Gauss-law projection, no fermion backreaction,
and no fermion-coupled compact dynamics yet.  The SU(3) left force still uses
reverse-mode autodiff through compact perturbation exponentials, so
JIT-compiling the full SU(3) leapfrog is intentionally not part of the default
test suite until a staple/vectorized force is implemented.

## Session 34 result

- A public basis-based compact Lie JAX API now builds algebra matrices, compact
  links, site gauges, pure gauges, left forces, compact descent, momentum
  transforms, Hamiltonian densities, momentum updates, and leapfrog steps from
  an explicit generator basis.
- Coordinate projection uses the real Hilbert-Schmidt Gram matrix, so
  non-fundamental and non-unit-normalized representations are handled
  correctly.
- The generic momentum kinetic energy uses the same Gram metric, not a naive
  coordinate Euclidean norm.
- `jax_patisalam.py` adapts the exact `lepton` chiral16 `Spin(0,6) ~= SU(4)`
  basis to JAX.
- The Pati-Salam SU(4) adapter exposes compact links in the `32 x 32`
  chiral16 representation with only 15 algebra coordinates.
- Chiral16 SU(4) projection recovers all 15 coordinates.
- Chiral16 SU(4) compact links preserve unitarity.
- Chiral16 SU(4) pure gauges have zero Wilson action.
- Chiral16 SU(4) finite-difference left force is finite and lowers a
  deterministic non-flat Wilson action under compact descent.
- Chiral16 SU(4) momentum transforms preserve Gram-metric kinetic energy.
- Chiral16 SU(4) compact momentum updates and leapfrog updates preserve
  compact links.
- The Pati-Salam SU(4) left force defaults to finite differences to avoid the
  memory failure mode from reverse-mode autodiff through many `32 x 32`
  exponentials.

Interpretation: the Pati-Salam SU(4) force-projection gap is closed without
pretending that a `32 x 32` chiral16 representation is fundamental SU(32).
This is still a compact gauge-dynamics control; Gauss-law projection, fermion
backreaction, and dynamical Higgs/Yukawa fields remain open.

## Session 35 result

- `jax_patisalam.py` now exposes a sector-generic chiral16 gauge adapter.
- Supported sectors are `su4`, `su2_l`, `su2_r`, `pati_salam`, `su3_c`,
  `u1_y`, and `sm`.
- Existing `jax_patisalam_su4_*` functions remain stable as wrappers over
  `sector="su4"`.
- Every sector uses explicit `lepton` generator bases and the Session 34
  Gram-metric projection.
- Sector dimensions are `15`, `3`, `3`, `21`, `8`, `1`, and `12`
  respectively.
- Projection recovers coordinates, compact exponentials preserve unitarity,
  and pure gauges have zero Wilson action in the representative sectors.
- Independent SM factor links commute for `SU(3)_c`, `SU(2)_L`, and `U(1)_Y`.
- Representative finite-difference forces lower non-flat Wilson action for
  `SU(2)_L`, `SU(3)_c`, and `U(1)_Y`.
- Momentum transforms preserve Gram-metric kinetic energy for representative
  sectors and for the combined `sm` basis.

Interpretation: the module now has a uniform compact gauge adapter for the
full Pati-Salam and SM gauge-content layer.  Combined `pati_salam` and `sm`
sectors are representation-level convenience bases; independent physical
couplings and Gauss-law constraints remain future work.

## Session 36 result

- `jax_fermion_gauge.py` adds the first no-backreaction coupling between the
  BCC Dirac fermion step and evolving Pati-Salam/SM links.
- `jax_patisalam_dirac_step` validates and applies the BCC Dirac step through
  `(nx, ny, nz, 8, 32, 32)` chiral16 links.
- `jax_patisalam_fermion_gauge_step` evolves `(links, momenta)` by pure-gauge
  leapfrog, then transports the fermion state through the updated links.
- `jax_transform_patisalam_dirac_state` supplies the site-local internal gauge
  transform for `(nx, ny, nz, 4, 32)` states.
- Identity links and zero momenta reduce the wrapper to the ordinary Dirac
  step.
- A zero fermion state leaves gauge evolution identical to pure-gauge
  leapfrog, making the no-backreaction convention explicit.
- The Dirac step is site-local gauge covariant, and the zero-time coupled
  wrapper is covariant under the existing pull-link convention.

Interpretation: the module now has an executable bridge between fermion
transport and evolving Pati-Salam/SM link fields.  It is still a
no-backreaction background-gauge simulation wrapper: fermion current,
Gauss-law constraints, and dynamical Higgs/Yukawa coupling remain the next
roadmap items.

## Session 37 result

- `jax_gauss.py` adds a target-site electric-divergence residual in the BCC
  pull-link convention.
- Link momenta can be converted from sector coordinates to anti-Hermitian
  `32 x 32` algebra matrices.
- Fermion charge density is computed from `Q_a = i T_a`, with both raw moments
  and Gram-dual basis coordinates.
- The Gauss residual is `divE - g rho`.
- A finite-difference link current is defined from the compact left variation
  of `Re <psi, D_U psi>`.
- `jax_patisalam_apply_fermion_backreaction` applies the explicit source kick
  `P -> P + dt * g * J`.
- `jax_patisalam_fermion_gauge_step_with_backreaction` adds the current kick
  before calling the Session 36 no-backreaction wrapper.
- Focused tests verify anti-Hermiticity, zero-divergence/zero-charge controls,
  gauge covariance of divergence, charge, and residual, zero-state current,
  finite `U(1)_Y` current invariance, nonabelian current covariance, linear
  source kicks, and zero-coupling regression to Session 36.

Interpretation: this is the first constrained matter/gauge prototype.  It is
not yet a full Gauss-law projector or production dynamical gauge simulation:
analytic current formulas, constraint solving, physical coupling constants,
and long-time stability remain open.

## Session 38 result

- `yukawa.py` adds a deterministic two-complex static Higgs API:
  `Phi = (phi_plus, phi_zero)` with exact `(real, imag)` SymPy coordinates.
- The selected map slice is `(upper[0], upper[1], lower[0], lower[1])` from
  the existing Higgs-like doublet map spaces.
- `higgs_phi_raising_map` builds the non-Hermitian charge-raising map
  `A(Phi)`.
- `hermitian_yukawa_internal_control` builds
  `Y_internal(Phi) = A(Phi) + A(Phi)^T`.
- `hermitian_yukawa_hamiltonian` builds `beta x Y_internal(Phi)`.
- Neutral helpers implement the static `Phi = (0, vev)` direction.
- `jax_yukawa.py` mirrors the construction for numerical code with native
  complex scalar inputs.
- Focused tests verify zero `Phi`, linearity, Hermiticity, neutral-VEV
  `Q_em` preservation, charged-component `Q_em` breaking, and JAX/SymPy
  parity.

Interpretation: this closes the static Yukawa insertion gap.  It is still a
fixed-background Hermitian Yukawa layer, not a site-local dynamical Higgs
field, Higgs potential, or Yukawa hierarchy.

## Session 39 result

- `jax_higgs.py` adds a site-local Higgs field layout:
  `Phi.shape == (nx, ny, nz, 2)` with `(phi_plus, phi_zero)`.
- The finite gauge coordinate order is
  `(su2_x, su2_y, su2_z, u1_y)`.
- Fundamental anti-Hermitian generators use
  `T_a = -i sigma_a / 2` and `T_Y = -i I_2 / 2`, so
  `i(T_3 + T_Y)` has charges `(+1, 0)`.
- BCC covariant differences use the existing pull-link convention:
  `D_h Phi[x] = U[x,h] Phi[x+h] - Phi[x]`.
- Kinetic and Mexican-hat potential diagnostics are implemented sitewise.
- Pure-gauge links make a gauge-transformed constant Higgs field covariantly
  constant.
- Sitewise bridge helpers evaluate Session 38's `Y(Phi[x])` and
  `beta x Y(Phi[x])` on every lattice site.
- The heavy sitewise Yukawa bridge tests are marked `slow`; the gauge-field
  and potential tests stay in the fast suite.

Interpretation: this closes the first dynamical-Higgs infrastructure gap.
`Phi(x)` is now a gauge-covariant lattice field with energy diagnostics.  It is
not yet a complete Higgs time-evolution rule, and it is not yet inserted into
the coupled fermion/gauge update.

## Session 24 result

- Position-dependent internal link fields are keyed by
  `(target_site, displacement)` in the pull convention
  `target_site <- target_site + displacement`.
- `dirac_step_with_link_field` implements
  `out[x] = sum_h (W_h x U[x <- x+h]) psi[x+h]`.
- Site-local gauge transforms act as
  `psi[x] -> (I_space x G[x]) psi[x]` and
  `U[x <- y] -> G[x] U[x <- y] G[y]^-1`.
- Exact finite-lattice covariance is tested:
  `step(G psi, GUG^-1) = G step(psi, U)`.
- Identity and constant link fields regress to the previous Session 22 paths.
- Pure-gauge links preserve norm on the deterministic plane-wave test state.

Interpretation: this is a background-link gauge-covariant finite QCA rule.
It is not yet a plaquette/curvature construction or a dynamical gauge field.

## Session 24b result

- Elementary BCC plaquettes are four-hop non-backtracking parallelograms
  `(a, b, -a, -b)` built from body-diagonal displacements.
- Modulo cyclic rotation and orientation reversal, there are six canonical
  unoriented elementary plaquette shapes.
- Plaquette holonomy uses the pull convention:
  `H[x0] = U[x0 <- x1] U[x1 <- x2] U[x2 <- x3] U[x3 <- x0]`.
- Holonomy transforms by base-site conjugation:
  `H[x0] -> G[x0] H[x0] G[x0]^-1`.
- Identity and pure-gauge link fields have identity holonomy.

Interpretation: the module now has the BCC loop geometry needed for curvature
observables.  This is not yet a plaquette action or dynamical gauge-field
update rule.
