# Project Status

One-page status of each module in this workspace.

## obstruction_r10 — frozen for publication

**Result**: Five propositions and two corollaries closing specified primitive
classes for the carrier-first QCA derivation of the Spin(10) chiral-16 on
`R^10`.

| # | Statement | Status | Witness |
|---|---|---|---|
| Prop 1 | Coarse block-blind primitives produce no rule-derived (6,4) center. | Structurally closed | E1/E2 sprint, `explore_rule_space.py` |
| Prop 2 | Commuting non-scalar second-layer locks produce lower-rank central idempotents. | Structurally closed | Proof (Bezout + commutativity) |
| Prop 3 | Floquet-α-type noncommuting on-site rules force lower-rank center, full addressability, or commutativity. | Structurally closed | Wedderburn proof + 3840 → 0 exhaustive scan |
| Cor 3.1 | On-site bridge closure under Floquet-α hypothesis. | Structural consequence | Combines Props 1, 2, 3 |
| Prop 4a | Full 2400-candidate projector-free monomial-hop Bloch census: 1320 closed candidates, all with split-real projected centralizers, zero compatible-J. | Empirically closed (finite census) | `bloch_path_a_stepwise.py`, 10 batch summaries |
| Lemma 4b.1 | Split-real projected centralizer admits no real orthogonal J. | Structurally closed | Character proof (one paragraph) |
| Prop 4b | Coprime monomial-hop incompatibility (all coprime cases have split-real centralizers). | **Conjectural** with finite witness | Open structural extension |
| Prop 5 | Route-1 compatible J orbits are SM-inequivalent; gauge-equivalence relaxation fails for fixed SU(3)×SU(2)×U(1). | Structurally closed | Hodge-complement argument + `gauge_equivalence_check.py` |

**Next action**: publication. See [`docs/PUBLICATION_PLAN.md`](docs/PUBLICATION_PLAN.md).

**No further research expected** on closed primitive classes. The open
mechanism classes (non-translation-invariant defects, parameterized rule
families, higher-dim coarse-graining, Proposition 4b structural extension)
are identified for future work but are not gated on this paper.

## lepton — positive result, active development possible

**Result**: Cl(0,10) factored as `Cl(0,6) ⊗ Cl(0,4)` (Pati-Salam) produces:

- Chiral-16 of Spin(10) carrier on R^32.
- Compatible J from the Cl(0,4) right-quaternionic commutant.
- Full Pati-Salam algebra `SU(4) × SU(2)_L × SU(2)_R` (dim 21).
- SM gauge group `SU(3)_c × SU(2)_L × U(1)_Y` after PS → SM breaking.
- Correct one-generation hypercharge spectrum: `{1/6: 6, -2/3: 3, 1/3: 3, -1/2: 2, 1: 1, 0: 1}`.
- 1+1D massless Dirac/Weyl continuum dynamics with covariant background gauge.

**Sessions completed**: 13 (Cl(8) octonion audit), 14 (rigid Clifford dynamics),
15 (Lie data), 16 (1+1D checkerboard continuum), 17 (chiral-16 lift),
18 (Pati-Salam factorization), 19a (SM algebra extraction), 19b (hypercharge
table verification).

**Declared choices** (documented, not derived):
- Cl(0,10) signature.
- Cl(0,6) ⊗ Cl(0,4) Pati-Salam factorization.
- Fano octonion multiplication table.
- e_7 imaginary direction.
- Right-quaternionic J.
- T_{3R} factor 1/2 and B-L factor 2/3 normalizations.
- PS → SM breaking pattern.

**Open**: mass / Yukawa, dynamical gauge fields, 3+1D Lorentz boost recovery,
three generations.

**Tests**: 120+ green.

## sim — shared infrastructure

**Goal**: shared JAX simulation helpers that can be reused by sidecars without
encoding Pati-Salam, Spin(10), BCC Dirac, Wilson plaquette, or gauge-force
physics policy.

**Implemented**:
- JAX dtype/device helpers and compile/run timing.
- Periodic 3D pull-roll helpers.
- Generic JAX state allocation, flattening, and norm diagnostics.
- Generic identity/constant pull-link fields and finite site-local gauge
  transforms.
- Basic finite-value/state-transition diagnostics and benchmark wrapper.
- Physics-agnostic recorded loop/scan runners.  The scan runner now advances
  between requested record points and computes observables only for recorded
  steps.
- Generic `.npz` plus JSON sidecar persistence and JSON-safe profiling
  payloads.

**Boundary**: BCC Weyl/Dirac kernels, BCC plaquettes, Wilson observables, and
SO(2)/SU(2)/SU(3) force policy remain in `spacetime_qca`.

## qca_smv0 — Stage 39 physical-right production local-force recorded rollout

**Goal**: focused simulator sidecar for the next Standard-Model QCA prototype,
using the shared `sim` infrastructure and local `qca_smv0` kernels only.

**Implemented**:
- Package scaffold.
- `README.md`, `PLAN.md`, and `STATUS.md`.
- `scripts/` and `tests/` directories.
- Two-component free BCC Weyl state layout `(nx, ny, nz, 2)`.
- Pauli/projector helpers and eight BCC hop matrices.
- Pull-convention JAX step `psi_next[x]=sum_h A_h psi[x+h]`.
- Split-product and hop-sum Bloch symbols.
- Focused diagnostics for hop completeness, symbol unitarity, norm drift, and
- small-k Weyl speed.
- Directional anisotropy scaling audit: phase-speed spread decreases under
  momentum halving toward the continuum.
- Massless Dirac assembly from two opposite-chirality Weyl blocks, with Dirac
  hop completeness, symbol unitarity, periodic norm conservation, and JIT
  compatibility.
- Session 01 script and focused tests.
- Local 32-component SM internal carrier.
- Anti-Hermitian `SU(3)_c x SU(2)_L x U(1)_Y` generator basis.
- Static BCC SM link fields with shape `(nx, ny, nz, 8, 32, 32)`.
- Static gauge-covariant Dirac BCC transport and site-local gauge
  transformations.
- Identity-link reduction to the free internal Dirac step.
- Wilson plaquette traces over selected BCC parallelograms.
- Weak-link linearization check for continuum covariant derivative scaling.
- Session 02 script and focused tests.
- Real SM link momenta with shape `(nx, ny, nz, 8, 12)`.
- Algebra-coordinate projection and target-site adjoint momentum gauge
  transforms.
- Legacy finite-difference left Wilson-force oracle.
- Local analytic BCC plaquette-staple force as the production
  left-trivialized Wilson force.
- Pure-gauge Hamiltonian density and reversible leapfrog update.
- Electric-divergence / Gauss covariance diagnostics.
- Pure-gauge zero-momentum Gauss-preservation audit.
- Weak-field plaquette field-strength linearization and Wilson-action /
  Yang-Mills-density matching.
- No-backreaction fermion/gauge wrapper with spectator fermion transport.
- Session 03 script and focused tests.
- Local Higgs doublet fields with shape `(nx, ny, nz, 2)`.
- Constant unitary-gauge Higgs helper `H=(0,v/sqrt(2))`.
- Conjugate Higgs helper `H_tilde=i sigma_2 H^*`.
- Hermitian one-generation Yukawa matrix on the 32-component SM carrier.
- Unitary-gauge SM door audit: `H_tilde` opens up/neutrino couplings and `H`
  opens down/electron couplings.
- Exact site-local Higgs/Yukawa collision `exp(-i step_size beta Y(H))`.
- Zero-step and zero-Higgs identity controls.
- Chirality-flip and massive-dispersion diagnostics.
- Session 04 script and focused tests.
- Family dimension `3`.
- Empirical Wolfenstein and shear-candidate attenuation inputs.
- Local two-state unitary beam splitter for one FN recirculation step.
- Finite hidden-path unitary chains with endpoint transfer `lambda^n`.
- Default simulator charges `Q=(3,2,0)`, `U=(5,2,0)`, `D=(1,0,0)`.
- Quark path-length matrices `n^u_ij=Q_i+U_j`, `n^d_ij=Q_i+D_j`.
- Standard diagonal FN scaling audits `lambda^8:lambda^4:1` and
  `lambda^4:lambda^2:1`.
- Left-frame Wolfenstein scaling `lambda^abs(Q_i-Q_j)`.
- Generated quark Yukawa matrices `Y_ij=c_ij lambda^(Q_i+R_j)`.
- Singular masses and CKM-like left-frame mismatch from the same generated
  matrices.
- Session 05 script and focused tests.
- `SU(3)_c` center phase helper `omega=exp(2 pi i / 3)`.
- Explicit up/down center-power matrices decorating FN order-one coefficients.
- Center phase unit-modulus and `phase^3=1` audits.
- Center-decorated coefficients preserving order-one magnitudes.
- Antiparticle Yukawas as complex conjugates.
- Quark/antiquark singular-mass equality audit.
- Nonzero CKM Jarlskog and nonzero commutator CP trace from center-decorated
  coefficients.
- All-real coefficient zero-Jarlskog control.
- Session 06 script and focused tests.
- Family-extended fermion state layout `(nx, ny, nz, 4, 32, 3)`.
- Local internal/family dimension `96`.
- Stage 5/6 generated quark Yukawa matrices embedded into the local Higgs
  doors.
- Explicit diagonal placeholder lepton matrices as simulator inputs.
- Hermitian three-family local Yukawa matrix.
- Unitary-gauge quark block extraction and wrong-door controls.
- CKM-like left-frame mismatch consistency from embedded blocks.
- Exact local three-family collision `exp(-i step_size beta Y_family(H))`.
- Zero-step, zero-Higgs, norm, chirality-flip, and JIT audits.
- Session 07 script and focused tests.
- Higgs electroweak `SU(2)_L x U(1)_Y` generator basis on the doublet.
- Finite Higgs BCC links with shape `(nx, ny, nz, 8, 2, 2)`.
- Higgs field and momentum layout `(nx, ny, nz, 2)`.
- Kinetic, gauge-covariant gradient, and quartic-potential energy densities.
- Covariant Higgs force.
- Unitary-gauge vacuum force audit.
- Pure-gauge vacuum gradient-energy audit.
- Force covariance and Hamiltonian gauge-invariance checks.
- Reversible no-fermion Higgs leapfrog update.
- Small-step Hamiltonian drift and JIT audits.
- Session 08 script and focused tests.
- Higgs electroweak link momenta with shape `(nx, ny, nz, 8, 4)`.
- Projection between Higgs algebra matrices and electroweak coordinates.
- Target-site adjoint transforms for Higgs link momenta.
- Left-trivialized Higgs gauge force from the covariant-gradient energy.
- Zero-force covariantly constant vacuum control and nonzero current control.
- Local Higgs charge density and Higgs Gauss diagnostic
  `electric divergence - Higgs charge`.
- Covariance checks for Higgs gauge force, charge, and Gauss diagnostic.
- Higgs electroweak momentum embedding into the full SM 12-generator momentum
  layout with zeros on color and components on `8..11`.
- Coupled no-fermion Higgs/gauge leapfrog with link-unitarity, reversibility,
  small Hamiltonian-drift, and JIT audits.
- Session 09 script and focused tests.
- Local Yukawa energy density `E_Y=Re psi^dagger beta Y(H) psi` from the same
  Stage 7 three-family Yukawa matrix used by the collision.
- Higgs source force `-dE_Y/dH*` from real/imaginary automatic differentiation.
- Deterministic fermion source state with nonzero Yukawa bilinears.
- Zero-state and nonzero-source controls.
- Explicit local Yukawa-door electroweak gauge convention using physical
  right-handed hypercharges for the covariance audit.
- Yukawa energy gauge-invariance and Higgs-source covariance checks in that
  local door convention.
- Reversible Higgs-momentum source kick and local collision-plus-kick wrapper.
- Fermion norm, source-after-collision, and JIT audits.
- Session 10 script and focused tests.
- BCC streaming bilinear `E_stream=Re sum_x,h psi[x]^dag D_h U_h[x]
  psi[x+h]`.
- Left-trivialized fermion gauge current from the streaming bilinear.
- Zero-state and nonzero-current controls.
- Streaming-energy gauge-invariance audit.
- Target-site adjoint covariance for the fermion current.
- Local streaming fermion charge density and Gauss diagnostic
  `electric divergence - fermion charge`.
- Reversible fermion-current momentum kick and kick-then-transport wrapper.
- Kicked-link unitarity, spectator fermion norm, and JIT audits.
- Session 11 script and focused tests.
- Coupled sourced SM link force as
  `Wilson + embedded Higgs gauge force + BCC streaming fermion current`.
- Side-by-side SM transport links and Higgs electroweak links updated from one
  12-coordinate SM momentum field.
- Sourced Gauss diagnostic
  `electric divergence - fermion charge - embedded Higgs charge`.
- Shared electroweak covariance checks for sourced force and sourced Gauss.
- Coupled tick with fermion transport, Higgs-field advance, SM/Higgs link
  unitarity, fermion norm, and JIT audits.
- Session 12 script and focused tests.
- Family-summed BCC streaming current on the Stage 7 family carrier
  `(nx, ny, nz, 4, 32, 3)`.
- Single-family embedding/extraction and one-family reduction controls against
  Stage 11 energy, current, charge, and transport.
- Family-summed local fermion charge and Gauss diagnostic
  `electric divergence - family charge`.
- Family-current covariance, reversible momentum kick, kick-then-transport,
  link unitarity, family-state norm, and JIT audits.
- Session 13 script and focused tests.
- Family-sourced SM link force as
  `Wilson + embedded Higgs gauge force + family-summed BCC fermion current`.
- One-family reduction controls against the Stage 12 sourced force, sourced
  Gauss diagnostic, and coupled tick.
- Family-sourced Gauss diagnostic
  `electric divergence - family charge - embedded Higgs charge`.
- Shared electroweak covariance checks for the family-sourced force and Gauss
  diagnostic.
- Reversible family-sourced momentum kick.
- Family-sourced tick with family transport, Higgs-field advance, SM/Higgs
  link unitarity, family-state norm, JIT audits, and Session 14 focused tests.
- Full family production tick merging the Stage 10 local Yukawa/Higgs source
  and exact Stage 7 family Yukawa collision into the Stage 14 family-sourced
  tick.
- Zero-Yukawa reduction controls against the Stage 14 family-sourced tick.
- Production Higgs force as Higgs dynamics plus local Yukawa Higgs source.
- Nonzero deterministic Yukawa-source and zero-state/vacuum-source controls.
- Reversible production Higgs momentum kick for frozen fields.
- Production tick with symmetric local half-collision, BCC family transport,
  Higgs-field advance, SM/Higgs link unitarity, family-state norm, and JIT
  audits.
- Explicit status boundary preserving the separation between the Stage 10
  physical local Higgs-door gauge convention and the Stage 14 transport/current
  gauge convention.
- Session 15 script and focused tests.
- Explicit Stage 2 transport electroweak generators and Stage 10 physical
  Yukawa-door electroweak generators.
- Finite exponentiation from the Stage 16 Yukawa-door generators reproduces the
  Stage 10 Yukawa-door helper.
- Exact generator agreement on duplicated left doublet labels.
- Exact zero `SU(2)_L` action on right singlet labels in both conventions.
- Exact charge-conjugation relation for right-singlet hypercharge.
- Nonzero full generator difference and nonzero sorted-hypercharge spectral
  mismatch proving there is no unitary similarity on the fixed 32-component
  carrier.
- Physical Yukawa-door energy covariance and transport-convention
  non-invariance controls.
- JIT audits for the gauge-convention energy residuals.
- Session 16 script and focused tests.
- Physical-right full SM generators built from Stage 2 transport generators by
  left-linear / right-antilinear projector bridge.
- Exact left-linear and right-antilinear generator residuals.
- Electroweak slice matching the Stage 10/16 physical Yukawa-door generators.
- Finite physical-right site gauge equal to the projected antiunitary finite
  bridge from the transport gauge.
- Finite bridge unitarity audit.
- Full physical-right gauge covariance for the local Yukawa energy and
  unbridged transport-gauge non-invariance control.
- JIT audits for the full bridge energy residuals.
- Session 17 script and focused tests.
- Physical-right finite BCC link bridge
  `U_phys=P_L U_transport P_L + P_R conj(U_transport) P_R`.
- Identity-link bridge control.
- Finite bridged-link unitarity audit.
- Bridged-link covariance under physical-right site gauges.
- Family BCC transport through physical-right bridged links.
- Identity-link reduction to the existing family transport.
- Physical-right family transport covariance and norm-preservation audits.
- Nontrivial transport-kernel difference from the unbridged Stage 13 transport
  convention.
- JIT audit for the physical-right transport kernel.
- Session 18 script and focused tests.
- Physical-right streaming bilinear through bridged finite BCC links.
- Left-trivialized physical-right current in the same 12 transport coordinates
  used by the gauge links and momenta.
- Zero-state and nonzero-current controls.
- Nontrivial current difference from the unbridged Stage 13
  transport-convention family current.
- Physical-right charge density and Gauss diagnostic
  `electric divergence - physical-right family charge`.
- Covariance checks for physical-right streaming energy, current, charge, and
  Gauss.
- Reversible physical-right current momentum kick and
  kick-then-physical-right-transport wrapper.
- Link unitarity, spectator family-state norm, and JIT audits.
- Session 19 script and focused tests.
- Physical-right sourced link force
  `Wilson + embedded Higgs gauge force + physical-right family current`.
- Nontrivial force difference from the Stage 14 transport-convention
  family-sourced force.
- Physical-right sourced Gauss diagnostic
  `physical-right electric divergence - physical-right family charge -
  embedded Higgs charge`.
- Covariance checks for physical-right sourced force and Gauss.
- Zero-source and deterministic nonzero-source controls.
- Reversible physical-right sourced SM momentum kick.
- Physical-right sourced tick with bridged family transport, Higgs-field
  advance, SM/Higgs link unitarity, family-state norm, and JIT audits.
- Nontrivial state difference from the Stage 14 transport-convention sourced
  tick.
- Session 20 script and focused tests.
- Physical-right production tick merging the Stage 10 local Yukawa/Higgs source
  and exact Stage 7 family Yukawa collision into the Stage 20 physical-right
  sourced tick.
- Zero-Yukawa reduction controls against the Stage 20 physical-right sourced
  tick.
- Production Higgs force as Higgs dynamics plus local Yukawa Higgs source.
- Nonzero deterministic Yukawa-source and zero-state/vacuum-source controls.
- Reversible production Higgs momentum kick for frozen fields.
- Production tick with symmetric local half-collision, physical-right BCC
  family transport, Higgs-field advance, SM/Higgs link unitarity, family-state
  norm, and JIT audits.
- Nontrivial state difference from the Stage 15 transport-convention production
  tick.
- Session 21 script and focused tests.
- Physical-right production rollout state carrying family fermions, Higgs
  fields/momenta, SM links and momenta, and Higgs-representation links.
- Direct final-state rollout and sparse recorded rollout through the shared
  `sim.runner` loop/scan interface.
- One-step direct-tick agreement and loop/scan agreement audits.
- Recorded observables for family norm, Higgs norm, Higgs momentum norm, SM
  momentum norm, and SM/Higgs link-unitarity residuals.
- Multi-tick family-norm drift, link-unitarity, finite-observable, and
  zero-Yukawa difference controls.
- Session 22 script and focused tests.
- Physical-right production Gauss monitor reading the existing sourced Gauss
  diagnostic on production rollout states.
- Exact zero-source vacuum control: zero family state, unitary-gauge Higgs
  vacuum, zero momenta, identity links, and zero Gauss throughout rollout.
- Deterministic nonzero production Gauss history and final-Gauss contrast
  between default and zero-Yukawa rollouts.
- Monitored rollout family-norm and link-unitarity controls.
- Session 23 script and focused tests.
- Physical-right production energy monitor recording pure SM gauge
  Hamiltonian, Higgs Hamiltonian, physical-right streaming bilinear, and local
  three-family Yukawa energy.
- Exact vacuum-zero monitored-total-energy control.
- Deterministic component-energy history and default-vs-zero-Yukawa
  monitored-total-energy contrast.
- Session 24 script and focused tests.
- Higgs force normalization repaired so the production Higgs force is the
  `-dE/dphi*` force of the monitored Higgs Hamiltonian density plus local
  Yukawa energy.
- Variational audit of the physical-right production forces: exact
  link/Higgs-force decomposition checks, one selected color-link
  finite-difference check, one selected complex Higgs-component
  finite-difference check, vacuum force controls, and deterministic
  nonzero-force controls.
- Session 25 script and focused tests.
- Fixed-time timestep refinement audit for the monitored physical-right
  production total energy.
- The refined rollout remains finite, family-norm controlled, and link-unitary,
  while the zero-source vacuum remains zero-energy.
- The current hybrid production tick does not improve monitored energy drift
  under timestep halving, so timestep-refined energy convergence is explicitly
  not claimed.
- Session 26 script and focused tests.
- Physical-right BCC family transport adjoint step.
- Frozen production fermion substage adjoint audit: old-Higgs half collision,
  updated-link transport, and updated-Higgs half collision restore under the
  explicit adjoint.
- Local family Yukawa collision inverse audit under `step_size -> -step_size`.
- Full production tick followed by a negative-timestep production tick is
  explicitly detected as not being the inverse of the full map.
- Session 27 script and focused tests.
- Explicit inverse helper for the current physical-right production tick:
  reconstructs final half-step momenta from final production forces, rewinds
  the sourced link update and Higgs field, applies the frozen fermion-stage
  adjoint, and recovers initial momenta from reconstructed first forces.
- Forward-then-inverse and inverse-then-forward audits restore the rollout
  state to float32 precision.
- The explicit inverse improves strongly over the naive negative-timestep
  production tick while preserving family norm and SM/Higgs link unitarity.
- Session 28 script and focused tests.
- Multi-step inverse rollout by repeated Stage 28 inverse steps.
- Forward rollout followed by inverse rollout restores the initial state, and
  the restored state replayed forward reproduces the stored final state.
- Each inverse step is audited against the stored forward trajectory.
- Multi-step naive negative-timestep control remains far worse than the
  explicit inverse trajectory.
- Forward/inverse family-norm and SM/Higgs link-unitarity controls remain
  bounded, with fixed-step JIT compatibility for the inverse rollout.
- Session 29 script and focused tests.
- Loschmidt echo diagnostic using the explicit inverse trajectory.
- A local final-time SM momentum perturbation produces a finite nonzero
  initial-surface echo after inverse rollback.
- Doubling the final-time perturbation doubles the echo to local linear
  precision.
- The unperturbed base roundtrip and perturbed inverse link unitarity remain
  controlled.
- Session 30 script and focused tests.
- Finite tangent-response echo diagnostic using independent final-time SM
  momentum and Higgs momentum perturbations.
- Separate inverse echoes are finite and nonzero.
- The inverse echo of the combined perturbation matches the sum of the
  separate inverse echoes to local finite-difference precision.
- Combined perturbed inverse link unitarity remains controlled.
- Session 31 script and focused tests.
- Local echo Gram matrix from three independent final-time perturbations:
  SM momentum, Higgs momentum, and family state.
- Echo norms are finite and nonzero; the Gram matrix is symmetric, positive,
  well-conditioned on the certificate trajectory, and has bounded off-diagonal
  correlations.
- Inverse-pulled echo-state link unitarity remains controlled.
- Session 32 script and focused tests.
- Echo-Gram scale audit comparing `epsilon` and `2 epsilon`.
- Echo norms scale linearly and Gram eigenvalues scale quadratically in the
  certificate finite-difference regime.
- Dimensionless Gram data, including condition number and off-diagonal
  correlation, remain stable across the two perturbation sizes.
- Session 33 script and focused tests.
- Finite-horizon echo-spectrum audit comparing one-tick and two-tick production
  trajectories.
- Echo gains derived from Gram eigenvalues remain finite, nonzero, and locally
  bounded across the two horizons.
- Condition number, off-diagonal correlations, base roundtrips, and
  inverse-pulled link unitarity remain controlled across horizons.
- Session 34 script and focused tests.
- Finite-stencil locality audit from exact integer BCC displacement sets.
- The BCC transport stencil is the eight body-diagonal hops and is closed under
  inversion.
- Higgs-gradient terms stay within the one-hop envelope; plaquette/current
  force terms are conservatively bounded by the two-hop envelope.
- The one-tick production envelope has radius `2`; the two-tick inverse echo
  envelope has radius `4`.
- Session 35 script and focused tests.
- Dense production-workload audit from the implemented register dimensions.
- The dense rollout state is linear in site count and remains modest on the
  `3^3` certificate lattice.
- The pre-Stage-37 finite-difference Wilson force evaluated a global plaquette
  action for every `sites x 8 links x 12 generators` coordinate.
- The finite-difference plaquette workload scales quadratically in site count;
  on the `3^3` certificate it is `5184x` the linear local staple-force target.
- Session 36 script and focused tests.
- Local analytic Wilson-force replacement for the production
  `sm_left_wilson_force` path.
- Automatic-differentiation coordinate audit against exact left-update
  derivatives of the Wilson action.
- Identity, pure-gauge, covariance, and coarse legacy finite-difference
  controls for the local force.
- Session 37 script and focused tests.
- Local-force production-rollout smoke test on a `2 x 2 x 1` lattice.
- The production step is finite, keeps SM/Higgs links unitary, and produces
  nonzero Higgs, SM-link, and SM-momentum updates.
- Changing the legacy `wilson_epsilon` keyword leaves the production step
  unchanged, confirming the old finite-difference Wilson knob no longer drives
  the tick.
- On the Stage 38 certificate lattice the local Wilson-force path uses `24`
  plaquette holonomies versus `18432` for the legacy finite-difference path
  (`768x` reduction).
- Session 38 script and focused tests.
- Multi-step sparse recorded rollout on the same `2 x 2 x 1` local-force
  certificate lattice.
- Python-loop and `lax.scan` recorded runners agree on final state and
  observations over two production steps.
- Recorded observations remain finite; family norm drift and SM/Higgs link
  unitarity remain controlled while Higgs, SM-link, and SM-momentum changes
  stay nonzero.
- Session 39 script and focused tests.

**Boundary**: no boundary condition, quantized scalar/gauge registers, full
microscopic BCC derivation of the antiunitary bridge, Gauss projection, exact
full-energy conservation claim, timestep-refined energy-convergence claim, or
energy-convergent reversible-integrator claim beyond the explicit inverse
helper, continuum Lyapunov claim, or large-lattice spatial echo measurement is
implemented yet.  The derivation of the flavor/Higgs inputs is also
not implemented: FN charges, `lambda`, order-one coefficients, center-power
matrices, placeholder lepton matrices, and Higgs potential parameters are
explicit simulator inputs rather than BCC-bulk derivations.

## scalar_clebsch — V3 conditional pass, active

**Goal**: test whether corrected quark mass Clebsch coefficients have a scalar
boundary-response origin rather than borrowing coherent CKM current-amplitude
Clebsches.

**Result**:
- Up sector: a length-3 nilpotent Taylor response
  `exp(xN)=I+xN+x^2 N^2/2` with `x=1/sqrt(2)` gives
  `(1/4,1/sqrt(2),1)`.
- Down sector: the natural S3/projector baseline gives counts `(6,2,4)` and
  vector `(1,1/sqrt(3),sqrt(2/3))`.
- Down data-improved candidate: the odd-shell bottom repair gives counts
  `(6,2,5)` and vector `(1,1/sqrt(3),sqrt(5/6))`; the bottom `+1` is recorded
  as open, not derived.
- S3 projector audit: `(6,2,5)` is available in the regular S3 algebra, but
  not forced by S3 alone. Rank 2 requires a defect-selected standard copy, and
  rank 5 requires choosing which one-dimensional irrep is excluded.
- The old up `sqrt(2)` coefficient is rejected as a scalar positive mass
  coefficient; it belongs to coherent current-amplitude Clebsches.

**Open**: derive the microscopic two-successor/no-leakage repair condition
used by `radial_response` R5, and derive or kill the down-sector
defect-selection rule.

## radial_response — R1-R13 pass, active

**Goal**: reframe the quark mass sector as QCA boundary recirculation residues /
pole shifts instead of direct finite-group outputs.

**Result**:
- R1 proves the Feshbach/Schur form
  `Sigma(z)=V^T(z-H_Q)^-1V`, so masses are repeated `P -> Q -> P`
  boundary returns.
- R2 separates up-sector radial stacking laws:
  exponential/Poisson gives `C_u C_t / C_c^2 = 1/2`, while
  geometric/resolvent gives `1`.
- R3 kills literal `exp(xN)` as a family-space Yukawa matrix: at `x=1`, its
  singular values are `(2,1,1/2)` and its left metric is non-diagonal.
- R4 imports the S3 projector audit and reframes `C_b^2=5/6` as a regular
  S3 shell minus one dark line, available but not derived.
- R5 proves that two equal no-leakage scalar repair successors force the
  amplitude `x=1/sqrt(2)`; the microscopic successor/no-leakage condition
  remains the load-bearing hypothesis.
- R6 builds a minimal exact unitary S3 Floquet defect form `U=S C` and verifies
  the unitary Schur self-energy. Coin-angle and defect-vector controls show
  that this form does not by itself force the physical phase or radial values.
- R7 certifies R5's two-channel no-leakage condition in a finite modeled scalar
  successor basis: only the Z2-conjugate `triality_plus/triality_minus` pair is
  allowed, while same-state, wrong-height, two-tick, leakage, asymmetric, and
  third-successor controls are vetoed.
- R8 proves that this pair is complete inside the S3 regular shell: the
  identity is same-state, the two non-identity cycles are the scalar
  holomorphic successors, and the three transpositions belong to the
  Hermitian/Z2 repair sector.
- R9 reduces the S3 shell to vacuum-frame-preserving BCC tetrahedral exit
  automorphisms: the selected-exit stabilizer induces residual S3, and scalar
  holomorphic restriction leaves the same triality pair. The premise that
  actual one-tick scalar repair is such an automorphism remains open.
- R10 derives the R9 automorphism premise from a declared scalar-local,
  deterministic exit-map class, while rejecting nonlocal, non-scalar, generic
  linear-mixture, frame-breaking, same-state, and Hermitian/Z2 controls.
- R11 inherits the silver transfer root from the existing boundary-response /
  flavor-A stack: `epsilon=sqrt(2)-1`, `eta=epsilon^2`, `r=epsilon^4`.
- R12 proves that finite S3/silver-transfer data do not force pole/residue
  values: admissible baths with the same channel grammar give different
  self-energies, poles, and residues.
- R13 shows that the target quark textures are positive finite spectral
  measures with inverse Jacobi-bath reconstructions, but the existing R12
  baths, P3 Jacobi control, silver-tail control, and minimal unitary S3 toy do
  not select them. This is reconstruction-only, not a mass derivation.

**Open**: derive the scalar-local deterministic exit-map class from the actual
BB/QCA scalar boundary update; identify a forward spectral-density principle
that selects the target measure rather than reconstructing it; derive or kill
the down dark-line selection rule; eventually build a simulator-backed boundary
spectral density.

## universal_bath — parked after Session 24

**Goal**: replace inverse spectral-measure reconstruction with a forward bath
architecture:

```text
sector response = finite Lanczos head + universal retarded silver tail
```

**Result**:
- Session 01 proves the common Schur/Jacobi spine.
- The universal period-one tail satisfies `t = 1 / (z - t)`.
- At the BB marginal probe, `t(2 sqrt(2)) = sqrt(2) - 1`.
- Finite scalar moments round-trip through a Jacobi head.
- Finite-head Schur response equals its continued fraction.
- Alternate terminators change the response, so the universal silver tail is a
  physical closure principle rather than a continued-fraction tautology.
- The reduction taxonomy is fixed: positive sectors use scalar Jacobi, real
  non-positive shells use indefinite look-ahead Jacobi, and chiral/unitary
  sectors use CMV/OPUC.
- Session 02 freezes the supported lepton-side source anchors:
  `neutrino_collective_u` (`u`, depth 1), `neutrino_edge_b` (`b`, depth 0),
  and `charged_lepton_active_e1` (`e1`, depth 2).
- The BB source survival identity
  `B_+^*B_+ + B_-^*B_-=I/2` gives every frozen normalized source first-hop
  radial survival weight `1/2`.
- Session 03 certifies the neutrino core only inside the product half-line bath:
  `H_Q = H_chain tensor I_family` gives equal `u`/`b` diagonal returns,
  zero cross-return moments through the checked finite orders, and normalized
  response `epsilon^2 P_u + P_b` at the BB silver probe.
- The resulting neutrino ratios are
  `m2/m3 = epsilon^2` and `Delta m^2_21 / Delta m^2_31 = epsilon^4`.
- Session 04 builds the charged-lepton finite CMV head:
  `alpha_e = sqrt(3/2) epsilon^2 exp(-5 pi i / 12)` followed by the free
  OPUC tail `alpha_n = 0`; the associated two-state CMV/Givens block is
  exactly unitary.
- Session 05 derives the charged-lepton torsion value as the frozen source
  occupation moment:
  `e1 = sqrt(2/3) a + 1/sqrt(3) u`, so `p_a p_u = (2/3)(1/3) = 2/9`.
  The coherent amplitude `sqrt(2)/3`, equal-weight, and one-port controls are
  rejected; this does not by itself rederive the CMV phase.
- Session 06 implements the conditional up-quark nilpotent CMV head:
  the BB survival weight `1/2` gives injection `x=1/sqrt(2)`, and the
  length-3 Taylor head gives `(x^2/2,x,1) = (1/4,1/sqrt(2),1)` followed by the
  free CMV tail. The up source vector remains unresolved.
- Session 07 implements the conditional down-quark real/symmetric Jacobi head:
  the 3-port shell gives `(3,1,2)/3 -> (1,1/sqrt(3),sqrt(2/3))`; the regular
  S3 shell gives the baseline `(6,2,4)/6` and can host the candidate
  `(6,2,5)/6 -> (1,1/sqrt(3),sqrt(5/6))`; S3 does not select the rank-5 line.
- Session 08A audits the quark height doors: hypercharge forces `H_tilde` for
  up and `H` for down, with neutral Higgs components, while the declared repair
  split maps up to the oriented length-3 nilpotent and down to the Hermitian
  path closure. Swapping repair modes remains hypercharge-allowed, so the
  coherent-up / Hermitian-down split is a height-dynamics premise, not an
  electroweak charge theorem.
- Session 08B audits color lifting: a fixed visible color vector is rejected by
  `SU(3)_c`; a color-scalar spectator embedding preserves visible color but
  stays on the three-port shell; an active hidden color-return lift also
  preserves visible color while reaching `1_direct + 2_BCC + 3_color`, making
  the regular-S3 baseline and rank-five candidate available but not selected by
  gauge covariance alone.
- Session 09 audits the neutrino product-bath upgrade gate: the exact
  microscopic BB edge update supplies the q=0 scar and q=+-2 leakage blocks
  with norm split `I/2 + I/2 = I`, but the current edge graph has no `u,b`
  family-port nodes. Therefore `<u|H_BCC^k|b>` is not yet a BCC
  walk-counting observable; the `epsilon^4` neutrino result remains protected
  by the product ansatz until that graph is built.
- Session 10 supplies the selected internal family-port graph
  `H_fam = H_chain tensor (P_u + P_b) + I tensor Lambda P_a`. Direct graph
  moments give zero `u/b` cross returns, equal `u/b` diagonal returns, and
  radial-active separation through the checked finite powers. Closing the
  active plane with the universal tail gives `epsilon^2 P_u + P_b`. The
  remaining physical gate is deriving the selected active-plane projector
  `P_u + P_b` from the microscopic BB edge update rather than imposing it.
- Session 11 derives the active-plane projector from selected residual-port
  incidence: `e1 = sqrt(2/3) a + u/sqrt(3)`, so detracing `e1` against the
  collective mode fixes the radial line `a`, and `a^perp = span(u,b)`. Selected
  `S2` symmetry alone is shown not to be enough.
- Session 12 connects that active plane to the exact BB q-mismatch and
  retarded-compression model: same-normal and mixed-normal blocks split the BB
  norm as `1/2+1/2`, `g q^2` gives mixed-sector Schur feedback
  `I/[2(z-4g)] -> 0`, and retarded clock-error leads make visible powers equal
  q=0 survival powers. A recurrent wedge control is rejected.
- Session 13 audits the remaining material-origin premise. The local mismatch
  coordinate `q=r1-r2` is unique and `K^T K=q^2` if the constraint is admitted,
  but bare BB blocks contain no stiffness/gap parameter and do not select
  outgoing clock-error leads over recurrent wedge return.
- Session 14 constructs the minimal colorless active charged-lepton family-port
  graph. The two-sided residue is exactly `sqrt(2) P_u + R_theta P_perp`, acting
  on `e1` gives trace/traceless equipartition and Koide `K=2/3`, and one-trace
  plus one-sided-Hermitian controls are rejected. The trace paths and `2/9`
  torsion dynamics remain inputs.
- Session 15 assembles the quark source dependency graph. The common residual
  family incidence basis, SM charge doors, conditional up nilpotent head, and
  conditional down real-symmetric heads are available. The source freeze is
  not derived: `V_u,V_d` still lack port vectors, residual components, and
  normal-depth placements, and four microscopic inputs remain open.
- Session 16 audits the normal-depth blocker against `depth_scar`. The
  nilpotent repair flag and `{0,2,6}` normal-mode spectrum are exact, but the
  path-scar operator is not a diagonal port-depth assignment and doubled port
  heights are `(0,2,4)`, so this does not freeze `V_u.normal_depth` or
  `V_d.normal_depth`.
- Session 17 reduces the active hidden color-return blocker. Inside the
  primitive six-label shell, equal-degeneracy microcanonical reduction gives
  `I_6/6` and selects the active `1_direct + 2_BCC + 3_color` shell over the
  compressed three-port spectator control. The remaining input is the
  equal-degeneracy / max-entropy prior.
- Session 18 reduces the down rank-five blocker. In the active primitive shell,
  the bottom candidate is the full odd shell, the complement of the even direct
  line, and the strange channel is the BCC odd doublet. The remaining premise
  is that the physical down bottom readout selects the full primitive odd
  shell.
- Session 19 audits the charged-lepton trace-path and torsion origin. Inside
  the minimal graph, `n` coherent trace paths give trace weight `n/(n+2)`, so
  trace/traceless equipartition uniquely forces `n=2`. The Session 14 graph
  supplies exactly two trace-only pole rows, and one/three-trace controls fail.
  This still does not derive those rows from a microscopic colorless BCC/Higgs
  boundary, and the `2/9` occupation moment is still inserted as a real
  rotation angle rather than produced by an occupation-to-angle dynamics.
- Session 20 reduces the quark height-door premise. The depth-scar successor
  certificate supplies the oriented flag `a -> u`, `b -> a`; the up repair is
  this nilpotent flag and the down repair is its Hermitian flag-Laplacian
  closure. Thus the two repair objects are not independent. The remaining
  open input is the Higgs-door orientation coupling that maps `H_tilde` to the
  retarded flag readout and `H` to the Hermitian closure.
- Session 21 implements the colored active-current quark ansatz. Selected
  incidence gives `P_act=P_u+P_b`, and the non-scalar current condition selects
  `b`; first-passage from `b` under the certified flag gives orders `(2,1,0)`,
  hence up depths `(6,3,0)` and down depths `(6,4,2)`. The coherent up readout
  `exp(N/sqrt(2))b` gives `(1/4,1/sqrt(2),1)`. The down readout is kept as a
  Hermitian current covariance over shell measures, with baseline `(6,2,4)`
  and odd-shell `(6,2,5)` alternatives visible until the identity-return veto
  is derived.
- Session 22 reduces the current-source selector. The selected-port `S2` swap
  acts on `(u,a,b)` as `diag(+,+,-)`, so `P_odd=P_b`; the oriented current
  across the two unselected ports is `(e2-e3)/sqrt(2)=b`; and
  `P_act P_odd=P_b`. Thus the Session 21 source line is no longer just the
  active non-scalar choice. It is the unique selected-`S2` odd boundary-current
  line, conditional on interpreting colored quark mass events as such currents.
- Session 23 decides the down bottom fork inside the retarded-current model.
  The primitive shell has one direct even identity/contact line and five odd
  hidden returns `2_BCC+3_color`. Requiring a down mass event to leave the
  visible sheet before returning vetoes the identity line and selects
  `(6,2,5)/6 -> (1,1/sqrt(3),sqrt(5/6))`. The contact/S3 baseline
  `(6,2,4)/6 -> (1,1/sqrt(3),sqrt(2/3))` remains the rejected control under
  that predicate.
- Session 24 audits the Higgs-door orientation coupling. Endpoint reflection
  maps the certified flag as `R N R=N.T`, while the Hermitian down closure
  `Delta_N=NN.T+N.TN-(N+N.T)` is reflection-invariant and is not produced by
  reversal alone. The declared assignment `H_tilde -> N`, `H -> Delta_N` and
  the swapped assignment both survive the current gauge/current/flag
  constraints, so the door-to-readout rule remains a dynamical boundary
  premise.

**Open**: derive, or audit as irreducible, the deeper boundary-material origin
of the single-clock locking field and outgoing clock-error asymptotics
(Session 13 shows it is not derived from bare BB block algebra); derive the
charged-lepton microscopic two-trace-path origin and active `2/9`
occupation-to-angle dynamics;
derive or replace the Higgs-door orientation-coupling rule and active hidden
color-return rule needed to freeze the up/down quark BCC source vectors and
normal-depth placements without flavor data; derive or kill the down rank-5
bottom-line selection rule; Session 15 records these as the exact quark
source-freeze blockers, and Session 16 sharpens the normal-depth blocker into
a source placement theorem; Session 17 reduces active color return to a
primitive-shell microcanonical prior; Session 18 reduces the rank-five bottom
line to the primitive odd-shell readout premise; Session 19 reduces the
charged-lepton trace-path question to the microscopic origin of two coherent
trace returns; Session 20 reduces the quark height premise to a Higgs-door
orientation-coupling rule; Session 21 proposes the active-current source `b`
and first-passage depth theorem; Session 22 reduces the colored-current
selection to the selected-`S2` odd-current premise; Session 23 proves the down
rank-five bottom line inside the non-contact retarded-current model, while the
material/dynamical origin of that non-contact criterion remains open; Session
24 shows that Higgs conjugation supplies orientation reversal, not the
Hermitian closure operation needed for the down readout;

**Parked**: do not continue to CKM/mixing inside this sidecar until a new
physical principle supplies the Higgs-door pairing rule or another parked gate.
The closure note is
[`universal_bath/CLOSURE.md`](src/clifford_3plus2_d5/universal_bath/CLOSURE.md).

## threeclocks — on hold after Session 02

**Goal**: explore a simpler quark-model branch based on finite clocks, without
importing the full parked universal-bath machinery.

**Result**:
- Session 01 implements the exact finite-clock spine.
- The default prototype is:

```text
Z3 x Z3 x Z3
```

- The product dimension is `27` and common closure order is `3`.
- Each local clock has exact shift/phase matrices satisfying:

```text
Z X = omega X Z
```

- Clock words compose, invert, and report closure order exactly.
- A non-uniform control `Z2 x Z3 x Z5` passes with dimension and closure order
  `30`, so the `Z3^3` default is not hard-wired as a theorem.
- Session 02 attaches a selected `D3 ~= S3` clock to the three-port family
  boundary.  With

```text
u=(1,1,1)/sqrt(3), a=(2,-1,-1)/sqrt(6), b=(0,1,-1)/sqrt(2),
```

  it proves the exact identities:

```text
(C e1 - C^-1 e1)/sqrt(2) = b
(2 e1 - C e1 - C^-1 e1)/sqrt(6) = a
```

  so `b` is the oriented tangent current and `a` is the radial second
  difference of the selected tooth.
- Controls show `spec(2I-C-C^-1)={0,3,3}`, so the three-port D3 Laplacian is
  not the down shell, and a literal port-basis cut does not equal the target
  repair flag `N=|u><a|+|a><b|`.
- Conditional profiles are recorded only as next-gate targets:
  `C_u=(1/4,1/sqrt(2),1)`, rank-five `C_d=(1,1/sqrt(3),sqrt(5/6))`, and the
  contact-allowed down control `(1,1/sqrt(3),sqrt(2/3))`.

**Open**: derive or kill the representation-basis repair flag from the D3
clock defect; embed `B_+ tensor C + B_- tensor C^-1` in a unitary dilation;
select the active quark shell over the spectator embedding; audit the bottom
contact veto; derive closure exponents; compute CKM from two-sided clock
kernels.

**Parked**: this sidecar is on hold until the D3-clock route is resumed.

## cusp — MicroCUSP A-H pass

**Goal**: implement `CUSP-PLAN.md`: Froggatt-Nielsen flavor from the valuation
filtration of a `(2,3)` recirculation cusp.

**Implemented**:
- Target A: the local BCC edge q-clock has two q-preserving same-normal
  branches and two q-changing leakage branches; the q-preserving pair supplies
  weak/BCC multiplicity `2`, while the exact BB edge blocks supply the
  same-normal/leakage norm split `I/2 + I/2 = I`.  Stacking the exact same/mixed
  BB blocks gives an `8 x 2` isometry with finite unitary completion; q-changing
  feedback vanishes under the hard-gap/outgoing condition, while recurrent
  leakage has a nonzero visible return and is rejected.  A finite
  material-origin audit derives `p(q)=gq^2` as the unique lowest-degree
  q-local, q-reflection-even positive penalty that vanishes on `q=0`, rejects
  linear and constant controls, and selects the no-incoming retarded closure
  over recurrent leakage.  The color `SU(3)` center supplies primitive closure
  `3`.  These sources now generate local center-charge recirculation automata:
  one tick advances center charge, visible readout occurs only at charge `0`,
  and the weak/color automata force primitive returns `2` and `3`.  Closed-walk
  enumeration gives valuation algebra `C[t^2,t^3]`, first valuations
  `(0,2,3)`, and gap `1`; `C[t]`, `C[t^2]`, `C[t^3]`, one-step-loop,
  weak-only, color-only, and wrong-color-length graph controls give different
  low valuations.
- Target B: normalized color/weak amplitude shear gives
  `lambda_rec=sqrt(3/2)-1=0.224744871...` as the unique stable minimum of the
  squared mismatch functional.  Session 07 promotes this to a finite
  one-sided matching theorem: the cusp readout solves
  `(1+t)sqrt(2)=sqrt(3)`, while the ordinary reflection coefficient solves the
  different two-sided scattering control
  `(1+r)sqrt(2)=(1-r)sqrt(3)` and is rejected by the retarded recirculation
  boundary because it leaves a nonzero one-sided residual.  Count-ratio and
  inverse-amplitude shear controls fail the same one-sided equation.  No CKM
  data are used.
- Target C: SM hypercharge conservation forces the up Yukawa to use `H_tilde`
  and the down Yukawa to use `H`, rejecting swapped doors.  Given that door
  orientation, `S=<2,3>` derives conductor `c=2`.  Session 08 selects the down
  right charges as the conductor-ideal residue `D=max(Q-c,0)=(1,0,0)` and
  selects the up lift factor from the weak/BCC primitive closure order `2`,
  giving `U=(5,2,0)` without using fitted mass data as an input.
  Wrong-conductor, trivial-lift, and color-order-lift controls miss the target
  exponent skeleton.  The resulting charges reproduce diagonal FN exponents
  `(8,4,0)` and `(4,2,0)` plus CKM powers `V_us=lambda`, `V_cb=lambda^2`,
  `V_ub=lambda^3`; the diagonal solver is retained only as a consistency check.
  MicroCUSP Session 18 recovers this module from microscopic Schur moments:
  `(0,2,3)` gives `Q=(3,2,0)`, conductor `c=2` gives `D=(1,0,0)`, and the
  weak double cover gives `U=(5,2,0)` without using diagonal targets or mass
  fits.
- Target D: coefficients are finite path sums
  `c_ij=sum_gamma A_gamma Omega_gamma` with `Omega_gamma` in the color center;
  Session 10 selects the up center powers as geodesic distances on the
  non-cyclic length-3 cusp flag and the down center powers as the unit bilinear
  pairing on `F3` center labels.  This rule pair gives nonzero
  `Im tr([YuYu^dagger,YdYd^dagger]^3)`, while all-real, one-sector, and
  separable row/column controls give zero and full field rephasing leaves the
  invariant unchanged.  Session 09 derives the positive amplitude weights from
  the cusp-module path-count rule
  `A_ij=max(1,# decompositions of q_i+r_j in <2,3>)`, with the gap exponent `1`
  treated as one irreducible conductor-module/contact path.  This derived
  measure remains CP-active, while the all-real control with the same
  amplitudes stays CP-zero.  The earlier fixed non-unit positive deformation is
  retained only as a robustness control.
  MicroCUSP Session 19 recovers this topology from microscopic boundary data:
  up powers are geodesic distances on the non-cyclic cusp flag, down powers are
  the unit bilinear pairing of `F3` color-center labels, the CP invariant is
  nonzero, and all-real / one-sector / separable controls are zero.

**Boundary**:
- Target A/B are exact certificates inside the minimal recirculation model.
- Target C is a microscopic module audit inside the current conductor-module /
  weak-double-cover boundary-register model.
- Target D now has microscopic topology selection and a finite cusp-module
  amplitude measure inside the current boundary-register model.
- The finite BB/material dilation and material-origin audits still are not a
  full local boundary theorem.  MicroCUSP Session A now derives the q-local
  positive q-reflection stiffness from a two-normal harmonic boundary material,
  and Session B derives the no-incoming retarded boundary as `R=0` outgoing
  mixed-normal asymptotics with recurrent/reflecting/symmetric controls
  rejected.  Sessions C-D derive the weak `Z2` branch parity from same-normal
  BCC signs and the color `Z3` return from closed `SU(3)` center holonomy.  The
  Session 15 global quotient audit rejects the correlated diagonal `Z6` rule
  and U(1)-collapsed control.  Sessions 16-19 derive the Schur semigroup,
  `lambda_rec`, Target-C module, and Target-D topology.  No MicroCUSP A-H
  gates remain open.

## spacetime_qca — in progress

**Goal**: 3D BCC Weyl walk (Bialynicki-Birula 1994) → 4D Dirac chiral
assembly → tensor lift to `lepton`'s chiral-16 internal carrier → finite
simulation controls for background-gauge and coupled gauge/Higgs/Yukawa
experiments.

**Sessions**: 20-62 complete through BCC Dirac kinematics, real-space stepping,
position-dependent gauge covariance, BCC plaquettes, Wilson action/forces,
compact SU(2)/SU(3)/SU(4)/Pati-Salam/SM gauge prototypes, Gauss/backreaction
controls, dynamical Higgs/Yukawa infrastructure, anomaly and Lorentz audits,
simulation runner split, profiling/force optimizations, exact-unitary Yukawa
fast path, sparse recorded scans, and finite-difference Higgs-current
backreaction plus Higgs charge in the combined Gauss residual, opt-in Gauss
projection controls, and bounded projection sweeps.

**Implemented**:
- BCC geometry (8 body diagonals).
- Bialynicki-Birula 2-component Weyl Bloch symbol on BCC, pinned to
  Phys. Rev. D 49, 6920 (1994), Section II.
- 4-component Dirac chiral assembly.
- Hypercubic doubling control.
- First-order continuum expansion utilities.
- Constant-background gauge tensor lift against a real 32×32 Pati-Salam
  `SU(2)_L` generator from `lepton`.
- Position-dependent internal link fields and exact finite-lattice site-local
  gauge covariance for the Dirac step.
- BCC elementary plaquette shapes and holonomy covariance.
- Shared generic JAX state/link/diagnostic helpers are imported from `sim`;
  BCC/QCA physics remains local to `spacetime_qca`.
- JAX numerical backend for flat Dirac states, `Dirac x internal` tensor
  states, identity/constant/position-dependent link fields, and BCC Dirac
  steps.
- Exact and JAX Wilson plaquette observables over the six canonical BCC
  plaquette shapes, including action density and total Wilson action.
- SO(2)-parameterized JAX Wilson-action gradients, finite-difference checks,
  pure-gauge flatness checks, and `jax.jit` gradient support.
- SU(2)-parameterized JAX Wilson-action gradients, finite gauge-transform
  invariance, pure-gauge controls, and `jax.jit` gradient support.
- SU(2) left-trivialized Wilson force, compact left updates, and deterministic
  action-descent controls.
- SU(2)/SU(3) compact momentum fields, Hamiltonian-density helpers, and
  reversible leapfrog updates.
- Basis-based Pati-Salam and SM compact gauge adapters over the chiral-16
  internal carrier.
- Gauss residuals, fermion charge density, finite-difference link current, and
  explicit momentum-source backreaction controls.
- Combined fermion/Higgs Gauss residuals, opt-in bounded Gauss-descent
  projection controls, and tiny-lattice projection sweeps.
- Site-local Higgs field with electroweak gauge covariance, BCC covariant
  differences, Mexican-hat potential diagnostics, conjugate momentum, and
  fixed-link leapfrog controls.
- Static Hermitian `Y(Phi)` and exact site-local unitary Yukawa insertion.
  The production exact path now uses the selected cubic-polynomial identity;
  the old eigensolve path is retained as an oracle.
- Scan-backed main simulator with `.npz`/JSON output, small presets, lab/main
  split, focused profiling CLIs, analytic compact Wilson force path, and sparse
  observation recording via `record_every`.
- Real-form internal convention: the internal chiral-16 is kept as `R^32`
  with compatible `J`, equivalent to `C^16` but not compressed to a
  `16 x 16` complex basis.  Tensor-lift matrices currently have size
  `4 x 32 = 128` over Bloch complex scalars.
- Scalar Dirac mass layer `beta x M_internal`.
- Finite periodic real-space BCC Weyl/Dirac step.
- Representation-level Higgs-like internal charge-shift map, its conjugate
  transpose component, the `SU(2)_L`-generated lower map space, and static
  Hermitian controls.
- Session 20 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_20_BCC_DIRAC.md`.
- Session 21 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_21_MASS_LAYER.md`.
- Session 22 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_22_REAL_SPACE_STEP.md`.
- Session 23 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_23_YUKAWA_REPRESENTATION.md`.
- Session 24 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_24_GAUGE_COVARIANCE.md`.
- Session 24b report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_24B_PLAQUETTE_HOLONOMY.md`.
- Session 25 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_25_STATIC_YUKAWA.md`.
- Session 26 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_26_JAX_BACKEND.md`.
- Session 27 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_27_WILSON_OBSERVABLES.md`.
- Session 28 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_28_WILSON_ACTION.md`.
- Session 29 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_29_WILSON_GRADIENTS.md`.
- Session 30 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_30_SU2_FORCE.md`.
- Session 31 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_31_SU2_LEFT_FORCE.md`.
- Session 32 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_32_SU2_LEAPFROG.md`.
- Session 33 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_33_SU3_DYNAMICS.md`.
- Latest detailed status and Session 34-58 reports live in
  `src/clifford_3plus2_d5/spacetime_qca/STATUS.md`.

**Result**:
- `H_R(k) = sigma . k`.
- `H_L(k) = -sigma . k`.
- `H_D(k) = alpha . k`.
- Hypercube control has 8 literal cubic BZ-corner doublers.
- BCC cubic-corner gapless representatives are reciprocal-lattice origin
  equivalents, not claimed as independent doublers.
- BCC Bloch unitarity is sample-checked; the all-momentum identity is trusted
  from the pinned Bialynicki-Birula source convention.
- Hypercube control is Hamiltonian-form and used as a naive-lattice doubling
  diagnostic, not as a unitary cubic-walk comparison.
- Constant background gauge lift gives
  `H_eff = alpha.k x I_internal + I_spacetime x iA`.
- Scalar mass control gives
  `H_m(k)^2 = (k_x^2 + k_y^2 + k_z^2 + m^2) I`.
- Scalar internal mass commutes with Pati-Salam and SM generators; a
  non-scalar projector control breaks SM symmetry.
- Finite real-space pull-form step matches the BCC Bloch symbol on plane waves.
- Weyl and Dirac finite-lattice steps preserve norm on deterministic
  plane-wave states.
- Exact color-singlet Higgs-like internal maps exist with
  `Delta Y = +1/2` and `Delta T3_L = +1/2`; the transpose gives the
  conjugate `(-1/2, -1/2)` component; solution space dimension `4`.
- The `SU(2)_L` action generates the lower Higgs-like map component with
  `Delta Y = +1/2` and `Delta T3_L = -1/2`; the combined upper/lower module
  has real dimension `8`.
- Static Higgs controls are Hermitian after the `beta x Y_static` lift.  The
  neutral lower-component VEV preserves color and `Q_em = Y + T3_L` while
  breaking `Y` and `T3_L` separately.
- Default static Higgs controls are low-rank (`rank 2`, `nullity 30`), so they
  are background probes rather than realistic mass matrices.
- Position-dependent background links satisfy exact finite-lattice covariance:
  `step(G psi, GUG^-1) = G step(psi,U)`.
- BCC plaquette holonomies transform by base-site conjugation and are identity
  on identity/pure-gauge link fields.
- JAX finite-lattice steps match the exact SymPy backend for ungauged,
  constant-link, and position-dependent-link cases; the linked step is
  `jax.jit` compilable and runs on the default JAX device.
- Wilson-loop traces and normalized averages are implemented exactly and in
  JAX; identity/pure-gauge fields give normalized Wilson loop `1`, and traces
  are gauge invariant under finite site-local transforms.
- Wilson plaquette energy `1 - Re(Tr(H)/N)`, average action density, and total
  action are implemented exactly and in JAX; identity/pure-gauge fields have
  zero action density, and action values are gauge invariant.
- SO(2) compact-link Wilson-action gradients are implemented in JAX; zero and
  pure-gauge fields have zero gradient, non-flat fields have non-zero
  gradient, the gradient matches centered finite differences, and the force is
  orthogonal to pure-gauge directions.
- SU(2) compact-link Wilson-action gradients are implemented in JAX; zero
  fields, Cartan pure-gauge coordinates, and finite pure-gauge links are flat;
  non-flat fields have non-zero gradient; finite gauge transforms preserve
  action density.
- SU(2) compact-link left-trivialized forces are implemented in JAX; zero and
  finite pure-gauge links have zero left force; non-flat fields have non-zero
  force; compact descent preserves SU(2) links and lowers the Wilson action.
- SU(2) compact gauge dynamics now has a reversible leapfrog prototype:
  momentum fields transform by target-site adjoint action, Hamiltonian-density
  drift is small and step-size sensitive, compactness is preserved, and finite
  gauge covariance holds.
- SU(3) compact-link left-trivialized forces and reversible leapfrog dynamics
  are implemented in JAX; zero and finite pure-gauge links have zero force,
  non-flat fields have non-zero force, compact descent lowers Wilson action,
  momentum kinetic energy is gauge invariant, compactness is preserved, and
  finite gauge covariance holds.  The SU(3) force still differentiates through
  compact perturbation exponentials, so full SU(3) leapfrog JIT is deferred
  until a staple/vectorized force is implemented.
- Full-SM analytic-force profiling shows Wilson force is no longer the
  immediate simulator bottleneck in warm runs.
- Sparse recorded scans make `record_every` reduce actual observable work,
  not only saved output density.

**Open**:
- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Full symbolic all-momentum BCC Bloch-symbol unitarity proof.
- Optional conversion to a `J`-adapted explicit `C^16` internal basis.
- Exact Gauss-law projection / constraint solving beyond the combined residual
  and bounded diagnostic descent controls.
- Realistic Yukawa mass matrices and family structure.
- Long-time stability tests and larger-lattice simulator campaigns.

## boundary_response — theorem-gated boundary-response sidecar

**Goal**: test whether BCC boundary response can derive flavor-sector
structure from explicit `H_Q, V -> Sigma(z)` mechanisms, while keeping every
extra assumption named and falsifiable.

**Current result**: V1 through V43 are implemented. The selector sector is
closed for polishing modulo one explicit intermediate axiom:
`positive_quartic_backreaction_bounds_selector_radius`.

The closed selector chain is:

```text
single-Weyl BB walk
  -> real helicity-locked A2u selector term
  -> accepted tetrahedral branch selected
  -> free BB even energy destabilizes r=0
  -> positive quartic backreaction bounds radius
  -> finite nonzero selector vacuum
```

**Implemented theorem gates**:
- Exact residual transfer invariant `epsilon = sqrt(2)-1`.
- Product sterile-tail and semi-infinite Weyl-function theorem deriving the
  framed neutrino core `K_nu = epsilon^2 P_u + P_b`.
- Charged-lepton leakage and conditional leptonic phase-word gates.
- Quark boundary-shell gates: transfer hierarchy, color return, conditional
  CKM assembly, entropy/conservation reduction, and explicit records of the
  remaining boundary-shell assumptions.
- Chiral BB selector gates V35-V38: filled-band selector sign, branch
  selection, microscopic filled-band potential, and free-BB radial no-go.
- Higgs/backreaction closure gates V39-V43: radial stabilizer sufficiency,
  gauge-invariant Landau uniqueness, BB-induced breaking, analytic radial
  theorem, and closure ledger.

**Honest limits**:
- The positive quartic coefficient is not yet derived from a microscopic
  gauge/Higgs/backreaction update; it is the one named intermediate axiom left
  by V43.
- PMNS and CKM textures remain gated by their respective boundary-shell
  assumptions. They are not presented as closed derivations.
- The sidecar is a boundary-response theory audit, not a full dynamical SM
  simulation.

**Primary docs**:
- `docs/bcc_qca_boundary_response_research_note_v_2.md`
- `docs/epsilon_provenance.md`
- `src/clifford_3plus2_d5/boundary_response/README.md`
- `src/clifford_3plus2_d5/boundary_response/STATUS.md`
- `src/clifford_3plus2_d5/boundary_response/PLAN.md`
- `src/clifford_3plus2_d5/boundary_response/parameter_ledger.md`

## depth_scar — boundary repair-scar depth audit

**Goal**: test whether the quark family depth spectrum `{0,2,6}` can be
represented by a graph-native boundary repair operator rather than by a
hand-written diagonal spurion.

**Result**: V1-V12 PASS.  An `S3 -> Z2` repair scar with path graph `P3` gives

```text
D_scar = 2 Delta(P3)
Spec(D_scar) = {0, 2, 6}
epsilon^D_scar = diag(1, epsilon^2, epsilon^6)
```

in the defect normal-mode basis.

V2 separates built-in outputs from genuine predictions.  The sidecar now records
the exact projector transfer kernel

```text
T = P0 + epsilon^2 P2 + epsilon^6 P6
```

with fixed port relations, a democratic rank-one leading response, CKM exponents
`lambda:lambda^2:lambda^3`, and the no-go that a pure path graph supplies no
intrinsic graph-holonomy CP phase.

V3 derives the scar at the effective edge-weight level.  For nonnegative repair
weights `w=(w_ua,w_ab,w_ub)`, the symmetric potential

```text
V=(S1-2)^2+(S2-1)^2+S3
```

has exactly the three missing-edge path scars `(1,1,0)` and permutations as
zero-energy minima.  Each minimum gives `Spec(Delta)={0,1,3}` before doubling.

V4 proves the loop-healing CP separation.  The pure path has cycle rank `0`, so
all tree phases are removable and no intrinsic graph-holonomy CP phase exists.
Restoring the missing edge gives one cycle.  The real healed spectrum is
`{0,1+2 delta,3}` before doubling and `{0,2+4 delta,6}` after doubling; the
complex healed triangle carries one gauge-invariant loop phase `phi`.

V5 derives the same path scar from a canonical length-3 nilpotent repair flag:

```text
N = |u><a| + |a><b|,
Delta_flag = N N.T + N.T N - N - N.T = Delta(P3).
```

The rank-one nilpotent and cyclic closure controls are rejected; weighted flags
hit the target spectrum only for unit adjacent amplitudes up to sign.

V6 removes the remaining normalization freedom on that support.  For the generic
local flag `r exp(i alpha)|u><a| + s exp(i beta)|a><b|`, partial-isometry
conditions force nonzero `r^2=s^2=1`, and tree rephasing removes both phases.

V7 classifies all `64` no-self-loop binary directed supports on three ports.
Minimal rank-2 length-3 nilpotent supports with exactly two edges form one `S3`
orbit, represented by the V5 path flag.  Dropping the two-edge minimality
condition admits six shortcut supports, so minimality is load-bearing.

V8 promotes the minimality condition to a finite constrained variational
principle.  Over all rank-2, length-3 nilpotent, all-port-active supports,
there are twelve feasible supports: six path flags and six acyclic shortcuts.
Minimizing `cost(N)=edge_count(N)` gives minimum cost `2` and selects exactly
the six path flags; shortcuts have cost `3`.

V9 separates support minimality from normalization minimality.  Given a
three-level filtration `h(u,a,b)=(0,1,2)`, monotone repair allows
`a->u`, `b->a`, and `b->u`; one-tick residual geometry `u-a-b` forbids the
shortcut `b->u` by distance `2` / same BCC bipartite parity.  Rank-complete
local repair therefore has the unique path support.  Equal unit weights remain
conditional on active repair partial-isometry saturation; without it the result
is a weighted path, and `{0,1,3}` forces `w1=w2=1`.

V10 proves the active repair-isometry saturation identity.  For
`N=P_R U P_A` and leakage `L=(I-P_R) U P_A`, unitarity gives
`N.H N + L.H L = I_A`.  Hence the active repair block has unit edge weights
exactly when `L=0`.  Symmetric leakage preserves the `1:3` weighted-path ratio
but rescales the spectrum; unequal leakage breaks the ratio.

V11 proves the selection-signature no-leakage bridge.  If the microscopic
successor sets are `Omega(a)={u}` and `Omega(b)={a}`, then no output lies
outside the repaired range, so `L=0`; V10 then gives the unit path flag.  V11
does not perform the microscopic enumeration, but reduces the next gate to the
finite successor count.

V12 implements that finite successor certificate for the currently modeled
local boundary candidate basis.  Every active source/candidate transition has an
`ALLOW` or `FORBID` row with exact vetoes.  The certificate gives
`Omega(a)={u}`, `Omega(b)={a}`; classifies `b->u` as shortcut repair; and
rejects bulk/spectator/wrong-sector candidates as external leakage.

**Controls**:
- Unbroken `K3` remains degenerate: `{0,3,3}` before doubling and `{0,6,6}`
  after doubling.
- The diagonal `diag(0,2,6)` control has the right spectrum but is rejected as
  non-graph-native.
- The weighted-scar variants `(1,1,0)` and `(1/3,1/3,4/3)` reproduce
  `{0,1,3}` before doubling, while the symmetric triangle remains degenerate.

**Honest limit**: the sidecar does not derive the height filtration or
one-tick residual geometry from the actual BCC-QCA update, does not prove that
the V12 finite candidate basis is the complete microscopic local boundary
basis, does not make `P3` a mass model without a left/right Yukawa assignment,
and does not derive the microscopic loop-healing parameters `delta` and `phi`.

**Primary docs**:
- `src/clifford_3plus2_d5/depth_scar/README.md`
- `src/clifford_3plus2_d5/depth_scar/THEORY.md`
- `src/clifford_3plus2_d5/depth_scar/STATUS.md`
- `src/clifford_3plus2_d5/depth_scar/PLAN.md`
- `src/clifford_3plus2_d5/depth_scar/parameter_ledger.md`

## triality — closed: negative result

**Goal**: test whether the explicit Spin(8) triality outer automorphism can
produce three equivalent SM-generation carriers from one chiral-16, without
declaring three generations by hand.

**Result**: K1 FAIL.  Triality maps each of the three SM-inside-Spin(8)
Cartan generators outside the SM Cartan subspace:

| SM Cartan generator | Cartan coords | Image under triality |
|---|---|---|
| SU(3)_c v_0 | (-1, 1, 0, 0) | (0, 0, -1, 1) — outside |
| SU(3)_c v_1 | (-1, 0, 1, 0) | (0, -1, 0, 1) — outside |
| Y'         | (1/3, 1/3, 1/3, 1/2) | (3/4, -1/12, -1/12, -1/12) — outside |

Therefore the Spin(8) triality outer automorphism does not preserve
`g_SM(8) = SU(3)_c ⊕ U(1)_{Y'}` as a subalgebra, and the three rotated
chiral-16 carriers carry inequivalent SM-content positions.

**Underlying tension**: Spin(8) triality is the D_4 Dynkin Z/3 symmetry
acting symmetrically on the four Cartan elements `H_0..H_3`, but the
Pati-Salam factorization `Cl(0,6) ⊗ Cl(0,4)` makes `H_3` qualitatively
different.  Triality does not respect the Pati-Salam asymmetry.

**Implemented**:
- 28 Spin(8) generators on the chiral-16 via Cl(0,10) index `{0..7}`
  embedding.
- 4×4 triality Cartan matrix `T`: orthogonal, order 3, det +1.
- Spin(8)-restricted hypercharge `Y' = (1/3, 1/3, 1/3, 1/2)`.
- K1 test (Cartan-subspace preservation) and K2 spectrum diagnostic.

**Tests**: 21 passing.

**What this rules out vs preserves**: see
`src/clifford_3plus2_d5/triality/SESSION_T_KILL_TEST.md`.

**Open** (intentionally out of scope): non-natural Spin(8) embeddings,
approximate / broken triality (see ``broken_triality``), discrete-flavor
models without Spin(8) origin.

## broken_triality — closed: negative result

**Goal**: follow-up after the exact-triality K1 failure.  Reinterpret the
failure direction as a forced flavor structure: build a 3×3 Yukawa from
triality-rotated, SM-projected vectors and ask whether the resulting
spectrum can support SM mass hierarchy + CP phase.

**Result**: BT-1 PASS (with caveat), BT-2 FAIL.

| Kill | Verdict | Detail |
|---|---|---|
| BT-1 (Yukawa overlap structure)  | PASS  | 3 distinct eigenvalues {5/7, 31/72, 0}, off-diagonals non-zero, rank 2 |
| BT-2 (mass hierarchy)            | **FAIL** | non-zero ratio 360/217 ≈ 1.66, far below fail threshold 10 |
| BT-3 (CP phase)                  | skipped | program closed at BT-2 |
| BT-4 (parameter audit)           | skipped | program closed at BT-2 |

The default starting vector `v_* = Y'` has an `H_1 ↔ H_2` swap symmetry
that survives the triality cycle and SM projection, forcing the Yukawa
to be rank 2.  The two non-zero eigenvalues differ by a factor of only
~1.66 — essentially flat compared to the SM's 10²–10⁵ ratios within a
sector.

**Combined message with triality/**: both sidecars died on the same
structural mismatch.  Spin(8) triality is symmetric across `H_0..H_3`;
the Pati-Salam-aligned SM Cartan picks out `H_3`.  Whichever direction
you push the construction, the alignment fails.

**Implemented**:
- BT-1 (`yukawa_overlaps.py`): triality orbit + SM Cartan projection +
  3×3 overlap matrix + audit payload.
- BT-2 (`mass_hierarchy.py`): non-zero eigenvalue ratio against pass/fail
  thresholds.
- No new octonion / Clifford / Pati-Salam code; all algebra comes from
  ``triality/reuse.py``.

**Tests**: 17 passing.

**What this rules out vs preserves**: see
`src/clifford_3plus2_d5/broken_triality/SESSION_BT_KILL_TESTS.md`.  The
"approximate embedding / CP from BCC lattice anisotropy" hope is
**orthogonal** to triality and not addressed here.

**Effort spent**: ~4 hours of focused work.  Original sidecar sketch
budget was 3-6 months; the early-kill discipline returned a publishable
negative result in ~1% of that.

## topology — closed: negative result

**Goal**: kill-disciplined audit of topological mechanisms for three SM
generations on the BCC × chiral-16 carrier.  Four candidates ordered
cheapest-first: spatial body-diagonal Z_3, color SU(3)_c Z_3 center,
``π_3(G/H)`` for carrier-relevant cosets, and discrete anomaly forcing
N = 3.

**Result**: all four phases produced clean negatives.

| Phase | Verdict | Detail |
|---|---|---|
| D-1 (spatial Z_3 on BCC × chiral-16) | TOPOLOGY KILL | chiral-16 internal is built from Cl(0,10) gammas; spatial Z_3 acts trivially → 16 = 16 + 0 + 0 |
| D-2 (color Z_3 center)                | COLOR Z_3 KILL | chiral-16 = (8, 4, 4) under ``g_3 = diag(1, ω, ω²)`` — asymmetric, not three equal generations |
| D-3 (π_3 literature note)             | PI3 KILL | every carrier-relevant ``G/H`` has ``π_3`` ∈ {0, Z}; no Z/3 torsion |
| D-5 (discrete anomaly forcing N = 3)  | ANOMALY KILL | SM anomalies cancel per generation; Witten satisfied for any N; constraint reduces to 0 = 0 |

**Bonus finding**: the Bialynicki-Birula BCC hops are NOT Z_3-equivariant
in the spatial × Dirac sector (all 8 hop residuals non-zero).  Orthogonal
to the three-generation question — recorded for a future walk-symmetry
audit.

**Implemented**:
- ``bcc_z3_rotation.py``: 3×3 cyclic rotation + spinor SU(2) lift
  (cubes to -I) + 4-spinor block-diagonal lift.
- ``hop_equivariance.py``: BB hops Z_3-equivariance audit (bonus
  finding).
- ``internal_triviality.py``: chiral-16 spatial Z_3-trivial by
  construction.
- ``color_center_z3.py``: 16 = 8 + 4 + 4 decomposition.
- ``PI3_LITERATURE_NOTE.md``: 1-page markdown note citing
  Mimura-Toda, Husemoller, Bott-Tu.
- ``anomaly_cancellation/``: continuous SM anomaly polynomial,
  Witten global SU(2) check, combined N-constraint extraction,
  audit payload.

**Tests**: 53 passing.

**What this rules out vs preserves**: see
`src/clifford_3plus2_d5/topology/SESSION_TOPOLOGY.md`.  The closure is
specific to the BCC × chiral-16 carrier; higher-rank carriers (E_6, E_8),
instanton / Pontryagin (D-4 deferred), and cobordism / TQFT routes
remain open and orthogonal.

**Effort spent**: well within the ~4-5 week committed budget;
kill-discipline ordering meant D-1 / D-2 / D-3 closed in days and
D-5 closed structurally rather than through a multi-week lattice
computation.

## sme — closed: UNFALSIFIABLE PASS

**Goal**: Bold A audit.  Bridge the cp/ sidecar's (CP-odd, T_{2g}) cell
of H^(1) to the Standard Model Extension (SME) framework and compute
the maximum allowed ε from current experimental bounds.

**Result**: UNFALSIFIABLE PASS at ε ≲ 2 × 10⁻³³ m (~10² × Planck length).

| Phase | Verdict | Detail |
|---|---|---|
| A-1 (SME framework identification) | dim-5 non-minimal SME, fermion sector, CP-odd spin-tensor | five symmetry classes pinned by cp/ verified directly |
| A-2 (H^(1) → SME tensor mapping)   | three non-zero d^{(5)} components | axial-vector × 2 derivatives, CPT-even, CP-odd; per chirality: σ^x k_y k_z, −σ^y k_x k_z, σ^z k_x k_y |
| A-3 (SME_LITERATURE_NOTE.md)       | |d^{(5)}| ≲ 10⁻¹⁷ GeV⁻¹ representative | Kostelecky-Russell entry-id verification + 2024-2025 atom-interferometry cross-check flagged for follow-up |
| A-4 (symbolic ε constraint)        | ε ≲ 2 × 10⁻³³ m, ~10² ℓ_P | UNFALSIFIABLE PASS verdict class |
| A-5 (combined audit)               | program alive but unfalsifiable in this channel | clean PASS, not a kill |

**Underlying message**: the program is structurally consistent with
current experimental bounds at the representative ε scale.  No
experimental discrepancy forces a re-examination of the BCC × chiral-16
carrier or the cp/ dual-positive result.  However, the bound on ε from
this channel is ~10⁸ above current observational reach — the program
is currently unfalsifiable in the d^{(5)} channel.

**Implemented**:
- ``sme_framework_identification.py``: re-verify 5 symmetry classes from
  cp/; declare SME sector.
- ``sme_tensor_mapping.py``: Pauli decomposition of (CP-odd, T_{2g})
  cell → 3 non-zero T^{aij} entries → ``d^{(5)}_{αβγ}`` SME components.
- ``SME_LITERATURE_NOTE.md``: cited representative bounds with
  Kostelecky-Russell, Kostelecky-Mewes, Mattingly, Liberati references.
- ``epsilon_constraint.py``: Planck-length conversion + tightest-face
  rule + four-class verdict.
- ``sme_audit.py``: combined payload.

**Tests**: 37 passing.

**Follow-up flags**:
1. Verify Kostelecky-Russell entry ids for the three d^{(5)} components.
2. Cross-check against 2024-2025 atom-interferometry papers; update
   ``DIM5_FERMION_BOUND_GEV_INVERSE`` if tightened.
3. Check field-redefinition triviality (F-sme-5).
4. Consider parallel photon-sector audit using (k_F)^(5) bounds.

**Effort spent**: well within the ~2-week committed budget.  Scaffold
+ all five phases + tests in a single session.

## strongcp — closed: STRONG-CP TRIVIAL (structural + direct-computation)

**Goal**: kill-disciplined audit of whether the BCC Bialynicki-Birula
walk's effective action contributes to θ_QCD (bounded ≤ 10⁻¹⁰ by
neutron-EDM measurements).  Combines a cubic-group parity selection-rule
argument with direct chiral-anomaly checks.

**Result**: STRONG-CP TRIVIAL at O(ε) and O(ε²) by structural lattice
symmetry; SAFE at higher orders.  The program **naturally satisfies the
strong-CP problem without invoking an axion or accidental cancellation**.

| Phase | Status | Verdict |
|---|---|---|
| SC-1 (degree-3 cubic harmonics) | done | A_{2u} ⊕ T_{2u} ⊕ T_{1u} projectors built |
| SC-2 (BCC centrosymmetry)        | done | lattice + BB walk + Bloch symbol all centrosymmetric |
| SC-3 (higher-order H^(n) parity) | done | H^(2) is 100% T_{1u} with ZERO A_{2u} content |
| SC-4 (lattice Q(x) density)      | done | spatial Q dimensionally trivial; BCC plaquette rep parity-even; A_{2u} = 0 for SU(2)_L + SU(2)_R + SU(4)_PS; gauge-content independent |
| SC-5 (chiral anomaly + θ̄ shift)  | done | tr(γ^5 H^(1)) = tr(γ^5 H^(2)) = 0 |
| SC-6 (combined audit)            | done | STRONG-CP TRIVIAL aggregated and reported |

**Mechanism**: BCC centrosymmetry forces every H^(n) into a definite
parity class under k → -k.  H^(1) lives in T_{2g} (a g-irrep, verified
by cp/); H^(2) lives in T_{1u} but with ZERO A_{2u} content.  The
θ_QCD operator's momentum-shape ``k_x k_y k_z`` lives in A_{2u} (a u-
irrep).  By the cubic-group selection rule (``g × g = g``,
``u × u = g``, ``g × u = u``), no product of H^(n) operators (each
in g- or non-A_{2u} u-irreps) can populate A_{2u}.  Independently,
``tr(γ^5 H^(n)) = 0`` for n = 1, 2 — H^(1) and H^(2) are individually
vector (not axial), inducing no chiral rotation on the fermion
measure.

**Implemented**:
- ``cubic_harmonics_degree3.py``: full degree-3 O_h decomposition
  with projectors for A_{2u}, T_{2u}, T_{1u}.
- ``bcc_centrosymmetry.py``: lattice, walk, and Bloch-symbol
  inversion-symmetry verification.
- ``higher_order_parity.py``: H^(1) BCH cross-check vs cp/, H^(2)
  extraction via L_3 BCH coefficient, irrep decomposition.
- ``chiral_anomaly_check.py``: tr(γ^5 H^(n)) tests for n = 1, 2;
  cross-term tr(γ^5 H^(1) H^(2)) recorded as O(ε³) observation.
- ``theta_bar_constraint.py``: neutron-EDM bound comparison.
- ``lattice_topological_charge.py``: SC-4 direct lattice-gauge
  computation; F_{ij} = (H − H†)/(2i) extraction, plaquette
  inversion permutation, A_{2u} projection verification, gauge-
  group independence for SU(2)_L, SU(2)_R, SU(4)_PS.
- ``strong_cp_audit.py``: combined SC-1..SC-5 payload (structural
  + direct-computation).

**Tests**: 69 passing.

**Significance**: publishable structural + direct-computation positive.
Combined with cp/'s dual-positive (CP-violation at O(ε) in T_{2g}) and
sme/'s UNFALSIFIABLE PASS (ε ≲ 2 × 10⁻³³ m), the program now has a
coherent story: lattice CP-violation lives in a g-irrep that cannot
leak into the A_{2u} θ_QCD channel, so the program is automatically
strong-CP-safe regardless of ε.  SC-4 confirms this at the gauge
sector: the BCC 6-dim Wilson-plaquette permutation rep is parity-even
(inversion permutation is identity), so tr(F·F) ⊂ Sym²(g-rep) ⊂
g-irreps — A_{2u} absent.  Holds for all three Pati-Salam gauge
factors (SU(2)_L, SU(2)_R, SU(4)_PS).

**Follow-up flags**:
1. Temporal Wilson plaquettes (would enable non-trivial 4D Q
   computation) — ~2-3 weeks independent follow-up.
2. Extend H^(n) audit beyond H^(2) (selection rule predicts the
   pattern; confirming H^(3), H^(4) would be a sanity check).
3. Photon-sector dim-5 θ-term has different selection rules; separate
   audit.

**Effort spent**: SC-1, SC-2, SC-3, SC-5, SC-6 in one session
(~3 days); SC-4 in a follow-up session (~half a day with the
structural framing).  Total against the ~3-4 week committed budget.

## koide — closed: KOIDE CONSISTENT (PDG NOT IN LOCUS)

**Goal**: audit whether the BCC body-diagonal Z₃ structure of the
program naturally predicts/permits/forbids the empirical Koide formula
K = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 2/3 for charged-lepton
masses.  Triggered by the observation that the Koide cone direction
(1,1,1)/√3 is exactly the BCC body-diagonal axis — a striking coincidence
demanding direct test.

**Result**: KOIDE CONSISTENT.  The BCC body-diagonal Z₃ structure
admits Koide-satisfying Yukawa solutions but does not uniquely predict
the PDG mass triple.

| Phase | Verdict | Detail |
|---|---|---|
| KO-1 (empirical + cone geometry) | done | K_PDG = 0.666661 vs 2/3, deviation 6×10⁻⁶; three equivalent forms verified |
| KO-2 (BCC body-diagonal Z₃ on σ^a) | done | R fixes (1,1,1)/√3; trace + 2D-non-trivial Z₃ irrep projectors commute with R |
| KO-3 (BCC-Z₃-orbit 3×3 Yukawa) | done | Eigenvalues (3|v_t|², (3/2)|v_o|², (3/2)|v_o|²) — always 2-fold degenerate |
| KO-4 (cone vs locus) | done | L_Z3 ∩ C = 1-parameter family at |v_t|/|v_o| = 3+2√2; L_Z3 ⊄ C; PDG ∉ L_Z3 |
| KO-5 (combined audit) | done | KOIDE CONSISTENT aggregated and reported |

**Geometric mechanism**: the Koide cone direction (1,1,1)/√3 IS the
Z₃-trivial irrep direction of the BCC body-diagonal rotation.  The
coincidence is real, not accidental.  The Z₃-equivariant Yukawa from
the BCC body-diagonal orbit has a 1-parameter on-cone sub-family at
|v_t|/|v_o| = 3 + 2√2 ≈ 5.83, where the non-degenerate-to-degenerate
mass ratio is exactly 2(3+2√2)² = 2(17+12√2) ≈ 67.94.

**Phenomenological limit**: the Z₃-equivariant construction ALWAYS
gives 2-fold degenerate eigenvalues (m₂ = m₃ structurally), but PDG
charged-lepton masses are all distinct.  Therefore PDG ∉ Z₃-equivariant
locus.  Three distinct PDG masses require **Z₃-breaking input** —
naturally via a dynamical Higgs VEV alignment that tilts the Yukawa
off the equivariant locus.  Bold-B is the natural follow-up.

**Implemented**:
- ``koide_geometry.py``: PDG verification, three equivalent geometric
  forms, cone parametrization.
- ``bcc_z3_on_flavor.py``: BCC R on σ^a-space, Z₃-irrep projectors,
  σ^a ↔ generation identification.
- ``yukawa_eigenvalue_locus.py``: BCC-Z₃-orbit Yukawa with analytic
  eigenvalue structure; Koide special ratio derivation.
- ``cone_locus_compatibility.py``: PREDICTED/CONSISTENT/CONFLICT
  classifier with PDG-in-locus tag.
- ``koide_audit.py``: combined KoideAuditPayload.

**Tests**: 54 passing.

**Significance**: the audit settles the BCC↔Koide coincidence with a
clear structural positive (the cone direction IS the Z₃-trivial axis,
not accidental) and a clear phenomenological limit (PDG mass hierarchy
requires Z₃-breaking input beyond carrier structure).

**Follow-up flags**:
1. Bold-B dynamical Higgs sector for VEV-driven cone selection.
2. Quark Koide audit (K_up ≈ 0.85, K_down ≈ 0.73 — imprecise but
   suggestive).
3. Non-equivariant Yukawa scan to characterize the broader locus.

**Effort spent**: all 5 phases completed in one session (~half a day
against the ~3-4 week committed budget); the analytical structure of
the circulant Yukawa made KO-3 / KO-4 tractable without numerical scan.

## Workspace meta

- [`docs/PUBLICATION_PLAN.md`](docs/PUBLICATION_PLAN.md) — publication plan for the obstruction_r10 paper.
- [`docs/REORG_PLAN.md`](docs/REORG_PLAN.md) — record of the workspace reorganization that moved the trunk into `obstruction_r10/`.
- [`docs/project_conventions.md`](docs/project_conventions.md) — shared coding conventions across modules.

## Cross-module imports

| From | To | Why |
|---|---|---|
| `lepton.*` | `obstruction_r10.qca.rule_verdict` | The trunk's VerdictProfile / predicate machinery is reused by Lab A / Lab B verdicts. |
| `lepton.*` | `algebra.*` | Shared exact rational matrix algebra. |
| `spacetime_qca.*` | `algebra.*` | Shared matrix utilities. |
| `spacetime_qca.*` | `sim.*` | Shared JAX state/link/diagnostic infrastructure; physics-specific kernels stay in `spacetime_qca`. |
| `spacetime_qca.tests.*` | `lepton.checkerboard_patisalam`, `lepton.clifford_patisalam`, `lepton.patisalam_sm`, `lepton.sm_hypercharge` | Sessions 20-21 verify tensor lift and mass compatibility with real Pati-Salam / SM internal generators. |
| `triality.reuse` | `lepton.clifford_octonion`, `lepton.clifford_patisalam`, `lepton.patisalam_sm`, `lepton.sm_hypercharge` | Single import surface for the triality kill test. |
| `broken_triality.reuse` | `triality.*` | All algebra from triality (which itself imports from lepton); no new octonion / Clifford code. |
| `topology.reuse` | `spacetime_qca.bcc_weyl`, `spacetime_qca.dirac`, `lepton.patisalam_sm` | Single import surface for the topology audits (BB hops, Dirac gammas, chiral-16 SU(3)_c). |
| `sme.reuse` | `cp.continuum_cp`, `cp.cubic_harmonics`, `spacetime_qca.dirac` | Single import surface for the SME audit (H^(1), T_{2g} projector, γ matrices). |
| `strongcp.reuse` | `cp.continuum_cp`, `cp.cubic_harmonics`, `cp.discrete_symmetries`, `spacetime_qca.bcc_weyl`, `spacetime_qca.continuum` | Single import surface for the Strong-CP audit (H^(1), γ matrices, BCC walk, BCH machinery). |
| `koide.reuse` | `cp.j_misalignment`, `cp.cubic_harmonics`, `cp.continuum_cp`, `topology.bcc_z3_rotation`, `broken_triality.yukawa_overlaps` | Single import surface for the Koide audit (Higgs map basis, BCC R rotation, 3×3 Yukawa-from-orbit template). |
| `obstruction_r10.qca.*` | `algebra.*` | Shared matrix utilities. |

Runtime sidecar code remains mostly factored.  The visible sidecar couplings
today: `spacetime_qca` verifies tensor compatibility against a small set of
real internal generators from `lepton`; `triality` and `broken_triality`
import from `lepton` through their own thin `reuse.py` modules without
duplicating any algebra.
