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

## universal_bath — Session 09 audit, active

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

**Open**: build the microscopic BCC family-port graph and compute
`<u|H_BCC^k|b>` without inserting `I_family`; derive the charged-lepton
holonomy selection dynamics microscopically;
derive or replace the height-dynamics rule and active hidden color-return rule
needed to freeze the up/down quark BCC source vectors and normal-depth
placements without flavor data; derive or kill the down rank-5 bottom-line
selection rule; assemble mixing from Krylov/CMV basis overlaps.

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
