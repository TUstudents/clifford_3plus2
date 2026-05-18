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

**Boundary**: BCC Weyl/Dirac kernels, BCC plaquettes, Wilson observables, and
SO(2)/SU(2) force policy remain in `spacetime_qca`.

## spacetime_qca — in progress

**Goal**: 3D BCC Weyl walk (Bialynicki-Birula 1994) → 4D Dirac chiral
assembly → tensor lift to `lepton`'s chiral-16 internal carrier →
background gauge covariance audit.

**Sessions**: 20 BCC Dirac Bloch-symbol audit complete; 21 mass-layer audit
complete; 22 finite real-space step complete; 23 Higgs/Yukawa representation
audit complete; 24 position-dependent background gauge covariance complete;
24b BCC plaquette holonomy complete; 25 static Higgs/Yukawa map-layer audit
complete; 26 JAX numerical backend complete; 27 Wilson plaquette observables
complete; 28 Wilson action normalization complete; 29 SO(2) Wilson-action
gradients complete; 30 SU(2) nonabelian Wilson-force controls complete; 31
left-trivialized SU(2) force and compact descent complete; 32 reversible SU(2)
leapfrog gauge dynamics complete.

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
- SU(2) compact momentum fields, Hamiltonian-density helper, and reversible
  leapfrog update.
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
- 114 passing tests.

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

**Open**:
- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Full symbolic all-momentum BCC Bloch-symbol unitarity proof.
- Optional conversion to a `J`-adapted explicit `C^16` internal basis.
- Dynamical Higgs-Yukawa layer and realistic Yukawa mass matrices.
- SU(3), SU(4), or Pati-Salam Wilson-action gradients / force projection and
  dynamical gauge fields.
- Gauss-law constraints, fermion backreaction, and full physical gauge-field
  evolution beyond the current SU(2) leapfrog prototype.
- Numerical performance benchmarks and long-time stability tests.

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
| `obstruction_r10.qca.*` | `algebra.*` | Shared matrix utilities. |

Runtime sidecar code remains mostly factored.  The visible sidecar couplings
today: `spacetime_qca` verifies tensor compatibility against a small set of
real internal generators from `lepton`; `triality` and `broken_triality`
import from `lepton` through their own thin `reuse.py` modules without
duplicating any algebra.
