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

## spacetime_qca — in progress

**Goal**: 3D BCC Weyl walk (Bialynicki-Birula 1994) → 4D Dirac chiral
assembly → tensor lift to `lepton`'s chiral-16 internal carrier →
background gauge covariance audit.

**Sessions**: 20 BCC Dirac Bloch-symbol audit complete; 21 mass-layer audit
complete; 22 finite real-space step complete; 23 Higgs/Yukawa representation
audit complete; 24 position-dependent background gauge covariance complete;
24b BCC plaquette holonomy complete; 25 static Higgs/Yukawa map-layer audit
complete.

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
- 63 passing tests.

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

**Open**:
- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Full symbolic all-momentum BCC Bloch-symbol unitarity proof.
- Optional conversion to a `J`-adapted explicit `C^16` internal basis.
- Plaquette action / Wilson observable normalization.
- Dynamical Higgs-Yukawa layer and realistic Yukawa mass matrices.
- Dynamical gauge fields.

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
| `spacetime_qca.tests.*` | `lepton.checkerboard_patisalam`, `lepton.clifford_patisalam`, `lepton.patisalam_sm`, `lepton.sm_hypercharge` | Sessions 20-21 verify tensor lift and mass compatibility with real Pati-Salam / SM internal generators. |
| `obstruction_r10.qca.*` | `algebra.*` | Shared matrix utilities. |

Runtime sidecar code remains mostly factored.  The visible sidecar coupling
today is test-level: `spacetime_qca` verifies tensor compatibility against a
small set of real internal generators from `lepton`.
