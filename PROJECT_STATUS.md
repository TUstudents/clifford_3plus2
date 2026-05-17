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

**Sessions**: 20 BCC Dirac Bloch-symbol audit complete.

**Implemented**:
- BCC geometry (8 body diagonals).
- Bialynicki-Birula 2-component Weyl Bloch symbol on BCC, pinned to
  Phys. Rev. D 49, 6920 (1994), Section II.
- 4-component Dirac chiral assembly.
- Hypercubic doubling control.
- First-order continuum expansion utilities.
- Constant-background gauge tensor lift against a real 32×32 Pati-Salam
  `SU(2)_L` generator from `lepton`.
- Session 20 report: `src/clifford_3plus2_d5/spacetime_qca/SESSION_20_BCC_DIRAC.md`.
- 20 passing tests.

**Result**:
- `H_R(k) = sigma . k`.
- `H_L(k) = -sigma . k`.
- `H_D(k) = alpha . k`.
- Hypercube control has 8 literal cubic BZ-corner doublers.
- BCC cubic-corner gapless representatives are reciprocal-lattice origin
  equivalents, not claimed as independent doublers.
- Constant background gauge lift gives
  `H_eff = alpha.k x I_internal + I_spacetime x iA`.

**Open**:
- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Finite real-space BCC `step(state, links)`.
- Position-dependent gauge links and site-local gauge covariance.
- Mass / dynamical gauge in subsequent sessions.

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
| `spacetime_qca.tests.*` | `lepton.checkerboard_patisalam` | Session 20 verifies the tensor lift with a real Pati-Salam internal generator. |
| `obstruction_r10.qca.*` | `algebra.*` | Shared matrix utilities. |

Runtime sidecar code remains mostly factored.  The visible sidecar coupling
today is test-level: `spacetime_qca` verifies tensor compatibility against a
real generator from `lepton`.
