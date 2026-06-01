# clifford-3plus2-d5

Workspace for several QCA-derivation modules. Each module is a self-contained
sub-project with its own source, tests, scripts, and documentation. They share
a single Python environment and a small set of common utilities (rational
matrix algebra, exterior algebra, Spin(10) branching).

## Modules

### Active

| Module | Status | What it is |
|---|---|---|
| [`obstruction_r10`](src/clifford_3plus2_d5/obstruction_r10/) | **Frozen for publication** | Carrier-first QCA derivation of the chiral-16 of Spin(10) on `R^10`, with five structural obstruction propositions. The five propositions plus two corollaries close specified primitive classes; conjectural Proposition 4b has a finite census witness. |
| [`lepton`](src/clifford_3plus2_d5/lepton/) | **Positive result, active** | Cl(0,10) Pati-Salam factorization producing the full SM gauge content (`SU(3) × SU(2)_L × U(1)_Y`), a compatible complex structure, and the correct one-generation hypercharge spectrum, with 1+1D massless-Dirac/Weyl continuum dynamics under background gauge links. |
| [`spacetime_qca`](src/clifford_3plus2_d5/spacetime_qca/) | **In progress** | 3D BCC Weyl/Dirac walk, tensor lift to the lepton internal carrier, position-dependent gauge covariance, Wilson plaquette dynamics, Pati-Salam/SM compact gauge prototypes, dynamical Higgs/Yukawa controls, Gauss projection diagnostics, scan-backed simulator with sparse observation recording. Sessions 20–58. |
| [`boundary_response`](src/clifford_3plus2_d5/boundary_response/) | **V1–V43 implemented** | Theorem-gated sidecar for BCC-QCA boundary-response flavor. It derives the framed neutrino core `K_ν = ε²P_u + P_b`, keeps PMNS/CKM behind explicit boundary-shell gates, and closes the vacuum-selector thread through V43 modulo one named intermediate axiom: positive quartic backreaction bounds the selector radius. |
| [`depth_scar`](src/clifford_3plus2_d5/depth_scar/) | **V1–V5 PASS, active** | Boundary repair-scar sidecar for the quark depth spectrum. V1 proves that an `S3 → Z2` path repair scar has `D_scar = 2 Δ(P3)` with spectrum `{0,2,6}`. V2 records the prediction ledger. V3 derives the path scars as minima of a symmetric effective edge-weight potential. V4 proves that CP holonomy requires loop healing. V5 derives `Δ(P3)` from a canonical length-3 nilpotent repair flag. |

### Shared infrastructure

| Module | Status | What it is |
|---|---|---|
| [`sim`](src/clifford_3plus2_d5/sim/) | **Shared infrastructure** | Generic JAX simulation helpers: array conversion, periodic rolls, state/link layout, finite-value diagnostics, benchmark wrappers, sparse recorded scans, callable profiling, and `.npz`/JSON persistence. |
| [`cp`](src/clifford_3plus2_d5/cp/) | **Dual-positive, reused** | Continuum CP/T audits and cubic-harmonic decomposition used as the import surface for `sme/`, `strongcp/`, `koide/`. Dual-positive verdict: CP-violation at O(ε) in T_{2g}. |

### Closed sidecars

| Module | Verdict | Why |
|---|---|---|
| [`triality`](src/clifford_3plus2_d5/triality/) | **K1 FAIL** | The natural Pati-Salam-aligned SM Cartan is not preserved as a subspace by the Spin(8) order-3 outer automorphism. Three triality copies cannot represent three equivalent SM generations under this embedding. |
| [`broken_triality`](src/clifford_3plus2_d5/broken_triality/) | **BT-2 FAIL** | BT-1 passed (3 distinct Yukawa eigenvalues, non-zero mixing), but BT-2 failed: ratio `360/217 ≈ 1.66`, essentially flat. Triality projection of a hypercharge-aligned vector does not produce SM mass hierarchy. |
| [`exceptional`](src/clifford_3plus2_d5/exceptional/) | **FULL KILL** | All four candidates closed negative: Bi(O) = Spin(8) (inherits triality fail), three Fano lines through e₇ (no Lie closure), J_3(O) under Spin(10) (`27 = 16 + 10 + 1`; one chiral-16, not three), J_3^C(O) under Spin(10) × U(1) (`54 = 16 + 16* + 10 + 10* + 1 + 1*`; particle + antiparticle, not three generations). |
| [`topology`](src/clifford_3plus2_d5/topology/) | **All four phases KILL** | D-1 spatial Z₃ on BCC × chiral-16: trivial. D-2 color SU(3)_c center Z₃: 16 = 8+4+4, asymmetric. D-3 π₃ literature: no Z/3 torsion. D-5 anomaly forcing N=3: SM anomalies cancel per generation, constraint is `0 = 0`. |
| [`sme`](src/clifford_3plus2_d5/sme/) | **UNFALSIFIABLE PASS** | Three non-zero `d⁽⁵⁾` SME components for the cp/ (CP-odd, T_{2g}) cell. ε ≲ 2 × 10⁻³³ m (~10² × Planck length), ~10⁸ above current observational reach. |
| [`strongcp`](src/clifford_3plus2_d5/strongcp/) | **STRONG-CP TRIVIAL** | Cubic-group parity selection rule + tr(γ⁵ H⁽ⁿ⁾) = 0 give automatic strong-CP safety without invoking an axion or accidental cancellation. SC-4 direct lattice computation confirms across SU(2)_L, SU(2)_R, SU(4)_PS. |
| [`koide`](src/clifford_3plus2_d5/koide/) | **KOIDE CONSISTENT** | The Koide cone direction `(1,1,1)/√3` IS the BCC body-diagonal Z₃-trivial axis (not accidental). Z₃-equivariant Yukawa always gives 2-fold degenerate eigenvalues; PDG mass triple requires Z₃-breaking input. |

## Read the project status

For an at-a-glance overview of each module's current verdict, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Shared utilities

These live at the package root and are imported by multiple modules:

- [`algebra/`](src/clifford_3plus2_d5/algebra/) — exact rational matrix algebra, real carrier construction, `IncrementalMatrixSpan` and `RationalMatrixSpan` for sparse exact echelon, commutant/centralizer helpers.
- [`exterior.py`](src/clifford_3plus2_d5/exterior.py) — chiral exterior-algebra basis machinery.
- [`branching.py`](src/clifford_3plus2_d5/branching.py) — Spin(10) → SM branching tables.

## Cross-module dependencies

Each sidecar imports from upstream modules through a thin `reuse.py` shim
(no duplicated octonion/Clifford code). The runtime couplings are:

- `lepton` → `obstruction_r10.qca.rule_verdict` (VerdictProfile-based verdict path).
- `spacetime_qca` → `sim` (JAX state/link/diagnostic infrastructure); test-level imports from `lepton` (Pati-Salam / SM generators).
- `triality` → `lepton` (Cl(0,10) gammas, SM gauge generators, hypercharge).
- `broken_triality` → `triality`.
- `exceptional` → `lepton` (octonion, Pati-Salam, SM helpers).
- `topology` → `spacetime_qca` (BCC walk, Dirac gammas) + `lepton` (chiral-16 SU(3)_c).
- `sme` → `cp` (H⁽¹⁾, T_{2g} projector) + `spacetime_qca.dirac`.
- `strongcp` → `cp` (continuum CP, cubic harmonics, discrete symmetries) + `spacetime_qca` (BCC walk, continuum BCH).
- `koide` → `cp` (Higgs map basis, BCC R rotation) + `topology` + `broken_triality` (3×3 Yukawa-from-orbit template).
- `boundary_response` -> `spacetime_qca` (BB Weyl/Dirac walk imports for the selector gates); otherwise mostly local exact/symbolic machinery.
- `depth_scar` -> `boundary_response.transfer` (exact silver-ratio transfer factor).

## Working in a module

Each module is self-contained. Tests and scripts live inside the module:

```bash
# Run all tests across all modules
uv run pytest -q

# Run tests for a single module
uv run pytest src/clifford_3plus2_d5/obstruction_r10/tests/ -q
uv run pytest src/clifford_3plus2_d5/lepton/tests/ -q
uv run pytest src/clifford_3plus2_d5/sim/tests/ -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/ -q
uv run pytest src/clifford_3plus2_d5/cp/tests/ -q
uv run pytest src/clifford_3plus2_d5/boundary_response/tests/ -q
uv run pytest src/clifford_3plus2_d5/triality/tests/ -q
uv run pytest src/clifford_3plus2_d5/broken_triality/tests/ -q
uv run pytest src/clifford_3plus2_d5/exceptional/tests/ -q
uv run pytest src/clifford_3plus2_d5/topology/tests/ -q
uv run pytest src/clifford_3plus2_d5/sme/tests/ -q
uv run pytest src/clifford_3plus2_d5/strongcp/tests/ -q
uv run pytest src/clifford_3plus2_d5/koide/tests/ -q
uv run pytest src/clifford_3plus2_d5/depth_scar/tests/ -q

# Run a script via module invocation
uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.gauge_equivalence_check --check

# Run the scan-backed spacetime simulator
uv run python -m clifford_3plus2_d5.spacetime_qca.simulator.scripts.run_sim
```

Heavy JAX dynamics and exact-symbolic tests are tagged with `pytest.mark.slow`.
Use `-m "not slow"` for the fast lane.

## Workspace-level docs

These describe the workspace itself, not module results:

- [`docs/PUBLICATION_PLAN.md`](docs/PUBLICATION_PLAN.md) — plan for the obstruction_r10 paper.
- [`docs/REORG_PLAN.md`](docs/REORG_PLAN.md) — record of the workspace reorganization.
- [`docs/project_conventions.md`](docs/project_conventions.md) — shared coding conventions.
- [`docs/bcc_qca_boundary_response_research_note_v_2.md`](docs/bcc_qca_boundary_response_research_note_v_2.md) — research note motivating the `boundary_response` sidecar.
- [`docs/epsilon_provenance.md`](docs/epsilon_provenance.md) — disambiguates the dimensionless silver-ratio flavor invariant from lattice/BCH expansion epsilons.
- [`docs/depth_hierarchy_mechanism_review.md`](docs/depth_hierarchy_mechanism_review.md) — kill-gated review of the quark family depth hierarchy mechanism.

## Honest framing

This workspace does not derive the Standard Model from QCA in any single
module. Each module asks a different question and answers honestly:

- `obstruction_r10` classifies *what cannot be derived* in specified primitive classes.
- `lepton` *constructs* the SM gauge content from a declared Clifford framework, accepting specific algebraic choices as input.
- `spacetime_qca` builds the 3+1D BCC-lattice simulation arena around that internal carrier.
- `boundary_response` derives the framed neutrino core and closes the vacuum-selector sector through V43 modulo the positive-quartic backreaction axiom; PMNS/CKM remain gated by explicit boundary-shell assumptions.
- `depth_scar` upgrades the `{0,2,6}` quark depth spectrum to a graph-Laplacian transfer operator, derives the path scar from an effective edge-weight potential and a nilpotent repair flag, and identifies loop healing as the minimal graph-native CP location, while keeping the microscopic flag origin and loop parameters open.
- The closed sidecars (`triality`, `broken_triality`, `exceptional`, `topology`) cumulatively rule out the algebraic routes to three generations.
- The closed-PASS sidecars (`sme`, `strongcp`, `koide`) check consistency with current experimental bounds and identify the residual physical inputs.

The workspace is the honest superposition of these answers. The kill-disciplined
sidecar pattern — single audit module per gate, dataclass payload, controls +
verdict + interpretation — is the project's working method.
