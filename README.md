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
| [`qca_smv0`](src/clifford_3plus2_d5/qca_smv0/) | **Stage 31 PASS** | Focused simulator sidecar for the next Standard-Model QCA prototype. It now includes the free BCC Weyl/Dirac bulk walk, static and dynamic SM gauge transport, Higgs/Yukawa collisions, FN recirculation, center-holonomy CP inputs, three-family production ticks, physical-right bridged transport/current/source updates, rollout/Gauss/energy/variational monitors, refinement and adjoint limitation audits, an explicit inverse helper for the current physical-right production tick, a multi-step trajectory reversibility audit, a Loschmidt echo diagnostic, and a finite tangent-response echo audit. Boundary, quantized registers, microscopic bridge/input derivations, exact energy conservation, and timestep-convergence claims remain future sessions. |
| [`boundary_response`](src/clifford_3plus2_d5/boundary_response/) | **V1–V43 implemented** | Theorem-gated sidecar for BCC-QCA boundary-response flavor. It derives the framed neutrino core `K_ν = ε²P_u + P_b`, keeps PMNS/CKM behind explicit boundary-shell gates, and closes the vacuum-selector thread through V43 modulo one named intermediate axiom: positive quartic backreaction bounds the selector radius. |
| [`depth_scar`](src/clifford_3plus2_d5/depth_scar/) | **V1–V12 PASS, active** | Boundary repair-scar sidecar for the quark depth spectrum. V1 proves that an `S3 → Z2` path repair scar has `D_scar = 2 Δ(P3)` with spectrum `{0,2,6}`. V2 records the prediction ledger. V3 derives the path scars as minima of a symmetric effective edge-weight potential. V4 proves that CP holonomy requires loop healing. V5 derives `Δ(P3)` from a canonical length-3 nilpotent repair flag. V6 fixes unit flag normalization by local partial-isometry. V7 classifies minimal binary nilpotent supports. V8 shows edge-count minimization over feasible rank-2 length-3 all-port repairs uniquely selects the path-flag orbit. V9 refines minimality into a locality theorem: one-tick residual geometry forbids the shortcut support. V10 proves unit edge weights are exactly the no-leakage / repair-isometry condition. V11 proves unique allowed successors imply no leakage. V12 adds the finite successor certificate for the modeled candidate basis. |
| [`scalar_clebsch`](src/clifford_3plus2_d5/scalar_clebsch/) | **V3 conditional pass, active** | Scalar quark mass-Clebsch sidecar. V2 separates scalar mass coefficients from CKM current amplitudes: the up-sector vector `(1/4,1/√2,1)` follows from a length-3 nilpotent Taylor response with `x=1/√2`; the down sector records the natural S3/projector baseline `(6,2,4) -> (1,1/√3,√(2/3))` and the data-improved odd-shell candidate `(6,2,5) -> (1,1/√3,√(5/6))`. V3 proves `(6,2,5)` is available in the regular S3 algebra but not forced by S3 alone: rank 2 needs a defect-selected standard copy and rank 5 needs a chosen excluded one-dimensional line. |
| [`radial_response`](src/clifford_3plus2_d5/radial_response/) | **R1–R13 PASS, active** | Radial mass-response sidecar. It reframes quark masses as QCA boundary recirculation residues / pole shifts, proves the Feshbach self-energy form, separates exponential vs geometric up-sector stacking (`1/2` vs `1`), kills literal `exp(xN)` as the family-space Yukawa matrix, derives `x=1/sqrt(2)` only under two-channel no-leakage repair hypotheses, reduces the scalar repair shell to vacuum-framed S3 under named locality premises, inherits the silver transfer root from existing boundary-response ledgers, proves finite S3/silver data do not force poles/residues, and shows the target quark textures are inverse spectral-measure reconstructions rather than forward QCA mass derivations. |
| [`universal_bath`](src/clifford_3plus2_d5/universal_bath/) | **Parked after Session 24** | Universal spectral-bath sidecar. It turns the bath idea into `finite Lanczos head + universal retarded silver tail`, proves the common Jacobi/Schur spine, records the positive/indefinite/CMV reduction taxonomy, inherits the period-one silver terminator from the BB band-edge theorem, freezes supported lepton-side source anchors, certifies the neutrino core only inside the product half-line bath, audits that the raw microscopic BB edge update lacks the `u,b` family-port graph, supplies the selected internal family-port graph whose direct moments decouple `u` and `b`, derives the `P_u+P_b` active plane from selected-port incidence by detracing `e1`, connects it to BB q-mismatch hard-gap and retarded compression inside the single-clock/outgoing model, audits that the deeper boundary-material origin is not derived from bare BB blocks, constructs the minimal charged-lepton two-sided boundary graph with exact Koide equipartition, derives `2/9` as a source occupation moment, proves that charged-lepton equipartition forces exactly two coherent trace paths inside the minimal graph while leaving their BCC/Higgs origin and the occupation-to-angle dynamics open, localizes the conditional up/down quark finite heads, implements the colored active-current quark ansatz where source `b` gives first-passage depths `(2,1,0)`, proves that `b` is the unique selected-`S2` odd boundary-current line, and selects the down rank-five bottom profile inside the non-contact retarded-current model. Session 24 closes it for now: Higgs conjugation gives flag reversal `N.T`, not the Hermitian closure `Delta_N`, so the door-to-readout coupling remains a dynamical premise. |
| [`threeclocks`](src/clifford_3plus2_d5/threeclocks/) | **On hold after Session 02** | Simpler quark-model sidecar based on finite clocks. Session 01 implements exact clock specs, shift/phase matrices, Weyl relations, word composition, inverses, and closure orders. Session 02 attaches a selected `D3 ~= S3` clock to the three-port boundary and proves the exact source identities `(C e1-C^-1 e1)/sqrt(2)=b` and `(2e1-Ce1-C^-1e1)/sqrt(6)=a`, identifying `b` as the oriented tangent current and `a` as the radial second difference. Controls keep the repair flag, down shell, contact veto, closure exponents, and CKM as open gates rather than mass claims. |
| [`cusp`](src/clifford_3plus2_d5/cusp/) | **MicroCUSP A-H pass** | CUSP-PLAN sidecar for Froggatt-Nielsen flavor from a `(2,3)` recirculation cusp. Target A derives weak/BCC multiplicity `2` from the local BCC edge q-clock's two q-preserving same-normal branches, imports the exact BB norm split `I/2 + I/2 = I`, certifies that the same/mixed outgoing block stack is an `8 x 2` isometry with finite unitary completion, derives `p(q)=gq^2` as the unique lowest-degree q-local even positive penalty vanishing on q=0, and keeps color multiplicity `3` from the `SU(3)` center, then replaces abstract fixed-length paths with local center-charge automata: one tick advances center charge, visible readout occurs only at charge `0`, and the weak/color automata force primitive returns `2` and `3`; closed-walk enumeration gives valuation algebra `C[t^2,t^3]`, first valuations `(0,2,3)`, and gap `1`, while `C[t]`, `C[t^2]`, `C[t^3]`, one-step-loop, weak-only, color-only, and wrong-color-length graph controls fail. Target B certifies `lambda_rec=sqrt(3/2)-1` as the unique stable minimum of the one-sided retarded recirculation match `(1+t)sqrt(2)=sqrt(3)`; the ordinary reflection coefficient is shown to solve a different two-sided scattering control and is rejected by the cusp boundary, while count-ratio and inverse-amplitude shears fail the same one-sided equation. Target C verifies the SM hypercharge doors (`H_tilde` for up, `H` for down), derives conductor `c=2` from `S=<2,3>`, selects `D=(1,0,0)` as the conductor-ideal residue and `U=(5,2,0)` from the weak/BCC order-2 oriented lift, then rejects wrong-conductor, trivial-lift, and color-order-lift controls while keeping the nonnegative-charge solver as a consistency check. Target D builds coefficients as finite center-holonomy path sums; up center powers are finite-selected as geodesic distances on the non-cyclic cusp flag and down center powers as the unit bilinear pairing on `F3` center labels, giving a nonzero CP commutator invariant while all-real, one-sector, and separable row/column controls give zero and full field rephasing leaves the invariant unchanged. Its positive amplitudes come from the cusp-module path-count rule `A_ij=max(1,# decompositions of q_i+r_j in <2,3>)`. MicroCUSP Sessions A-H derive q-local stiffness, no-incoming retarded asymptotics, weak `Z2` branch parity, color `Z3` center holonomy, the independent SM center-axis gate, Schur semigroup, `lambda_rec`, Target-C module, and Target-D topology inside the current boundary-register model. |

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
- `scalar_clebsch` -> `boundary_response.quark_boundary_shell` (six-channel quark primitive labels).
- `radial_response` -> `scalar_clebsch` (S3 projector count audit).
- `universal_bath` -> `boundary_response` (lepton source anchors and holonomy),
  `scalar_clebsch` (quark Clebsch/count gates), and `radial_response`
  (up-sector stacking fork).

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
uv run pytest src/clifford_3plus2_d5/qca_smv0/tests/ -q
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
uv run pytest src/clifford_3plus2_d5/scalar_clebsch/tests/ -q
uv run pytest src/clifford_3plus2_d5/radial_response/tests/ -q
uv run pytest src/clifford_3plus2_d5/universal_bath/tests/ -q
uv run pytest src/clifford_3plus2_d5/threeclocks/tests/ -q
uv run pytest src/clifford_3plus2_d5/cusp/tests/ -q

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
- `qca_smv0` is a focused simulator sidecar for the next SM-QCA prototype; it currently implements the BCC/SM/Higgs/FN/production stack through an explicit inverse helper, multi-step reversibility audit, Loschmidt echo diagnostic, and finite tangent-response echo audit for the current physical-right production tick, but should not be read as a boundary, quantized-register, microscopic bridge, Gauss-projection, exact full-energy conservation, timestep-refined energy convergence, energy-convergent reversible-integrator, or flavor/Higgs-parameter derivation claim.
- `boundary_response` derives the framed neutrino core and closes the vacuum-selector sector through V43 modulo the positive-quartic backreaction axiom; PMNS/CKM remain gated by explicit boundary-shell assumptions.
- `depth_scar` upgrades the `{0,2,6}` quark depth spectrum to a graph-Laplacian transfer operator, derives the path scar from an effective edge-weight potential, a locally normalized nilpotent repair flag, a minimal support classification, a finite shortest-repair variational principle, a conditional one-tick locality theorem, an active repair isometry/no-leakage equivalence, a unique-successor no-leakage bridge, and a finite successor certificate, and identifies loop healing as the minimal graph-native CP location, while keeping the microscopic origin of the height filtration, basis-completeness theorem, and loop parameters open.
- `scalar_clebsch` separates scalar mass Clebsches from coherent CKM current Clebsches, deriving the corrected up vector from a nilpotent Taylor response with `x=1/sqrt(2)` and splitting the down sector into an S3/projector baseline plus a data-improved odd-shell candidate; its S3 audit shows the candidate counts are available but not forced until the defect-selection rule is derived.
- `radial_response` reframes the mass sector as boundary Green-function recirculation, preserving the up factorial relation, deriving `x=1/sqrt(2)` only under a two-channel no-leakage repair condition, proving that pair is complete inside the S3 scalar shell, reducing that shell to vacuum-framed BCC tetrahedral automorphisms under named premises, killing literal nilpotent Yukawa matrices, constructing the minimal exact unitary S3 defect form, inheriting the silver transfer root from existing ledgers, proving that finite S3/silver data do not force poles/residues, and showing that the target quark textures are positive inverse spectral-measure reconstructions until a forward bath-selection principle is derived.
- `universal_bath` upgrades the forward bath-selection program into a
  session-gated architecture: lepton sources are frozen, the neutrino product
  bath is proved inside its ansatz, the selected internal neutrino family-port
  graph now computes zero `u/b` cross moments directly, selected-port incidence
  derives the active plane `P_u+P_b` by detracing `e1`, BB q-mismatch and
  retarded clock-error leads derive the selected graph inside the
  single-clock/outgoing-boundary model, the deeper boundary-material origin
  is audited as not derivable from bare BB blocks,
  charged-lepton `2/9` is an exact source occupation moment rather than a phase
  derivation, the minimal charged-lepton two-sided graph realizes the residue
  and Koide equipartition once the angle is supplied, quark finite heads are
  implemented conditionally, the selected-`S2` odd-current theorem selects the
  `b` source line, the down identity-return veto selects the rank-five bottom
  profile inside the retarded-current model, the Higgs-door audit shows
  conjugation gives `N.T` rather than `Delta_N`, and the
  height-door/color-lift/current-parity audits reduce the unresolved up/down
  source-vector problem to named microscopic selection rules.
- `threeclocks` is on hold as a simpler quark-model branch with exact finite-clock
  infrastructure and the first D3 source identity: `b` is the selected clock's
  oriented tangent current while `a` is its radial second difference.  It keeps
  the representation-basis repair flag, active down shell, contact veto,
  closure exponents, and CKM frames behind explicit finite gates.
- `cusp` implements the first CUSP-PLAN target certificates: the `(2,3)`
  valuation spine is generated by a finite BB/material source and dilation
  audit plus local weak/color center-charge automata; the `q^2` locking and
  outgoing closure are finite consequences of explicit local material axioms,
  and the Cabibbo shear is the one-sided cusp uniformizer rather than the
  ordinary two-sided reflection coefficient, while
  the right-handed charge completion is promoted to a finite conductor-module /
  weak-double-cover audit and the center-holonomy CP mechanism now uses a
  cusp-module amplitude measure plus finite center-topology selection;
  MicroCUSP Sessions A-H recover the microscopic q-stiffness, retarded
  boundary, weak/color recirculation, SM quotient, Schur semigroup,
  `lambda_rec`, Target-C module, and Target-D topology gates inside the current
  boundary-register model.
- The closed sidecars (`triality`, `broken_triality`, `exceptional`, `topology`) cumulatively rule out the algebraic routes to three generations.
- The closed-PASS sidecars (`sme`, `strongcp`, `koide`) check consistency with current experimental bounds and identify the residual physical inputs.

The workspace is the honest superposition of these answers. The kill-disciplined
sidecar pattern — single audit module per gate, dataclass payload, controls +
verdict + interpretation — is the project's working method.
