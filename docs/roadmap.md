# Roadmap And Working Index

This is the main working index for the `clifford-3plus2-d5` side project.
The governing attack is now the enhanced J-first route:

```text
J first,
then forced 3+2,
then no-locking commutant,
then finite-depth QCA,
then Spin(10) spinor.
```

The bridge is not accepted by finding five complex anticommuting matrices.
It is accepted only if microscopic real QCA data force a local complex
structure `J`, a structural `3+2` split, and a geometric gate algebra that
cannot resolve individual color or weak axes.

## Source Documents

- [README](../README.md): project boundary, commands, and current status.
- [Enhanced J-First Attack Plan](qca_3plus2_j_first_enhanced_attack_plan.md):
  current main theorem and implementation plan.
- [Original J-First Attack Plan](qca_3plus2_j_first_attack_plan.md): earlier
  route ranking and motivation.
- [Theory Summary](theory.md): current mathematical baseline and audit
  boundary.
- [Falsifiers](falsifiers.md): conditions that force `notation_only` or
  `falsified`.
- [Results Index](results/index.md): reproducible command outputs.
- [Projector Lattice Report](literature/projector_lattice_report.md):
  Phase 3 projector and addressability checks.
- [Gate Classification Report](literature/gate_classification_report.md):
  Phase 4 SM commutant oracle.
- [QCA Update Certificate](literature/qca_update_certificate.md):
  Phase 5 finite-depth update candidate.
- [Spinor 16 Report](literature/spinor16_report.md):
  Phase 6 guarded spinor reconstruction.
- [Normalizer Report](literature/normalizer_report.md):
  Phase 7 normalizer and addressability algebra oracle.
- [Forcedness Certificate](literature/forcedness_certificate.md):
  Phase 7 conservative forcedness verdict.
- [Handover Compliance](handover_compliance.md): Phase 0 closeout checklist.
- [Side Project Plan](clifford_3plus2_to_d5_side_project_plan.md): original
  geometry-to-D5 project plan.
- [Archived Agent Handover](archive/clifford_3plus2_empty_repo_agent_handover.md):
  completed bootstrap and first implementation contract.

## Current Verdict

```text
phase_0_audit_contract: complete
phase_1_real_carrier_check_passed: true
forced_j_check_passed: true
forced_j_verdict: candidate_only
structural_split_check_passed: true
structural_split_verdict: candidate_only
gate_classification_check_passed: true
qca_geometric_gate_algebra_safe: false
qca_update_check_passed: true
finite_depth_qca_verdict: candidate_only
spinor16_check_passed: true
spinor16_verdict: candidate_only
normalizer_check_passed: true
forcedness_verdict: candidate_only
branching_check_passed: true
qca_split_audit_verdict: notation_only
load_bearing_qca_bridge: false
```

Meaning:

- The textbook Spin(10) branching arithmetic passes.
- The QCA bridge has not passed.
- No `data/qca_data.json` is present.
- The project must not claim a geometry-to-D5 theorem yet.

## Enhanced Theorem Target

The target theorem is carrier-first.

Input that must be derived before invoking Spin(10):

```text
K_x ~= R^10
J in SO(K_x), J^2 = -I
K_x = K_3 ⊕ K_2
dim_R K_3 = 6
dim_R K_2 = 4
J K_3 = K_3
J K_2 = K_2
A_geom subset Comm(SU(3) x SU(2) x U(1))
```

Then:

```text
W := (K_x, J) ~= C^5 = C^3 ⊕ C^2
S^+ := Lambda^even(W)
Y = -1/3 N_3 + 1/2 N_2
```

The bridge survives only if `J`, `P_3/P_2`, and the safe gate algebra are
produced by QCA rule data, not selected because they reproduce `Spin(10)`.

## Central Falsifier

The QCA may distinguish the whole `3` block from the whole `2` block. It may
not distinguish individual axes inside either block.

Allowed structural projectors:

```text
P_3
P_2
```

Forbidden addressability:

```text
|color 1><color 1|
|color 2><color 2|
|color 3><color 3|
|weak 1><weak 1|
|weak 2><weak 2|
Hom(C^3, C^2)
Hom(C^2, C^3)
```

If the microscopic update contains direction-conditioned terms such as

```text
S_1 ⊗ |1><1| + S_2 ⊗ |2><2| + S_3 ⊗ |3><3|
```

then it breaks the would-be `SU(3)` unless all `S_i` are identical.

## Phase 0: Audit Infrastructure And Contract

Status: complete.

Purpose: provide a reproducible exact-arithmetic audit contract that blocks
false bridge claims.

Completed:

- `uv` package scaffold.
- Exact rational matrix parsing.
- Default-failing QCA split audit.
- One-particle gate algebra safety checks.
- `data/qca_data.schema.json`.
- Invalid fixtures for hand-chosen `J`, unknown `J`, off-block gates, color
  projectors, and malformed data.
- Machine-readable JSON output from `qca_split_audit.py`.
- [Handover Compliance](handover_compliance.md) closeout checklist.
- Reproducible scripts:
  - [`scripts/branching_check.py`](../scripts/branching_check.py)
  - [`scripts/qca_split_audit.py`](../scripts/qca_split_audit.py)

Acceptance:

```text
The audit interface is documented, tested, and impossible to pass by
omission, floats, unknown origins, or hand-chosen J.
```

## Phase 1: Algebra Kernel And Real Carrier

Status: complete.

Purpose: implement the exact real carrier model used by the enhanced attack.
Do not introduce complex notation before `J` is constructed as a real matrix.

Core ansatz:

```text
K_x = R^2_clock ⊗ (R^3 ⊕ R^2)
basis = x_1,...,x_5,y_1,...,y_5
J = epsilon ⊗ I_5
P_3 = I_2 ⊗ diag(1,1,1,0,0)
P_2 = I_2 ⊗ diag(0,0,0,1,1)
```

Implementation deliverables:

```text
src/clifford_3plus2_d5/algebra/real_carrier.py
src/clifford_3plus2_d5/algebra/matrices.py
scripts/real_carrier_check.py
tests/test_real_carrier.py
docs/literature/real_carrier_report.md
```

Required exact checks:

```text
dim_R K_x = 10
J^2 = -I_10
J^T J = I_10
P_3 + P_2 = I_10
P_i^2 = P_i
P_3 P_2 = 0
rank_R(P_3) = 6
rank_R(P_2) = 4
[J,P_3] = [J,P_2] = 0
```

Acceptance:

```text
The real carrier and algebraic ansatz are exact, documented, and tested.
No claim is made yet that QCA dynamics force J or P_3/P_2.
```

## Phase 2: Forced `J` Search

Status: complete as candidate-only checker; not yet QCA-forced.

Purpose: prove or falsify that microscopic QCA data generate `J`.

Candidate routes:

- Period-four micromotion:
  ```text
  J = U(T/4)
  J^2 = U(T/2) = -I on K_x
  J^4 = U(T) = I
  ```
- Gate-word generation:
  ```text
  J = G_q ... G_2 G_1
  ```

Implementation deliverables:

```text
src/clifford_3plus2_d5/search/gate_words.py
src/clifford_3plus2_d5/search/forced_j.py
scripts/forced_j_check.py
tests/test_j_structure.py
tests/test_gate_word_search.py
docs/literature/forced_j_report.md
```

Acceptance:

```text
J is produced by microscopic QCA data,
J^2 = -I exactly,
J is local/finite-depth,
J is unique up to harmless equivalence,
rank-one pair rotations J_a are not independently addressable.
```

Current outcome:

```text
generated_by_gate_word = true
qca_forces_j = false
forced_j_verdict = candidate_only
```

Interpretation:

```text
The checker can certify a declared gate word. It does not prove microscopic
QCA rule data force that word.
```

Failure:

```text
J is chosen after the fact,
J is ambient scalar i,
many inequivalent J choices are equally valid,
or the rule set allows independently addressable J_a.
```

## Phase 3: Structural `3+2` Split

Status: complete as candidate-only checker; not yet QCA-forced.

Purpose: prove or falsify that the QCA rule data force only the block
projectors `P_3` and `P_2`, not rank-one projectors inside them.

Implementation deliverables:

```text
src/clifford_3plus2_d5/algebra/projectors.py
src/clifford_3plus2_d5/search/addressability.py
tests/test_projectors.py
tests/test_no_rank_one_addressability.py
docs/literature/projector_lattice_report.md
```

Current outcome:

```text
projector_identities_passed = true
projectors_commute_with_j = true
addressability_algebra_safe = true
qca_supplies_structural_3plus2_split = false
structural_split_verdict = candidate_only
```

Falsifiers implemented:

```text
rank-one color projectors -> falsified
rank-one weak projectors -> falsified
off-block mixers -> falsified
non-scalar within-block controls -> falsified
```

Acceptance:

```text
P_3 and P_2 are QCA-structural,
P_3 and P_2 commute with J,
the split is unique up to block rotations,
there are no rank-one projectors inside either block.
```

Failure:

```text
the 3+2 split is arbitrary,
the QCA can resolve individual color axes,
the QCA can resolve individual weak axes,
or the split is introduced only after SM labels appear.
```

## Phase 4: SM Commutant And Gate Classifier

Status: complete as exact classifier oracle; not yet a safe QCA gate set.

Purpose: build the permanent exact oracle for geometric gate safety.

Expected theorem:

```text
Comm(SU(3) x SU(2) x U(1) on C^3 ⊕ C^2) = C P_3 ⊕ C P_2
```

Gate classes:

```text
safe_sm_commutant
block_mixing_fail
color_breaking_fail
weak_breaking_fail
antilinear_fail
unknown_fail
```

Implementation deliverables:

```text
src/clifford_3plus2_d5/algebra/commutants.py
src/clifford_3plus2_d5/sm/embedding.py
src/clifford_3plus2_d5/sm/classification.py
tests/test_commutant.py
tests/test_gate_classification.py
docs/literature/gate_classification_report.md
```

Current outcome:

```text
gate_classification_check_passed = true
commutant_basis_matches_expected = true
safe_algebra_closure_passed = true
qca_geometric_gate_algebra_safe = false
load_bearing_qca_bridge = false
```

Canonical unsafe witnesses:

```text
block_mixer -> block_mixing_fail
rank_one_color_projector -> color_breaking_fail
rank_one_weak_projector -> weak_breaking_fail
real_conjugation -> antilinear_fail
```

Acceptance:

```text
every geometric QCA gate generator lies in C P_3 ⊕ C P_2,
all block mixers are rejected,
all rank-one color projectors are rejected,
all rank-one weak projectors are rejected,
closure of the safe algebra is proven.
```

## Phase 5: Finite-Depth Microscopic QCA Candidate

Status: complete as finite-depth candidate certificate; not yet QCA-forced.

Purpose: construct and certify an actual local update.

Candidate period-four structure:

```text
U(T/4) = J
U(T/2) = -I on K_x
U(T) = I on K_x
```

Layer certificate fields:

```text
finite_depth
layer_count
max_locality_radius
all_layers_orthogonal
quarter_period_operator
half_period_operator
J_certificate
gate_classification_summary
unsafe_gate_witnesses
```

Implementation deliverables:

```text
src/clifford_3plus2_d5/qca/gates.py
src/clifford_3plus2_d5/qca/layers.py
src/clifford_3plus2_d5/qca/locality.py
src/clifford_3plus2_d5/qca/certificates.py
tests/test_qca_layers.py
tests/test_qca_update_certificate.py
tests/test_no_locking.py
docs/literature/qca_update_certificate.md
```

Current outcome:

```text
finite_depth = true
period_four_check_passed = true
quarter_period_is_j = true
half_period_is_minus_identity = true
full_period_is_identity = true
all_internal_actions_safe = true
qca_rule_forces_update = false
finite_depth_qca_verdict = candidate_only
load_bearing_qca_bridge = false
```

Falsifiers implemented:

```text
rank-one color spacetime shift -> falsified
rank-one weak spacetime shift -> falsified
block-mixing spacetime shift -> falsified
effective-Hamiltonian-only update -> falsified
```

Acceptance:

```text
the update is finite-depth,
each layer is local,
each layer is exactly real orthogonal/unitary,
J appears as actual micromotion,
spacetime coupling does not use within-block projectors,
all internal geometric actions pass the SM commutant classifier.
```

## Phase 6: Spinor Reconstruction

Status: complete as guarded candidate-only reconstruction.

Purpose: only after `J` and the structural split pass, construct
`Lambda^even(C^5)` from the derived real carrier.

Implementation:

- [`src/clifford_3plus2_d5/exterior.py`](../src/clifford_3plus2_d5/exterior.py)
- [`src/clifford_3plus2_d5/branching.py`](../src/clifford_3plus2_d5/branching.py)
- [`src/clifford_3plus2_d5/sm/spinor16.py`](../src/clifford_3plus2_d5/sm/spinor16.py)
- [`src/clifford_3plus2_d5/sm/hypercharge.py`](../src/clifford_3plus2_d5/sm/hypercharge.py)
- [`tests/test_branching.py`](../tests/test_branching.py)
- [`tests/test_spinor16.py`](../tests/test_spinor16.py)
- [`tests/test_hypercharge.py`](../tests/test_hypercharge.py)
- [`docs/literature/spinor16_report.md`](literature/spinor16_report.md)

Current outcome:

```text
spinor16_dimension = 16
degree_dimensions = {0: 1, 2: 10, 4: 5}
hypercharge_check_passed = true
branching_table_check_passed = true
uses_existing_j_and_split = true
introduces_new_complex_structure = false
introduces_new_3plus2_split = false
spinor16_verdict = candidate_only
load_bearing_qca_bridge = false
```

Acceptance:

```text
the even Fock basis has dimension 16,
the hypercharge table is exact,
the branching uses the previously derived J and P_3/P_2,
no new choice of C^5 appears in this phase.
```

## Phase 7: Forcedness And Normalizer Analysis

Status: complete as candidate-only forcedness oracle.

Purpose: prove the construction is not one convenient coordinate choice among
many. The current implementation does not prove that yet; it provides the
normalizer and falsifier machinery needed to test source-backed data.

Tasks:

- Compute or approximate `Aut(Data)`, the real orthogonal transformations
  preserving microscopic rules.
- Study admissible `J'` with `(J')^2=-I`.
- Study admissible rank-three `J`-invariant projectors.
- Compute the addressability algebra generated by switchable controls.

Implementation deliverables:

```text
src/clifford_3plus2_d5/search/normalizer.py
scripts/normalizer_check.py
tests/test_normalizer.py
docs/literature/normalizer_report.md
docs/literature/forcedness_certificate.md
```

Current outcome:

```text
centralizer_dimension = 52
orthogonal_normalizer_dimension = 21
addressability_algebra_dimension = 2
candidate_j_valid = true
candidate_split_valid = true
candidate_j_preserved_by_normalizer = false
normalizer_preserves_declared_split = true
continuous_j_alternatives_not_excluded = true
rank_three_projector_family_not_excluded = true
addressability_algebra_safe = true
j_unique_or_forced = false
split_unique_or_forced = false
forcedness_verdict = candidate_only
load_bearing_qca_bridge = false
```

Falsifiers implemented:

```text
rank-one color projectors -> falsified
rank-one weak projectors -> falsified
off-block controls -> falsified
full-U(5)-like mode-resolving controls -> falsified
```

Acceptance:

```text
J is unique or canonically forced,
P_3/P_2 are unique structural central projectors,
the microscopic controls do not generate smaller projectors,
the normalizer does not erase the 3+2 split.
```

Current acceptance status:

```text
not accepted as a QCA bridge proof.
The normalizer preserves the declared split, but J uniqueness and projector
forcedness are not derived from microscopic rule data.
```

## Phase 8: Route Branches If Minimal Ansatz Fails

Status: fallback only.

Branch order:

1. Stronger real-QCA first route.
2. Floquet-Kahler micromotion.
3. Defect/monodromy complex structure.
4. Five-axis parent QCA.
5. `D5` / `SU(5)` root-lattice QCA.
6. Passive `Spin(10)` fiber fallback.

Fallback interpretation:

```text
If no route derives J, QCA geometry may still supply topology and locality,
but Spin(10) remains an independent internal fiber.
```

## Phase 9: Family Number Status

Status: out of scope for the bridge.

Purpose: prevent hidden insertion of three generations.

Rejected as derivations:

```text
choose winding n = 3
stack three copies
choose three defects
repeat a unit pump three times
choose a degree-three map
```

Acceptance for any future family claim:

```text
the microscopic rules forbid n = 1 and force n = 3,
or a stable index canonically fixed by the geometry equals 3.
```

Until then:

```text
one-generation internal carrier only.
```

## Immediate Work Packages

1. **Minimal Real Period-Four Ansatz**
   - Implement exact `J`, `P_3`, `P_2`.
   - Verify all carrier/projector identities.
   - Document that this is algebraic only until QCA rules force it.

2. **Structural Split And Addressability**
   - Keep `P_3/P_2` as exact candidate projectors.
   - Reject rank-one color and weak addressability.
   - Require real QCA data before marking the split structural.

3. **Commutant Classifier**
   - Maintain exact SM commutant basis check.
   - Classify safe block scalars and unsafe rank-one/block-mixing/antilinear gates.
   - Require an actual QCA gate set before marking geometric gates safe.

4. **Finite-Depth Candidate**
   - Maintain exact period-four update certificate.
   - Reject spacetime shifts with rank-one or block-mixing internal projectors.
   - Require real QCA rule data before marking the update forced.

5. **Spinor Reconstruction**
   - Maintain guarded `Lambda^even(C^5)` reconstruction from prior candidates.
   - Verify exact hypercharge and sector table.
   - Require earlier QCA-forced data before treating the spinor as load-bearing.

## Non-Goals

Do not add claims about:

- three families
- mirror gap
- Golterman-Shamir inertness
- continuum gauge dynamics
- Higgs/Yukawa realism
- phenomenology
- a completed QCA construction

Do not add `data/qca_data.json` unless it contains real candidate QCA data
with exact rational matrices and explicit gate restrictions.

## Working Rule

Derive `J`; derive only `P_3` and `P_2`; never derive or allow
`|1><1|`, `|2><2|`, `|3><3|`, `|4><4|`, or `|5><5|`.
