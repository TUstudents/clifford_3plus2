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
- [Real-QCA Branch Report](literature/real_qca_branch_report.md):
  Phase 8A stronger real-QCA-first branch checker.
- [Rule-Space Exploration Report](literature/rule_space_exploration_report.md):
  E1 bounded microscopic rule-space search.
- [Unseeded Projector Discovery Report](literature/unseeded_projector_discovery_report.md):
  E2 bounded search for rule-generated central `6+4` projectors without
  seeding `P_3/P_2`.
- [Rule-To-Verdict Report](literature/rule_to_verdict_report.md):
  collapsed finite-depth rule checker for the actual bridge criterion.
- [Floquet Alpha Report](literature/floquet_alpha_report.md):
  first physical primitive family replacing abstract depth escalation.
- [Floquet Alpha J Obstruction](literature/floquet_alpha_j_obstruction.md):
  theorem-standard decision between spectral and strict forced `J`.
- [Spatial 1D Sidecar Report](literature/spatial_1d_report.md):
  Route-2 winding prototype for coupling alpha/eta orientation signs.
- [Defect Beta Report](literature/defect_beta_report.md):
  second physical primitive family using computed wall-cycle monodromy.
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
real_qca_branch_check_passed: true
real_qca_branch_verdict: candidate_only
e1_exploration_check_passed: true
e1_forced_candidate_hits: 0
e1_surviving_candidates: 0
e2_projector_discovery_check_passed: true
e2_unseeded_projector_pairs_found: 0
e2_unsafe_rank_one_projectors: 3
rule_to_verdict_interface: implemented
rule_to_verdict_bridge_candidate: false
floquet_alpha_candidate_count: 10
floquet_alpha_rank_6_4_pair_candidates: 10
floquet_alpha_plus_polarization_j_candidates: 10
floquet_alpha_plus_scaled_polarization_certified_candidates: 10
floquet_alpha_plus_compatible_centralizer_dimension: 26
floquet_alpha_plus_compatible_j_moduli_dimension: 9
floquet_alpha_plus_local_compatible_operator_dimension: 4
floquet_alpha_plus_local_compatible_j_moduli_dimension: 0
floquet_alpha_plus_local_compatible_complex_structure_count: 4
floquet_alpha_plus_strict_bridge_candidates: 0
floquet_alpha_second_layer_commuting_candidates: 10
floquet_alpha_second_layer_center_dimension: 10
floquet_alpha_second_layer_compatible_centralizer_dimension: 10
floquet_alpha_second_layer_lower_rank_projector_ranks: [2, 2, 2]
floquet_alpha_second_layer_no_locking_guardrail_passed_candidates: 0
floquet_alpha_second_layer_strict_bridge_candidates: 0
floquet_alpha_noncommuting_candidate_count: 1
floquet_alpha_noncommuting_block_preserving_candidates: 1
floquet_alpha_noncommuting_coarse_center_preserved_candidates: 1
floquet_alpha_noncommuting_compatible_centralizer_dimension: 6
floquet_alpha_noncommuting_compatible_j_moduli_dimension: 0
floquet_alpha_noncommuting_compatible_j_count: 4
floquet_alpha_noncommuting_compatible_j_in_generated_algebra_count: 0
floquet_alpha_noncommuting_compatible_j_in_rule_local_center_count: 0
floquet_alpha_noncommuting_spectral_polarization_j_matched_count: 0
floquet_alpha_noncommuting_completion_generated_algebra_dimension: 26
floquet_alpha_noncommuting_completion_center_dimension: 4
floquet_alpha_noncommuting_completion_lower_rank_central_idempotents: 0
floquet_alpha_noncommuting_completion_local_compatible_j_count: 4
floquet_alpha_noncommuting_completion_strict_unique_j_found: false
floquet_alpha_noncommuting_completion_pass_bridge: false
spatial_1d_alpha_candidate_count: 1
spatial_1d_alpha_unitary_candidates: 1
spatial_1d_alpha_coarse_6_4_band_candidates: 1
spatial_1d_alpha_orientation_choices_before_transport: 4
spatial_1d_alpha_orientation_choices_after_transport: 2
spatial_1d_alpha_sign_coupled_to_global_pm: true
spatial_1d_alpha_route_label: spatial_signs_coupled_to_global_pm
spatial_1d_alpha_local_hopping_term_count: 2
spatial_1d_alpha_local_hopping_shifts: [3, 4]
spatial_1d_alpha_local_hopping_mode_windings: [4, 4, 4, 3, 3]
spatial_1d_alpha_local_hopping_reconstructs_transfer: true
spatial_1d_alpha_local_hopping_route_label: spatial_local_hopping_signs_coupled
spatial_1d_alpha_local_qca_term_count: 2
spatial_1d_alpha_local_qca_shifts: [3, 4]
spatial_1d_alpha_local_qca_laurent_orthogonal: true
spatial_1d_alpha_local_qca_central_idempotent_ranks: [0, 4, 6, 10]
spatial_1d_alpha_local_qca_lower_rank_central_idempotents: 0
spatial_1d_alpha_local_qca_route_label: spatial_local_qca_signs_coupled_not_load_bearing
spatial_1d_unseeded_candidate_count: 4
spatial_1d_unseeded_guardrail_rejections: 1
spatial_1d_unseeded_coarse_6_4_center_candidates: 0
spatial_1d_unseeded_sign_coupled_candidates: 0
spatial_1d_unseeded_strict_bridge_candidates: 0
spatial_1d_unseeded_route_label: unseeded_spatial_no_bridge_candidates
spatial_1d_alpha_strict_bridge_candidates: 0
floquet_alpha_noncommuting_forced_j_candidates: 0
floquet_alpha_noncommuting_strict_bridge_candidates: 0
floquet_alpha_bridge_candidates: 0
defect_beta_monodromy_candidates: 10
defect_beta_scaled_monodromy_certified_candidates: 10
defect_beta_monodromy_equals_matching_floquet_alpha: true
defect_beta_transition_functions_commute: false
defect_beta_rule_generated_algebra_dimension: 8
defect_beta_rule_center_dimension: 2
defect_beta_compatible_centralizer_dimension: 13
defect_beta_compatible_j_moduli_dimension: null
defect_beta_local_compatible_operator_dimension: 2
defect_beta_local_compatible_j_moduli_dimension: null
defect_beta_local_compatible_complex_structure_count: 0
defect_beta_strict_bridge_candidates: 0
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

Status: Branch A implemented as candidate-only checker; remaining branches are
fallback only.

Branch order:

1. Stronger real-QCA first route. Implemented as Phase 8A.
2. Floquet-Kahler micromotion.
3. Defect/monodromy complex structure.
4. Five-axis parent QCA.
5. `D5` / `SU(5)` root-lattice QCA.
6. Passive `Spin(10)` fiber fallback.

Phase 8A implementation:

```text
src/clifford_3plus2_d5/search/real_qca_branch.py
scripts/real_qca_branch_check.py
tests/test_real_qca_branch.py
docs/literature/real_qca_branch_report.md
```

Current outcome:

```text
candidate_word_found = true
candidate_word = global_clock_tick
finite_depth = true
translation_invariant = true
generates_j = true
generates_split = true
j_forced_by_rule_space = false
split_forced_by_rule_space = false
normalizer_verdict = candidate_only
real_qca_branch_verdict = candidate_only
load_bearing_qca_bridge = false
```

Falsifiers implemented:

```text
rank-one color projectors -> falsified
rank-one weak projectors -> falsified
rank-one pair rotations -> falsified
off-block controls -> falsified
```

Fallback interpretation:

```text
If no route derives J, QCA geometry may still supply topology and locality,
but Spin(10) remains an independent internal fiber.
```

## E1: Microscopic Rule-Space Exploration Sprint 1

Status: complete; no forced survivors.

Purpose: start real bounded exploration instead of adding another certificate
wrapper.

Implementation:

```text
src/clifford_3plus2_d5/explore/rule_space.py
src/clifford_3plus2_d5/explore/primitives.py
src/clifford_3plus2_d5/explore/projector_discovery.py
src/clifford_3plus2_d5/explore/search_runner.py
scripts/explore_rule_space.py
tests/test_rule_space_exploration.py
docs/literature/rule_space_exploration_report.md
data/exploration/e1_summary.json
data/exploration/e1_survivors.jsonl
data/exploration/e1_rejections.jsonl
```

Current outcome:

```text
primitive_sets_scanned = 10
words_scanned = 170
j_hits = 73
period_four_hits = 73
split_candidates = 170
addressability_safe_hits = 158
normalizer_candidate_hits = 158
forced_candidate_hits = 0
surviving_candidates = 0
load_bearing_qca_bridge = false
```

Top rejection reasons:

```text
normalizer_too_large = 158
no_j = 97
no_period_four = 97
rank_one_addressability = 38
unsafe_addressability = 12
normalizer_falsified = 12
off_block_addressability = 4
```

Acceptance:

```text
The sprint scanned more than one primitive set and more than 100 words,
detected sanity J/period-four candidates,
rejected forbidden rule families,
wrote reproducible artifacts,
and found zero forced bridge candidates.
```

Negative proposition:

```text
Within the E1/E2 coarse primitive class, increasing search depth alone is not
expected to produce the missing object. Fully coarse generators preserve too
much symmetry; generators that distinguish 3 from 2 have already supplied the
coarse split; rank-one distinguishers are no-locking failures.
```

## E2: Unseeded Projector Discovery Sprint

Status: complete; no unseeded complementary `6+4` projector pair.

Purpose: remove seeded `P_3/P_2` controls from the default projector search and
ask whether small exact rule-generated algebras derive a central complementary
rank-`6+4` pair.

Implementation:

```text
src/clifford_3plus2_d5/explore/unseeded_projectors.py
scripts/discover_projectors.py
tests/test_unseeded_projector_discovery.py
docs/literature/unseeded_projector_discovery_report.md
data/exploration/e2_summary.json
data/exploration/e2_projector_candidates.jsonl
data/exploration/e2_rejections.jsonl
```

Current unseeded outcome:

```text
primitive_sets_scanned = 6
algebra_elements_considered = 1924
candidate_projectors = 3
rank_2_projectors = 3
rank_6_projectors = 0
rank_4_projectors = 0
complementary_pairs = 0
unsafe_rank_one_projectors = 3
unseeded_projector_pairs_found = 0
discovery_verdict = not_found
load_bearing_qca_bridge = false
```

Comparison checks:

```text
sanity_seeded_standard_projectors -> projector_pair_found
block_reflection_candidate -> projector_pair_found
rank_one_color_control_falsifier -> discovery_check_passed = false
```

Acceptance:

```text
The default search does not seed the answer,
the checker still finds the pair when it is genuinely inserted,
unsafe rank-one routes are reported,
the deterministic artifacts are committed,
and no load-bearing bridge claim is made.
```

Next pressure:

```text
Stop depth-only escalation in the E1/E2 primitive class.
Move to primitive families with physical 3-versus-2 asymmetry, starting with
Floquet polarization and defect monodromy.
```

## Unified Rule-To-Verdict Interface

Status: implemented; no bridge candidate.

Purpose: collapse the actual physics criterion into one exact function from a
finite-depth layer list to a verdict.

Implementation:

```text
src/clifford_3plus2_d5/qca/rule_verdict.py
scripts/rule_to_verdict.py
tests/test_rule_to_verdict.py
docs/literature/rule_to_verdict_report.md
```

Inputs:

```text
U_1, ..., U_q
real on-site 10x10 layer matrices
1D locality metadata
```

Outputs:

```text
exact Floquet spectrum
generated real algebra dimension
center basis and central idempotent ranks
rank-(6,4) complementary central pairs
lower-rank central idempotent witnesses
rule-generated J candidates
compatible J solve status
pass_rule_to_bridge
```

Current controls:

```text
minimal_period_four -> falsified_no_rank_6_4_center
clock_block_reflection -> candidate_only_j_not_forced
clock_rank_one_color_reflection -> falsified_rank_one_center
```

This is now the preferred first-pass checker for Floquet-polarization and
defect-monodromy candidates.

## Floquet-Alpha Physical Primitive Family

Status: implemented; coarse center found and polarization `J` produced, but
strict compatible-commutant uniqueness still fails.

Purpose: replace depth escalation over abstract primitive classes with one
physical resonance mechanism.

Implementation:

```text
src/clifford_3plus2_d5/qca/floquet_alpha.py
scripts/floquet_alpha_search.py
scripts/floquet_alpha_plus_search.py
tests/test_floquet_alpha.py
docs/literature/floquet_alpha_report.md
```

Family:

```text
one mandatory quantized on-site Floquet layer
three mode pairs with phase 2 pi / 3
two mode pairs with phase pi / 2
ten resonance patterns enumerated
```

Current outcome:

```text
candidate_count = 10
rank_6_4_pair_candidates = 10
rank_one_falsified_candidates = 0
bridge_candidates = 0
verdict_counts = {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge = false
```

Alpha-plus outcome:

```text
polarization_j_candidates = 10
scaled_polarization_certified_candidates = 10
generated_j_moduli_dimension = 0
compatible_centralizer_dimension = 26
compatible_j_moduli_dimension = 9
locality_radius_bound = 0
local_compatible_operator_dimension = 4
local_compatible_j_moduli_dimension = 0
local_compatible_complex_structure_count = 4
strict_compatible_j_forced_candidates = 0
strict_bridge_candidates = 0
verdict_counts = {'polarization_j_produced_not_strictly_unique': 10}
load_bearing_qca_bridge = false
```

Interpretation:

```text
Floquet-alpha produces the desired coarse [0,4,6,10] central idempotent
lattice without rank-one centers. Alpha-plus now certifies the scaled exact
operator K_alpha = (2U+I)P_alpha with K_alpha^2 = -3P_alpha over QQ(zeta_12);
the normalized J_alpha is derived afterward over QQ(sqrt(3)). It still does
not pass the strict rule-to-verdict bridge criterion because the
compatible centralizer is M_3(C) plus M_2(C), real dimension 26, with a
9-dimensional compatible-J family. The rule-generated local-center search
already finds a 4-dimensional operator space with four discrete J choices, so
strict forcedness fails without claiming to exhaust every on-site local
operator in M_10(R).
```

Alpha cycle/swap second-layer outcome:

```text
commuting_second_layer_candidates = 10
order_certified_candidates = 10
compatible_centralizer_collapsed_candidates = 10
generated_algebra_dimension = 10
center_dimension = 10
compatible_centralizer_dimension = 10
explicit_lower_rank_projector_ranks = [2, 2, 2]
no_locking_guardrail_passed_candidates = 0
strict_bridge_candidates = 0
load_bearing_qca_bridge = false
```

Interpretation:

```text
The literal commuting cycle/swap lock is now a checked negative. It does
reduce the compatible centralizer, but because it commutes with U its
within-block spectral projectors are rule-generated central idempotents. The
explicit rank-2 witnesses fail the no-locking guardrail. This upgrades the
next theorem target: a successful second layer cannot be a commuting
semisimple lock whose spectral projectors are available inside the generated
rule algebra. The proposition is stated in [Theory](theory.md).
```

Noncommuting signed-twist route:

```text
src/clifford_3plus2_d5/qca/floquet_alpha_noncommuting.py
scripts/floquet_alpha_noncommuting_search.py
scripts/floquet_alpha_noncommuting_j_gap.py
scripts/floquet_alpha_noncommuting_completion.py
tests/test_floquet_alpha.py
```

Representative outcome:

```text
candidate_count = 1
noncommuting_candidates = 1
block_preserving_candidates = 1
coarse_center_preserved_candidates = 1
compatible_j_zero_dimensional_candidates = 1
forced_j_candidates = 0
strict_bridge_candidates = 0
best_compatible_centralizer_dimension = 6
route_label = coarse_center_preserved_compatible_j_not_rule_generated
load_bearing_qca_bridge = false
```

J-gap extraction:

```text
compatible_j_count = 4
pair_orientation_signs =
  (+,+,-,+,-)
  (+,+,-,-,+)
  (-,-,+,+,-)
  (-,-,+,-,+)
compatible_j_in_generated_algebra_count = 0
compatible_j_in_rule_local_center_count = 0
spectral_polarization_j_matched_count = 0
reason_for_forced_j_failure = compatible_j_finite_but_not_generated_or_rule_local
```

Minimal completion experiment:

```text
Declare W = compatible J candidate 0.
w_in_previous_generated_algebra = false
w_in_completed_generated_algebra = true
w_in_completed_center = true
generated_algebra_dimension = 26
center_dimension = 4
central_idempotent_ranks = [0,4,6,10]
lower_rank_central_idempotents = 0
compatible_centralizer_dimension = 4
local_compatible_complex_structure_count = 4
strict_unique_j_found = false
pass_completion_to_bridge = false
completion_label = completion_no_lower_rank_but_j_still_block_sign_ambiguous
```

Interpretation:

```text
This is the active Route-1 laboratory. The signed orientation twist preserves
the alpha/eta projectors but does not commute with the Floquet layer, so the
commuting second-layer no-go does not apply. It keeps the coarse [0,4,6,10]
central idempotent lattice and avoids lower-rank central projectors. The
compatible J variety is zero-dimensional, but the current strict checker does
not certify a rule-generated/local J. The extracted finite solutions are
local-looking pair-orientation structures, but none is generated by the rule
or by the rule-local center. The next microscopic move must supply that
missing local J source without reintroducing lower-rank central idempotents.
The completion experiment shows this is not enough on its own: declaring one
pair-orientation J leaves four local compatible J choices. The remaining
physics target is a mechanism that couples the alpha/eta orientation signs so
only a global ±J remains.
```

## Spatial 1D Sidecar Route

Status: finite-radius local-QCA sidecar implemented; not load-bearing.

Implementation:

```text
src/clifford_3plus2_d5/qca/spatial_1d.py
scripts/spatial_1d_alpha_search.py
scripts/spatial_1d_unseeded_search.py
tests/test_spatial_1d.py
docs/literature/spatial_1d_report.md
```

Prototype:

```text
period = 12
alpha_winding = 4
eta_winding = 3
gcd(4,3) = 1
lcm(4,3) = 12
local_hopping_shifts = [3,4]
local_qca_shifts = [3,4]
mode_windings = [4,4,4,3,3]
```

Current outcome:

```text
candidate_count = 1
unitary_candidates = 1
coarse_6_4_band_candidates = 1
orientation_choices_before_transport = 4
orientation_choices_after_transport = 2
sign_coupled_to_global_pm = true
local_hopping_reconstructs_transfer_on_samples = true
local_hopping_orientation_choices_after_transport = 2
local_qca_laurent_orthogonal = true
local_qca_symbol_reconstructs_transfer_on_samples = true
local_qca_central_idempotent_ranks = [0,4,6,10]
local_qca_lower_rank_central_idempotents = 0
local_qca_orientation_choices_after_transport = 2
strict_bridge_candidates = 0
route_label = spatial_signs_coupled_to_global_pm
local_hopping_route_label = spatial_local_hopping_signs_coupled
local_qca_route_label = spatial_local_qca_signs_coupled_not_load_bearing
unseeded_candidate_count = 4
unseeded_guardrail_rejections = 1
unseeded_coarse_6_4_center_candidates = 0
unseeded_sign_coupled_candidates = 0
unseeded_strict_bridge_candidates = 0
unseeded_route_label = unseeded_spatial_no_bridge_candidates
load_bearing_qca_bridge = false
```

Interpretation:

```text
This tests the Route-2 idea in the smallest exact setting. The sidecar now
uses a real finite-radius 1D local QCA layer with Laurent coefficients
`T(z) = P_alpha z^4 + P_eta z^3`. Exact Laurent orthogonality proves the layer
is unitary as a locality-preserving QCA, and the coefficient algebra has only
the coarse central idempotent ranks `[0,4,6,10]`. The resulting spatial cycle
has the right sign-coupling shape: independent alpha/eta orientations reduce
to global ±J. It still is not a load-bearing bridge because `P_alpha/P_eta`
enter as the layer coefficients rather than being derived from more primitive
microscopic gates. The next Route-2 task is to factor or replace this
projector-shift layer with microscopic local gates that do not seed the answer.
The first unseeded scan checks three block-blind finite-radius layers plus the
seeded projector-shift guardrail. The unseeded layers either have no coarse
`6+4` center or generate lower-rank central idempotents; the seeded layer is
correctly rejected because its coefficients are exactly `P_alpha/P_eta`.
```

## Defect-Beta Transition-Pair Family

Status: implemented but parked; monodromy center and canonical `J` found, but
strict compatible-commutant uniqueness still fails. It is retained as a
regression target, not the active load-bearing route, until rebuilt as a
genuine higher-dimensional defect calculation.

Implementation:

```text
src/clifford_3plus2_d5/qca/defect_beta.py
scripts/defect_beta_search.py
tests/test_defect_beta.py
docs/literature/defect_beta_report.md
```

Family:

```text
two distinct orientation-reversing wall-cycle transition functions
clutching reflection C = diag(-I_5, I_5)
round-trip monodromy computed by T_entry = C, T_exit = M C, T_exit T_entry = M
three mode pairs with omega = exp(2 pi i / 3)
two mode pairs with i
ten defect-charge patterns enumerated
```

Current outcome:

```text
monodromy_candidates = 10
scaled_monodromy_certified_candidates = 10
monodromy_equals_matching_floquet_alpha = true
transition_functions_commute = false
rule_generated_algebra_dimension = 8
rule_center_dimension = 2
rule_center_solved = true
generated_j_moduli_dimension = 0
compatible_centralizer_dimension = 13
compatible_j_moduli_dimension = None
locality_radius_bound = 1
local_compatible_operator_dimension = 2
local_compatible_j_moduli_dimension = None
local_compatible_complex_structure_count = 0
strict_compatible_j_forced_candidates = 0
strict_bridge_candidates = 0
verdict_counts = {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge = false
```

Interpretation:

```text
Defect-beta is redundant with Floquet-alpha at the monodromy level:
the round-trip monodromy equals the matching alpha Floquet operator. The
transition-pair rule is nevertheless distinct because the wall entry and exit
maps do not commute. Feeding `(T_entry, T_exit)` into the checker gives a
different algebra, with generated dimension 8 and compatible centralizer
dimension 13, but it still does not force J. β remains parked until rebuilt as
a genuine higher-dimensional defect calculation whose transition algebra
supplies independent microscopic structure.
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

6. **Exploration Sprints**
   - Keep E1/E2 artifacts reproducible.
   - Expand primitive families only when the falsifier outputs remain explicit.
   - Treat seeded or rank-one-derived projector pairs as controls, not bridge
     evidence.

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
