# Results

## 2026-05-14

This is a historical result bundle. Current validation convention is documented
in [Project Conventions](../project_conventions.md): full `uv run pytest -q` is
a slow archival suite, while normal work uses Ruff plus focused route checks.

Commands:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/real_carrier_check.py --check
uv run python scripts/forced_j_check.py --check
uv run python scripts/forced_j_check.py --include-addressable-rank-one --expect-verdict falsified
uv run python scripts/structural_split_check.py --check
uv run python scripts/structural_split_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/structural_split_check.py --include-rank-one-weak --expect-verdict falsified
uv run python scripts/gate_classification_check.py --check
uv run python scripts/qca_update_check.py --check
uv run python scripts/qca_update_check.py --include-rank-one-color-shift --expect-verdict falsified
uv run python scripts/qca_update_check.py --include-rank-one-weak-shift --expect-verdict falsified
uv run python scripts/spinor16_check.py --check
uv run python scripts/normalizer_check.py --check --expect-verdict candidate_only
uv run python scripts/normalizer_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/normalizer_check.py --include-rank-one-weak --expect-verdict falsified
uv run python scripts/normalizer_check.py --include-off-block --expect-verdict falsified
uv run python scripts/normalizer_check.py --include-full-u5-controls --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --check --expect-verdict candidate_only
uv run python scripts/real_qca_branch_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-rank-one-weak --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-rank-one-pair --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-off-block --expect-verdict falsified
uv run python scripts/explore_rule_space.py --check --output-dir data/exploration
uv run python scripts/discover_projectors.py --check --mode unseeded --expect-verdict not_found --output-dir data/exploration
uv run python scripts/discover_projectors.py --check --mode sanity-seeded --expect-verdict projector_pair_found --output-dir /tmp/e2_sanity
uv run python scripts/discover_projectors.py --check --mode block-reflection-candidate --expect-verdict projector_pair_found --output-dir /tmp/e2_block
uv run python scripts/rule_to_verdict.py --case minimal-period-four --expect-verdict falsified_no_rank_6_4_center
uv run python scripts/rule_to_verdict.py --case clock-block-reflection --expect-verdict candidate_only_j_not_forced
uv run python scripts/rule_to_verdict.py --case clock-rank-one-color-reflection --expect-verdict falsified_rank_one_center
uv run python scripts/floquet_alpha.py --check
uv run python scripts/floquet_alpha.py --variant plus --check
uv run python scripts/floquet_alpha.py --variant time-reversal --check
uv run python scripts/floquet_alpha.py --variant second-layer --check
uv run python scripts/floquet_alpha.py --variant noncommuting --check
uv run python scripts/floquet_alpha.py --variant noncommuting-gap --check
uv run python scripts/floquet_alpha.py --variant noncommuting-exhaustive --check
uv run python scripts/floquet_alpha.py --variant noncommuting-completion --check
uv run python scripts/spatial_1d_alpha_search.py --check
uv run python scripts/spatial_1d_alpha_search.py --variant combined --check
uv run python scripts/spatial_1d_unseeded_search.py --check
uv run python scripts/bloch_path_a_search.py --check
uv run python scripts/defect_beta_search.py --check
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

Verification:

```text
ruff: passed
pytest: prior full-suite baseline 229 passed; Route-3 focused file 18 passed
```

Real carrier check:

```text
This verifies the exact real carrier ansatz only.
It does not prove QCA dynamics force J.
carrier: R^2_clock tensor (R^3 plus R^2)
dimension: 10
mode_dimension: 5
J_squared_minus_identity: true
J_orthogonal: true
projector_3_rank: 6
projector_2_rank: 4
projectors_commute_with_J: true
phase_1_real_carrier_check_passed: true
qca_forces_j: false
load_bearing_qca_bridge: false
```

Forced J check:

```text
This checks whether declared exact gate words can produce J.
It does not prove microscopic QCA rule data force J.
candidate_name: standard_clock_j
generated_by_gate_word: true
gate_word: global_clock_tick
J_squared_minus_identity: true
J_orthogonal: true
rank_one_pair_rotations_addressable: false
forced_j_check_passed: true
qca_forces_j: false
forced_j_verdict: candidate_only
load_bearing_qca_bridge: false
```

Forced J addressability falsifier:

```text
rank_one_pair_rotations_addressable: true
forced_j_verdict: falsified
load_bearing_qca_bridge: false
```

Structural split check:

```text
This verifies the exact structural split candidate only.
It does not prove QCA rule data force P_3/P_2.
projector_identities_passed: true
projectors_commute_with_J: true
projector_3_rank: 6
projector_2_rank: 4
rank_one_color_projectors_addressable: false
rank_one_weak_projectors_addressable: false
addressability_algebra_safe: true
qca_supplies_structural_3plus2_split: false
structural_split_verdict: candidate_only
structural_split_check_passed: true
load_bearing_qca_bridge: false
```

Structural split rank-one color falsifier:

```text
rank_one_color_projectors_addressable: true
addressability_algebra_safe: false
structural_split_verdict: falsified
load_bearing_qca_bridge: false
```

Structural split rank-one weak falsifier:

```text
rank_one_weak_projectors_addressable: true
addressability_algebra_safe: false
structural_split_verdict: falsified
load_bearing_qca_bridge: false
```

Gate classification check:

```text
This verifies the exact SM commutant gate classifier only.
It does not prove QCA rule data supply only safe geometric gates.
P_3: safe_sm_commutant
P_2: safe_sm_commutant
J_P_3: safe_sm_commutant
J_P_2: safe_sm_commutant
block_mixer: block_mixing_fail
rank_one_color_projector: color_breaking_fail
rank_one_weak_projector: weak_breaking_fail
real_conjugation: antilinear_fail
commutant_basis_matches_expected: true
safe_algebra_closure_passed: true
gate_classification_check_passed: true
qca_geometric_gate_algebra_safe: false
load_bearing_qca_bridge: false
```

QCA update check:

```text
This verifies the finite-depth QCA update candidate only.
It does not prove microscopic QCA rule data force this update.
finite_depth: true
layer_count: 4
max_locality_radius: 0
all_layers_local: true
all_layers_orthogonal: true
period_four_check_passed: true
quarter_period_is_j: true
half_period_is_minus_identity: true
full_period_is_identity: true
all_internal_actions_safe: true
unsafe_gate_witnesses:
qca_rule_forces_update: false
finite_depth_qca_verdict: candidate_only
qca_update_check_passed: true
load_bearing_qca_bridge: false
```

QCA update rank-one color shift falsifier:

```text
all_internal_actions_safe: false
unsafe_gate_witnesses: rank_one_color_shift
finite_depth_qca_verdict: falsified
load_bearing_qca_bridge: false
```

QCA update rank-one weak shift falsifier:

```text
all_internal_actions_safe: false
unsafe_gate_witnesses: rank_one_weak_shift
finite_depth_qca_verdict: falsified
load_bearing_qca_bridge: false
```

Spinor 16 check:

```text
This verifies guarded Spin(10) spinor reconstruction only.
It does not prove QCA rule data force J or the 3+2 split.
spinor16_dimension: 16
degree_dimensions: {'0': 1, '2': 10, '4': 5}
hypercharge_check_passed: true
branching_table_check_passed: true
uses_existing_j_and_split: true
introduces_new_complex_structure: false
introduces_new_3plus2_split: false
spinor16_verdict: candidate_only
spinor16_check_passed: true
load_bearing_qca_bridge: false
```

Normalizer forcedness check:

```text
This checks forcedness and normalizer proxies for the candidate data.
It does not prove microscopic QCA rule data force J or P_3/P_2.
rule_data_source: candidate_only
centralizer_dimension: 52
orthogonal_normalizer_dimension: 21
addressability_algebra_dimension: 2
candidate_j_valid: true
candidate_split_valid: true
candidate_j_preserved_by_normalizer: false
normalizer_preserves_declared_split: true
continuous_j_alternatives_not_excluded: true
rank_three_projector_family_not_excluded: true
rank_one_color_projectors_addressable: false
rank_one_weak_projectors_addressable: false
off_block_controls_addressable: false
addressability_algebra_safe: true
j_unique_or_forced: false
split_unique_or_forced: false
forcedness_verdict: candidate_only
normalizer_check_passed: true
load_bearing_qca_bridge: false
```

Normalizer rank-one color falsifier:

```text
rank_one_color_projectors_addressable: true
addressability_algebra_safe: false
forcedness_verdict: falsified
load_bearing_qca_bridge: false
```

Normalizer rank-one weak falsifier:

```text
rank_one_weak_projectors_addressable: true
addressability_algebra_safe: false
forcedness_verdict: falsified
load_bearing_qca_bridge: false
```

Normalizer off-block falsifier:

```text
off_block_controls_addressable: true
addressability_algebra_safe: false
forcedness_verdict: falsified
load_bearing_qca_bridge: false
```

Normalizer full-U(5)-like controls falsifier:

```text
rank_one_color_projectors_addressable: true
rank_one_weak_projectors_addressable: true
off_block_controls_addressable: true
addressability_algebra_safe: false
forcedness_verdict: falsified
load_bearing_qca_bridge: false
```

Real-QCA branch check:

```text
This checks the stronger real-QCA-first branch candidate.
It does not prove microscopic QCA rule data force J or P_3/P_2.
branch_name: phase_8a_stronger_real_qca_first
candidate_word_found: true
candidate_word: global_clock_tick
word_depth: 1
finite_depth: true
translation_invariant: true
generates_j: true
generates_split: true
j_forced_by_rule_space: false
split_forced_by_rule_space: false
forbidden_rank_one_controls_present: false
forbidden_off_block_controls_present: false
addressability_algebra_safe: true
normalizer_verdict: candidate_only
real_qca_branch_verdict: candidate_only
real_qca_branch_check_passed: true
load_bearing_qca_bridge: false
```

Real-QCA branch rank-one color falsifier:

```text
forbidden_rank_one_controls_present: true
addressability_algebra_safe: false
normalizer_verdict: falsified
real_qca_branch_verdict: falsified
load_bearing_qca_bridge: false
```

Real-QCA branch rank-one weak falsifier:

```text
forbidden_rank_one_controls_present: true
addressability_algebra_safe: false
normalizer_verdict: falsified
real_qca_branch_verdict: falsified
load_bearing_qca_bridge: false
```

Real-QCA branch rank-one pair falsifier:

```text
forbidden_rank_one_controls_present: true
addressability_algebra_safe: true
normalizer_verdict: candidate_only
real_qca_branch_verdict: falsified
load_bearing_qca_bridge: false
```

Real-QCA branch off-block falsifier:

```text
forbidden_off_block_controls_present: true
addressability_algebra_safe: false
normalizer_verdict: falsified
real_qca_branch_verdict: falsified
load_bearing_qca_bridge: false
```

E1 rule-space exploration:

```text
This runs bounded exact rule-space exploration sprint E1.
It does not prove microscopic QCA rule data force J or P_3/P_2.
rule_space_name: e1_bounded_real_rule_space
primitive_family_count: 5
primitive_set_count: 10
primitive_sets_scanned: 10
words_scanned: 170
j_hits: 73
period_four_hits: 73
split_candidates: 170
addressability_safe_hits: 158
normalizer_candidate_hits: 158
forced_candidate_hits: 0
rank_one_rejections: 38
off_block_rejections: 4
normalizer_too_large_rejections: 158
top_rejection_reasons: normalizer_too_large=158,no_j=97,no_period_four=97,rank_one_addressability=38,unsafe_addressability=12,normalizer_falsified=12,off_block_addressability=4
surviving_candidates: 0
exploration_check_passed: true
load_bearing_qca_bridge: false
```

E2 unseeded projector discovery:

```text
This runs E2 projector discovery.
It separates unseeded discovery from seeded sanity checks.
mode: unseeded
primitive_sets_scanned: 6
algebra_elements_considered: 1924
candidate_projectors: 3
rank_2_projectors: 3
rank_6_projectors: 0
rank_4_projectors: 0
complementary_pairs: 0
unsafe_rank_one_projectors: 3
unseeded_projector_pairs_found: 0
seeded_projector_pairs_found: 0
block_reflection_pairs_found: 0
discovery_verdict: not_found
discovery_check_passed: true
load_bearing_qca_bridge: false
```

E2 seeded sanity check:

```text
mode: sanity-seeded
candidate_projectors: 2
rank_6_projectors: 1
rank_4_projectors: 1
complementary_pairs: 1
seeded_projector_pairs_found: 1
discovery_verdict: projector_pair_found
discovery_check_passed: true
load_bearing_qca_bridge: false
```

E2 block-reflection candidate check:

```text
mode: block-reflection-candidate
candidate_projectors: 2
rank_6_projectors: 1
rank_4_projectors: 1
complementary_pairs: 1
block_reflection_pairs_found: 1
discovery_verdict: projector_pair_found
discovery_check_passed: true
load_bearing_qca_bridge: false
```

Unified rule-to-verdict minimal clock:

```text
rule_name: minimal_period_four_clock_candidate
natural_eigenvalue_field: QQ
floquet_spectrum: [{'eigenvalue': '1', 'multiplicity': 10}]
generated_algebra_dimension: 2
center_dimension: 2
central_idempotent_ranks: [0, 10]
generated_j_moduli_dimension: 0
generated_complex_structures: 2
compatible_j_moduli_dimension: None
forced_j_found: false
pass_rule_to_bridge: false
verdict: falsified_no_rank_6_4_center
load_bearing_qca_bridge: false
```

Unified rule-to-verdict block reflection:

```text
rule_name: clock_block_reflection_rule
natural_eigenvalue_field: QQ(-I, I)
generated_algebra_dimension: 4
center_dimension: 4
central_idempotent_ranks: [0, 4, 6, 10]
complementary_rank_6_4_pairs: 1
lower_rank_central_idempotents: 0
generated_j_moduli_dimension: 0
generated_complex_structures: 4
compatible_centralizer_dimension: 26
compatible_j_solved: false
compatible_j_moduli_dimension: None
forced_j_found: false
pass_rule_to_bridge: false
verdict: candidate_only_j_not_forced
load_bearing_qca_bridge: false
```

Unified rule-to-verdict rank-one reflection:

```text
rule_name: clock_rank_one_color_reflection_rule
central_idempotent_ranks: [0, 2, 8, 10]
lower_rank_central_idempotents: 1
pass_rule_to_bridge: false
verdict: falsified_rank_one_center
load_bearing_qca_bridge: false
```

Floquet-alpha search:

```text
This searches the Floquet-alpha physical primitive family.
It exposes one mandatory quantized resonance layer per candidate.
candidate_count: 10
rank_6_4_pair_candidates: 10
rank_one_falsified_candidates: 0
bridge_candidates: 0
verdict_counts: {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge: false
```

Floquet-alpha-plus search:

```text
This searches Floquet-alpha plus canonical polarization J extraction.
It reports the strict compatible-J obstruction separately.
candidate_count: 10
polarization_j_candidates: 10
scaled_polarization_certified_candidates: 10
generated_j_moduli_dimension: 0
compatible_centralizer_dimension: 26
compatible_j_moduli_dimension: 9
locality_radius_bound: 0
local_compatible_operator_dimension: 4
local_compatible_j_moduli_dimension: 0
local_compatible_complex_structure_count: 4
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'polarization_j_produced_not_strictly_unique': 10}
load_bearing_qca_bridge: false
```

Floquet-alpha time-reversal sidecar:

```text
This checks Floquet-alpha with a declared time-reversal involution.
K is diagnostic only unless generated by microscopic rule data.
candidate_count: 10
k_real_orthogonal_candidates: 10
k_involution_candidates: 10
k_conjugates_floquet_to_inverse_candidates: 10
k_anticommutes_with_canonical_j_candidates: 10
k_in_generated_algebra_candidates: 0
compatible_j_moduli_dimension_before_k: 9
k_fixed_compatible_j_moduli_dimension: 3
k_fixed_local_compatible_complex_structure_count: 4
k_fixed_local_matches_canonical_orbit_count: 2
k_reduces_full_moduli_candidates: 10
k_reduces_to_global_pm_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'declared_time_reversal_reduces_moduli_not_unique': 10}
load_bearing_qca_bridge: false
```

Floquet-alpha cycle/swap second-layer search:

```text
This checks the literal Floquet-alpha commuting cycle/swap second layer.
candidate_count: 10
commuting_second_layer_candidates: 10
order_certified_candidates: 10
compatible_centralizer_collapsed_candidates: 10
no_locking_guardrail_passed_candidates: 0
strict_bridge_candidates: 0
generated_algebra_dimension: 10
center_dimension: 10
compatible_centralizer_dimension: 10
explicit_lower_rank_projector_ranks: [2, 2, 2]
load_bearing_qca_bridge: false
```

Spatial 1D sidecar search:

```text
candidate_count: 1
unitary_candidates: 1
coarse_6_4_band_candidates: 1
period: 12
alpha_winding: 4
eta_winding: 3
winding_gcd: 1
winding_lcm: 12
locality_radius: 4
sample_count: 12
transfer_unitary_on_samples: true
alpha_projector_rank: 6
eta_projector_rank: 4
orientation_choices_before_transport: 4
orientation_choices_after_transport: 2
sign_coupled_to_global_pm: true
strict_bridge_candidates: 0
route_label: spatial_signs_coupled_to_global_pm
local_hopping_term_count: 2
local_hopping_shifts: [3, 4]
local_hopping_mode_windings: [4, 4, 4, 3, 3]
local_hopping_reconstructs_transfer_on_samples: true
local_hopping_orientation_choices_after_transport: 2
local_hopping_route_label: spatial_local_hopping_signs_coupled
local_qca_layer_name: spatial_1d_alpha_projector_shift_qca
local_qca_term_count: 2
local_qca_shifts: [3, 4]
local_qca_locality_radius: 4
local_qca_finite_radius: true
local_qca_laurent_orthogonal: true
local_qca_symbol_reconstructs_transfer_on_samples: true
local_qca_symbol_unitary_on_samples: true
local_qca_coefficient_algebra_dimension: 2
local_qca_coefficient_center_dimension: 2
local_qca_central_idempotent_ranks: [0, 4, 6, 10]
local_qca_lower_rank_central_idempotents: 0
local_qca_orientation_choices_after_transport: 2
local_qca_route_label: spatial_local_qca_signs_coupled_not_load_bearing
load_bearing_qca_bridge: false
```

Spatial 1D unseeded search:

```text
candidate_count: 4
unseeded_candidate_count: 3
seeded_guardrail_rejections: 1
laurent_orthogonal_candidates: 4
unseeded_coarse_6_4_center_candidates: 0
unseeded_sign_coupled_candidates: 0
unseeded_strict_bridge_candidates: 0
lower_rank_center_rejections: 1
route_label: unseeded_spatial_no_bridge_candidates
load_bearing_qca_bridge: false
candidate: unseeded_uniform_identity_shift, center_ranks=[0, 10], route=unseeded_spatial_no_coarse_6_4_center
candidate: unseeded_uniform_clock_shift, center_ranks=[0, 10], route=unseeded_spatial_no_coarse_6_4_center
candidate: unseeded_mode_5_cycle_shift, center_ranks=[0, 2, 4, 4, 6, 6, 8, 10], route=unseeded_spatial_lower_rank_center_rejected
candidate: spatial_1d_alpha_projector_shift_qca, witnesses=['shift_3:P_eta', 'shift_4:P_alpha'], route=unseeded_spatial_seeded_coefficient_rejected
```

Bloch Path-A search:

```text
candidate_count: 6
seed_guardrail_rejections: 4
unseeded_candidate_count: 2
stable_6_4_band_candidates: 0
topological_pm_candidates: 4
rule_generated_j_section_candidates: 0
strict_bridge_candidates: 0
route_label: bloch_path_a_seeded_shape_only
projector_free_rule_verdict: not_solved
projector_free_rule_generated_algebra_dimension: 16
projector_free_rule_generated_algebra_closed: false
projector_free_rule_pass_rule_to_bridge: false
load_bearing_qca_bridge: false
```

Bloch Path-A stepwise projector-free search:

```text
uv run python scripts/bloch_path_a_stepwise.py --max-candidates 6 --max-algebra-dim 48 --jobs 2 --check

candidate_count: 6
closed_count: 6
coarse_6_4_count: 0
candidate: path_a_projector_free_cycle_combined, dim=34, closed=true, center=None, ranks=[], label=closes_center_not_checked
candidate: path_a_projector_free_p0_c12340_s44343, dim=34, closed=true, center=None, ranks=[], label=closes_center_not_checked
candidate: path_a_projector_free_p0_c12340_s44334, dim=34, closed=true, center=None, ranks=[], label=closes_center_not_checked
candidate: path_a_projector_free_p0_c12403_s44433, dim=34, closed=true, center=None, ranks=[], label=closes_center_not_checked
candidate: path_a_projector_free_p0_c12403_s44343, dim=34, closed=true, center=None, ranks=[], label=closes_center_not_checked
candidate: path_a_projector_free_p0_c12403_s44334, dim=34, closed=true, center=None, ranks=[], label=closes_center_not_checked
```

Follow-up staged calculation on the base projector-free candidate:

```text
uv run python scripts/bloch_path_a_stepwise.py --max-candidates 1 --max-algebra-dim 48 --center-top 1 --idempotents --centralizer --j-solve --check

generated_algebra_dimension: 34
generated_algebra_closed: true
center_dimension: 4
central_idempotent_ranks: [0,4,6,10]
compatible_centralizer_dimension: 4
generated_j_count: 0
compatible_j_count: 0
bridge_j_status: no_rule_generated_j
route_label: closes_coarse_6_4_center
```

Defect-beta search:

```text
This searches defect-beta using the entry/exit transition-pair rule.
It also records that the round-trip monodromy equals Floquet-alpha.
candidate_count: 10
monodromy_candidates: 10
scaled_monodromy_certified_candidates: 10
monodromy_equals_matching_floquet_alpha: true
transition_functions_commute: false
rule_generated_algebra_dimension: 8
rule_center_dimension: 2
rule_center_solved: true
generated_j_moduli_dimension: 0
compatible_centralizer_dimension: 13
compatible_j_moduli_dimension: None
locality_radius_bound: 1
local_compatible_operator_dimension: 2
local_compatible_j_moduli_dimension: None
local_compatible_complex_structure_count: 0
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge: false
```

Branching check:

```text
This verifies standard Spin(10) branching only.
It does not prove the QCA supplies the split.
hypercharge_formula: Y = 0 + (-1/3) N_3 + (1/2) N_2
total_multiplicity: 16
sector: N_3=0, N_2=0, multiplicity=1, Y=0, label=nu^c
sector: N_3=0, N_2=2, multiplicity=1, Y=1, label=e^c
sector: N_3=1, N_2=1, multiplicity=6, Y=1/6, label=Q
sector: N_3=2, N_2=0, multiplicity=3, Y=-2/3, label=u^c
sector: N_3=2, N_2=2, multiplicity=3, Y=1/3, label=d^c
sector: N_3=3, N_2=1, multiplicity=2, Y=-1/2, label=L
branching_check_passed: true
load_bearing_qca_bridge: false
```

QCA split audit:

```text
This audits the load-bearing QCA-to-3+2 bridge claim.
No qca_data.json, no bridge claim.
qca_split_audit_verdict: notation_only
qca_supplies_structural_3plus2_split: false
complex_structure_compatible_with_3plus2_split: false
load_bearing_qca_bridge: false
```

QCA split audit JSON mode:

```json
{
  "anticommutation_matrix": [],
  "block_diagonal_gate_algebra": false,
  "candidate_generators": [],
  "complex_structure_compatible_with_3plus2_split": false,
  "complex_structure_in_allowed_gate_algebra": false,
  "complex_structure_operator": null,
  "complex_structure_origin": "unknown",
  "complex_structure_preserves_3plus2_split": false,
  "complex_structure_squares_to_minus_one": false,
  "load_bearing_qca_bridge": false,
  "off_block_gate_generators_present": false,
  "qca_supplies_structural_3plus2_split": false,
  "signature": "unknown",
  "sm_commutant_gate_algebra": false,
  "structural_origin": "unknown",
  "verdict": "notation_only"
}
```

Interpretation:

```text
The textbook branching arithmetic passes.
The QCA bridge has not passed.
Current bridge status: notation_only.
Phase 0 audit contract is closed.
Phase 1 real-carrier algebra is exact but not yet QCA-forced.
Phase 2 can certify a declared J gate word, but J is still not QCA-forced.
Phase 3 can certify a declared P_3/P_2 candidate and rank-one falsifiers, but
the split is still not QCA-forced.
Phase 4 proves the SM commutant classifier oracle on canonical safe and unsafe
gates, but no actual QCA gate set has been certified safe.
Phase 5 certifies a declared finite-depth period-four update candidate and
no-locking falsifiers, but the update is still not QCA-forced.
Phase 6 reconstructs the guarded 16-dimensional spinor table from prior
candidate data, but it does not make the QCA bridge load-bearing.
Phase 7 computes normalizer and forcedness proxies. The declared split is
preserved by the candidate data, but `J` and the rank-three projector are not
forced by source-backed microscopic rule data.
Phase 8A adds a stronger real-QCA-first branch checker. The declared
period-four word still generates `J`, but the branch remains candidate-only
because the rule space does not force `J` or the split.
E1 performs bounded rule-space exploration. It scans 170 exact words, finds
73 `J`/period-four hits, and finds zero forced surviving candidates.
E2 performs bounded unseeded projector discovery. It scans 1924 exact algebra
elements across 6 primitive sets, finds zero complementary `6+4` projector
pairs, and records three unsafe rank-2 projector candidates.
The unified rule-to-verdict checker now collapses the real physics criterion
into one interface. Current controls are negative: minimal clock has no
`6+4` center, block reflection has the center but does not force `J`, and
rank-one reflection is falsified by a lower central idempotent.
Floquet-alpha replaces the abstract primitive-family direction with a physical
quantized resonance layer. It produces the coarse `6+4` center in all ten
patterns, without rank-one centers. Alpha-plus extracts a canonical
spectral-polarization `J` from the same rule operator in all ten patterns, but
the global compatible centralizer has dimension 26 with a 9-dimensional
compatible-`J` family. The rule-generated local-center search already finds
four discrete `J` choices, so strict uniqueness fails without claiming to
exhaust every on-site local `J in M_10(R)`.
The declared time-reversal sidecar satisfies `KUK=U^-1` and `KJK=-J`, but
`K` is not rule-generated and only reduces the full compatible-`J` moduli
estimate from 9 to 3. It does not reduce the local choices to global `±J`.
The literal commuting cycle/swap second layer reduces the compatible
centralizer to dimension 10 but generates rank-2 central projectors, so the
no-locking guardrail rejects it.
The noncommuting Floquet-alpha signed twist keeps the coarse center and removes
the continuous compatible-`J` family, but the finite compatible `J` choices
are not rule-generated/local. The spatial 1D sidecar is the current Route-2
follow-up: an explicit finite-radius local QCA layer satisfies exact Laurent
orthogonality, reconstructs the period-12 transfer, and couples independent
alpha/eta block signs down to global `±J`, while remaining non-load-bearing
because the coarse projectors are still supplied as coefficients.
The unseeded spatial scan makes the remaining gap concrete: conservative
block-blind finite-radius layers do not produce a bridge candidate, and the
working projector-shift layer is rejected when seeded coefficients are
forbidden.
The exhaustive discrete Route-1 scan is now checked as a focused result:
`3840` signed-twist candidates reduce to `96` exact symmetry classes. It finds
`720` generated-compatible-`J` hits, including `240` minimal four-`J` hits, but
zero generated-`J` hits in the no-locking shape and zero bridge candidates.
This is now the third no-go proposition: block-preserving noncommuting on-site
locking cannot supply a rule-generated `J` while preserving no-locking and the
SM commutant.
The combined Route-1/Route-2 sidecar composes the noncommuting on-site update
with the `(4,3)` winding hops. It reaches the topological `±J` shape
(`4` compatible signs reduced to `2` transported signs), but the joint rule
algebra still generates zero transported `J`s, so the strict bridge remains
false.
The Bloch Path-A checker now makes this boundary explicit over sampled
root-of-unity Bloch symbols: seeded candidates have the topological shape but
fail the coefficient-algebra guardrail, while the first unseeded full-shift
candidates produce no stable rank-`(6,4)` band split.
The main `rule_to_verdict` path now also samples Bloch symbols as a single
joint algebra. Its first projector-free combined Route-1/Route-2 candidate
uses source-mode shifts `(4,4,4,3,3)` without raw `P_alpha/P_eta`
coefficients. Raising the closure cap shows this was a real structured
candidate, not a dead computational boundary: the base candidate closes at
dimension `34`, has center dimension `4`, and its central idempotent ranks are
`[0,4,6,10]`. Six nearby projector-free monomial-hop variants close to the
same algebra dimension. The bounded split-center `J` solver settles the base
candidate negatively: generated and compatible `J` counts are both `0`. The
remaining search gap is therefore not the coarse center, but a microscopic
unseeded hopping primitive whose structured algebra also carries a complex
structure.
Defect-beta is retained as a regression target but parked as a load-bearing
route until rebuilt as a genuine higher-dimensional defect calculation. Its
round-trip monodromy is exactly the matching Floquet-alpha operator. The
noncommuting transition-pair rule has a distinct generated algebra
(dimension 8, compatible centralizer dimension 13), but still does not force
`J`.
```

## Active Roadmap Update

The active plan is now the enhanced J-first attack:

```text
J first,
then forced 3+2,
then no-locking commutant,
then finite-depth QCA,
then Spin(10) spinor.
```

No new bridge result is claimed by this roadmap update.
