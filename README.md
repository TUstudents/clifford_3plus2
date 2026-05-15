# clifford-3plus2-d5

Do not fool yourself.

This repo tests whether QCA data can produce a real geometry-to-Spin(10)
one-generation bridge. It does not derive the Standard Model, three families,
mirror decoupling, continuum gauge dynamics, or phenomenology.

The active roadmap is the enhanced J-first attack:

```text
real finite-depth QCA data
  -> forced local J on R^10
  -> structural J-invariant 6+4 real split
  -> C^5 = C^3 ⊕ C^2
  -> geometric gate algebra in the SM commutant
  -> Lambda^even(C^5) = Spin(10) chiral 16
```

Current honest status:

```text
phase_0_audit_contract: complete
phase_1_real_carrier_check: passes
forced_j_check: candidate_only
structural_split_check: candidate_only
gate_classification_check: oracle_passes
qca_update_check: candidate_only
spinor16_check: candidate_only
normalizer_check: candidate_only
real_qca_branch_check: candidate_only
rule_space_exploration: no forced survivors
unseeded_projector_discovery: no complementary 6+4 pair
rule_to_verdict_check: unified negative interface with exact field and J-moduli metadata
floquet_alpha: canonical CLI for alpha variants; coarse 6+4 center found
floquet_alpha_time_reversal: declared K reduces J moduli 9->3, not ±J
floquet_alpha_second_layer: cycle/swap lock fails no-locking guardrail
floquet_alpha_noncommuting: block-preserving signed twist under exploration
floquet_alpha_noncommuting_exhaustive: generated J hits exist, none in no-locking shape
floquet_alpha_noncommuting_completion: no lower center, still four J signs
spatial_1d_alpha: local-QCA sidecar couples four J signs to global ±
spatial_1d_combined: Route 1 + Route 2 gives ± shape, still no rule-generated J
spatial_1d_unseeded: no bridge candidate; seeded projector-shift rejected
defect_beta_search: monodromy equals alpha; transition-pair rule still negative
Spin(10) branching check: passes
QCA load-bearing bridge: notation_only
```

## Active Theorem Target

The bridge is accepted only if QCA rule data produce, before invoking
Spin(10):

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

If `J`, `P_3/P_2`, or the gate commutant are chosen because they reproduce
Spin(10), the project remains `notation_only`.

## Main Working Docs

- [Roadmap and working index](docs/roadmap.md)
- [Project conventions](docs/project_conventions.md)
- [Enhanced J-first attack plan](docs/qca_3plus2_j_first_enhanced_attack_plan.md)
- [Real carrier report](docs/literature/real_carrier_report.md)
- [Forced J report](docs/literature/forced_j_report.md)
- [Projector lattice report](docs/literature/projector_lattice_report.md)
- [Gate classification report](docs/literature/gate_classification_report.md)
- [QCA update certificate](docs/literature/qca_update_certificate.md)
- [Spinor 16 report](docs/literature/spinor16_report.md)
- [Normalizer report](docs/literature/normalizer_report.md)
- [Forcedness certificate](docs/literature/forcedness_certificate.md)
- [Real-QCA branch report](docs/literature/real_qca_branch_report.md)
- [Rule-space exploration report](docs/literature/rule_space_exploration_report.md)
- [Unseeded projector discovery report](docs/literature/unseeded_projector_discovery_report.md)
- [Rule-to-verdict report](docs/literature/rule_to_verdict_report.md)
- [Floquet alpha report](docs/literature/floquet_alpha_report.md)
- [Floquet alpha J obstruction](docs/literature/floquet_alpha_j_obstruction.md)
- [Spatial 1D sidecar report](docs/literature/spatial_1d_report.md)
- [Defect beta report](docs/literature/defect_beta_report.md)
- [Theory summary](docs/theory.md)
- [Falsifiers](docs/falsifiers.md)
- [Phase 0 handover compliance](docs/handover_compliance.md)
- [Results](docs/results/index.md)

## Implemented Checks

`scripts/branching_check.py` verifies the standard exterior-algebra branching:

```text
Y = -1/3 N_3 + 1/2 N_2
dim Lambda^even(C^5) = 16
```

It must print:

```text
branching_check_passed: true
load_bearing_qca_bridge: false
```

`scripts/qca_split_audit.py` audits the older exact-input bridge contract.
Without valid exact input at `data/qca_data.json`, it must report:

```text
qca_split_audit_verdict: notation_only
load_bearing_qca_bridge: false
```

The Phase 0 audit contract is closed in
[docs/handover_compliance.md](docs/handover_compliance.md).

`scripts/real_carrier_check.py` verifies the exact Phase 1 real-carrier
ansatz:

```text
K_x = R^2_clock ⊗ (R^3 ⊕ R^2)
J = epsilon ⊗ I_5
P_3 = I_2 ⊗ diag(1,1,1,0,0)
P_2 = I_2 ⊗ diag(0,0,0,1,1)
```

It must also print:

```text
qca_forces_j: false
load_bearing_qca_bridge: false
```

`scripts/forced_j_check.py` verifies that a declared exact gate word can
produce the Phase 1 `J`. By default this is still only a candidate:

```text
generated_by_gate_word: true
qca_forces_j: false
forced_j_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/structural_split_check.py` verifies that the exact Phase 1
projectors form a `J`-compatible `3+2` candidate and rejects rank-one
addressability inside either block. By default this is still only a candidate:

```text
qca_supplies_structural_3plus2_split: false
structural_split_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/gate_classification_check.py` verifies the exact Phase 4 SM
commutant classifier:

```text
safe_sm_commutant
block_mixing_fail
color_breaking_fail
weak_breaking_fail
antilinear_fail
unknown_fail
```

It must also print:

```text
gate_classification_check_passed: true
qca_geometric_gate_algebra_safe: false
load_bearing_qca_bridge: false
```

`scripts/qca_update_check.py` verifies the exact Phase 5 period-four
finite-depth update candidate:

```text
U(T/4) = J
U(T/2) = -I
U(T) = I
```

It must also print:

```text
qca_rule_forces_update: false
finite_depth_qca_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/spinor16_check.py` verifies the guarded Phase 6 reconstruction of
`Lambda^even(C^5)` from the prior `J` and `P_3/P_2` candidates:

```text
spinor16_dimension: 16
hypercharge_check_passed: true
branching_table_check_passed: true
spinor16_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/normalizer_check.py` verifies the Phase 7 forcedness and normalizer
proxies. It checks whether the declared data preserve the split, whether they
force the candidate `J`, and whether switchable controls generate forbidden
within-block or off-block addressability. By default this is still only a
candidate:

```text
candidate_j_preserved_by_normalizer: false
normalizer_preserves_declared_split: true
j_unique_or_forced: false
split_unique_or_forced: false
forcedness_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/real_qca_branch_check.py` verifies the Phase 8A stronger
real-QCA-first branch interface. It composes the gate-word, structural split,
and normalizer checks for a finite-depth real gate-word search space. By
default this is still only a candidate:

```text
candidate_word_found: true
candidate_word: global_clock_tick
generates_j: true
generates_split: true
j_forced_by_rule_space: false
split_forced_by_rule_space: false
real_qca_branch_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/explore_rule_space.py` runs the E1 bounded rule-space exploration.
It enumerates small exact primitive sets and finite-depth words, then records
which constraints reject them:

```text
words_scanned: 170
j_hits: 73
period_four_hits: 73
forced_candidate_hits: 0
surviving_candidates: 0
load_bearing_qca_bridge: false
```

`scripts/discover_projectors.py` runs the E2 bounded unseeded projector
discovery pass. The default mode does not seed the standard `P_3/P_2`
projectors:

```text
primitive_sets_scanned: 6
algebra_elements_considered: 1924
candidate_projectors: 3
rank_2_projectors: 3
rank_6_projectors: 0
rank_4_projectors: 0
complementary_pairs: 0
unseeded_projector_pairs_found: 0
discovery_verdict: not_found
load_bearing_qca_bridge: false
```

`scripts/rule_to_verdict.py` is the collapsed rule-to-verdict interface. It
takes finite-depth real on-site layer data and reports the Floquet spectrum,
generated algebra center, central idempotents, exact `J` solve status, and one
bridge-candidate boolean. The built-in block-reflection control still does not
pass because `J` is not forced:

```text
central_idempotent_ranks: [0, 4, 6, 10]
complementary_rank_6_4_pairs: 1
forced_j_found: false
pass_rule_to_bridge: false
verdict: candidate_only_j_not_forced
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha.py` is the canonical Floquet-α diagnostic CLI. Its
default variant searches the active physical primitive family and replaces
depth escalation over abstract E1/E2 primitives with one mandatory quantized
resonance layer:

```text
candidate_count: 10
rank_6_4_pair_candidates: 10
rank_one_falsified_candidates: 0
bridge_candidates: 0
verdict_counts: {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha.py --variant plus` extracts the canonical spectral
polarization `J` from the same mandatory Floquet layer:

```text
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

The load-bearing α certificate is exact over `QQ(zeta_12)`: it verifies
`K_alpha = (2U+I)P_alpha` and `K_alpha^2 = -3P_alpha` before deriving the
normalized `J_alpha = K_alpha/sqrt(3)`.

`scripts/floquet_alpha.py --variant time-reversal` checks the Route-3
antilinear-orientation idea with a declared real involution `K`. The declared
`K` is exact, but it is not generated by the Floquet rule algebra and it does
not reduce the compatible `J` choices to global `±J`:

```text
k_real_orthogonal_candidates: 10
k_involution_candidates: 10
k_conjugates_floquet_to_inverse_candidates: 10
k_anticommutes_with_canonical_j_candidates: 10
k_in_generated_algebra_candidates: 0
compatible_j_moduli_dimension_before_k: 9
k_fixed_compatible_j_moduli_dimension: 3
k_fixed_local_compatible_complex_structure_count: 4
k_reduces_to_global_pm_candidates: 0
verdict_counts: {'declared_time_reversal_reduces_moduli_not_unique': 10}
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha.py --variant second-layer` checks the proposed commuting
cycle/swap lock layer:

```text
commuting_second_layer_candidates: 10
order_certified_candidates: 10
compatible_centralizer_collapsed_candidates: 10
generated_algebra_dimension: 10
center_dimension: 10
compatible_centralizer_dimension: 10
explicit_lower_rank_projector_ranks: [2, 2, 2]
no_locking_guardrail_passed_candidates: 0
strict_bridge_candidates: 0
load_bearing_qca_bridge: false
```

This is a checked negative. The layer does reduce the compatible centralizer,
but because it commutes with `U`, its spectral projectors are central
idempotents. The no-locking guardrail rejects the resulting rank-2 projectors.

`scripts/floquet_alpha.py --variant noncommuting` starts the viable Route-1
escape: a block-preserving signed orientation twist `U2` with `[U1,U2] != 0`.
The representative twist preserves the coarse alpha/eta projectors, keeps the
rank `6+4` central pair without lower-rank central idempotents, and reduces
compatible `J` to a finite zero-dimensional set. It still does not pass the
strict bridge because the current checker cannot certify the compatible `J` as
rule-generated/local:

```text
candidate_count: 1
noncommuting_candidates: 1
block_preserving_candidates: 1
coarse_center_preserved_candidates: 1
compatible_j_zero_dimensional_candidates: 1
forced_j_candidates: 0
strict_bridge_candidates: 0
best_compatible_centralizer_dimension: 6
route_label_counts: {'coarse_center_preserved_compatible_j_not_rule_generated': 1}
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha.py --variant noncommuting-gap` extracts the finite `J`
gap for that representative. The four compatible complex structures are exact
pair-orientation matrices with signs
`(+,+,-,+,-)`, `(+,+,-,-,+)`, `(-,-,+,+,-)`, and `(-,-,+,-,+)`. None lies in
the generated algebra, none lies in the rule-local center, and none equals
the spectral-polarization `J` or its negative:

```text
compatible_j_count: 4
generated_algebra_dimension: 22
center_dimension: 3
compatible_centralizer_dimension: 6
compatible_j_moduli_dimension: 0
generated_j_solved: false
generated_complex_structure_count: 0
local_compatible_complex_structure_count: 0
compatible_j_in_generated_algebra_count: 0
compatible_j_in_rule_local_center_count: 0
spectral_polarization_j_matched_count: 0
reason_for_forced_j_failure: compatible_j_finite_but_not_generated_or_rule_local
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha.py --variant noncommuting-exhaustive` exhausts the
discrete block-preserving signed-twist class: `10` alpha/eta mode patterns,
`32` sign patterns per mode pattern, and `12` block-preserving permutation
choices. It finds generated compatible `J` hits, but none in the no-locking
shape. The strict bridge remains false:

```text
candidate_count: 3840
evaluated_symmetry_classes: 96
noncommuting_candidates: 2400
commuting_candidates: 1440
oversized_algebra_rejections: 0
compatible_j_count_distribution: ((4, 560), (8, 1200), (16, 640))
minimal_four_j_candidates: 560
no_locking_shape_candidates: 240
compatible_j_in_generated_algebra_candidates: 720
minimal_four_j_in_generated_algebra_candidates: 240
no_locking_shape_j_in_generated_algebra_candidates: 0
bridge_candidate_count: 0
route_label: discrete_signed_twist_generated_j_hits_fail_no_locking_shape
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha.py --variant noncommuting-completion` performs the
minimal completion experiment: declare one of those finite compatible `J`s as
a third mandatory layer `W`. This does not create lower-rank central
idempotents, but it still does not force a unique `±J`; the rule-local center
contains four block-sign choices:

```text
w_in_previous_generated_algebra: false
w_in_completed_generated_algebra: true
w_in_completed_center: true
central_idempotent_ranks: [0, 4, 6, 10]
lower_rank_central_idempotents: 0
compatible_centralizer_dimension: 4
compatible_j_moduli_dimension: 0
compatible_complex_structure_count: 4
local_compatible_complex_structure_count: 4
strict_unique_j_found: false
pass_completion_to_bridge: false
completion_label: completion_no_lower_rank_but_j_still_block_sign_ambiguous
load_bearing_qca_bridge: false
```

`scripts/spatial_1d_alpha_search.py` is a Route-2 sidecar prototype. It does
not replace `rule_to_verdict`; it checks an exact finite-radius 1D local QCA
layer whose Laurent symbol reconstructs a period-12 root-of-unity transfer,
then asks whether coprime alpha/eta windings couple the remaining four
block-sign choices to global `±J`:

```text
candidate_count: 1
unitary_candidates: 1
coarse_6_4_band_candidates: 1
period: 12
alpha_winding: 4
eta_winding: 3
winding_gcd: 1
winding_lcm: 12
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

`scripts/spatial_1d_alpha_search.py --variant combined` composes the Route-1
on-site noncommuting rule with the Route-2 `(4,3)` winding hops:

```text
U_local = U2 * U1
T(z) = U_local * (P_alpha z^4 + P_eta z^3)
```

This is the direct synthesis experiment. It preserves the Route-1 coarse
center, keeps the finite four-`J` compatible set, and spatial transport
reduces those four signs to the two global `±J` choices. It still does not
pass the strict bridge because the two transported `J`s are not generated by
the joint rule algebra:

```text
qca_shifts: [3, 4]
laurent_orthogonal: true
symbol_unitary_on_samples: true
onsite_central_idempotent_ranks: [0, 4, 6, 10]
onsite_compatible_j_count: 4
transported_compatible_j_count: 2
transported_j_commute_on_samples: true
coefficient_algebra_generates_alpha_eta_projectors: true
joint_rule_algebra_dimension: 22
joint_rule_generated_transported_j_count: 0
topological_pm_shape_candidate: true
strict_bridge_candidate: false
route_label: combined_route_signs_coupled_but_j_not_rule_generated
load_bearing_qca_bridge: false
```

`scripts/spatial_1d_unseeded_search.py` starts the unseeded Route-2 search.
It scans conservative block-blind finite-radius QCA layers, then includes the
projector-shift layer above as a guardrail case that must be rejected because
its coefficients already contain `P_alpha/P_eta`:

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
```

`scripts/defect_beta_search.py` computes round-trip monodromy from wall
transition functions. It is retained as a regression target, but parked as a
load-bearing route until rebuilt as a genuine higher-dimensional defect
calculation. Its monodromy is exactly the matching Floquet-α operator, so the
actual β rule verdict is now computed from the noncommuting entry/exit
transition pair:

```text
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

The β wall transitions now avoid half-angle factors: `T_entry = C`,
`T_exit = M C`, and `T_exit T_entry = M`, so no accidental `sqrt(2)` extension
enters the exact certificate.

The strict obstruction is not a four-sign ambiguity. In both α and β, the
compatible centralizer is `M_3(C) ⊕ M_2(C)` of real dimension `26`, and the
compatible orthogonal complex structures contain a 9-dimensional
`U(3)/O(3) × U(2)/O(2)` family.
The checker treats `forced_j_found` as a rule-generated local-algebra
statement: finite candidate matching is considered only after
`local_compatible_j_moduli_dimension = 0`. For α, the rule-generated local
center has dimension `4` and already produces four `J` candidates. This does
not enumerate all on-site local `J in M_10(R)`; it is a falsifier showing that
the rule's own algebra has not singled out a unique `±J`.

## QCA Input Contract

The current audit reads nontrivial input only from `data/qca_data.json`.
The expected shape is documented in `data/qca_data.schema.json`.
Exact matrix entries must be strings parseable as rational numbers, such as
`"0"`, `"1"`, `"-1"`, or `"1/2"`. Floating-point matrix entries are rejected.
Internal symbolic primitive families may use declared algebraic fields, such
as `QQ(zeta_12)`, but must certify the relevant polynomial identities exactly.

Do not add `data/qca_data.json` unless it contains real source-backed QCA data.

## Development

Default validation is route-focused. Full `uv run pytest -q` is a slow
archival regression suite, not the normal commit gate; see
[project conventions](docs/project_conventions.md).

```bash
uv sync --dev
uv run ruff check .
uv run python scripts/<active_route_check>.py --check
uv run pytest tests/test_<active_route>.py -q
```

Historical and route-specific check catalogue:

```bash
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
uv run python scripts/defect_beta_search.py --check
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

## Claim Boundary

Do not call a representation identity a theorem about QCA geometry. Do not flip
a boolean unless code computes it or a proof derives it. Do not treat "`n = 3`
was chosen" as "`n = 3` was selected." Do not hide hand-chosen complex
structures, SU(5) embeddings, gate restrictions, or within-block projectors.
