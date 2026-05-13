# Results

## 2026-05-14

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
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

Verification:

```text
ruff: passed
pytest: 140 passed
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
