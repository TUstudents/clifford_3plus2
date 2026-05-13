# Results

## 2026-05-13

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
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

Verification:

```text
ruff: passed
pytest: 82 passed
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
