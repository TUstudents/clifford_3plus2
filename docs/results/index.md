# Results

## 2026-05-13

Commands:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

Verification:

```text
ruff: passed
pytest: 34 passed
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
