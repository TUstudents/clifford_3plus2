# Results

## 2026-05-13

```bash
uv run python scripts/branching_check.py --check
```

```text
branching_check_passed: true
load_bearing_qca_bridge: false
```

This verifies standard Spin(10) branching only. It does not prove the QCA
supplies the split.

```bash
uv run python scripts/qca_split_audit.py --check
```

```text
qca_split_audit_verdict: notation_only
qca_supplies_structural_3plus2_split: false
complex_structure_compatible_with_3plus2_split: false
load_bearing_qca_bridge: false
```

No `data/qca_data.json` is present, so there is no bridge claim.
