# Handover Compliance

This document checks the implemented repository against
[the archived agent handover](archive/clifford_3plus2_empty_repo_agent_handover.md).

## Status Summary

```text
handover_short_term_success: complete
phase_0_audit_contract: complete
real_qca_bridge: notation_only
open_handover_implementation_points: none
```

No real `data/qca_data.json` is present. That is intentional. Without real
QCA data, the bridge claim must remain blocked.

The original handover is archived because its implementation checklist is
closed. Future work should follow [Roadmap And Working Index](roadmap.md).

## Checklist

| Handover item | Status | Implementation |
| --- | --- | --- |
| Highest mantra and claim boundary | Implemented | [README](../README.md), [Theory Summary](theory.md), [Falsifiers](falsifiers.md) |
| `uv` package scaffold | Implemented | `pyproject.toml`, `uv.lock`, `src/clifford_3plus2_d5/` |
| README requirements | Implemented | [README](../README.md) |
| Core `QCASplitAudit` dataclass | Implemented | `src/clifford_3plus2_d5/status.py` |
| Default values do not pass | Implemented | `tests/test_clifford_audit.py` |
| Phase A branching check | Implemented | `src/clifford_3plus2_d5/exterior.py`, `src/clifford_3plus2_d5/branching.py`, `scripts/branching_check.py` |
| Branching script marks result non-load-bearing | Implemented | `tests/test_scripts.py` |
| Phase B default-failing QCA audit | Implemented | `src/clifford_3plus2_d5/clifford_audit.py`, `scripts/qca_split_audit.py` |
| Exact rational matrix parsing | Implemented | `src/clifford_3plus2_d5/gate_algebra.py` |
| `data/qca_data.json` contract | Implemented | `data/qca_data.schema.json` |
| Missing, incomplete, or invalid QCA data returns `notation_only` | Implemented | `tests/test_clifford_audit.py`, `tests/fixtures/qca/` |
| Explicit `J` checks | Implemented | `J^2=-I`, split preservation, allowed-gate inclusion |
| Hand-chosen `J` cannot pass | Implemented | `tests/fixtures/qca/hand_chosen_j.json` |
| Unknown `J` cannot pass | Implemented | `tests/fixtures/qca/unknown_j_origin.json` |
| Phase C gate algebra audit | Implemented | `src/clifford_3plus2_d5/gate_algebra.py`, `tests/test_gate_algebra.py` |
| Off-block gates fail | Implemented | `tests/fixtures/qca/off_block_gate.json` |
| Color-basis projectors fail | Implemented | `tests/fixtures/qca/color_projector_gate.json` |
| Phase D documentation | Implemented | [Roadmap](roadmap.md), [Theory Summary](theory.md), [Falsifiers](falsifiers.md), [Results](results/index.md) |
| Reproducible command suite | Implemented | Historical Phase 0 suite: Ruff, pytest, branching script, QCA audit script |

## Intentional Deferrals

The handover distinguishes infrastructure from a real theorem. These items are
not open handover points; they are later scientific milestones:

- Real `data/qca_data.json`.
- Proof that QCA supplies a structural two-plane.
- Proof that QCA supplies a non-hand-chosen `J`.
- Closed one-generation bridge theorem.
- Integration into the main BCC-QCA program.

## Synthetic Positive Fixture

No synthetic `structural_bridge` fixture is included in Phase 0. The current
audit intentionally avoids a fake bridge-pass example because a real pass must
come from explicit QCA data and a resolved representation-space convention for
Clifford generators, `J`, projectors, and gate generators.

Synthetic safe/unsafe gate checks are present in unit tests, but they are not
claimed as QCA evidence.

## Phase 0 Closure Criteria

This is the historical Phase 0 closure record. Current validation convention is
route-focused and documented in [Project Conventions](project_conventions.md);
the full pytest suite is no longer the normal commit gate.

Phase 0 is closed when all commands pass:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

Expected scientific status after Phase 0:

```text
branching_check_passed: true
qca_split_audit_verdict: notation_only
load_bearing_qca_bridge: false
```
