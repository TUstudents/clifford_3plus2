# sme — Status

**Status**: CLOSED — UNFALSIFIABLE PASS at ε ≲ 2 × 10⁻³³ m
(~10² × Planck length).

See ``PLAN.md`` for the full design and decision tree, and
``SESSION_SME.md`` for the combined verdict.

## Phase status

| Phase | Status | Verdict |
|---|---|---|
| Phase A-1 — SME framework identification     | done | dim-5 non-minimal SME, fermion sector, CP-odd spin-tensor |
| Phase A-2 — H^(1) → SME tensor mapping       | done | maps to 3 non-zero d^{(5)} components (axial-vector × 2 derivatives, CPT-even) |
| Phase A-3 — SME_LITERATURE_NOTE.md           | done | representative bound |d^(5)| ≲ 10⁻¹⁷ GeV⁻¹ (KR entry-id verification pending) |
| Phase A-4 — symbolic ε constraint            | done | ε ≲ 2 × 10⁻³³ m → UNFALSIFIABLE PASS |
| Phase A-5 — combined audit + SESSION report  | done | program alive but unfalsifiable in this channel |

## Cross-module dependency

Consumes the H^(1) (CP-odd, T_{2g}) cell from `cp/continuum_cp.py` and
the T_{2g} projector from `cp/cubic_harmonics.py`.  Uses γ matrices
from `spacetime_qca.dirac` for the bilinear projection in Phase A-2.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/sme/tests -q
```
