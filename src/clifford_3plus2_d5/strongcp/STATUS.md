# strongcp — Status

**Status**: STRUCTURAL + DIRECT-COMPUTATION ARGUMENT COMPLETE.
Verdict: STRONG-CP TRIVIAL at O(ε) and O(ε²); SAFE at higher
orders; SC-4 direct lattice-gauge computation CONFIRMS.

See ``PLAN.md`` for the full design and decision tree, and
``SESSION_STRONGCP.md`` for the combined verdict.

## Phase status

| Phase | Status | Verdict |
|---|---|---|
| Phase SC-1 — degree-3 cubic harmonics            | done | A_{2u} ⊕ T_{2u} ⊕ T_{1u} projectors built; idempotent + orthogonal + complete |
| Phase SC-2 — BCC centrosymmetry audit            | done | lattice + BB walk + Bloch symbol all centrosymmetric |
| Phase SC-3 — higher-order H^(n) parity audit     | done | H^(2) is 100% T_{1u} with ZERO A_{2u} content |
| Phase SC-4 — lattice topological-charge density  | done | spatial-only Q dimensionally trivial; 6-dim BCC plaquette rep parity-even; A_{2u} = 0 for SU(2)_L, SU(2)_R, SU(4)_PS — CONFIRMS |
| Phase SC-5 — chiral anomaly + θ̄ shift            | done | tr(γ^5 H^(n)) = 0 for n = 1, 2; no chiral rotation |
| Phase SC-6 — combined audit + SESSION report     | done | STRONG-CP TRIVIAL verdict aggregated and reported |

## Cross-module dependency

Imports from `cp/` (cubic harmonics degree-2 baseline, H^(1), BCH
machinery, P/T/C operators) and `spacetime_qca/` (BCC geometry, BB
walk, Wilson plaquettes, continuum expansion).

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/strongcp/tests -q
```
