# koide — Status

**Status**: CLOSED — **KOIDE CONSISTENT** (PDG NOT IN LOCUS).

See ``PLAN.md`` for the full design and decision tree, and
``SESSION_KOIDE.md`` for the combined verdict report.

## Phase status

| Phase | Status | Verdict |
|---|---|---|
| Phase KO-1 — empirical Koide + 45° cone geometry      | done | K_PDG = 0.666661, deviation 6×10⁻⁶ from 2/3; three forms agree |
| Phase KO-2 — BCC body-diagonal Z₃ on σ^a-axes         | done | R fixes (1,1,1)/√3; trace + 2D-non-trivial Z₃ irrep projectors |
| Phase KO-3 — BCC-Z₃-orbit 3×3 Yukawa locus            | done | Always 2-fold degenerate eigenvalues; cone IFF |v_t|/|v_o| = 3+2√2 |
| Phase KO-4 — cone vs. locus comparison                | done | L ∩ C = 1-parameter family; L ⊄ C; PDG ∉ L |
| Phase KO-5 — combined audit + SESSION report          | done | KOIDE CONSISTENT verdict aggregated and reported |

## Cross-module dependency

Imports from cp/ (T_{2g} projector, H^(1), J-decomposition, Higgs
map basis), topology/ (BCC body-diagonal rotation), and
broken_triality/ (3×3 Yukawa-from-orbit template).

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/koide/tests -q
```
