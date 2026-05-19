# topology — Status

**Status**: CLOSED — all four phases produced negative verdicts.

See ``PLAN.md`` for the full design and decision tree, and
``SESSION_TOPOLOGY.md`` for the combined verdict report.

## Phase status

| Phase | Status | Verdict |
|---|---|---|
| Phase D-1 — BCC body-diagonal Z_3   | done | TOPOLOGY KILL (chiral-16 internal Z_3-trivial; BB hops not equivariant — orthogonal finding) |
| Phase D-2 — color Z_3 center        | done | COLOR Z_3 KILL (16 = 8 + 4 + 4 asymmetric, not three equal generations) |
| Phase D-3 — π_3 literature note     | done | PI3 KILL (no carrier-relevant coset has Z/3 torsion in π_3) |
| Phase D-5 — discrete anomaly        | done | ANOMALY KILL (cancellation per generation; no constraint on N) |

## Cross-module dependency

Imports from ``spacetime_qca`` (BCC walk, Dirac gammas) and ``lepton``
(chiral-16, SU(3)_c, hypercharge). No new octonion / Clifford code.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/topology/tests -q
```
