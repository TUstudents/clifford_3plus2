# broken_triality — Status

**Status**: closed.  Kill test ran, BT-2 failed.

This sidecar asked:

> Can a broken Z/3 cycle — inherited from the failed exact-triality
> attempt — produce the measured mass hierarchy and CKM CP phase with
> O(few) free parameters, while preserving CPT?

**Answer: no, under the natural Spin(8) embedding and hypercharge-aligned
``v_*``.**  The Yukawa overlap matrix has rank 2 with non-zero
eigenvalue ratio ``360/217 ≈ 1.66`` — essentially flat, far below the
fail threshold of 10.

See ``SESSION_BT_KILL_TESTS.md`` for the full verdict.

## Kill test results

| Kill | Verdict | Detail |
|---|---|---|
| BT-1 (Yukawa overlap structure)  | PASS  | 3 distinct eigenvalues {5/7, 31/72, 0}, rank 2 |
| BT-2 (mass hierarchy)            | **FAIL** | non-zero ratio 1.66, below threshold 10 |
| BT-3 (CP phase)                  | skipped | program closed at BT-2 |
| BT-4 (parameter audit)           | skipped | program closed at BT-2 |

## What was built

- ``__init__.py`` — package init.
- ``reuse.py`` — re-exports from ``triality/`` (which re-exports from
  ``lepton/``).  No new Clifford / Pati-Salam / octonion code.
- ``yukawa_overlaps.py`` — BT-1 implementation.  Triality orbit on a
  starting vector, SM Cartan projection, 3×3 overlap matrix, audit
  payload.
- ``mass_hierarchy.py`` — BT-2 implementation.  Non-zero eigenvalue
  ratio against thresholds.
- ``parameter_ledger.md`` — choices made.
- ``PLAN.md`` — four-kill-test plan.
- ``SESSION_BT_KILL_TESTS.md`` — full verdict.
- 17 passing tests.

## What was NOT built

- ``cp_phase.py`` (BT-3) — not implemented; program closed at BT-2.
- A formal ``parameter_count_audit.py`` (BT-4) — not implemented;
  program closed at BT-2.

## Combined message with exact-triality result

Both the exact-triality (``../triality/``) and broken-triality
(this sidecar) programs died on the same structural issue, framed
two ways:

- **Exact-triality K1**: Spin(8) triality does not preserve the
  Pati-Salam-aligned SM Cartan as a subspace.
- **Broken-triality BT-2**: Triality projection of the SM Cartan
  produces an essentially flat (and rank-deficient) Yukawa.

Both reflect: **Spin(8) triality and the Pati-Salam factorization are
not aligned**.  Triality is symmetric across the four Cartan elements;
Pati-Salam picks out one of them.

## What remains open

The negative result does not address:

- Non-natural ``v_*`` (different starting vector with no H_1↔H_2
  symmetry).
- Non-natural Spin(8) embeddings.
- Discrete flavor symmetries with no Spin(8) origin (A_4, T', etc.).
- The "approximate embedding from BCC anisotropy" hope from the user's
  original spec — this is orthogonal to triality entirely.
- The Furey octonion three-copy program.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/broken_triality/tests -q
```

Expected: 17 passing.

## Cross-module dependency

Imports from ``triality/`` only.  No dependency on ``spacetime_qca/``.
No new lepton imports beyond what ``triality/`` already exposes.
