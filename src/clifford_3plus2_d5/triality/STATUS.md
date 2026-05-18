# triality — Status

**Status**: closed.  Kill test ran, K1 failed.

This sidecar asked one load-bearing question:

> Can explicit Spin(8) triality produce three equivalent SM-generation
> carriers without declaring three generations by hand?

**Answer: no, under the natural Pati-Salam-aligned Spin(8) embedding.**

See ``SESSION_T_KILL_TEST.md`` for the full verdict.

## Result summary

K1 (Cartan necessary condition): **FAIL** on all 3 SM Cartan generators.

Under the pinned ``Spin(8) ⊂ Spin(10)`` embedding (gamma indices
``{0..7}``, Cartan ``H_k = (1/2) γ_{2k} γ_{2k+1}``), triality maps each
of the three SM-inside-Spin(8) Cartan generators to a vector outside the
SM Cartan subspace:

```text
SU(3)_c v_0 = (-1, 1, 0, 0)  →  ( 0, 0, -1, 1)        outside
SU(3)_c v_1 = (-1, 0, 1, 0)  →  ( 0, -1, 0, 1)        outside
Y'         = (1/3,1/3,1/3,1/2) → (3/4,-1/12,-1/12,-1/12)  outside
```

Therefore the Spin(8) triality outer automorphism does not preserve the
SM-inside-Spin(8) subalgebra, and the three triality-rotated chiral-16
carriers cannot represent three equivalent SM generations.

## What was built

- ``__init__.py`` — package init.
- ``reuse.py`` — single import surface from ``lepton``; no duplication.
- ``spin8_triality.py`` — 28 Spin(8) generators on chiral-16, Cartan
  basis, triality Cartan matrix `T_cartan`, witness predicates.
- ``sm_restriction.py`` — restricted hypercharge `Y'`, SU(3)_c Cartan
  extraction, K1 test with witness diagnostics, K2 Y' spectrum, audit
  payload.
- ``parameter_ledger.md`` — every choice made (2 free, 2 forced).
- ``PLAN.md`` — load-bearing question, kill-test definitions, failure
  modes.
- ``SESSION_T_KILL_TEST.md`` — verdict report.
- 21 passing tests.

## What was NOT built (deferred because K1 failed)

The original sidecar sketch included:

- mass / Yukawa hierarchy from triality breaking;
- Z/3 character analysis for CP phase;
- PDG numerical fit for CKM/PMNS;
- breaking pattern parameterization;
- multiple session reports T1..T6.

All deferred indefinitely.  None of these make sense without a triality
that preserves the SM, and the K1 result shows there is no such triality
on the natural Spin(8) embedding.

## What remains open

The K1 failure does not address:

- whether a non-natural Spin(8) embedding might pass K1 (untested);
- whether a different ambient group with Z/3 outer automorphism could
  carry triality-based generation structure (none of the classical
  simple Lie groups beyond Spin(8) have one, by Dynkin inspection);
- whether triality as an *approximate* / *broken* symmetry could give
  the observed three-generation pattern with mass hierarchy and CP from
  the breaking — this is a different program (the "CP from quantization"
  hope) and is not addressed by this kill test.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/triality/tests -q
```

Expected: 21 passing.

## Cross-module dependency

This sidecar imports from ``clifford_3plus2_d5.lepton`` only through
``reuse.py``.  It does not import from ``spacetime_qca``.  The spacetime
side is orthogonal to the (failed) triality question.
