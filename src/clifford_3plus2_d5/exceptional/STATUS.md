# exceptional — Status

**Status**: closed.  All four candidate mechanisms produced clean
negative verdicts.  Three SM generations do NOT emerge from the
exceptional-algebra family under Spin(10) representation theory.

See ``SESSION_EXCEPTIONAL.md`` for the full verdict report.

## Result summary

| Phase | Candidate | Verdict |
|---|---|---|
| 0a  | Bi(O) bimultiplication      | KILL (= Spin(8), inherits triality fail) |
| 0b  | Three Fano lines through e_7 | KILL (no Lie closure; overlapping SU(2)s) |
| 2   | J_3(O) under Spin(10)        | KILL (27 = 16 + 10 + 1; 3 × 16 > 24 dim) |
| 2b  | J_3^C(O) under Spin(10) × U(1) | KILL (54 = 16 + 16* + 10 + 10* + 1 + 1*) |

Combined: **FULL KILL of the exceptional-algebra approach to three
generations**.

## What was built

- ``__init__.py`` — package init.
- ``reuse.py`` — single import surface (octonion + Pati-Salam + SM helpers from lepton).
- ``bimultiplication.py`` — Phase 0a triage.
- ``fano_lines.py`` — Phase 0b triage.
- ``j3o_algebra.py`` — Phase 1: 3×3 Hermitian octonion matrices,
  Jordan product, cubic norm, basis.
- ``spin10_on_j3o.py`` — Phase 2 kill test (27 = 16 + 10 + 1).
- ``j3o_complex.py`` — Phase 2b kill test (J_3^C(O) extension).
- ``PLAN.md`` — original audit design.
- ``SESSION_EXCEPTIONAL.md`` — full verdict report.
- ``parameter_ledger.md`` — choices made.
- ``tests/`` — 39 passing tests.

## What was NOT built (Phase 3, deferred indefinitely)

Phase 3 (Yukawa + CP across generations) was gated on Phase 2 OR
Phase 2b producing a three-generation structure.  Both closed negative,
so Phase 3 is unwritten.  If a future investigation finds a
three-generation mechanism (Bold D topological, or other), Phase 3's
infrastructure can be built on top of the existing j3o_algebra.

## Combined message with the triality family

Three sidecars investigating three-generation derivation have now closed
negative:

| Sidecar | Mechanism | Status |
|---|---|---|
| ../triality/ | Spin(8) triality | K1 FAIL |
| ../broken_triality/ | Broken Z/3 Yukawa | BT-2 FAIL |
| ../exceptional/ | Exceptional Jordan family | ALL 4 sub-candidates FAIL |

**Cumulative signal**: three generations is unlikely to emerge from
algebraic structure alone.  The program's honest direction:

1. Treat three generations as empirical input.
2. Investigate non-algebraic mechanisms (topology, anomaly, RG) only
   with new theoretical input justifying the attempt.
3. Focus positive content on what IS derived: one SM generation
   (lepton), BCC Dirac kinematics (spacetime_qca), CP-from-quantization
   T_{2g} cubic-anisotropy (cp).

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/exceptional/tests -q
```

Expected: 39 passing.

## Cross-module dependency

Imports octonion / Clifford / Pati-Salam / SM helpers from ``lepton``
through ``reuse.py``.  No dependency on ``triality`` /
``broken_triality`` / ``cp``.  Each sidecar's verdict is independent.
