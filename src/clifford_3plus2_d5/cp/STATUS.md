# cp — Status

**Status**: DUAL PASS.  Both alpha and beta audits produce positive
CP/CPT structure from the existing infrastructure.  See
``SESSION_CP_ALPHA_BETA.md`` for the full verdict.

## Result summary

| Audit | Verdict | Key finding |
|---|---|---|
| alpha-2 (bare walk symmetries) | **PASS**  | CPT, P, CT exact; CP, T, C, PT broken at lattice |
| alpha-3 (Yukawa-perturbed)     | confirms  | trivial internal action preserves same pattern |
| beta (J-misalignment)          | **PASS**  | Higgs map has 50/50 J-commuting vs J-anticommuting parts |

This is the strongest CP result the program has produced.  The
CP-from-quantization hope is **vindicated at the structural level**:
the existing infrastructure contains two independent CP-violating,
CPT-preserving slots.

## What exists

- ``__init__.py`` — package init.
- ``reuse.py`` — single import surface.  No duplication.
- ``discrete_symmetries.py`` — P, T, C operators + 7-symmetry composite
  framework + γ-matrix conjugation patterns.
- ``walk_symmetries.py`` — alpha-2 (massless audit) and alpha-3
  (Yukawa-perturbed audit).
- ``j_misalignment.py`` — beta audit: J candidate enumeration,
  J-decomposition of the Higgs map, CP-violating fraction.
- ``PLAN.md`` — original audit design.
- ``SESSION_CP_ALPHA_BETA.md`` — combined verdict report.
- ``parameter_ledger.md`` — choices made.
- ``tests/`` — 29 passing tests.

## Headline findings

**Alpha (walk symmetries)**: the bare massless BCC Dirac walk has
lattice-level CP-violation that vanishes in the continuum limit.  The
Bialynicki-Birula construction's complex ``q_± = (1±i)/4`` coefficients
select a chirality direction, breaking T/C/CP/PT at the lattice while
preserving CPT, P, CT.

**Beta (J-misalignment)**: the Session 23 Higgs-like internal map ``M``
satisfies ``||M_c||² = ||M_a||² = 128`` (Frobenius), giving CP-violating
fraction **exactly 1/2**.  M has maximal CP mixing under the chosen J.

## What this DOES NOT yet prove

This audit is structural, not phenomenological.  Open:

- Continuum-limit suppression order of the alpha CP-violation vs
  experimental Lorentz-violation bounds.
- Robustness of the beta 50/50 mixing across the dim-4 Higgs basis.
- Three-generation embedding (the triality / broken-triality sidecars
  both failed; this CP structure has no flavor-context yet).
- Magnitude matching to measured CKM CP-violation parameters.

Each is a separate, larger investigation.  This audit only establishes
the **existence** of CP slots.

## What remains open

The cp sidecar produces structural verdicts only.  Follow-up work could
include:

- Continuum expansion of the BCC walk to identify the ``O(ε^n)`` order
  of CP-violating corrections.
- Multi-element audit of the dim-4 Higgs space (test all 4 basis
  elements, not just the first).
- Pursue a three-generation mechanism independent of triality (e.g.,
  Furey octonions, family Spin(8), explicit copies) and combine with the
  beta CP slot.
- Magnitude estimates and comparison to PDG.

Each is its own scope.  The cp module remains as a stable foundation for
any of these.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/cp/tests -q
```

Expected: 29 passing.

## Cross-module dependency

Imports from ``spacetime_qca``, ``lepton`` (via ``triality.reuse`` and
direct).  No dependency on ``triality`` or ``broken_triality`` results —
the cp audit is independent of both.
