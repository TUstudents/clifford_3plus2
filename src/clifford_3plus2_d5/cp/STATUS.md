# cp — Status

**Status**: DUAL ROBUST PASS — strengthened with multi-element + O(ε)
audits.  See ``SESSION_CP_ALPHA_BETA.md`` and ``SESSION_CP_ORDER_EPS2.md``.

## Result summary

| Audit | Verdict | Key finding |
|---|---|---|
| alpha-2 (bare walk symmetries) | **PASS**  | CPT, P, CT exact; CP, T, C, PT broken at lattice |
| alpha-3 (Yukawa-perturbed)     | confirms  | trivial internal action preserves same pattern |
| **alpha-cont (O(ε) continuum)**| **PASS**  | H^(1) is 100% CP-odd, lives entirely in T_{2g} cubic-harmonic irrep |
| beta (J-misalignment, basis[0])| **PASS**  | Higgs map basis[0] has 50/50 J-commuting vs J-anticommuting |
| **beta-multi (all 4 basis)**   | **ROBUST PASS** | All 8 elements (4 basis + 4 transposes) give CP-violating fraction = 1/2 |

This is the strongest CP result the program has produced.  Two
independent CP slots, both structurally clean:

1. **Walk-level**: lattice CP violation at O(ε), localized in the
   T_{2g} cubic-harmonic irrep.  CPT preserved; CP vanishes in
   continuum limit ε → 0 with specific angular signature.
2. **Algebra-level**: the dim-4 Higgs-like space has universal 50/50
   J-commuting vs J-anticommuting Frobenius content.  Robust across
   all 8 tested elements.

## What exists

- ``__init__.py`` — package init.
- ``reuse.py`` — single import surface.  No duplication.
- ``discrete_symmetries.py`` — P, T, C operators + 7-symmetry composite
  framework + γ-matrix conjugation patterns.
- ``walk_symmetries.py`` — alpha-2 (massless audit) and alpha-3
  (Yukawa-perturbed audit).
- ``j_misalignment.py`` — beta audit: J candidate enumeration,
  J-decomposition of the Higgs map, CP-violating fraction, multi-element
  audit over the full dim-4 basis.
- ``cubic_harmonics.py`` — minimal O_h projector framework for degree-2
  momentum polynomials (A_{1g}, E_g, T_{2g}).
- ``continuum_cp.py`` — O(ε) effective-Hamiltonian extraction via BCH +
  CP × cubic-harmonic decomposition.
- ``PLAN.md`` — original audit design.
- ``SESSION_CP_ALPHA_BETA.md`` — baseline verdict report.
- ``SESSION_CP_ORDER_EPS2.md`` — multi-element β + O(ε) continuum report.
- ``parameter_ledger.md`` — choices made.
- ``tests/`` — 57 passing tests (29 baseline + 4 multi-element + 10
  cubic-harmonic + 14 continuum-CP).

## Headline findings

**Alpha (walk symmetries)**: the bare massless BCC Dirac walk has
lattice-level CP-violation that vanishes in the continuum limit.  The
Bialynicki-Birula construction's complex ``q_± = (1±i)/4`` coefficients
select a chirality direction, breaking T/C/CP/PT at the lattice while
preserving CPT, P, CT.

**Alpha-continuum (O(ε))**: the leading lattice correction
``H^(1)(k) = i (B_2 - B_2^†)/2`` is **purely CP-odd** (CP-violating
fraction = 1) and **localized in the T_{2g} cubic-harmonic irrep**
(``k_x k_y, k_y k_z, k_z k_x``).  Norm ``||H^(1)||² = 12``.

**Beta (J-misalignment)**: the Session 23 Higgs-like internal map ``M``
satisfies ``||M_c||² = ||M_a||² = 128`` (Frobenius), giving CP-violating
fraction **exactly 1/2**.  M has maximal CP mixing under the chosen J.

**Beta-multi**: this 50/50 mixing is **universal** across the dim-4
Higgs space.  All 4 basis elements + 4 transpose components yield
CP-violating fraction = exactly 1/2.

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
