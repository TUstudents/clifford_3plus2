# cp — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`cp` is a small (~2k lines, 57 tests), exact, and unusually self-honest module
that audits CP/T structure of the Bialynicki-Birula BCC Dirac walk. Its central
result is genuine and I verified it: the walk preserves **CPT, P, CT** exactly
and breaks **T, C, PT, CP** at the lattice, and the leading O(ε) lattice
correction `H^(1)` is **100% CP-odd and lives entirely in the T₂g cubic-harmonic
irrep** (`‖H^(1)‖²=12`, all of it in CP-odd×T₂g), vanishing in the continuum.
The module also contains documented self-corrections that *raise* confidence.
Main caveat is presentational: the "dual positive" branding leans on a second
"slot" (β, the J-misalignment 50/50 result) that the module itself reclassified
as an *algebraic* property, not a physical-CP measurement.

- **Verdict:** one clean, verified, physically-meaningful CP result (α-continuum
  T₂g) + one honest algebraic observation (β). Correctly scoped, reliable as the
  shared foundation for `sme`/`strongcp`/`koide`.
- **Confidence:** high on the α-continuum result and discrete-symmetry pattern;
  low on β *as CP* (the module agrees).

## What it claims

PROJECT_STATUS: *"dual-positive verdict: CP-violation at O(ε) in T₂g."* STATUS:
*"DUAL ROBUST PASS … two independent CP slots."* The α slot is a physical CP
statement; the β slot is, per the module's own ledger, an algebraic one.

## Progress

| Audit | Verdict | Finding |
|---|---|---|
| α-2 (bare massless walk) | PASS | CPT, P, CT exact; T, C, CP, PT broken at lattice |
| α-3 (Yukawa-perturbed) | confirms | trivial internal action preserves the pattern |
| **α-cont (O(ε))** | **PASS** | `H^(1)` purely CP-odd, entirely T₂g, `‖·‖²=12` |
| β (J-misalignment, basis[0]) | PASS | 50/50 J-commuting vs J-anticommuting |
| β-multi (all 8 elements) | ROBUST | 50/50 split universal across dim-4 Higgs space |

Tests: 57 passing (13 discrete + 9 walk + 11 J-misalignment + 10 cubic-harmonic
+ 14 continuum-CP).

## Assumptions / inputs

- **Discrete operators (standard, chiral basis):** P=γ⁰, T=γ²γ⁰ (antiunitary),
  C=iγ² (antiunitary). The C definition was *corrected* on 2026-05-20 from a
  Bloch-level particle-hole matrix to the standard physical `Cγ^μC⁻¹=−(γ^μ)^T`.
- **BB walk** from `spacetime_qca`; **Higgs-like map** and **J** from `lepton`
  (so β inherits `lepton`'s declared `J`).
- **Trivial internal action** `S_internal = I` for the discrete symmetries.
- **Zero continuous parameters; six discrete choices**, all forced once
  conventions are fixed — "falsifiability maximal," fully exact-symbolic.

## Soundness

Everything I checked is exact and correct.

- **Discrete-symmetry pattern** — I ran it: `{P, CT, CPT}` exact, `{T, C, PT,
  CP}` broken. This is physically correct: CPT preserved as required, and the
  BB complex coefficients `q_± = (1±i)/4` break the antiunitary T (hence C, CP,
  PT) at the lattice while the continuum `α·k` restores CP. ✔
- **α-continuum** — I ran it: `H^(1)` Hermitian, `‖H^(1)‖²=12`, and the CP×irrep
  norm table is `{(CP-odd, T₂g): 12}` with **every other cell exactly 0**. So
  the result is not approximate or fractional — it is a *clean* 100% CP-odd,
  100% T₂g localization. ✔
- **Self-corrections (strong positive signal):** the 2026-05-20 convention audit
  (a) fixed a CP-action bug — the momentum-flip `k→−k` was omitted, which gave
  the *right* verdict on degree-2 `H^(1)` "by coincidence" but misclassified
  `H^(0)=α·k` (CP-even → CP-odd); and (b) renamed the β metric
  `cp_violating_fraction → j_anticommuting_fraction` to drop an unjustified
  physical-CP identification. Both are exactly the kind of catch-your-own-error
  hygiene that increases trust.

No soundness problems. The cubic-harmonic O_h projector machinery (A₁g⊕Eg⊕T₂g
at degree 2) is standard group theory, applied correctly.

## Novelty

- **Genuine (if narrow):** the specific observation that the BB walk's *leading*
  lattice correction is purely CP-odd and sits in a *single* cubic irrep (T₂g),
  with CPT exact and continuum CP recovery, is a neat, exact result about a
  known (1994) walk — and it is the precise bridge `sme` uses to map to SME
  T₂g/`d^(5)` operators.
- **Standard:** discrete-symmetry analysis of lattice Dirac operators and O_h
  cubic-harmonic decomposition are textbook; the contribution is the clean
  application + localization, not new machinery.
- **β** is more an algebraic curiosity (Frobenius 50/50 split under the chosen
  J); the module correctly stops short of calling it CP violation.

## Gaps

1. **Structural only** — no magnitude vs experiment (correctly deferred to
   `sme`, which finds the ε bound is ~10⁸ beyond reach).
2. **"Dual" framing is half-earned** — after the β reclassification, there is
   *one* physical CP slot (α-continuum T₂g) and one algebraic observation (β).
   The "DUAL ROBUST PASS / two independent CP slots" wording in STATUS slightly
   over-sells; a careful reader should treat β as algebra, which the ledger now
   states.
3. **No three-generation context** — so no link to CKM δ, ε_K (the flavor
   sidecars `triality`/`broken_triality` failed; CP here has no flavor home).
4. **Degree-2 / O(ε) only** — higher-order corrections not pursued (scoped out).
5. **Inherited `J`** — β rests on `lepton`'s declared complex structure.

**Highest-leverage next step:** none urgent — the module is a finished, correctly
scoped foundation. If extended, the physically interesting direction is the one
`sme` already took: pin the T₂g CP-odd operator to an SME coefficient and a
lattice spacing, turning the clean structural result into a (currently
unfalsifiable) bound.

## Confidence (calibrated)

- α-continuum (`H^(1)` CP-odd, T₂g, O(ε)): **high** — exact, verified, clean.
- Discrete-symmetry pattern: **high** — verified, physically correct.
- β as a *CP* result: **low** (it's algebraic; module agrees); as an algebraic
  fact: high but of limited significance.
- As the import surface for `sme`/`strongcp`/`koide`: **high** — stable and
  correct.

## Verdict

`cp` is one of the cleanest modules in the workspace: small, fully exact, and
written by someone willing to correct their own conventions in public. Its real
result — the Bialynicki-Birula walk's leading lattice correction is purely
CP-odd and entirely T₂g, with CPT exact and continuum CP restored — is genuine,
verified to the digit, and physically sensible, and it is exactly the hook the
downstream sidecars need. The self-corrections (the CP-action k-flip fix, the β
de-physicalization) are a credit to the module and raise my confidence rather
than lower it. The only thing to read carefully is the "dual" branding: the
truly physical content is the single α-continuum CP slot; the β 50/50 result is
an algebraic property of the Higgs-map space under a declared J, not a second
independent CP violation — and the module's own ledger now says so.
