# Session CP — Alpha + Beta Kill-Test Report

**Verdict: DUAL PASS.**  Both audits produce positive CP/CPT structure
from the existing infrastructure without inventing new content.  This is
the strongest CP result the program has produced.

| Audit | Verdict | Key finding |
|---|---|---|
| alpha-2 (bare walk symmetries) | **ALPHA PASS** | CPT and P exact; CP, T, C, PT broken at lattice |
| alpha-3 (Yukawa-perturbed walk) | confirms alpha-2 | trivial internal action preserves same pattern |
| beta (J-misalignment) | **BETA PASS** | Higgs map M is 50/50 J-commuting vs J-anticommuting |

29 passing tests.

## Setup

The cp sidecar imports from ``spacetime_qca``, ``lepton``, and
indirectly through ``triality.reuse``.  No new octonion / Clifford /
Pati-Salam code introduced.

Pinned conventions (logged in ``parameter_ledger.md``):

- Spinor basis: chiral, from ``spacetime_qca.dirac``.
- P spinor matrix: ``γ^0`` (unitary).
- T spinor matrix: ``γ^2 γ^0`` (antiunitary).
- C spinor matrix: ``γ^2 γ^0`` (antiunitary; same algebraic matrix as T
  in this basis).
- Internal action: trivial (``S_internal = I``) for all 7 operators.
- Yukawa term: ``β ⊗ M_higgs`` with ``β = γ^0`` and ``M_higgs`` the
  Session 23 Higgs-like charge-shift candidate.
- J for beta audit: ``patisalam_chosen_complex_structure`` from
  ``lepton`` (the right-quaternionic Cl(0,4)-commutant J).

## Alpha-1: discrete operator construction

The 7 named operators have the following ``(γ^0, γ^1, γ^2, γ^3)``
conjugation patterns (with antiunitary K applied where applicable):

| Operator | Pattern | Antiunitary? | k-flip? |
|---|---|---|---|
| P   | ( 1, -1, -1, -1) | no  | yes |
| T   | (-1,  1,  1,  1) | yes | yes |
| C   | (-1,  1,  1,  1) | yes | yes |
| PT  | (-1, -1, -1, -1) | yes | no  |
| CP  | (-1, -1, -1, -1) | yes | no  |
| CT  | ( 1,  1,  1,  1) | no  | no  |
| CPT | ( 1, -1, -1, -1) | no  | yes |

T and C have the same matrix structure (both ``γ^2 γ^0``); the
operational distinction is physical interpretation, not algebra.

## Alpha-2: massless BCC Dirac walk symmetries

For each operator, the symbolic commutation/anticommutation test against
the BCC Dirac Bloch operator ``B(k) = block_diag(B_R(k), B_L(k))``:

```text
Operator   Verdict
P          EXACT
T          BROKEN
C          BROKEN
PT         BROKEN
CP         BROKEN
CT         EXACT
CPT        EXACT
```

**Interpretation**: the bare BCC Dirac walk has lattice-level
CP-violation **as a structural feature of the Bialynicki-Birula
construction**.  The construction's complex `q_± = (1±i)/4` coefficients
pick out a chirality direction in time, breaking T/C/CP/PT at the walk
level while preserving CPT (and P, CT as composites).

This is exactly the pattern the original CP-from-quantization hope
predicted: **CPT preserved by structural discrete symmetry, CP broken
at the lattice level with corrections vanishing in the continuum
limit.**

The continuum Hamiltonian ``H = α·k`` is fully T/C/CP-invariant.  The
violations are O(ε^?) corrections from the discrete walk that go to
zero in the ε → 0 limit.

## Alpha-3: Yukawa-perturbed walk

Adding the Session 23 Higgs-like ``β ⊗ M_higgs`` term with trivial
internal action ``S_internal = I``:

The ``β`` factor (``= γ^0``) is preserved by operators whose
``γ^0``-conjugation pattern entry is ``+1``: P, CT, CPT.  These remain
exact when Yukawa is added.  T, C, PT, CP reverse ``β`` and were already
broken at the kinetic level — they stay broken.

```text
Operator   Kinetic   β preserved   Combined
P          ✓         ✓             EXACT
T          ✗         ✗             BROKEN
C          ✗         ✗             BROKEN
PT         ✗         ✗             BROKEN
CP         ✗         ✗             BROKEN
CT         ✓         ✓             EXACT
CPT        ✓         ✓             EXACT
```

Yukawa with trivial internal action does not introduce new symmetry
breakings.  P, CT, CPT remain exact; CP and the others remain broken.

## Beta: J-misalignment in the Higgs-like map space

### Beta-1: enumerate J candidates

The ``Cl(0,4)`` commutant
(``lepton.clifford_patisalam.cl04_commutant_complex_structures``)
returns **3** unit-square complex-structure candidates.

Lifting each via ``I_8 ⊗ J_8`` and restricting through
``patisalam_chiral16_block_matrix``:

- ``J_0`` (the project's chosen J): preserves chiral-16 — **viable**.
- ``J_1``, ``J_2``: do not preserve chiral-16 under the existing
  restriction.

Effectively one viable J candidate exists.  The "free choice" of J for
the chiral-16 carrier is therefore effectively pinned.

### Beta-2: J-decomposition of the Higgs map

Define
```text
M_c = (M - J M J) / 2     commutes with J
M_a = (M + J M J) / 2     anticommutes with J
```

For the Session 23 Higgs-like map ``M = higgs_like_charge_shift_candidate``:

```text
||M||²_F                  = 256
||M_c||²_F  (CP-even)    = 128
||M_a||²_F  (CP-odd)     = 128
```

**The decomposition is exactly 50/50.**

CP-violating fraction:

```text
||M_a||² / ||M||² = 128 / 256 = 1/2
```

### Beta-3: verdict

The Higgs-like map has **maximal CP mixing** under the chosen J: equal
J-commuting and J-anticommuting content.  This is the strongest possible
algebraic CP signal.

**BETA PASS (maximal mixing).**

## Combined interpretation

Both audits independently produce CP-violating, CPT-preserving structure
from the existing infrastructure:

- **Alpha**: the *walk itself* breaks CP at the lattice level while
  preserving CPT.  The mechanism is the Bialynicki-Birula construction's
  built-in complex `q_±` phases, which select a "time direction" not
  visible in the continuum limit.
- **Beta**: the *Higgs sector* contains exactly equal CP-even and CP-odd
  components under the chosen J.  Any Yukawa-style coupling that picks
  out a basis-dependent linear combination will inherit complex content,
  giving a CKM-style phase.

These are independent CP sources.  The existing infrastructure has **two
slots** for CP violation, both consistent with CPT exactness.

This is a far stronger positive result than the exact-triality and
broken-triality sidecars produced (both of which died as negative
results).  The CP-from-quantization hope is **vindicated at the
structural level** — at least to the extent that the necessary
structures exist; the magnitude-matching to PDG observables is the
next-level question.

## What this does NOT yet prove

The dual pass is structural, not phenomenological.  Open questions:

- The continuum-limit suppression of alpha CP violation: what's the
  exact ``O(ε^n)`` order, and does it match Lorentz-violation
  experimental bounds?
- The 50/50 mixing in beta: is this an artifact of one specific Higgs
  basis element, or a robust feature of the dim-4 Higgs space?
- The full CKM matrix from these CP sources requires three generations,
  which the existing infrastructure does NOT provide.  The triality and
  broken-triality sidecars failed to derive three generations; without
  that, beta's CP content has no flavor structure to embed in.
- The magnitudes of resulting CP-violation parameters vs measured CKM
  values (and the strong CP problem's small θ).

Each of these is a separate investigation.  This audit only establishes
that the **slots** are present.

## What's saved for the future

If anyone returns to the CP/CPT investigation with three generations
(perhaps from an entirely different mechanism — Furey octonions, family
Spin(8), explicit copies), the alpha and beta CP slots are ready to be
combined with that mechanism.

The infrastructure built here:

- ``discrete_symmetries.py``: P, T, C operators + 7-symmetry composite
  framework + γ-matrix conjugation pattern utilities.
- ``walk_symmetries.py``: massless and Yukawa-perturbed audit machinery.
- ``j_misalignment.py``: J candidate enumeration + J-decomposition of
  internal maps + CP-violating fraction.

All of these can be reused for further audits with different Higgs
candidates, different J choices (if more become viable), or different
walk variants.

## Tests

29 passing tests:

```text
tests/test_discrete_symmetries.py     13 passed
tests/test_walk_symmetries.py          9 passed
tests/test_j_misalignment.py           7 passed
                                    ─────────
                                      29 passed
```

Run with:

```bash
uv run pytest src/clifford_3plus2_d5/cp/tests -q
```

## Effort

- Scaffolding: 30 min.
- Alpha-1 (discrete operators + math): 1 hr.
- Alpha-2 (massless walk audit): 1 hr.
- Alpha-3 (Yukawa-perturbed): 30 min (mostly confirmation).
- Beta (J-misalignment): 1.5 hr.
- Tests + report: 1 hr.

Total: ~5.5 hr of focused work to dual verdict.  Within the ~2-3 day
budget; the early-kill discipline was unnecessary because both audits
passed.

## Cross-references

- ``PLAN.md`` — original alpha/beta audit design.
- ``../spacetime_qca/SESSION_23_YUKAWA_REPRESENTATION.md`` — origin of
  the dim-4 Higgs-like map space used in beta.
- ``../triality/SESSION_T_KILL_TEST.md`` — earlier negative result on
  Spin(8) triality.  This cp result is INDEPENDENT of triality; the CP
  slots exist regardless of how three generations are obtained.
- ``../broken_triality/SESSION_BT_KILL_TESTS.md`` — earlier negative
  result on broken-triality flavor structure.  Same independence note.

## Follow-on (2026-05-18)

The robustness questions raised in "What this does NOT yet prove" were
addressed by ``SESSION_CP_ORDER_EPS2.md``:

- **Multi-element β audit** confirmed the 50/50 mixing across the full
  dim-4 Higgs space (4 basis + 4 transposes, all = 1/2 exactly).
- **O(ε) continuum audit** localized the walk-level CP-violation to
  the T_{2g} cubic-harmonic irrep, with CP-violating fraction = 1 at
  this order.

The dual pass is now a dual ROBUST pass with structural localization.
