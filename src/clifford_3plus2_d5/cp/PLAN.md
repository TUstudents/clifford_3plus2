# CP / CPT Sidecar — Plan

Status: planning only.  No code, no audits yet.

## Load-bearing question

> Does the existing carrier (BCC Dirac walk × chiral-16 internal × J)
> contain a CP-violating slot that is CPT-preserving, derivable from the
> structure already implemented in ``spacetime_qca``, ``lepton``, and
> ``triality``?

The answer is binary.  Two audits run in parallel.  Either or both can
fail; the program is alive iff at least one of them produces a clean
CPT-exact + CP-broken pattern.

## Background

Earlier brainstorming triaged four candidate sources of "tiny CP from
quantization":

| Candidate | Verdict |
|---|---|
| 1: SU(2)_L → finite discrete subgroup | Experimentally ruled out (Lorentz bounds force order > 10^30). |
| 2: BCC anisotropy SO(3) → O_h | Produces Lorentz violation, not CP. |
| 3: Discrete time step | Produces energy bandwidth, not CP. |
| 4: Discrete internal index | Not actually an approximation. |

The brainstorming surfaced two new candidates with genuine CP potential
inside the existing infrastructure:

- **alpha (Candidate B in the brainstorm)**: CP / CPT structure of the
  BCC Dirac walk itself.  Whether the walk preserves CPT exactly but
  breaks CP at some finite order.
- **beta (Candidate A in the brainstorm)**: Whether the choice of
  Cl(0,4)-commutant complex structure ``J`` (currently a free discrete
  choice in ``lepton.clifford_patisalam.cl04_commutant_complex_structures``)
  forces complex entries in the Higgs-like map space from Session 23, and
  whether that complexity survives all valid J choices.

Both are computable from existing infrastructure in 1-2 days each.

## What this sidecar will NOT investigate

These were brainstormed but are deferred:

- **Candidate C**: anomalous CP violation from BCC chirality.  This
  requires an effective-action calculation at ``O(eps^2)`` and is at
  least 2-3 weeks of effective-action work, not a kill-test.
- **Candidate D**: Trotter splitting in the walk.  The current BCC
  Dirac walk is a single Floquet operator, not Trotter-split, so
  candidate D does not apply.  Documented for future reference.

## Audit alpha — CP/CPT walk symmetries

### Step alpha-1: build P, T, C discrete operators on the carrier

Concrete operators on the per-site carrier ``C^4_Dirac ⊗ R^32_internal``:

- **P (parity, unitary)**: ``ψ(x) → γ^0 · ψ(-x mod L)``.  On the lattice
  step, ``P · U · P^{-1} = U`` iff the hop matrices satisfy
  ``γ^0 W_D(-h) = W_D(h) γ^0``.  Already verifiable by hand from the
  Bialynicki-Birula chiral assembly.
- **T (time reversal, antiunitary)**: ``ψ(x) → T_spinor · K · ψ(x)``
  where ``K`` is complex conjugation and ``T_spinor`` is a fixed Dirac
  matrix.  Standard convention in chiral basis: ``T_spinor = i γ^1 γ^3``.
  T-invariance of the walk: ``T · U · T^{-1} = U^†``.
- **C (charge conjugation, antiunitary)**: ``ψ(x) → C_spinor · K · ψ(x)``
  with ``C_spinor`` involving the internal complex structure ``J``.
  Different conventions exist; this sidecar pins ``C_spinor`` from the
  product ``i γ^2 · J_internal`` and documents the choice.
- **Composites** PT, CP, CT, CPT formed by composition.

### Step alpha-2: massless audit table

Apply each composite to the massless BCC Dirac walk:

```text
P:     ?  exact / broken at O(ε^?)
T:     ?
C:     ?
PT:    ?
CP:    ?
CT:    ?
CPT:   ?
```

For free massless Dirac, the expected answer is: all symmetries exact.
This step is a sanity check, not a kill-test.

### Step alpha-3: Yukawa-perturbed audit table

Add the Session 23 Higgs-like map as a constant background insertion
(``β ⊗ M_higgs``):

```text
Operator       Massless walk   With Higgs map
P:             exact?           exact / broken?
T:             exact?           ?
C:             exact?           ?
CPT:           exact?           ?  (must remain exact)
CP:            exact?           ?  (the load-bearing one)
```

### Pass condition (alpha)

The alpha audit **passes** if:

- ``CPT`` is exact in both the massless and Yukawa-perturbed walks;
- AND ``CP`` is exact in the massless walk but broken in the
  Yukawa-perturbed walk.

The alpha audit **fails** if:

- ``CPT`` is not exact in either case (impossible per Lüders-Pauli for
  local Hermitian walks; if seen, indicates an implementation bug); OR
- ``CP`` remains exact after Higgs insertion (no CP slot from this
  audit).

### Effort: 1-2 days

All inputs exist: ``dirac_step`` from ``spacetime_qca``, Higgs-like maps
from ``spacetime_qca.yukawa``, the J from
``lepton.clifford_patisalam``.  The audit is a symbolic commutation
check on 128-dim matrices.  SymPy handles this in seconds.

## Audit beta — J-misalignment in the Higgs-like map space

### Step beta-1: enumerate compatible J candidates

The lepton module records that
``cl04_commutant_complex_structures()`` returns multiple unit-square
complex structures in the Cl(0,4) commutant.  Each is a valid
"compatible J" in the lepton sense.

Enumerate all J candidates and tabulate their action on the chiral-16
internal carrier.

### Step beta-2: Higgs map space under each J

For each candidate J:

1. Rebuild the Session 23 Higgs-like map space with this J playing the
   role of the canonical complex structure.
2. Decompose the dim-4 real space into J-eigenspaces (J · M = ±i · M).
   The eigenvalue structure is the "complex" content of the Higgs
   maps relative to this J.
3. Build the corresponding Yukawa-like 3-generation matrix (using the
   broken_triality construction) but with complex entries from the
   J-eigenspace decomposition.

### Step beta-3: Jarlskog-like invariant per J

For each J candidate, compute:

```text
J_beta = Im( det( Y · Y^† ) )    or similar phase-sensitive scalar
```

If ``J_beta`` is identically zero for all J candidates, beta fails (no
CP from J ambiguity).

If ``J_beta`` is non-zero for some J candidate, beta passes.

If ``J_beta`` is non-zero AND varies between J choices, the J ambiguity
is the source of a CP parameter that depends on the J choice.

### Pass condition (beta)

The beta audit **passes** if:

- there exists a J such that ``J_beta ≠ 0`` AND the value varies
  non-trivially among compatible J choices;
- OR there exists a J such that ``J_beta ≠ 0`` (forced complex
  structure).

The beta audit **fails** if ``J_beta = 0`` for all J candidates.

### Effort: 1-2 days

All inputs exist: the J candidates from
``cl04_commutant_complex_structures``, the Higgs map space from
``spacetime_qca.yukawa.color_singlet_charge_shift_basis``.  The audit
is linear-algebra plus a phase-sensitive scalar computation.

## Decision tree (combined verdicts)

```text
            β fails           β passes
α fails    program dies      partial: CP from J ambiguity only;
                              alpha-side closed, beta open
α passes   partial: CP from   both: program alive, multiple
           walk only;         CP slots — proceed to a fitting
           beta-side closed   investigation
```

The dual fail closes the entire program (no CP slot in existing
infrastructure).

The dual pass would be the strongest positive result the program has
seen — and would justify a major investigation into matching the
predicted CP magnitudes to observation.

A partial pass (one passes, one fails) keeps the program alive on one
side but documents the failure of the other.

## Reuse from existing infrastructure

```text
from clifford_3plus2_d5.spacetime_qca import (
    dirac_step,
    bcc_dirac_symbol,
    gamma0, gamma_matrices, gamma5,
    block_diag,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    cl04_commutant_complex_structures,
    cl04_chosen_commutant_j,
    patisalam_chosen_complex_structure,
)
from clifford_3plus2_d5.spacetime_qca.yukawa import (
    color_singlet_charge_shift_basis,
    higgs_like_charge_shift_candidate,
    higgs_like_charge_shift_pair,
    beta_is_off_diagonal_between_chiralities,
)
```

No new octonion / Clifford / Pati-Salam code.  The C, P, T spinor
matrices in chiral basis are the only new primitives.

## Parameter ledger structure (template)

```text
## Discrete choices
1. Spinor basis: chiral (inherited from spacetime_qca).
2. P spinor matrix: γ^0.
3. T spinor matrix: i γ^1 γ^3 (Pauli convention).
4. C spinor matrix: i γ^2 with internal J insertion.
5. Higgs map space basis: from color_singlet_charge_shift_basis.
6. Jarlskog convention: Im(det(Y · Y^†)).

## Continuous parameters
None at this stage.

## Total free parameter count
Six discrete, all forced once conventions are pinned.  Falsifiability is
maximal.
```

## Files to implement (after this plan is approved)

```text
src/clifford_3plus2_d5/cp/
  __init__.py
  PLAN.md                         (this file)
  STATUS.md                       (scaffolding status)
  parameter_ledger.md             (template; filled after each audit)
  reuse.py                        (single import surface)
  discrete_symmetries.py          (alpha-1: P, T, C operators)
  walk_symmetries.py              (alpha-2 + alpha-3: audit tables)
  j_misalignment.py               (beta-1, beta-2, beta-3)
  SESSION_CP_ALPHA_BETA.md        (final verdict report)
  tests/
    __init__.py
    test_discrete_symmetries.py
    test_walk_symmetries.py
    test_j_misalignment.py
```

Estimated effort: ~2-3 days to dual verdict.

## Failure precedents named in advance

**F-alpha-1**: CPT not exact in massless walk.  Diagnosis: implementation
bug in discrete operators or walk; CPT must hold by Lüders-Pauli.  Fix
the bug before continuing.

**F-alpha-2**: CP still exact after Higgs insertion.  Diagnosis: the
Higgs-like map used is real (CP-preserving).  Try a different basis
element from the dim-4 space.  If all choices give real CP, alpha
genuinely fails.

**F-beta-1**: Only one J candidate exists.  Diagnosis: lepton's
``cl04_commutant_complex_structures`` returns a singleton.  If so,
J is not free and beta is vacuous.  Document and close beta.

**F-beta-2**: All J choices give Jarlskog = 0.  Diagnosis: the Higgs
map space has no imaginary part in the J-eigenspace sense.  CP-violation
from J ambiguity is structurally absent.

**F-beta-3**: All J choices give the same Jarlskog ≠ 0.  Diagnosis: J
choice doesn't matter (any J gives the same CP), but there IS forced
CP from the algebraic structure.  This is actually a partial pass — CP
is generated by the algebra, just not dependent on J.  Document
clearly.

## What "do it" triggers

If the user approves this plan and says "do it", the sequence is:

1. Scaffold ``cp/`` with empty modules.
2. Implement alpha-1 (discrete symmetries) + tests.
3. Implement alpha-2 (massless audit).  Run.
4. Implement alpha-3 (Yukawa-perturbed audit).  Run.
5. Implement beta-1, beta-2, beta-3.  Run.
6. Write ``SESSION_CP_ALPHA_BETA.md`` with combined verdict.
7. Update ``STATUS.md`` and ``parameter_ledger.md``.

Worst case: dual fail in ~1 day with a clean negative report.
Best case: dual pass in ~2-3 days, opening a new investigation track.

## Cross-references

- ``../spacetime_qca/`` — BCC Dirac walk, Higgs-like maps.
- ``../lepton/`` — Cl(0,4) J candidates, Pati-Salam structure.
- ``../triality/SESSION_T_KILL_TEST.md`` — established failure of exact
  triality (relevant context: triality is not the source of CP, but the
  walk and J ambiguity might be).
- ``../broken_triality/SESSION_BT_KILL_TESTS.md`` — established failure
  of broken triality (relevant context: triality-projection Yukawa is
  real / CP-preserving).
