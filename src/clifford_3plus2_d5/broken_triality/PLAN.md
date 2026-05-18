# Broken-Triality Sidecar — Plan

Status: planning only.  No code, no audits yet.

## The load-bearing question

> Given that **exact** Spin(8) triality does not preserve the SM-inside-Spin(8)
> subalgebra (proven in ``../triality/SESSION_T_KILL_TEST.md``), can a
> **broken** Z/3 cycle — using the same triality machinery as an
> approximate flavor structure — produce the measured mass hierarchy
> and CKM CP phase with O(few) free parameters, while preserving CPT?

The exact-triality program died.  The broken-triality program asks whether
the *failure mode* of the exact program is itself the source of the flavor
structure: triality maps SM Cartan generators **slightly outside** the SM
Cartan, and the "outside" components are exactly what would-be-symmetric
Z/3 breaking should populate.

This is a different hypothesis from exact triality and needs its own kill
discipline.

## Why this is not just "yet another discrete flavor model"

Standard discrete-flavor models (A_4, T', S_4, Δ(27), ...) introduce a
discrete symmetry **by hand** acting on flavor index, choose a breaking
pattern, and fit.  They are typically unfalsifiable because the breaking
pattern is selected to fit the data.

The broken-triality program is more constrained:

- The Z/3 is **not chosen**; it is the residual Z/3 cyclic part of the
  Spin(8) Out-automorphism group S_3, inherited from the exact triality
  attempt.
- The **specific failure direction** of K1 — the way each SM Cartan
  generator maps **outside** the SM Cartan — fixes the breaking
  structure, not as a free parametrization but as a forced consequence
  of the embedding choice.
- The breaking magnitude is a single parameter (overall scale of the
  triality-rejection part).  Everything else is forced.

If this works, the model has very few free parameters compared to flavor
observables, and is genuinely falsifiable.  If it doesn't work, it dies
cleanly at one of the kill tests below.

## What this program is NOT

- It is not a complete replacement for the Standard Model — the SM
  gauge structure remains as in ``lepton``.  Triality acts on flavor
  index only.
- It does not claim "exact triality is restored after symmetry breaking."
  The exact symmetry is dead.  What survives is a Z/3 flavor structure
  with a specific, forced breaking pattern.
- It does not predict three generations from triality alone in a
  hand-free way.  Three generations are imposed as three copies; the
  question is whether the *relations between them* are forced by
  triality.

## Kill tests, ordered cheapest-first

Each kill produces a binary verdict.  A fail closes the sidecar with a
published negative result.  The total budget to verdict on all four
kills is **~3 focused days**.

### Kill BT-1 — Yukawa structure from triality overlaps

**Question**: project each ``τ^i``-rotated SM Cartan basis vector back
onto the SM Cartan subspace.  Build a 3x3 overlap matrix from these
projections.  Is the resulting matrix non-trivial (off-diagonal entries
exist, eigenvalues non-degenerate)?

**Concrete construction**:

```text
For i, j ∈ {0, 1, 2}:
    Y_ij = ⟨ Π_SM ( τ^i · v_* ),  Π_SM ( τ^j · v_* ) ⟩
```

where:

- ``v_*`` is a fixed SM Cartan basis vector (e.g., ``Y'`` from the
  triality sidecar);
- ``Π_SM`` is orthogonal projection onto the 3-dim SM Cartan subspace of
  the 4-dim Spin(8) Cartan;
- ``⟨·,·⟩`` is the Euclidean inner product on R^4.

**Pass condition**: the resulting 3x3 matrix has

- non-zero off-diagonal entries (would-be Z/3-circulant structure
  visible);
- non-degenerate eigenvalues (mass hierarchy possible).

**Fail condition**: Y is proportional to identity or to a permutation
matrix.  In either case there is no flavor structure to break — the
program is dead.

**Effort**: 1 day.  All inputs (``T``, SM Cartan basis, projection
helper) already exist in ``../triality/``.

**Files to add**: ``yukawa_overlaps.py``, ``test_yukawa_overlaps.py``.

### Kill BT-2 — Mass eigenvalue ratios from circulant Yukawa

**Question**: does the natural mass hierarchy from a Z/3-circulant Yukawa
with O(1) breaking parameters span enough orders of magnitude to be
consistent with the observed SM ratios?

The relevant constraints:

- ``m_t / m_e ≈ 3 × 10^5`` (charged sector ratio across generations and
  sectors).
- ``m_t / m_c ≈ 130``, ``m_c / m_u ≈ 600`` — different ratios within the
  up-quark sector.
- Similar for down quarks and charged leptons.

**Concrete construction**: take the BT-1 Yukawa matrix and treat it as
the leading-order Yukawa.  Diagonalize.  Compute eigenvalues.

**Pass condition**: the ratio ``λ_max / λ_min`` for the diagonalized
matrix is at least ``~10^2``.  (We do not yet require hitting the full
six-orders-of-magnitude span; just enough to show the structure is
hierarchical, not flat.)

**Fail condition**: ``λ_max / λ_min < 10`` (essentially flat spectrum).
In that case the circulant + small breaking pattern cannot produce
sufficient hierarchy; we would need exponential / Froggatt-Nielsen-style
structure not present in pure triality breaking.

**Effort**: ½ day.  3x3 eigenvalue calculation in sympy.

**Files to add**: ``mass_hierarchy.py``, ``test_mass_hierarchy.py``.

### Kill BT-3 — CP phase from Z/3 character analysis

**Question**: when the Yukawa matrix is decomposed using Z/3 characters
``{1, ω, ω̄}``, does a non-zero CP phase (Jarlskog-like invariant) emerge
generically, or is it forced to zero by the structure?

**Concrete construction**:

1. Decompose the BT-1 Yukawa matrix into Z/3-irreducible pieces via the
   DFT:
   ```text
   F = (1/√3) [[1, 1, 1],
                [1, ω, ω̄],
                [1, ω̄, ω]]
   ```
   The matrix ``F^† Y F`` has block structure under Z/3.

2. Compute the Jarlskog-like invariant:
   ```text
   J = Im( det( Y · Y^† ) )    or similar phase-sensitive scalar
   ```

3. Check whether ``J`` is identically zero (Z/3-symmetric Yukawa forced
   to real) or non-zero (CP phase present).

**Pass condition**: ``J ≠ 0`` for generic O(1) parameters in the
Yukawa.

**Fail condition**: ``J = 0`` identically.  This would mean the
Z/3-circulant Yukawa structure has no CP-violating slot; the program's
main attractive feature (CP phase from complex characters) vanishes.

**Magnitude check (secondary)**: if ``J ≠ 0``, compare its natural order
of magnitude to the measured CKM Jarlskog ``J_CKM ≈ 3 × 10^-5``.  A
naïve Z/3 breaking gives ``J ~ O(1)``, which is far too large.  This is
not a kill on its own (suppression mechanisms exist), but it must be
documented as a tension to be addressed in any follow-on.

**Effort**: 1 day.  3x3 matrix algebra plus phase-sensitive scalar
extraction.

**Files to add**: ``cp_phase.py``, ``test_cp_phase.py``.

### Kill BT-4 — Parameter count audit

**Question**: how many real free parameters does the broken-triality
model have, and is it falsifiable against the ~22 SM Yukawa-sector
observables?

**Parameter ledger to enumerate**:

- Per sector (up quarks, down quarks, charged leptons, neutrinos):
  - 3 circulant magnitudes (the ``a, b, c`` of the Z/3-symmetric part).
  - 1 overall mass scale.
  - 1 breaking magnitude ``ε``.
  - 1 breaking-orientation phase (if real-valued breaking).
  - 1 breaking texture choice (discrete).
- Cross-sector mixing scales: any?
- Total per sector: ~6 continuous + ~1 discrete.
- Three sectors: ~18 continuous + ~3 discrete = ~21 free parameters.

**Observable count**: 22 SM Yukawa-sector observables (12 masses + 4
CKM + 4 PMNS Dirac + 2 Majorana phases).

**Pass condition**: free parameter count < observable count with margin
(say, at most 18 parameters for 22 observables, giving 4 over-constraint
slots).

**Fail condition**: parameters ≥ observables.  Model is unfalsifiable as
a flavor-physics statement.

**Effort**: ½ day.  Just an accounting exercise once the BT-1..BT-3
parameter structure is fixed.

**Files to add**: ``parameter_ledger.md`` (final form, after BT-1..BT-3
fix the parameter count).

## Decision tree

```text
START
  │
  ▼
BT-1: Yukawa overlap structure
  ├── FAIL: no flavor structure   → CLOSE sidecar
  └── PASS
       ├── BT-2: mass hierarchy span
       │   ├── FAIL: flat spectrum → CLOSE sidecar
       │   └── PASS
       │        │
       │        ▼
       └── BT-3: CP phase from Z/3 characters
             ├── FAIL: J = 0 forced → CLOSE sidecar
             └── PASS
                  │
                  ▼
             BT-4: parameter count audit
                  ├── FAIL: params ≥ obs → CLOSE sidecar
                  └── PASS  → SIDECAR EARNS THE RIGHT TO GROW
                              (next: PDG numerical fit)
```

A failure at any step closes the sidecar with a clean negative result
and a published report.  Success at all four earns the right to attempt
the actual PDG fit — which is a separate, larger investigation that this
plan does **not** scope.

## Reuse from existing infrastructure

The sidecar imports from ``triality/`` directly (which itself imports
from ``lepton/``).  No new octonion / Clifford / Pati-Salam code.

```text
from clifford_3plus2_d5.triality.spin8_triality import (
    triality_cartan_matrix,
    spin8_cartan_on_chiral16,
    apply_triality_to_cartan_vector,
    cartan_coordinates,
)
from clifford_3plus2_d5.triality.sm_restriction import (
    g_sm_8_cartan_basis_coords,
    restricted_hypercharge_cartan_coords,
    su3_c_cartan_coords,
)
from clifford_3plus2_d5.triality.reuse import (
    physical_hypercharge_generator,
    charge_observable,
    ...
)
```

All triality machinery (the 4x4 ``T``, the SM Cartan basis, the
projections) is already implemented and tested.  The broken-triality
sidecar only adds 3x3 Yukawa-matrix manipulation on top.

## Parameter ledger structure (template)

The ledger will be filled in as BT-1..BT-3 fix the parameter structure.
Initial template:

```text
## Discrete choices
1. Spin(8) embedding — inherited from triality/ ({0..7})
2. SM Cartan basis — inherited from triality/
3. Triality direction (τ vs τ²) — inherited from triality/
4. Choice of "starting" SM Cartan vector v_* for the BT-1 overlap

## Continuous parameters (to be enumerated after BT-1..BT-3)
- a, b, c circulant magnitudes per sector
- overall mass scale per sector
- breaking magnitude ε per sector
- breaking phase per sector
- breaking texture (discrete) per sector

## Total
TBD after BT-3
```

## What we won't do until all kills pass

These are out of scope for the kill-test phase, but become natural
follow-ons if all four kills pass:

- PDG numerical fit (CKM angles, CP phase, mass values).
- Neutrino sector (PMNS, Majorana vs Dirac, seesaw mechanism).
- Connection to spacetime QCA / Higgs mechanism / Yukawa as dynamical
  field.
- Higher-order corrections to the Z/3 breaking pattern.
- Comparison to other discrete-flavor-symmetry models (A_4, T', etc.).

## Failure precedents named in advance

**F-BT-1**: BT-1 overlap matrix is identity or zero.
Diagnosis: the SM Cartan is either invariant under ``Π_SM ∘ τ`` (would
contradict K1, so unexpected) or fully orthogonal to its ``τ`` images
(also unexpected — would mean ``τ`` rotates SM Cartan into pure
complement).  Either way, no flavor structure to break.

**F-BT-2**: Eigenvalues of the overlap matrix are all equal (degenerate).
Diagnosis: the would-be-circulant structure has full Z/3-symmetric
eigenvalue ``a+b+c`` with multiplicity 3 — no mass splitting without
explicit breaking.  The "natural" mass hierarchy from the projection
itself is too flat.

**F-BT-3**: ``J = Im(det(Y Y^†))`` is identically zero.
Diagnosis: the Yukawa matrix derived from triality projection is real
(no imaginary entries), so its Hermitian product is real-diagonalizable
with no CP phase.  Possible if the projection itself produces only real
coefficients — which would be the case if everything in the construction
lives in real Cartan coordinates only.  In that case, the CP phase has
to come from an additional complex structure not in the current
construction.

**F-BT-4**: Parameter count ≥ observable count.
Diagnosis: the model is no more predictive than generic SM Yukawa
matrices.  Z/3 structure was decorative, not constraining.

## What this kill test specifically rules out vs preserves

**A negative verdict (any kill fails)** rules out: broken Spin(8)
triality (with the natural Pati-Salam-aligned embedding) as the source
of generation structure with predictive flavor content.

**A negative verdict does not rule out**:

- Discrete flavor symmetries (A_4, T', S_4, Δ(27), etc.) — these are
  different programs.
- Non-natural Spin(8) embeddings inside Spin(10) (an alternative
  embedding could in principle give different overlaps; out of scope
  for this kill phase).
- The "CP from BCC lattice anisotropy" program (orthogonal to
  triality — different mechanism entirely).
- The "Furey octonion three-copy" program (different algebraic
  framework).

**A positive verdict (all four kills pass)** does not constitute a fit
to PDG.  It only earns the right to attempt the fit.  The fit itself is
the post-kill-test session — likely weeks of work, with its own
falsifiability bar.

## Cross-references

- ``../triality/SESSION_T_KILL_TEST.md`` — what the exact-triality
  program died on; the source of the structural hints used in BT-1.
- ``../triality/PLAN.md`` — the exact-triality kill-test design; this
  plan inherits the same discipline.
- ``../lepton/`` — Pati-Salam, SM, hypercharge infrastructure (via
  ``triality/reuse.py``).
- ``../spacetime_qca/`` — orthogonal; the broken-triality question is
  about flavor structure, not spacetime dynamics.

## Effort budget

Total to a four-kill verdict: **~3 focused days**.

```text
BT-1 (Yukawa overlaps)   1.0 day
BT-2 (mass hierarchy)    0.5 day
BT-3 (CP phase)          1.0 day
BT-4 (parameter audit)   0.5 day
                        ─────────
                         3.0 days
```

Plus ½ day for the session report at the end.

This is comparable to the exact-triality sidecar (~1 day to verdict) but
slightly more expensive because the broken-version computations are more
involved than the single Cartan-subspace check.

## What "go" means

If you say "do it" on this plan, I will:

1. Scaffold ``broken_triality/`` with empty modules + tests (same
   pattern as ``triality/``).
2. Implement BT-1.  Run it.  If FAIL, write report and stop.
3. Implement BT-2.  Run it.  If FAIL, write report and stop.
4. Implement BT-3.  Run it.  If FAIL, write report and stop.
5. Implement BT-4.  Run it.  If FAIL, write report and stop.
6. If all four PASS, write a positive report and stop — without doing
   the actual PDG fit (that is a separate decision).

This means the worst case is a clean negative report in ~1 day; the best
case is a clean positive report in ~3 days plus the right to start a
much bigger investigation.
