# Plan: koide sidecar — Koide-formula audit for the BCC carrier

## Context

The Koide formula (Y. Koide, 1981) is the empirical relation

    K = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²  =  2/3

verified to ~10⁻⁴ precision for charged leptons.  Geometrically this
is equivalent to the statement that the 3-vector

    v⃗ = (√m_e, √m_μ, √m_τ)

makes an angle of **exactly 45°** with the direction (1, 1, 1)/√3 in
flavor-space.  Equivalently: v⃗ lies on the half-angle-45° cone
around (1, 1, 1).

The striking coincidence: **(1, 1, 1)/√3 is the BCC body-diagonal
axis** — exactly the rotation axis we already audited in
topology/SC-2.  Koide's geometry singles out the same direction as
the BCC lattice's most natural Z₃ symmetry axis.  Either this is a
deep clue tying the program's lattice structure to the empirical
flavor sector, or it is a remarkable coincidence we should rule out.

Prior Z₃ attempts on the carrier closed negative (triality K1 fail;
broken_triality BT-2 fail; topology/D-1 chiral-16 Z₃-trivial; topology/D-2
color Z₃ asymmetric).  Koide is structurally **different**:

- It does NOT say Z₃ permutes generations (which would force
  degenerate masses — the broken_triality failure mode).
- It says: the mass-vector lies on a Z₃-invariant **cone**, but its
  specific position on the cone is broken by something else.

So Koide is consistent with a *broken* BCC body-diagonal Z₃: the cone
is exact, the position on the cone is selected by additional
dynamics.  Verifying this audits the natural lattice-flavor
connection precisely.

**The single load-bearing question**:

> Does the BCC body-diagonal Z₃ acting on the σ^a-axes of the
> chiral-16 T_{2g} structure naturally place the charged-lepton
> mass-vector on the Koide 45° cone, given three generations as
> phenomenological input?

Three pre-named outcomes (user-confirmed):

1. **KOIDE PREDICTED**: the program's algebraic constraints force
   v⃗ onto the cone.  Major positive — first natural derivation
   of Koide from lattice geometry.
2. **KOIDE CONSISTENT**: the program's constraints permit v⃗ on
   the cone but do not force it.  Honest pass; cone-selection
   requires additional input.
3. **KOIDE CONFLICT**: the program forbids v⃗ on the cone.  Kill
   — the apparent BCC↔Koide coincidence is broken by the program's
   structure, ruling out the connection.

## User-confirmed scope decisions

- **Sidecar name**: ``koide/``.
- **Mass-vector identification**: σ^a-axes from cp/H^(1) T_{2g}.  The
  natural 3-fold structure of the program's leading CP-odd
  correction is the cyclic Pauli-axis labelling

      H^(1)_chir = σ^x k_y k_z − σ^y k_x k_z + σ^z k_x k_y

  with BCC body-diagonal Z₃ cycling σ^a.  Identify
  (m_e, m_μ, m_τ) ↔ (m_{σx}, m_{σy}, m_{σz}).
- **Audit depth**: BOTH structural test (does the program's
  constraint force the 45° cone?) AND broken_triality-style
  numerical 3×3 Yukawa construction from a BCC-Z₃ orbit with
  explicit eigenvalue + K computation.  ~3-4 weeks.
- **Three-generation treatment**: phenomenological input.  The
  topology/exceptional/triality/broken_triality closures rule out
  deriving 3 generations from the carrier; add them by hand and
  test whether the program's algebraic constraints constrain the
  Yukawa to lie on the Koide cone.
- **Verdict thresholds**: PREDICTED / CONSISTENT / CONFLICT.

## Existing infrastructure to reuse (via `koide/reuse.py`)

Verified from exploration:

**From cp/**:
- ``cp/j_misalignment.viable_j_candidates()`` — J's preserving the
  chiral-16; needed for any Yukawa construction.
- ``cp/j_misalignment.j_decomposition()`` — splits a matrix into
  J-commuting (+) / J-anticommuting (−) parts.
- ``cp/j_misalignment.cp_violating_fraction()`` — Frobenius-norm
  CP fraction.
- ``cp/j_misalignment.color_singlet_charge_shift_basis()`` — the
  4-dim "Higgs map space" used by β-multi audit.
- ``cp/continuum_cp.cp_irrep_decomposition()`` — the (CP-odd, T_{2g})
  cell of H^(1).
- ``cp/cubic_harmonics.projector_T2g()`` — the T_{2g} projector.

**From topology/**:
- ``topology/bcc_z3_rotation.body_diagonal_rotation_matrix()`` — the
  3×3 cyclic rotation R; permutes (x, y, z) → (y, z, x).
- ``topology/bcc_z3_rotation.dirac_spinor_lift()`` — SU(2) spinor lift.
- ``topology/bcc_z3_rotation.bcc_direction_permutation()`` — BCC
  cycle structure (3, 3, 1, 1).

**From broken_triality/** (closest structural precedent):
- ``broken_triality/yukawa_overlaps.py`` — builds a 3×3 from a Z₃
  orbit + projection.  The pattern to adapt:

      Y_ij = ⟨Π(R^i v_*), Π(R^j v_*)⟩

  with R = triality cycle and Π = SM Cartan projector.  For Koide
  we replace R with the BCC body-diagonal cycle.
- ``broken_triality/mass_hierarchy.py`` — eigenvalue extraction +
  ratio testing pattern.

**From spacetime_qca/**:
- ``spacetime_qca/yukawa.hermitian_yukawa_internal_control(φ⁺, φ⁰)``
  — Session 38's Hermitian Y(Φ) on chiral-16.  Returns 32×32 real
  symmetric.  We may extract a 3×3 charged-lepton sub-block.

**What must be built new**:
- Koide cone parametrization + 45° / equipartition geometric test.
- σ^a ↔ generation dictionary on the chiral-16.
- BCC-Z₃-orbit 3×3 Yukawa construction (adaptation of
  broken_triality's pattern).
- Cone-locus vs. Yukawa-locus comparison (PREDICTED /
  CONSISTENT / CONFLICT classifier).
- Combined audit + SESSION report.

## Decision tree

```text
Phase KO-1 (empirical Koide + 45° cone geometry, ~2 days)
  ├── Verify K = 2/3 from PDG (m_e, m_μ, m_τ); accurate to 10⁻⁴.
  ├── Build cone parametrization in flavor-space.
  └── Establish equipartition form: Koide ⇔ |v_trace|² = |v_traceless|²
      where v_trace = ((v_x+v_y+v_z)/3)(1,1,1) along (1,1,1).
        │
        ▼  [empirical baseline established]
Phase KO-2 (BCC body-diagonal Z₃ on σ^a-axes, ~2-3 days)
  ├── Reuse topology/'s R matrix; verify R fixes (1,1,1) (trace direction).
  ├── Decompose generic σ^a-indexed 3-vectors under R:
  │   v = v_trace ⊕ v_traceless (Z₃-trivial + 2D non-trivial irrep).
  ├── Identify (1,1,1) in σ^a-space with BCC body-diagonal in real-space.
  └── Map (m_e, m_μ, m_τ) ↔ (m_{σx}, m_{σy}, m_{σz}).
        │
        ▼  [identification fixed; structural test ready]
Phase KO-3 (broken_triality-style 3×3 Yukawa from BCC-Z₃ orbit, ~5-7 days)
  ├── Adapt broken_triality's pattern with R = BCC body-diagonal Z₃.
  ├── Choose starting vectors v_* in flavor-space:
  │   • restricted hypercharge Y' (broken_triality's default)
  │   • J-aligned vectors from cp/'s viable J candidates
  │   • color-singlet charge-shift basis from cp/j_misalignment
  ├── Build Y_{ij} = ⟨Π(R^i v_*), Π(R^j v_*)⟩ for each v_*.
  ├── Extract eigenvalues, compute K from √(eigenvalues).
  └── Try systematic scan over the Higgs map space (dim-4).
        │
        ▼  [Yukawa-eigenvalue locus mapped]
Phase KO-4 (cone vs. locus comparison, ~3-4 days)
  ├── Locus L = the set of (m_e, m_μ, m_τ) reachable from the program's
  │   algebraic constraints (BCC-Z₃ orbit + J-decomposition + color-
  │   singlet + charge-shift + hermitian Yukawa).
  ├── Cone C = the 45° cone around (1,1,1).
  ├── Compute L ∩ C, |L ∩ C| / |L|, |L ∩ C| / |C|.
  └── Verdict classifier:
      • L ⊂ C  → PREDICTED
      • L ∩ C ≠ ∅, L ⊄ C  → CONSISTENT
      • L ∩ C = ∅  → CONFLICT
        │
        ▼
Phase KO-5 (combined audit + SESSION report, ~3 days)
  ├── Aggregate KO-1..KO-4 into KoideAuditPayload.
  ├── SESSION_KOIDE.md final verdict.
  └── Update STATUS.md, parameter_ledger.md, PROJECT_STATUS.md.
```

## Phase KO-1 — Empirical Koide + 45° cone geometry (~2 days)

### FKO-1: Numerical verification of Koide from PDG

PDG charged-lepton masses:
- m_e  = 0.51099895 MeV
- m_μ  = 105.6583755 MeV
- m_τ  = 1776.86 MeV

Compute K_PDG; assert |K_PDG − 2/3| < 10⁻³ (allowance for
experimental uncertainty on m_τ).

### FKO-2: Equivalent geometric form

Koide K = 2/3 ⇔ vec v⃗ = (√m_e, √m_μ, √m_τ) makes 45° with
(1, 1, 1)/√3.  Equivalent decomposition:

    v_trace      = (v⃗ · n̂) n̂                  where n̂ = (1,1,1)/√3
    v_traceless  = v⃗ − v_trace

    Koide  ⇔  |v_trace|² = |v_traceless|²

Verify both forms agree empirically from PDG.

### FKO-3: Cone parametrization

The 45° cone in ℝ³ around (1, 1, 1) is

    C = { v ∈ ℝ³ : (v · n̂)² = (1/2) |v|² }
      = { r · (n̂ · cos(45°) + ê(φ) · sin(45°)) : r > 0, φ ∈ [0, 2π) }

where ê(φ) is a unit vector in the plane orthogonal to n̂.  This
gives the (r, φ) parametrization.  PDG values map to specific
(r_PDG, φ_PDG).

Output: ``koide/koide_geometry.py``.

## Phase KO-2 — BCC body-diagonal Z₃ on σ^a-axes (~2-3 days)

### FKO-4: BCC Z₃ rotation on the σ^a 3-vector

Reuse ``topology/bcc_z3_rotation.body_diagonal_rotation_matrix()``:

    R = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

R fixes (1, 1, 1)/√3 (eigenvalue +1, the trace direction).  Its
orthogonal 2D action on the plane perpendicular to (1, 1, 1) is the
non-trivial Z₃ irrep with eigenvalues ω, ω² where ω = exp(2πi/3).

Verify: R³ = I, R^T R = I, det R = +1.

### FKO-5: σ^a ↔ generation identification

Per cp/'s H^(1) T_{2g} structure, the three Pauli axes σ^x, σ^y, σ^z
are cyclically permuted by the BCC body-diagonal Z₃ rotation.  The
identification:

    σ^x  ↔  generation e
    σ^y  ↔  generation μ
    σ^z  ↔  generation τ

is one of three cyclic conventions (each differs by a Z₃ phase).
For the audit we pin this convention and check consistency under
re-labelling.

### FKO-6: Trace/traceless decomposition under Z₃

For any v ∈ ℝ³ (interpreted as a σ^a-indexed mass-vector):

    P_trace = (1/3) (1,1,1)(1,1,1)^T   ← Z₃-trivial irrep
    P_traceless = I_3 − P_trace        ← Z₃-non-trivial 2D irrep

These are the Z₃ irrep projectors on ℝ³.

Verify ``v_PDG = P_trace · v_PDG + P_traceless · v_PDG`` and
``|P_trace · v_PDG|² ≈ |P_traceless · v_PDG|²`` (Koide cone
condition).

Output: ``koide/bcc_z3_on_flavor.py``.

## Phase KO-3 — Yukawa eigenvalue locus from BCC-Z₃ orbit (~5-7 days)

### FKO-7: Adapt broken_triality pattern

broken_triality builds:

    Y_{ij} = ⟨Π_SM(τ^i v_*), Π_SM(τ^j v_*)⟩

with τ = Spin(8) triality.  We replace τ with the BCC body-diagonal
Z₃ rotation R acting on flavor / σ^a-space, with the lifted spinor
action U_3 = ``dirac_spinor_lift()`` on the chiral-16 internal
carrier.

### FKO-8: Choice of starting vector v_*

Several natural candidates from the program:

1. **Restricted hypercharge Y'**: the broken_triality default; the
   SM hypercharge restricted to Spin(8)/PS Cartan.
2. **Color-singlet charge-shift basis elements** from
   ``cp/j_misalignment.color_singlet_charge_shift_basis()`` — the
   dim-4 "Higgs map space" — projected onto a 3-dim σ^a subspace.
3. **Pauli-axis directions** σ^x, σ^y, σ^z directly (would give
   trivial Yukawa unless modulated by Higgs VEV).
4. **J-aligned starting vectors** from ``cp/j_misalignment.viable_j_candidates()``.

### FKO-9: 3×3 Yukawa eigenvalue extraction

For each v_*:

    orbit = (v_*, R v_*, R² v_*)
    Y_{ij} = ⟨Π(orbit_i), Π(orbit_j)⟩   for some projection Π

(Π choices: trace projection, J-aligned, color-singlet — to be
tried.)

Compute eigenvalues {μ_1, μ_2, μ_3} of Y, take v⃗_Y = (√μ_1, √μ_2,
√μ_3), check Koide K_Y from v⃗_Y.

Pre-named failure mode: rank-2 + zero (broken_triality BT-1
outcome).  In that case v⃗_Y has a zero component and Koide is
ill-defined (K could be evaluated by limit but loses meaning).

Output: ``koide/yukawa_eigenvalue_locus.py``.

### FKO-10: Systematic scan over starting vectors

Iterate over the dim-4 Higgs map space; for each v_* compute Y,
eigenvalues, K_Y.  Build the LOCUS:

    L = { (μ_1, μ_2, μ_3) reachable from the program's
          algebraic constraints }

The locus is a sub-variety of ℝ³₊ parametrized by v_* + projection
choices.

## Phase KO-4 — Cone vs. locus comparison (~3-4 days)

### FKO-11: Locus-cone intersection

For each point in L (eigenvalue triple), compute K_Y.  Classify:

- K_Y = 2/3 within tolerance: point on the cone.
- |K_Y − 2/3| > tolerance: point off the cone.

Fraction of L on the cone, and where on the cone PDG sits:

    f_cone = |L ∩ C| / |L|              (cone density of locus)
    f_PDG_in_L = does PDG point ∈ L?    (PDG reachability)

### FKO-12: Verdict classifier

    if  L ⊂ C  →  KOIDE PREDICTED
    if  L ∩ C ≠ ∅ and L ⊄ C  →  KOIDE CONSISTENT
    if  L ∩ C = ∅  →  KOIDE CONFLICT
    if  PDG ∈ L  →  ADDITIONAL TAG "PDG IN LOCUS" (stronger positive)
    if  PDG ∉ L  →  ADDITIONAL TAG "PDG NOT IN LOCUS" (weaker)

Output: ``koide/cone_locus_compatibility.py``.

## Phase KO-5 — Combined audit + SESSION report (~3 days)

### FKO-13: KoideAuditPayload

Aggregate:

- ``empirical_K_pdg``: K computed from PDG.
- ``equipartition_holds_for_pdg``: |v_trace|² ≈ |v_traceless|²?
- ``z3_irrep_decomposition``: trace + 2D-non-trivial Z₃ irrep verification.
- ``yukawa_eigenvalue_locus``: from KO-3, the L set (or its
  characterization).
- ``locus_cone_classification``: PREDICTED / CONSISTENT / CONFLICT.
- ``pdg_in_locus``: yes / no.
- ``final_verdict``: from above.

### FKO-14: SESSION_KOIDE.md

Structure analogous to ``SESSION_TOPOLOGY.md``, ``SESSION_SME.md``,
``SESSION_STRONGCP.md``:

```markdown
# Koide — flavor 45°-cone audit verdict

**Status**: CLOSED — {PREDICTED / CONSISTENT / CONFLICT}

## Load-bearing question
## Phase-by-phase findings (KO-1..KO-4)
## Combined verdict
## What this does NOT close
## Test summary
```

## Pre-named failure modes

**F-koide-1**: empirical K_PDG ≠ 2/3 within 10⁻³.  Diagnosis:
PDG-value bug or m_τ uncertainty (small possibility); reset.

**F-koide-2**: σ^a → generation identification arbitrary; no
physical principle pins it.  Diagnosis: HONEST CAVEAT — document
the limitation in SESSION report.  The audit becomes "given this
identification, is Koide PREDICTED?"; weaker positive than a
derivation that picks the identification uniquely.

**F-koide-3**: broken_triality-style Yukawa collapses to rank-2 +
zero.  Diagnosis: BT-1-style outcome; the 3-vector v⃗_Y has a
zero component and Koide K_Y is ill-defined.  Try different v_*,
different projections, or report as a failure mode of the
construction.

**F-koide-4**: locus L is FAR from cone C (|K_Y − 2/3| >> 10⁻²
for typical v_*).  Diagnosis: CONFLICT verdict — the program
forbids Koide.  Apparent BCC↔Koide coincidence is broken by
program structure.

**F-koide-5**: locus L equals the cone C (or 1-parameter family
on C).  Diagnosis: PREDICTED verdict — major positive.

**F-koide-6**: locus L is broad and contains the cone but also
much else; PDG happens to land on C by phenomenological choice
of v_*.  Diagnosis: CONSISTENT but not PREDICTED.  Most likely
realistic outcome.

**F-koide-7**: dim-4 Higgs map space scan reveals MULTIPLE
disconnected cone-intersecting sub-loci; ambiguous classification.
Diagnosis: report ambiguity in SESSION; classify by the
specifically-reachable sub-locus that contains PDG (if any).

## Critical files

To create (under ``src/clifford_3plus2_d5/koide/``):

```text
koide/
  __init__.py
  PLAN.md                              (this plan as reference)
  STATUS.md
  parameter_ledger.md
  reuse.py                             # imports from cp/, topology/, broken_triality/, spacetime_qca
  koide_geometry.py                    # Phase KO-1: cone parametrization, PDG K verification
  bcc_z3_on_flavor.py                  # Phase KO-2: Z₃ irrep decomposition on σ^a-space
  yukawa_eigenvalue_locus.py           # Phase KO-3: BCC-Z₃-orbit 3×3 Yukawa + scan
  cone_locus_compatibility.py          # Phase KO-4: PREDICTED/CONSISTENT/CONFLICT classifier
  koide_audit.py                       # Phase KO-5: combined payload
  SESSION_KOIDE.md                     # final verdict
  tests/
    __init__.py
    test_koide_geometry.py
    test_bcc_z3_on_flavor.py
    test_yukawa_eigenvalue_locus.py
    test_cone_locus_compatibility.py
    test_koide_audit.py
```

To consult / read-only:

```text
src/clifford_3plus2_d5/cp/cubic_harmonics.py             # T_{2g} projector
src/clifford_3plus2_d5/cp/continuum_cp.py                # H^(1), CP-odd T_{2g} cell
src/clifford_3plus2_d5/cp/j_misalignment.py              # viable J's, color-singlet charge-shift basis, J-decomposition
src/clifford_3plus2_d5/topology/bcc_z3_rotation.py       # R matrix, Dirac spinor lift, permutation
src/clifford_3plus2_d5/broken_triality/yukawa_overlaps.py # 3×3 Yukawa-from-orbit template
src/clifford_3plus2_d5/broken_triality/mass_hierarchy.py  # eigenvalue + ratio test pattern
src/clifford_3plus2_d5/spacetime_qca/yukawa.py           # Hermitian Y(Φ) on chiral-16
```

## What this plan does NOT include (deferred)

- **Three-generation derivation**.  Closed negative by
  triality/broken_triality/exceptional/topology.  Three generations
  are phenomenological input.
- **Quark Koide / mass-vector audit**.  Up-quark K ≈ 0.85, down-quark
  K ≈ 0.73 — Koide is generation-precise for charged leptons but
  imprecise for quarks.  A future analog audit could pursue the
  partial pattern; out of scope here.
- **Dynamical Higgs VEV alignment with (1,1,1)**.  Bold B territory.
  If KO-4 returns CONSISTENT (cone-residing locus exists, requires
  VEV input), Bold B would investigate whether the VEV dynamics
  prefer the cone direction.  Bold B is ~2 months of separate work.
- **Neutrino Koide**.  m_ν ≪ m_e; neutrino sector requires separate
  modelling and is not part of the charged-lepton Koide formula.
- **Photon-sector audit** (k_F)^(5).  Orthogonal to Koide.
- **Strong-CP SC-4 direct lattice computation**.  Independent
  deferred follow-up from strongcp/.

## Verification

End-to-end test sequence after each phase:

```bash
# Phase KO-1
uv run pytest src/clifford_3plus2_d5/koide/tests/test_koide_geometry.py -q

# Phase KO-2
uv run pytest src/clifford_3plus2_d5/koide/tests/test_bcc_z3_on_flavor.py -q

# Phase KO-3
uv run pytest src/clifford_3plus2_d5/koide/tests/test_yukawa_eigenvalue_locus.py -q

# Phase KO-4
uv run pytest src/clifford_3plus2_d5/koide/tests/test_cone_locus_compatibility.py -q

# Phase KO-5
uv run pytest src/clifford_3plus2_d5/koide/tests/test_koide_audit.py -q

# Full sidecar
uv run pytest src/clifford_3plus2_d5/koide/tests -q

# Regression: existing sidecars stay green
uv run pytest src/clifford_3plus2_d5/cp/tests -q
uv run pytest src/clifford_3plus2_d5/topology/tests -q
uv run pytest src/clifford_3plus2_d5/broken_triality/tests -q
uv run pytest src/clifford_3plus2_d5/strongcp/tests -q
uv run pytest src/clifford_3plus2_d5/sme/tests -q
```

Verdict callable:

```bash
uv run --no-sync python -c "
from clifford_3plus2_d5.koide.koide_audit import koide_audit_payload
p = koide_audit_payload()
print(p.final_verdict)
print(f'PDG K = {float(p.empirical_K_pdg):.6f} vs 2/3 = {2/3:.6f}')
print(p.locus_cone_classification)
print(p.interpretation)
"
```

Expected outcome (a priori unknown — that is the point of the audit):

- Most likely **CONSISTENT** — the program's Yukawa construction
  admits cone-residing solutions but requires VEV / projection
  input to land specifically on the cone.
- Possible **PREDICTED** if there is a tight enough structural
  constraint (e.g., the BCC-Z₃ orbit + J-decomposition uniquely
  forces equipartition).
- Possible **CONFLICT** if the program rules out the cone — that
  would explicitly close the BCC↔Koide coincidence as not deep.

Any of the three outcomes is publishable.

## Effort budget

- Phase KO-1: ~2 days.
- Phase KO-2: ~2-3 days.
- Phase KO-3: ~5-7 days.
- Phase KO-4: ~3-4 days.
- Phase KO-5: ~3 days.

**Total committed budget**: ~3-4 weeks.

**Worst case**: ~3-4 weeks for a CONSISTENT verdict (the program
admits Koide but doesn't force it).  Best case: PREDICTED in ~2-3
weeks if the structural test in KO-4 closes positive quickly.
Kill-discipline ordering: KO-1, KO-2 are empirical / setup
(unlikely to surprise).  KO-3 (broken_triality-style Yukawa) is
the risk phase — could collapse to rank-2 + zero, requiring
recovery work.  KO-4 is the verdict.

Kill-discipline applies less directly here than for the
algebraic / topological sidecars (Koide's outcomes are not binary
pass/fail).  But the audit returns a single classifier with three
honest outcomes, all of which are publishable.
