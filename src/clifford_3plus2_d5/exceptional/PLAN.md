# Plan: Bold C sidecar — three-generation exceptional-algebra audit

## User-confirmed scope decisions

- **Sidecar name**: ``exceptional/`` (broader umbrella covering J_3(O) + complexified
  follow-on, not just Furey-bimultiplication).
- **Phase 0**: included — quick triages of bimultiplication and Fano-line candidates
  produce documented verdicts.
- **Phase 2b**: included — if J_3(O) under Spin(10) closes negative, run a 1-week
  follow-on testing the complexified J_3^C(O) extension that Boyle's three-generation
  argument actually uses.

## Context

The triality sidecar (closed negative, K1 fail) and broken_triality sidecar (closed
negative, BT-2 fail) both failed to derive three SM generations from Spin(8)-related
mechanisms. The cp sidecar produced a strong positive structural CP result, but it
remains one-generation-only.

Bold C is the next kill-disciplined attack on the three-generation gap. Rather than
testing exotic group structures one-at-a-time, the sidecar:

1. Quickly triages the "obvious" candidates that have plausible but likely-fatal
   issues (bimultiplication-of-T = Spin(8) triality; three Fano-line SU(2)s).
2. Commits the main effort to the most-promising untried candidate: the **Boyle
   exceptional Jordan algebra J_3(O)**, whose 27-dim representation decomposes under
   `E_6 ⊃ Spin(10) × U(1)` and whose three-row matrix structure is the cleanest
   non-triality three-fold mechanism in the literature.
3. Stops at any clean negative; commits only to the next phase if the previous
   phase's verdict justifies it.

The single load-bearing question:

> Can J_3(O) (or a closely-related exceptional Jordan / E_6 structure) carry
> three SM generations of the chiral-16 type without declaring "three" by hand?

Worst case: ~1-2 weeks for a clean negative. Best case: ~4-6 weeks for a positive
result that opens phenomenological work (Yukawa structure, CP across generations).

## Existing infrastructure to reuse (via `exceptional/reuse.py`)

Verified by exploration:

**Octonion ops** (`lepton/clifford_octonion.py`):
- `octonion_multiply(left, right)`: 8×1 × 8×1 → 8×1 SymPy.
- `octonion_left_multiplication(index)`, `octonion_right_multiplication(index)`: 8×8.
- `octonion_structure_constants()`, `octonion_fano_triples()`: Fano table data.
- `octonion_derivation_basis()`: 14-dim G_2 derivations as 8×8 matrices.
- `su3_stabilizer_basis(7)`: 8-dim SU(3) ⊂ G_2 fixing `e_7`.

**Clifford / Pati-Salam** (`lepton/clifford_patisalam.py`):
- `patisalam_cl010_gamma_matrices()`: 10 64×64 gammas.
- `patisalam_chiral16_block_matrix()`: project Spin(10) generators to chiral-16.
- `patisalam_chiral16_basis_matrix("+")`: 64×32 column matrix of chiral-16 basis.
- `su4_generators_from_spin06()`, `su2_l_generators_from_spin04()`, etc.

**SM / hypercharge** (`lepton/patisalam_sm.py`, `lepton/sm_hypercharge.py`):
- `su3_c_generators_from_su4`, `physical_hypercharge_generator`, etc.
- `EXPECTED_JOINT_Y_T3L_TABLE`: target one-generation charge table.

**CP machinery** (`cp/j_misalignment.py`):
- `cp_violating_fraction`, `j_decomposition` — reusable for CP across generations.

**What must be built new**: J_3(O) algebra (27-dim octonionic Hermitian matrices),
its Jordan product / trace / cubic norm, the 45 Spin(10) generators (21 already
accessible; 24 cross-terms constructible from `combinations(range(10), 2)`),
explicit Spin(10) action on J_3(O), the `27 = 16 + 10 + 1` decomposition.

**Not** required (deferred until/unless Phase 3 demands it): explicit E_6
generators (78-dim Lie algebra), F_4 generators (52-dim), full norm-preserving
group action. The kill condition can be evaluated using only Spin(10) and the
Jordan-algebra structure.

## Decision tree

```text
Phase 0 (triage, ~2-3 days)
  ├── C-1: bimultiplication → triality          [expected: trivial K1 fail]
  └── C-3: three Fano-line SU(2)s               [expected: SM-shape fail]
        │
        ▼
Phase 1 (J_3(O) construction, ~1 week)
  Build the 27-dim algebra, Jordan product, three rows, tests.
        │
        ▼
Phase 2 (KILL TEST: J_3(O) decomposition, ~1-2 weeks)
  Compute decomposition of 27 under Spin(10) ⊂ E_6.
  ├── If 27 = 1×(chiral-16) + 10 + 1:    CLOSE — one generation + extras, not three.
  ├── If 27 = 3×(chiral-16)/3 + ... :    PROCEED to Phase 3 (very surprising).
  └── If something else interesting:     report and decide case-by-case.
        │
        ▼ (if Phase 2 closes with standard 16+10+1, the EXPECTED outcome)
Phase 2b (J_3^C(O) extension, ~1 week)
  Build the complexified J_3^C(O) (54-dim) and check whether Boyle's actual
  three-generation argument survives.  Specifically: 27_C = 27 + 27* under
  E_6 × U(1)? Three "copies" via real/imaginary/conjugate?
  ├── If 27_C still decomposes as one-generation + extras:  FINAL KILL.
  └── If 27_C carries genuine 3-generation structure:        proceed to Phase 3.
        │
        ▼ (only if Phase 2 OR Phase 2b produces a three-generation structure)
Phase 3 (Yukawa + CP, ~3-4 weeks)
  Build off-diagonal Yukawa matrix from J_3(O) entries; CP analysis across
  generations using cp/ infrastructure.
```

A negative at any phase closes the sidecar with a clean negative report.

Phase 2b is a **decisive extension** of Phase 2.  If Phase 2 closes with the
standard 16+10+1 (the expected outcome), Phase 2b tests whether Boyle's
real three-generation argument — which uses J_3^C(O), not just J_3(O) — could
still work.  This way the sidecar's final negative is on the FULL J_3(O)-family
approach, not just the simplest version.

## Phase 0 — triage (~2-3 days)

### FT-0a — Bimultiplication-of-T quick check (~1 day)

The Furey "bimultiplication" candidate: combine left- and right-multiplications
on `T = C ⊗ H ⊗ O`. For the octonion factor, `L(O) + R(O)` generates the
bimultiplication algebra `Bi(O)`, which by classical results is `Spin(8)` via
triality. We have already shown Spin(8) triality fails K1 in `triality/`.

**What to do**: build the bimultiplication algebra symbolically (using
`octonion_left_multiplication` and `octonion_right_multiplication`), verify
it generates a Spin(8)-like algebra, confirm the triality fail still applies.

**Kill condition**: if Bi(O) is exactly Spin(8) and the K1 fail is reproduced,
close with a one-line "C-1 confirms triality" note.

**Unexpected outcome**: if Bi(O) provides a structure not reducible to triality,
that's interesting and would justify a longer investigation.

### FT-0b — Three Fano lines quick check (~1-2 days)

The three Fano lines through `e_7` give three H-subalgebras `{1, e_7, e_i, e_j}`,
hence three SU(2) actions. Three SU(2)s could-in-principle be three generations'
weak interactions.

**What to do**: build the three SU(2) actions on the chiral-16 carrier. Check
whether they:
1. Coincide as Lie algebras (a single SU(2) acting three ways on a tripled
   carrier).
2. Or are independent (three separate gauge groups, which contradicts SM).
3. Or split into "one diagonal SU(2)_L + 2 orthogonal" structure.

**Kill condition**: if the three SU(2)s don't combine into a single SU(2)_L
acting on three doublets (as the SM requires), close with a clean negative.

## Phase 1 — Build J_3(O) (~1 week)

### FJ-1 — Algebra construction

J_3(O) = 3×3 Hermitian octonion matrices. 27 real dimensions:
- 3 diagonal real entries.
- 3 off-diagonal octonion entries (above-diagonal); below-diagonal is determined
  by Hermiticity `M_ji = conj(M_ij)`.

**Implementation**:
- Represent `M ∈ J_3(O)` as a dataclass holding (3 real numbers, 3 octonion 8-vectors).
- Hermitian product: `M ·_J N = (MN + NM) / 2` where `MN` is matrix multiplication
  with the octonion product `octonion_multiply`. Note: J_3(O) is power-associative
  but not associative; the Jordan product is symmetric and ensures the result
  stays in J_3(O).
- Linear trace: `Tr(M) = m_11 + m_22 + m_33` (sum of diagonal reals).
- Bilinear form: `(M, N) = Tr(M · N)`. Used for orthogonality.
- Cubic norm: `N(M) = det(M)` defined by the unique cubic invariant
  `N(M) = m_11 m_22 m_33 - sum of off-diagonal terms`. Used to identify the
  E_6 group action (preserves this cubic form).

**Tests**:
- 27 real dimensions verified (basis count).
- Jordan product symmetric: `M · N = N · M`.
- Power-associativity: `M · (M · M) = (M · M) · M` for any specific test M.
- Trace is linear; cubic norm is cubic.

### FJ-2 — Three-row decomposition (preparatory)

Identify the three rows of J_3(O) as three "naive" generation slots. Each row
is a 9-dim subspace: 1 diagonal real + 1 off-diagonal octonion = 9 real DOF.

**Tests**: 3 × 9 = 27 ✓ (after removing double-count from Hermiticity).

This is preparatory — the actual three-generation question requires the
Spin(10) action (Phase 2), not just the matrix-row decomposition.

## Phase 2 — KILL TEST: Spin(10) decomposition of 27 (~1-2 weeks)

### FJ-3 — Build the 45 Spin(10) generators

Use `patisalam_cl010_gamma_matrices()` and `combinations(range(10), 2)`:

```text
for (i, j) in combinations(range(10), 2):
    bivector_full = (gammas[i] * gammas[j]) / 2
    bivector_on_chiral16 = patisalam_chiral16_block_matrix(bivector_full)
```

This gives 45 generators on the chiral-16 (32×32 real-skew matrices).

**Tests**: 45 generators, all skew-symmetric on chiral-16, the existing
`spin06_generators_chiral16()` (15) and `spin04_generators_chiral16()` (6) match
the corresponding subset of the new 45.

### FJ-4 — Identify Spin(10) ⊂ E_6 action on J_3(O)

This is the load-bearing step.

The standard construction (Boyle 2020, Krasnov 2018, Manogue-Dray classic):
E_6 acts on the 27-dim J_3(O) preserving its cubic norm. Spin(10) ⊂ E_6 acts on
27 reducibly:

```text
27 = 16 ⊕ 10 ⊕ 1
where 16 = Spin(10) chiral spinor (= our chiral-16),
      10 = Spin(10) vector rep,
      1  = singlet.
```

**Implementation**: identify the chiral-16, vector-10, and singlet subspaces
of J_3(O) explicitly. Standard recipe:
- 1 singlet: the trace `Tr(M)`.
- 10 vector: the traceless diagonal + 3 cross-block real parts of off-diagonal.
- 16 chiral spinor: built from off-diagonal octonion entries via the
  "octonion-as-spinor" identification `O ≅ chiral spinor of Spin(8)`. Combined
  with the 3 off-diagonal entries and a sign-pattern from the Hermitian
  structure, this should give a Spin(10) chiral-16.

**Tests**:
- 16 + 10 + 1 = 27 dimension check.
- All 45 Spin(10) generators preserve each subspace (so they act block-diagonally
  on the 16 + 10 + 1 decomposition).
- The 16 piece matches lepton's chiral-16 (verified by comparing SU(3)_c × SU(2)_L ×
  U(1)_Y charge content).

### FJ-5 — Kill test

The kill condition is now precise:

```text
Decomposition pattern         Verdict
─────────────────────────────  ────────────────────────────────────────────
27 = 16 + 10 + 1               KILL: J_3(O) is one generation + extras, NOT
                               three generations. Boyle interpretation must
                               rely on a different structure than just J_3(O)
                               under Spin(10). Close with negative.

27 = 16 + 16 + ... (multiple   PASS: a genuine multi-generation structure
chiral-16 copies)               exists. Proceed to Phase 3.

27 decomposes differently       INVESTIGATE: report the decomposition and
                               decide.
```

**Expected outcome (honest estimate)**: the standard E_6 representation theory
says 27 = 16 + 10 + 1 under Spin(10). This means one chiral-16 + extras, not
three. The sidecar will likely close cleanly here.

If the standard decomposition is reproduced, the verdict is:

> J_3(O) provides ONE chiral-16 of Spin(10) (matching lepton's existing
> carrier), plus a 10-dim vector representation and a singlet — NOT three
> generations. The three-row appearance of J_3(O) is misleading; the actual
> Spin(10)-irreducible decomposition does not split into three generation
> copies. Boyle's three-generation interpretation must rely on additional
> structure (e.g., complexified J_3(O), tensor with another algebra, or a
> different group action) not captured by J_3(O) under Spin(10) alone.

This is a clean, publishable negative result.

## Phase 2b — J_3^C(O) extension (~1 week, only if Phase 2 closes negative)

The complexification `J_3^C(O) = J_3(O) ⊗_R C` is a 54-real-dimensional space.
Under the natural extension `E_6 × U(1) ⊃ Spin(10) × U(1) × U(1)`, the 27_C
representation `54 = 27 + 27*` could in principle carry genuine multi-generation
structure that the real J_3(O) alone misses.

Boyle's actual three-generation proposal uses this complexified structure (the
27 and its conjugate 27* together provide the "two halves" that combine into
three SM-generation slots via specific projections).

### FJ-2b — Build J_3^C(O) and re-decompose

**Implementation**:
- Promote each octonion entry from 8 real DOF to 8 complex DOF.  Total: 3 real
  diagonals (complexified to 3 × 2 = 6 real) + 3 complex octonions
  (complexified to 3 × 16 = 48 real) = 54 real.
- Identify the real/imaginary subspaces under the new complex structure.
- Decompose under Spin(10) × U(1) using the 45 generators built in Phase 2.

**Tests**:
- 54-dim verified.
- Spin(10) action lifts cleanly.
- Decomposition 54 = 16 ⊕ 16* ⊕ ... reproducible.

### FJ-5b — Final kill test

```text
Decomposition under Spin(10) × U(1)             Verdict
──────────────────────────────────────────────  ────────────────────────────────
54 = (16 + 10 + 1) + (16* + 10* + 1*)            KILL: complexification just doubles
                                                  the existing one-generation
                                                  content into particle + antiparticle.
                                                  Not three generations.

54 contains 3 distinct chiral-16 copies          PASS: genuine three-generation
                                                  structure. Proceed to Phase 3.

Other decomposition                              INVESTIGATE.
```

**Expected outcome (honest estimate)**: most likely 16 + 16* + 10 + 10* + 1 + 1*,
giving a one-generation-with-antiparticle structure rather than three
generations.  This would close the sidecar with a stronger negative:

> Neither J_3(O) nor its complexification J_3^C(O) under Spin(10) gives three
> generation copies.  Boyle's three-generation interpretation must rely on
> additional structure beyond exceptional Jordan algebra under Spin(10) (e.g.,
> tensor with another algebra, or a different group than Spin(10)).  The
> exceptional-Jordan approach to three generations is closed.

## Phase 3 (conditional, ~3-4 weeks) — Yukawa + CP if Phase 2 OR Phase 2b passes

Only enter Phase 3 if Phase 2 reveals an unexpected decomposition that suggests
a genuine multi-generation structure.

**Sketch** (deferred details if reached):
- The off-diagonal octonion entries of J_3(O) provide 3 × 8 = 24 real DOF that
  could carry Yukawa-like inter-generation couplings.
- Apply the multi-element β audit pattern: compute CP-violating fraction per
  generation pair (3 × 3 = 9 entries → 6 independent pairs).
- Build a Yukawa matrix and check non-degenerate eigenvalues.
- Identify the CP-violating slot at the three-generation level.

If Phase 2 closes negative, Phase 3 is unwritten.

## Pre-named failure modes

**F-exc-1**: Bimultiplication doesn't generate Spin(8) cleanly. Diagnosis:
non-associativity gives extra structure; check whether it's a Spin(9) or larger
algebra. Unlikely to change the verdict (the failure of K1 will still apply at
the relevant subgroup), but worth documenting.

**F-exc-2**: The three Fano-line SU(2)s are mutually orthogonal Lie algebras
(no shared structure). Diagnosis: three independent SU(2)s ≠ SM's single
SU(2)_L. Close with note.

**F-exc-3**: J_3(O) Jordan product fails associativity tests beyond
power-associativity. Diagnosis: implementation bug in `octonion_multiply` lifting
to 3×3. Standard Jordan algebra theorem says J_3(O) IS power-associative;
failure indicates a bug.

**F-exc-4**: 45 Spin(10) generators on chiral-16 don't reproduce the existing
`su4_generators_from_spin06 + su2_l_generators_from_spin04` content. Diagnosis:
ordering / convention bug. Cross-check the index `combinations(range(10), 2)`
against the existing `range(6)` and `range(6,10)` subsets.

**F-exc-5**: 27 = 16 + 10 + 1 is the decomposition. EXPECTED. Close with the
clean negative summary in FJ-5 above.

**F-exc-6**: 27 decomposes as 16 + 16 + ... (multi-generation). UNEXPECTED.
Proceed to Phase 3 and write up the structural surprise.

**F-exc-7**: Spin(10) preserves J_3(O) but breaks at the cubic norm. Diagnosis:
the cubic norm requires the FULL E_6 (not Spin(10)) for preservation. If we only
care about decomposing 27 under Spin(10), the cubic norm is auxiliary.

## Critical files

To create (under `src/clifford_3plus2_d5/exceptional/`):

```text
exceptional/
  __init__.py
  PLAN.md
  STATUS.md
  parameter_ledger.md
  reuse.py                              # imports from lepton, cp
  bimultiplication.py                   # Phase 0a (C-1 quick check)
  fano_lines.py                         # Phase 0b (C-3 quick check)
  j3o_algebra.py                        # Phase 1: J_3(O) algebra
  spin10_on_j3o.py                      # Phase 2: 45 generators + decomposition
  decomposition_audit.py                # FJ-5 kill test payload
  j3o_complex.py                        # Phase 2b: J_3^C(O) extension
  complex_decomposition_audit.py        # FJ-5b kill test payload
  SESSION_EXCEPTIONAL.md                # final verdict report
  tests/
    __init__.py
    test_bimultiplication.py
    test_fano_lines.py
    test_j3o_algebra.py
    test_spin10_on_j3o.py
    test_decomposition_audit.py
    test_j3o_complex.py
    test_complex_decomposition_audit.py
```

To consult / read-only:

```text
src/clifford_3plus2_d5/lepton/clifford_octonion.py
src/clifford_3plus2_d5/lepton/clifford_patisalam.py
src/clifford_3plus2_d5/lepton/patisalam_sm.py
src/clifford_3plus2_d5/lepton/sm_hypercharge.py
src/clifford_3plus2_d5/triality/spin8_triality.py      # for C-1 cross-check
src/clifford_3plus2_d5/cp/j_misalignment.py            # for Phase 3 CP analysis
```

## What this plan does NOT include (deferred)

- Full E_6 generator construction (78 generators). The kill test only needs the
  45 Spin(10) generators acting on the 27 = 16 + 10 + 1 decomposition. Phase 2b
  may need one additional U(1) generator for the complexified version.
- Full F_4 (= Aut(J_3(O))) generator construction (52 generators). Not needed
  for the kill test.
- Sedenions / Cayley-Dickson higher steps. The plan explicitly does not pursue
  this beyond a brief note in the failure analysis.
- Magnitude / phenomenology matching to CKM. Even if Phase 3 produces a
  three-generation structure, phenomenology is a separate investigation.
- E_6 × G_2 / F_4 × G_2 tensor structures (sometimes invoked in three-generation
  programs). Deferred to a separate sidecar if the present one closes negative
  and the program continues to pursue exceptional structures.

## Verification

End-to-end test sequence after each phase:

```bash
# Phase 0 (after both triages)
uv run pytest src/clifford_3plus2_d5/exceptional/tests/test_bimultiplication.py -q
uv run pytest src/clifford_3plus2_d5/exceptional/tests/test_fano_lines.py -q

# Phase 1 (after J_3(O) construction)
uv run pytest src/clifford_3plus2_d5/exceptional/tests/test_j3o_algebra.py -q

# Phase 2 (after Spin(10) decomposition)
uv run pytest src/clifford_3plus2_d5/exceptional/tests/test_spin10_on_j3o.py -q
uv run pytest src/clifford_3plus2_d5/exceptional/tests/test_decomposition_audit.py -q

# Phase 2b (after J_3^C(O) extension, if Phase 2 closes negative)
uv run pytest src/clifford_3plus2_d5/exceptional/tests/test_j3o_complex.py -q
uv run pytest src/clifford_3plus2_d5/exceptional/tests/test_complex_decomposition_audit.py -q

# Full sidecar after each milestone
uv run pytest src/clifford_3plus2_d5/exceptional/tests -q

# Regression: existing tests stay green
uv run pytest src/clifford_3plus2_d5/cp/tests -q
uv run pytest src/clifford_3plus2_d5/lepton/tests -q
```

Verdict callable after Phase 2 (and Phase 2b if reached):

```bash
uv run --no-sync python -c "
from clifford_3plus2_d5.exceptional.decomposition_audit import j3o_decomposition_audit_payload
from clifford_3plus2_d5.exceptional.complex_decomposition_audit import j3o_complex_decomposition_audit_payload
p = j3o_decomposition_audit_payload()
print('Phase 2 verdict:', p.verdict)
print(p.interpretation)
if not p.passes:
    pc = j3o_complex_decomposition_audit_payload()
    print('Phase 2b verdict:', pc.verdict)
    print(pc.interpretation)
"
```

Expected output (per F-exc-5 most-likely scenario):

> Phase 2 verdict: J_3(O) KILL — 27 = 16 + 10 + 1
> J_3(O) provides one chiral-16 of Spin(10) plus a 10 + 1, not three
> generations. Standard E_6 representation theory confirmed.
>
> Phase 2b verdict: J_3^C(O) KILL — 54 = 16 + 16* + 10 + 10* + 1 + 1*
> Complexification doubles into particle/antiparticle, still one
> generation plus extras.  Exceptional Jordan approach to three
> generations is closed at the carrier level.

## Effort budget

- Phase 0 (triage): 2-3 days.
- Phase 1 (J_3(O) construction): ~1 week.
- Phase 2 (Spin(10) decomposition + kill test): ~1-2 weeks.
- Phase 2b (J_3^C(O) extension, if Phase 2 closes negative): ~1 week.
- Phase 3 (conditional, only if Phase 2 OR Phase 2b passes): ~3-4 weeks.

**Worst case**: ~3-4 weeks for a clean negative on the full J_3(O) family
(Phase 2 negative + Phase 2b negative).
**Best case**: ~5-7 weeks if Phase 3 is reached and produces three-generation
structure.

The kill-discipline ordering means Phase 0, Phase 2, and Phase 2b are the
load-bearing decision points. Phase 3 is the post-pass payoff and is not
committed to before Phase 2 or Phase 2b's verdict is in.
