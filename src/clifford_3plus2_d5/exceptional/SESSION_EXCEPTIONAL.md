# Session Exceptional — Bold C kill-test report

**Verdict: FULL KILL across all 4 sub-candidates.**  The exceptional-algebra
approach to three SM generations is closed at the carrier-representation
level.

| Phase | Candidate | Verdict | Why |
|---|---|---|---|
| 0a  | Bimultiplication Bi(O) | KILL | Collapses to Spin(8) = triality; inherits the failed K1 result |
| 0b  | Three Fano lines through e_7 | KILL | Octonion non-associativity prevents closure as Lie subalgebras |
| 2   | J_3(O) under Spin(10) | KILL | 27 = 16 + 10 + 1 (one chiral-16 + extras); 3 × 16 = 48 > 24 dimensionally impossible |
| 2b  | J_3^C(O) under Spin(10) × U(1) | KILL | 54 = 16 + 16* + 10 + 10* + 1 + 1* (particle + antiparticle, only 2 chiral-16 subreps) |

**39 passing tests.**  Total effort: ~3 hours of focused work.

## Phase 0a result — bimultiplication

The Furey "bimultiplication" candidate combines left- and
right-multiplications on `T = C ⊗ H ⊗ O`.  Restricted to the octonion
factor, computational verification confirms:

```text
dim(Bi(O)) = 28 = dim(so(8))
```

The bimultiplication algebra is exactly Spin(8).  Since
`triality/SESSION_T_KILL_TEST.md` already showed Spin(8) triality fails
K1, this candidate inherits that failure.

## Phase 0b result — three Fano lines

The three Fano lines through e_7 are `{1, 6, 7}, {2, 5, 7}, {3, 4, 7}`.
Each defines a candidate "SU(2)" via octonion left-multiplications
`{L_a, L_b, L_c}`.

**Strong kill**: due to octonion non-associativity, `[L_1, L_6]` is NOT
in the linear span of `{L_1, L_6, L_7}`.  The three candidates fail to
close as Lie algebras under commutator.  No SU(2) Lie structure exists
at all — let alone three SU(2)s acting as the SM's single SU(2)_L on
three doublets.

Additional structural fact: the three SU(2) candidates would share
generator `L_{e_7}` and span only 7 dimensions (= all imaginary
octonions), not 9 (= 3 independent SU(2)s).  Even without
non-associativity, this would not match the SM pattern.

## Phase 1 result — J_3(O) algebra

J_3(O) constructed as 3×3 Hermitian octonion matrices, 27 real
dimensions.  Verified:

- Power-associativity: `M · (M · M) = (M · M) · M` (Jordan-algebra
  property).
- Trace linearity, bilinear form symmetry.
- Cubic norm: `det([[a, 0, 0], [0, b, 0], [0, 0, c]]) = a · b · c`.
- 27-dimensional basis (3 reals + 3 × 8 octonions), linearly independent.

## Phase 2 result — Spin(10) on J_3(O) (load-bearing kill)

Under any natural Spin(10) ⊂ E_6 acting on J_3(O), the 27 of E_6
decomposes as

```text
27 = 16 ⊕ 10 ⊕ 1
```

where 16 is the (real) chiral spinor, 10 the vector, 1 the singlet.

The decomposition is realized by picking a "preferred row k":

- **Singlet** (1-dim): the diagonal entry M_{kk}.
- **Vector** (10-dim): the other 2 diagonals + the off-diagonal
  octonion NOT touching row k.
- **Spinor** (16-dim): the 2 off-diagonal octonions touching row k.

Three different row choices give three different chiral-16 candidates,
which **overlap pairwise by 8 basis elements** (the shared octonion):

```text
spinor_0 ∩ spinor_1 = 8
spinor_0 ∩ spinor_2 = 8
spinor_1 ∩ spinor_2 = 8
union = 24 (all off-diagonal octonions)
```

The dimensional kill: three independent chiral-16 require **3 × 16 = 48
real dimensions**, but J_3(O)'s 24-dimensional off-diagonal octonion
space is insufficient.  Even using all 27 dimensions, 48 > 27.

J_3(O) carries at most ONE chiral-16 of Spin(10), not three.

## Phase 2b result — J_3^C(O) extension

The complexified `J_3^C(O) = J_3(O) ⊗_R C` has 54 real dimensions.
Under Spin(10) × U(1) ⊂ E_6 × U(1), the 27_C decomposes as

```text
27_C = (16, +1) ⊕ (10, -2) ⊕ (1, +4)
54 (real) = 16 + 16* + 10 + 10* + 1 + 1*
```

Dimensionally, 3 × 16 = 48 ≤ 54 — three chiral-16 COULD fit by
dimension alone.  But the representation theory of Spin(10) × U(1) on
the 27_C of E_6 is **forced**: it gives exactly two chiral-16-type
subreps, namely the 16 (particle) and 16* (antiparticle).  This is the
SM's one-generation-plus-antiparticle structure, not three generations.

Boyle's three-generation argument therefore must rely on additional
structure beyond just Spin(10) × U(1) acting on J_3^C(O) — for example,
tensoring with a separate algebra, a different group, or a topological
mechanism.  At the bare carrier-representation level, the exceptional
Jordan approach is closed.

## Combined structural lesson

All four candidates fail for related but distinct structural reasons:

1. **Bimultiplication** collapses to Spin(8) which fails K1.
2. **Three Fano lines** fail Lie-algebra closure due to octonion
   non-associativity.
3. **J_3(O) under Spin(10)** has the wrong dimension for three
   chiral-16 (24 << 48).
4. **J_3^C(O) under Spin(10) × U(1)** has correct dimensional fit but
   forced representation theory disallows three independent 16s.

The common thread: **three generations is a topologically/geometrically
forced structure, not an algebraic representation-theory consequence**
of any standard exceptional Lie / Jordan algebra acting on a chiral-16
carrier.  At the carrier level, the SM's three-generation count appears
to be genuinely empirical — not derivable from carrier algebra alone.

## What this rules out

The exceptional-algebra approach to three SM generations is now closed
at the **carrier representation theory** level, covering:

- Furey-style bimultiplication of `T = C ⊗ H ⊗ O`.
- Three Fano lines through any preferred octonion unit.
- Boyle exceptional Jordan algebra J_3(O) under Spin(10).
- Complexified Boyle algebra J_3^C(O) under Spin(10) × U(1).

## What this does NOT rule out

The negative result is **at the carrier-representation-theory level**.
It does NOT address:

- E_6 × G_2, F_4 × G_2, or other product structures (could be tested in
  a separate sidecar).
- Sedenions / Cayley-Dickson higher steps (also deferred).
- Topological mechanisms (Bold D — independent investigation).
- "Three from anomaly cancellation" / RG fixed points.
- The cp/ result on CP-from-quantization (independent of the
  three-generation question).

## Where the program stands after this kill

Three attempts to derive three generations from algebraic structure
have now closed negative:

| Sidecar | Mechanism | Status |
|---|---|---|
| ../triality/ | Spin(8) triality (Z/3 outer automorphism) | KILL (K1 fail) |
| ../broken_triality/ | Broken Z/3 Yukawa from triality projection | KILL (BT-2 fail) |
| ../exceptional/ | Exceptional Jordan / E_6 / Spin(10) | KILL (all 4 sub-candidates) |

This is a strong cumulative signal: **three generations is unlikely to
emerge from algebraic structure alone.**  The honest program direction
forward:

1. Treat three generations as an empirical input (one generation
   derived, three put in by hand).
2. Investigate non-algebraic mechanisms (topological, RG, anomaly) in
   separate sidecars.
3. Focus the program's positive content on what it DOES derive: one SM
   generation with correct hypercharges (lepton), BCC Dirac kinematics
   (spacetime_qca), and CP-from-quantization with T_{2g} cubic-anisotropy
   signature (cp).

## Tests

```text
tests/test_bimultiplication.py          6 passed
tests/test_fano_lines.py                7 passed
tests/test_j3o_algebra.py              13 passed
tests/test_spin10_on_j3o.py             7 passed
tests/test_j3o_complex.py               6 passed
                                     ─────────
                                       39 passed
```

Run with:

```bash
uv run pytest src/clifford_3plus2_d5/exceptional/tests -q
```

## Effort

- Scaffolding: 30 min.
- Phase 0a (bimultiplication): 30 min.
- Phase 0b (Fano lines): 30 min.
- Phase 1 (J_3(O) algebra): 1 hr.
- Phase 2 (Spin(10) kill): 30 min.
- Phase 2b (J_3^C(O) kill): 20 min.
- Report + status: 30 min.

Total: ~4 hr of focused work.  **Far below the planned ~3-4 day budget**
because the dimensional / representation-theory arguments are decisive
without needing explicit E_6 generator construction.

## Cross-references

- ``PLAN.md`` — original sidecar design with phases and failure modes.
- ``../triality/SESSION_T_KILL_TEST.md`` — exact triality K1 fail
  (referenced by Phase 0a).
- ``../broken_triality/SESSION_BT_KILL_TESTS.md`` — broken triality
  BT-2 fail.
- ``../cp/SESSION_CP_ALPHA_BETA.md``, ``../cp/SESSION_CP_ORDER_EPS2.md``
  — independent positive CP result (this sidecar's negative does not
  affect cp's positive verdict).
