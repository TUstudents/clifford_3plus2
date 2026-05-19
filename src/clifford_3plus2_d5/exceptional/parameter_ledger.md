# Exceptional Sidecar Parameter Ledger

Final form — all phases closed negative.

## Inherited from lepton

1. Octonion multiplication table (Fano triples) from
   `lepton.clifford_octonion.octonion_fano_triples()`:
   `(1,2,3), (1,4,5), (1,6,7), (4,2,6), (2,5,7), (3,4,7), (3,5,6)`.
2. `e_7` as the preferred imaginary unit (for G_2 → SU(3) stabilizer
   and for the Phase 0b Fano-line investigation).
3. Cl(0,10) gamma matrix ordering (10 generators in fixed order).

## Phase 0a (bimultiplication) choices

4. **Generator set**: L_{e_1}, ..., L_{e_7}, R_{e_1}, ..., R_{e_7}
   (14 base + commutators).  Closure verified to dim 28 = so(8).

## Phase 0b (Fano lines) choices

5. **Preferred imaginary unit**: e_7.  Three Fano lines through e_7:
   {1,6,7}, {2,5,7}, {3,4,7}.
6. **Candidate SU(2) realization**: octonion left-multiplications
   {L_a, L_b, L_c} for each triple (a, b, c).  Fails Lie closure under
   commutator.

## Phase 1 (J_3(O) algebra) choices

7. **Basis ordering**: 3 real diagonals (m_11, m_22, m_33) +
   3 above-diagonal octonions (m_12, m_13, m_23) — total 3 + 3 × 8 = 27.
8. **Conjugation convention**: octonion conjugation flips sign of all
   imaginary components, fixes the real part.
9. **Jordan product**: ``M ·_J N = (MN + NM)/2`` using entry-wise
   octonion multiplication.
10. **Cubic norm convention**:
    `N(M) = a b c + 2 Re(x z y*) - a |z|² - b |y|² - c |x|²`
    where `(a, b, c)` are diagonal reals and `(x, y, z)` are
    above-diagonal octonions.

## Phase 2 (Spin(10) on J_3(O)) choices

11. **Decomposition convention**: pick a "preferred row" k ∈ {0, 1, 2}.
    The 27 splits as singlet (M_{kk}, 1-dim) ⊕ vector (2 other diagonals
    + the octonion NOT touching row k, 10-dim) ⊕ spinor (2 octonions
    touching row k, 16-dim).
12. **Off-diagonal pair indexing**:
    pair 0 = M_{12} (touches rows 0, 1),
    pair 1 = M_{13} (touches rows 0, 2),
    pair 2 = M_{23} (touches rows 1, 2).

## Phase 2b (J_3^C(O)) choices

13. **Complexification convention**: standard complexification
    `J_3^C(O) = J_3(O) ⊗_R C`, real dimension 54 = 2 × 27.
14. **Spin(10) × U(1) decomposition**:
    27_C = (16, +1) ⊕ (10, -2) ⊕ (1, +4); realified
    54 = 16 + 16* + 10 + 10* + 1 + 1*.  U(1) anomaly-free
    (16·1 + 10·(-2) + 1·4 = 0).

## Continuous parameters

None.  All audits are exact symbolic with no fitting parameters.

## Total free parameter count

Zero continuous, ~6 discrete (all forced by conventions inherited from
lepton + the decomposition recipe).  Falsifiability is maximal.

## Final verdicts

- Phase 0a (Bi(O) = Spin(8)):       **KILL** (inherits triality fail).
- Phase 0b (three Fano lines):     **KILL** (no Lie closure).
- Phase 2 (J_3(O) decomposition):  **KILL** (27 = 16 + 10 + 1).
- Phase 2b (J_3^C(O) decomposition): **KILL** (54 = 16 + 16* + 10 + 10* + 1 + 1*).

Combined: **FULL KILL of the exceptional-algebra approach to three
generations**.

See ``SESSION_EXCEPTIONAL.md`` for the full report.
