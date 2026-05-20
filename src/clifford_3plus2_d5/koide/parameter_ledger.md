# koide Sidecar Parameter Ledger

Filled in as each phase pins choices.

## Inherited from cp/, topology/, broken_triality/ (read-only)

1. Degree-2 cubic-harmonic decomposition: A_{1g} ⊕ E_g ⊕ T_{2g}.
2. H^(1) extraction at O(ε) via BCH.
3. BCC body-diagonal rotation R = [[0,1,0],[0,0,1],[1,0,0]].
4. Z₃ spinor lift U_3 = exp(-i (2π/3) (n̂·σ) / 2), n̂ = (1,1,1)/√3.
5. broken_triality 3×3 Yukawa-from-orbit pattern.

## Phase KO-1 choices

1. PDG 2024 charged-lepton masses (in MeV):
   m_e = 0.51099895, m_μ = 105.6583755, m_τ = 1776.86.
2. Koide K target: 2/3 (exact rational).
3. Empirical tolerance: 10⁻⁴ (consistent with m_τ uncertainty).
4. Three equivalent geometric forms verified:
   - K-ratio form: K = (Σm) / (Σ√m)² = 2/3.
   - Angle form: (v · n̂)² / |v|² = 1/2 (v at 45° to n̂).
   - Equipartition form: |P_trace v|² = |P_traceless v|².

## Phase KO-2 choices

1. BCC body-diagonal Z₃ rotation: R = [[0,1,0],[0,0,1],[1,0,0]]
   (reused from topology/SC-2).
2. Trace direction: n̂ = (1, 1, 1)/√3 (the Z₃-trivial irrep eigenvector).
3. σ^a ↔ generation identification: σ^x ↔ e, σ^y ↔ μ, σ^z ↔ τ.
   Convention; one of three cyclic equivalents.  Koide K is
   permutation-invariant.

## Phase KO-3 choices

1. Yukawa construction (broken_triality-style with BCC R):
   Y_ij = ⟨R^i v_*, R^j v_*⟩ (Euclidean inner product on ℝ³).
2. Z₃-equivariant Yukawa eigenvalues:
   λ_1 = 3|v_t|², λ_2 = λ_3 = (3/2)|v_o|².
3. Koide special ratio: r* = |v_t|/|v_o| = 3 + 2√2 ≈ 5.8284.
4. At cone, non-degenerate-to-degenerate mass ratio:
   m_1/m_2 = 2(3+2√2)² = 2(17+12√2) ≈ 67.94.

## Phase KO-4 choices

1. Verdict classifier rules:
   - PREDICTED ⇔ L_Z3 ⊂ C.
   - CONSISTENT ⇔ L_Z3 ∩ C ≠ ∅, L_Z3 ⊄ C.
   - CONFLICT ⇔ L_Z3 ∩ C = ∅.
2. Additional tags: "PDG IN LOCUS" / "PDG NOT IN LOCUS".

## Phase KO-5 choices

Aggregates KO-1..KO-4 sub-payloads into KoideAuditPayload with
final_verdict and pdg_in_locus tag.

## Continuous parameters

Two numerical inputs:

- PDG charged-lepton masses: m_e = 0.51099895 MeV,
  m_μ = 105.6583755 MeV, m_τ = 1776.86 MeV (PDG 2024).
- Koide K_target = 2/3 (exact rational).

No other continuous parameters; the audit is otherwise symbolic.
