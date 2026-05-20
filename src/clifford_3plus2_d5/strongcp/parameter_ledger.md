# strongcp Sidecar Parameter Ledger

Filled in as each phase pins choices.

## Inherited from cp/ (read-only)

1. Degree-2 cubic-harmonic decomposition: A_{1g} ⊕ E_g ⊕ T_{2g}.
2. H^(1) extraction via Baker-Campbell-Hausdorff at O(ε).
3. CP, P, T, C spinor matrices in chiral basis.
4. ε is symbolic positive (SymPy).

## Phase SC-1 choices

1. Degree-3 monomial basis ordering (fixed 10-tuple): k_x³, k_y³, k_z³,
   k_x²k_y, k_x²k_z, k_y²k_x, k_y²k_z, k_z²k_x, k_z²k_y, k_x k_y k_z.
2. Degree-3 O_h irreps appearing: A_{2u} (1-dim), T_{2u} (3-dim),
   T_{1u} (6-dim).  Other O_h irreps (A_{1g}, A_{2g}, E_g, T_{1g},
   T_{2g}, A_{1u}, E_u) absent at degree 3.
3. T_{2u} basis: (m_5 − m_7)/√2 (∝ k_x(k_y² − k_z²)) plus
   cyclic permutations.
4. T_{1u} treated as a single 6-dim block (two copies do not
   separate invariantly under O_h alone; sufficient for the
   selection rule).

## Phase SC-2 choices

1. BCC lattice site set: Z³ ∪ (Z + 1/2)³.
2. Dirac parity operator: γ⁰ (= ``parity_spinor()``); in chiral
   basis swaps the upper-left and lower-right 2×2 blocks.
3. Full BB Dirac hop: W_full(h) = block_diag(W_R(h), W_L(h)) with
   W_L(h) = W_R(−h) (opposite-helicity convention).
4. Inversion symmetry test: γ⁰ · W_full(h) · γ⁰⁻¹ = W_full(−h).

## Phase SC-3 choices

1. BCH expansion convention: U(k, ε) = exp(−i ε H_eff(k, ε));
   H_eff = sum_n ε^n H^(n)(k) with H^(0) = α·k.
2. log U Mercator-series truncation: log(I + X) = X − X²/2 + X³/3.
3. Cross-check: BCH-derived H^(1) agrees with cp/'s
   ``effective_hamiltonian_first_correction()``.
4. H^(2) extraction: H^(2)(k) = i · L_3 where L_3 is the ε³
   coefficient of log U.

## Phase SC-4 choices

(deferred — direct lattice topological-charge density computation
to follow up in a future session)

## Phase SC-5 choices

1. Chiral rotation generator: γ⁵.  Chiral-trace tests applied to
   H^(1), H^(2), and cross-products.
2. Direct-θ̄-shift test: ``tr(γ^5 H^(n))`` ≠ 0 indicates a
   chiral-rotation-induced shift.
3. Cross-term ``tr(γ^5 H^(1) H^(2))`` recorded as O(ε³) observation
   (non-zero, but ε³-suppressed below 10⁻¹⁰).

## Phase SC-6 choices

1. Neutron-EDM bound on θ̄: |θ̄| ≤ 10⁻¹⁰ (Abel et al., 2020,
   arXiv:2001.11966).
2. ε upper bound: ε ≲ 2 × 10⁻³³ m (inherited from sme/'s
   UNFALSIFIABLE PASS).
3. Hadronic scale for ε-suppression normalization: Λ_QCD⁻¹ ~ 1 fm
   = 10⁻¹⁵ m.
4. Suppression ratio per ε: ε / Λ_QCD⁻¹ ~ 2 × 10⁻¹⁸.
5. Verdict thresholds: TRIVIAL (θ̄ = 0 structurally), SAFE
   (|θ̄| ≪ 10⁻¹⁰), TENSION (|θ̄| ≳ 10⁻¹⁰).

## Continuous parameters

- ε (lattice scale, symbolic positive).  The single continuous
  parameter; constrained from neutron-EDM bound on θ_QCD.
- Neutron-EDM bound: |θ_QCD| ≤ 10⁻¹⁰ (numerical input, cited).
