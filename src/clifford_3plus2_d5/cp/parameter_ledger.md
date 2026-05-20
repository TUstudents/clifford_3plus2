# CP Sidecar Parameter Ledger

Final form — both audits completed.

## Discrete choices inherited from spacetime_qca / lepton

1. **Spinor basis**: chiral (``ψ = (ψ_R, ψ_L)^T``).  From
   ``spacetime_qca.dirac``.
2. **BCC Dirac walk**: Bialynicki-Birula hops.  From
   ``spacetime_qca.bcc_weyl.bialynicki_birula_hops``.
3. **Higgs-like map**: from
   ``spacetime_qca.yukawa.higgs_like_charge_shift_candidate`` (first
   basis element of the dim-4 color-singlet, (+1/2, +1/2)-charge-shift
   nullspace).
4. **J for beta**: ``lepton.clifford_patisalam.patisalam_chosen_complex_structure``
   (right-quaternionic Cl(0,4)-commutant J).

## Discrete choices made in cp/

5. **P spinor matrix**: ``γ^0`` (unitary).  Standard.
6. **T spinor matrix**: ``γ^2 γ^0`` (antiunitary).  Satisfies Dirac-equation
   T-conjugation conditions in our chiral basis: anticommutes with γ^0
   and γ^2, commutes with γ^1 and γ^3.
7. **C spinor matrix**: ``i·γ^2`` (antiunitary; standard chiral-basis
   physical charge conjugation).  Satisfies ``C γ^μ C^{-1} = -(γ^μ)^T``;
   conjugation pattern (-1, -1, -1, -1).  **2026-05-20 correction**:
   was previously ``γ^2 γ^0`` (Bloch-level particle-hole, same matrix as
   T).  Retained under ``bloch_particle_hole_spinor`` for backward
   compatibility.
8. **Composite ordering**: P then T then C, etc., in standard "rightmost
   first" notation; explicit ordering via ``compose(...)``.
9. **Internal action for all 7 operators**: ``S_internal = I``.  Trivial
   internal action — the discrete spacetime symmetries do not act
   non-trivially on the internal R^32 carrier.
10. **J-anticommuting fraction convention**: Frobenius-norm ratio
    ``||M_a||^2 / ||M||^2`` where ``M_a = (M + JMJ)/2`` is the
    J-anticommuting part.  **2026-05-20 rename**: was previously called
    "CP-violating fraction"; renamed to drop the unjustified physical-CP
    identification.  The 50/50 result remains valid as an algebraic
    structural property of the dim-4 Higgs-like map space.
11. **Walk-symmetry test criterion**: ``XNOR(antiunitary, hamiltonian_sign
    == +1)`` decides whether to compare to ``B(k_image)`` (no dagger) or
    ``B(k_image)†`` (dagger).  **2026-05-20 correction**: was previously
    "dagger for any antiunitary", which is the special case of the XNOR
    criterion when all operators have ``hamiltonian_sign = +1`` (true
    for the BCC massless walk, but not robust at this convention level).

## Continuous parameters

None.  All audits are exact symbolic; no fitting parameters.

## Total free parameter count

Zero continuous, six discrete (numbers 5-10).  All forced once
conventions are fixed.  Falsifiability is maximal.

## Multi-element β + O(ε²) audits (added 2026-05-18)

11. **Monomial basis ordering** for degree-2 polynomials in
    ``(k_x, k_y, k_z)``: ``(k_x², k_y², k_z², k_y k_z, k_z k_x, k_x k_y)``.
    From ``cubic_harmonics.monomial_basis``.  Pinned.
12. **O_h irrep choice for degree 2**: ``A_{1g} ⊕ E_g ⊕ T_{2g}``
    (dimensions 1 + 2 + 3 = 6).  T_{1g} does not appear at degree 2
    for symmetric rank-2 tensors.  Pinned.
13. **Projector construction**: direct linear-algebra (not character
    formula).  Equivalent to the 48-element O_h projector formula but
    avoids hand-coding group elements.  Justified by the explicit basis
    decomposition.  Pinned.
14. **BCH extraction for H^(1)**: ``H^(1) = i (B_2 - B_2^†) / 2``
    where ``B_2`` is the ε²-coefficient of the Bloch operator.  Pinned.
15. **Polynomial-matrix norm**: ``Σ_ij Σ_α |c_α(M_ij)|^2`` over the
    6-monomial basis, treating monomials as orthonormal.  Pinned in
    ``continuum_cp.polynomial_matrix_norm_squared``.
16. **CP action on polynomial-matrix operators**: ``CP_mat · M^* ·
    CP_mat^{-1}`` (k unchanged, since CP has no momentum flip; entries
    complex-conjugated by antiunitary).  Pinned.

## Final verdicts (updated)

- Alpha-2 (massless walk): **PASS** — CPT, P, CT exact; T, C, CP, PT
  broken at the lattice.
- Alpha-3 (Yukawa-perturbed): confirms — trivial internal action
  preserves the alpha-2 pattern.
- **Alpha-cont (O(ε) continuum)**: **PASS** — H^(1) is purely CP-odd
  (CP-violating fraction = exactly 1) and lives entirely in the
  T_{2g} cubic-harmonic irrep.  ||H^(1)||² = 12.
- Beta (J-misalignment, basis[0]): **PASS** — CP-violating fraction
  exactly 1/2.
- **Beta-multi (all 4 basis + 4 transposes)**: **ROBUST PASS** — all 8
  elements give fraction exactly 1/2.  50/50 mixing is universal.

Combined: **DUAL ROBUST PASS** with structural localization.

See ``SESSION_CP_ALPHA_BETA.md`` for the baseline and
``SESSION_CP_ORDER_EPS2.md`` for the multi-element + O(ε) report.
