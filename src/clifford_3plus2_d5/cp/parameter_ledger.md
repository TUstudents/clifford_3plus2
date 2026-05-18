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
7. **C spinor matrix**: ``γ^2 γ^0`` (antiunitary).  Same matrix as T;
   distinction is operational (C does not reverse walk direction in the
   antiunitary commutation; T does, via [S, H] = 0 forcing S U S^{-1} =
   U^{-1}).
8. **Composite ordering**: P then T then C, etc., in standard "rightmost
   first" notation; explicit ordering via ``compose(...)``.
9. **Internal action for all 7 operators**: ``S_internal = I``.  Trivial
   internal action — the discrete spacetime symmetries do not act
   non-trivially on the internal R^32 carrier.
10. **CP-violating fraction convention**: Frobenius-norm ratio
    ``||M_a||^2 / ||M||^2`` where ``M_a = (M + JMJ)/2`` is the
    J-anticommuting part.

## Continuous parameters

None.  All audits are exact symbolic; no fitting parameters.

## Total free parameter count

Zero continuous, six discrete (numbers 5-10).  All forced once
conventions are fixed.  Falsifiability is maximal.

## Final verdicts

- Alpha-2 (massless walk): **PASS** — CPT, P, CT exact; T, C, CP, PT
  broken at the lattice.
- Alpha-3 (Yukawa-perturbed): confirms — trivial internal action
  preserves the alpha-2 pattern.
- Beta (J-misalignment): **PASS** — Higgs map ``M`` has CP-violating
  fraction **exactly 1/2** under the chosen J.

Combined: **DUAL PASS**.  Both audits independently produce
CP-violating, CPT-preserving structure from the existing infrastructure.

See ``SESSION_CP_ALPHA_BETA.md`` for the full report.
