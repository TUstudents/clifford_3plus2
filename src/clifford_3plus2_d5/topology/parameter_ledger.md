# Topology Sidecar Parameter Ledger

Filled in as each phase pins choices.

## Inherited from spacetime_qca / lepton

1. Bialynicki-Birula directions in order ``product((1,-1), repeat=3)``.
2. Bialynicki-Birula hops (Phys. Rev. D 49, 6920).
3. Chiral basis γ⁰..γ³ in 4×4 SymPy form.
4. Chiral-16 carrier (32-dim real, complex-16) from patisalam_chiral16_block_matrix.
5. SU(3)_c basis from su3_c_generators_from_su4.

## Phase D-1 choices

1. Body-diagonal rotation axis: ``(1, 1, 1) / √3``.
2. 3×3 rotation matrix: ``R = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]``
   (the cyclic permutation ``(x, y, z) → (y, z, x)``).
3. Dirac spinor lift: ``U_3 = exp(-i (2π/3) (n · σ) / 2)`` with
   ``n = (1, 1, 1) / √3``; extended to 4-spinor as ``diag(U_3, U_3)``
   in chiral basis.
4. Permutation cycle structure on the 8 BCC directions: ``(3, 3, 1, 1)``
   (the two body-diagonal corners are fixed, the six others form two
   3-cycles).
5. Equivariance failure on BB hops is recorded as an orthogonal finding
   in ``hop_equivariance.py`` — it does not affect the chiral-16
   triviality verdict.

## Phase D-2 choices

1. SU(3)_c Cartan Z_3 generator: ``g_3 = diag(1, ω, ω²)`` on color
   triplets with ``ω = exp(2π i / 3)``.
2. Antitriplet character convention: charge-conjugated, ``g_3* =
   diag(1, ω², ω)``.  Counts as ``(1, 1, 1)`` over ``(1, ω, ω²)``
   because the three antitriplet colors permute through all three
   characters.
3. Per-generation field count: 6 (Q_L) + 3 (u_R^c) + 3 (d_R^c) +
   2 (L_L) + 1 (e_R^c) + 1 (ν_R^c) = 16.
4. Decomposition under color Z_3 center: ``(trivial, ω, ω²) = (8, 4, 4)``.

## Phase D-3 choices

Documented as a Markdown literature note (no code or tests).  Sources:
Mimura-Toda, Husemoller, Bott-Tu, Wikipedia.  No carrier-relevant
``G/H`` has Z/3 torsion in ``π_3``.

## Phase D-5 choices

1. Standard SM hypercharge assignments (all left-handed Weyl):
   ``Y(Q_L)=1/6, Y(u_R^c)=-2/3, Y(d_R^c)=1/3, Y(L_L)=-1/2,
   Y(e_R^c)=1, Y(ν_R^c)=0``.
2. Anomaly conditions checked: gravitational (Σ Y), U(1)_Y³ (Σ Y³),
   SU(2)²·Y (Σ Y over doublets), SU(3)²·Y (Σ Y over color reps).
3. Witten global SU(2) anomaly via ``π_4(SU(2)) = Z/2``: count of
   SU(2)_L doublets must be even.
4. Doublet count per generation: 3 (Q_L colors) + 1 (L_L) = 4 doublets.
5. Combined constraint on N: ``0 = 0`` (trivial — satisfied for every
   N ≥ 0).

## Continuous parameters

None. All audits are exact symbolic.
