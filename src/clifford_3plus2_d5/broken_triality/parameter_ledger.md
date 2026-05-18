# Broken-Triality Parameter Ledger

This is the final form — the program died at BT-2, so no continuous
breaking parameters were ever introduced.

## Discrete choices inherited from triality/

1. **Spin(8) embedding**: Cl(0,10) gamma indices `{0..7}`.
2. **Cartan basis**: ``H_k = (1/2) γ_{2k} γ_{2k+1}`` for `k = 0..3`.
3. **Triality direction**: ``α_1 → α_3 → α_4 → α_1`` (vs inverse).

## New discrete choices in broken_triality/

4. **Starting vector ``v_*``**: ``restricted_hypercharge_cartan_coords()``
   = ``Y' = (1/3, 1/3, 1/3, 1/2)`` in ``(H_0, H_1, H_2, H_3)`` basis.
   Aligned with the physical Higgs hypercharge direction.

5. **Inner product**: Euclidean on R^4.  Alternative: Killing-form
   weighted; not pursued because the program died before this choice
   mattered.

6. **Projection convention**: orthogonal projection (least-squares
   solution of ``span * c = v``).  Standard choice.

## Continuous parameters

None.  No continuous breaking parameters were introduced because the
pure-triality-projection Yukawa already fails BT-2 at the leading-order
level.  The program would have introduced continuous breaking magnitudes
only if leading-order pass had been reached.

## Total free parameter count

Zero continuous, six discrete (all forced once the embedding is chosen).

## Comparison to SM observable count

SM Yukawa-sector observables: ~22 (12 masses + 4 CKM + 4 PMNS Dirac + 2
Majorana phases).

Free parameter count from broken-triality: 0 continuous + 0 free
discrete (all six discrete choices are forced by the existing
``triality/`` infrastructure).

The program is therefore **maximally constrained** — there are no knobs
to turn.  This explains why it dies cleanly: there is no way to adjust
the leading-order Yukawa to fit the observed mass hierarchy.

This is the desired property of a well-constructed kill-test: the
absence of free parameters makes the prediction sharp and falsifiable.

## Final verdict

**BT-2 FAIL.**  See ``SESSION_BT_KILL_TESTS.md``.
