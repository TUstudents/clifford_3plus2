# universality Sidecar Parameter Ledger

Filled in as each gate pins choices.

## Inherited (read-only)

From `boundary_response/`:
1. Transfer invariant `epsilon = sqrt(2) - 1` and `epsilon^4 = 17 - 12 sqrt(2)`
   (`transfer.py`); the residual-graph derivation `residual_graph_decaying_factor`
   (V26), with `epsilon = residual_graph_decaying_factor(3)`.
2. Per-sector `epsilon`-formulas: neutrino mass ratio `epsilon^4`; charged-lepton
   depth-2 Weyl leakage `epsilon^2`; quark transfer depths `{1:0, 2:2, 3:6}` →
   amplitudes `epsilon^2, epsilon^4, epsilon^6`.
3. Quark primitive shell `1_even + 5_odd = 1_direct + (2_BCC + 3_color)`
   (`quark_boundary_shell.py`) and the conserved-label tuple
   `(parity, bcc_index, color_index)` (`conserved_label_partition.py`, V21).
4. Lepton K3 residual `(u, a, b)` = `1 (trivial) + 2 (doublet)`
   (`residual_basis.py`).

From `lepton/`:
5. One-generation field multiplicity table (Y, complex multiplicity)
   (`sm_hypercharge.sm_field_multiplicity_table`).

## U1 choices

1. Residual graph family: complete graphs `K_2, K_3, K_4` (degrees 1, 2, 3) with
   unit continuation; the shared root is `K_3`'s.
2. Sector `epsilon`-powers: `{neutrino:4, charged_lepton:2, quark_12:2,
   quark_23:4, quark_13:6}`.
3. Negative control: a `K_4`-rooted sector must fail the `K_3` universal
   prediction (sensitivity check).

## U2 choices

1. Lepton residual S_3 split: `{trivial: 1, doublet: 2}` (u trivial; a,b doublet),
   as stated by `residual_basis`.
2. Quark non-color core = `even_direct + bcc_odd` = `1 + 2`; color ports =
   `color_odd` = 3.
3. Color-label test: `color_index != "none"` selects the 3 color channels;
   expected split `(3 color, 3 non-color)`.

## U3 choices

1. Per-field quantum numbers `(color, weak, Y)`:
   Q=(3,2,1/6), u^c=(3,1,-2/3), d^c=(3,1,1/3), L=(1,2,-1/2), e^c=(1,1,1),
   nu^c=(1,1,0).
2. Factorization check: chiral-16 complex multiplicity must equal `color x weak`.
3. Explicit 32x32 chiral-16 sector projectors (via
   `joint_charge_decomposition` + `su3_c` generators) are **deferred to A3**; this
   gate works at the multiplicity level.

## Continuous parameters

None. The audit is exact-symbolic / integer-combinatorial; no fitting parameters.

## Remaining declared input (the deferral)

`unified_H_Q_on_chiral16_reproduces_all_Sigma_f` — the full numerical proof that
one `H_Q` on lepton's chiral-16 carrier reproduces every sector's `Sigma_f` via
its quantum-number projection `V_f`. This is roadmap A3; the present gate
establishes only the necessary conditions and can kill but not confirm.
