# unified_boundary Sidecar Parameter Ledger

Filled in as each gate pins choices.

## Inherited (read-only)

From `boundary_response/`:
1. Common boundary `H_Q` = semi-infinite unit sterile chain; Weyl function
   `m(z) = (z - sqrt(z^2-4))/2`, transfer probe `z = 2 sqrt(2)`, so
   `m(z) = sqrt(2) - 1 = epsilon` (`weyl_sterile.py`, `explicit_hq.py`).
2. Lepton Schur self-energy `Sigma_lepton = eps^2 P_u + P_b = K_nu`
   (`weyl_sterile.weyl_product_sterile_normalized_response`, `residual_basis.k_nu_operator`).
3. Quark transfer depths `{1:0, 2:2, 3:6}` → amplitudes `eps^2, eps^4, eps^6`
   (`quark_transfer_hierarchy.py`).
4. Quark Clebsch/coin data: `C_F = 4/3`, BCC `sqrt(2)` and `1/sqrt(2)`, coin
   phase `atan(sqrt(5))` (`quark_clebsch_factors.py`, `quark_boundary_shell.py`).
5. K_n graph roots for controls (`residual_graph_transfer.residual_graph_decaying_factor`).

From `universality/`:
6. The `V_f` quantum-number catalog and the quark-triplet / lepton-singlet color
   split (`u3_coupling_catalog.color_split_is_quark_triplet_lepton_singlet`).

## A3-1 choices

1. The common `H_Q` is fixed as the sterile chain; the transfer factor is read at
   the probe `z = 2 sqrt(2)`.

## A3-2 choices

1. Quark families couple to the *same* chain at depths `{0, 2, 6}`; transition
   depths are the absolute differences `{2, 4, 6}`.
2. Finite-shell convergence checked at `shells = 12`, tolerance `1e-6`.
3. Controls: odd depth (power 3) and the K_2 / golden chain root.

## A3-3 choices

1. SU(3) fundamental normalization `T^A = lambda^A / 2`, so `sum_A T^A T^A = (4/3) I`.
2. BCC path Clebsches `sqrt(2)` (symmetric) and `1/sqrt(2)` (antisymmetric).
3. Coin phase from the flat `Cl_5` coin: `atan(sqrt(5))`.

## Continuous parameters

None. The audit is exact-symbolic (with one finite-shell float convergence check);
no fitting parameters.

## Remaining declared inputs (the A3b deferral)

- `quark_family_depths_0_2_6_derived` — derive the depth embedding `{0,2,6}` from
  the chiral-16 / boundary-shell geometry (currently assigned).
- `clebsch_and_coin_factors_derived_from_chiral16` — derive `C_F`, the BCC
  Clebsches, and the coin phase as `V_f` quantum-number outputs of the chiral-16
  (currently taken from the SU(3)/BCC/Cl_5 structure as inputs).

A3a establishes the transfer-boundary unification and that sector structure lives
in `V_f`; A3b is the derivation of these inputs. The gate can kill but does not
prove the full flavor pattern.
