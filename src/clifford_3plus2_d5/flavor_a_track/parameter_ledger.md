# flavor_a_track Parameter Ledger

The A-track is exact-symbolic / integer-combinatorial throughout: **no continuous
fitting parameters** anywhere. The "free inputs" (A3b) are discrete structural
choices (integer depths, a binary CP branch, a max-entropy prior). What follows is
the inherited base, the per-phase choices, and the remaining-input chain.

## Inherited (read-only, via `reuse.py`)

From `boundary_response/`:
1. Transfer invariant `epsilon = sqrt(2) - 1` (`= residual_graph_decaying_factor(3)`,
   V26) and `epsilon^4 = 17 - 12 sqrt(2)` (`transfer.py`).
2. Common sterile chain `H_Q`: Weyl factor `m(2 sqrt(2)) = sqrt(2)-1`
   (`weyl_sterile.py`, `explicit_hq.py`); normalized Schur self-energy
   `K_nu = eps^2 P_u + P_b` (`residual_basis.k_nu_operator`).
3. Quark primitive shell `1_even + 5_odd = 1_direct + (2_BCC + 3_color)` and the
   conserved-label tuple `(parity, bcc_index, color_index)`
   (`quark_boundary_shell.py`, `conserved_label_partition.py`).
4. Lepton K3 residual `(u, a, b) = 1 (trivial) + 2 (doublet)` (`residual_basis.py`).
5. Quark Clebsch/coin data: `C_F = 4/3`, BCC `sqrt(2)` / `1/sqrt(2)`,
   `Gamma_q^2 = 5 I`, coin phase `atan(sqrt(5))` (`quark_clebsch_factors.py`,
   `quark_boundary_shell.py`).
6. Quark depth embedding `{1:0, 2:2, 3:6}` → amplitudes `eps^2, eps^4, eps^6`
   (`quark_transfer_hierarchy.py`).
7. Charged-lepton residual port `e1 = sqrt(2/3) a + 1/sqrt(3) u`, leakage depth 2
   (`charged_lepton_leakage.py`); V10 leptonic phase word `5 pi/12`
   (`leptonic_boundary_holonomy.py`); ergodicity-prior marker
   `physical_vacuum_order_parameter_exists` (`local_boundary_fiber.py`).

From `lepton/`:
8. One-generation field multiplicity table (Y, complex multiplicity)
   (`sm_hypercharge.sm_field_multiplicity_table`).

## A2 choices (universality)

- U1: residual graphs `K_2, K_3, K_4`; shared root is `K_3`'s. Sector
  `epsilon`-powers `{nu:4, e:2, q12:2, q23:4, q13:6}`. Negative control: a
  `K_4`-rooted sector must fail the `K_3` prediction.
- U2: lepton split `{trivial:1, doublet:2}`; quark non-color core
  `even_direct + bcc_odd = 1 + 2`; color ports `color_odd = 3`; expected label
  split `(3 color, 3 non-color)`. The U2 color check (shell level) is
  *complementary* to A3-3 (coupling level, `C_F`), not a duplicate.
- U3: per-field `(color, weak, Y)` — Q=(3,2,1/6), u^c=(3,1,-2/3), d^c=(3,1,1/3),
  L=(1,2,-1/2), e^c=(1,1,1), nu^c=(1,1,0). The explicit 32x32 chiral-16 projector
  build is deferred (the deferral flag is now conditional on the gate passing).

## A3a choices (unified transfer boundary)

- Common `H_Q` = sterile chain; transfer read at probe `z = 2 sqrt(2)`.
- Quark families couple to the *same* chain at depths `{0,2,6}`; transition depths
  `{2,4,6}`. Finite-shell convergence at `shells = 12`, tol `1e-6`. Controls: odd
  depth (power 3), K2/golden chain root.
- SU(3) fundamental `T^A = lambda^A/2` → `sum_A T^A T^A = (4/3)I`; BCC `sqrt(2)`,
  `1/sqrt(2)`; flat `Cl_5` coin phase `atan(sqrt(5))`.

## A3b ledger (derive or count)

**Derived (B1 — 7 factors, each machine-checked against source):**

| factor | value | source check |
|---|---|---|
| color Casimir `C_F` | `4/3` | `sum_A T^A T^A == (4/3) I_3` |
| coin base | `sqrt(5)` | `Gamma_q^2 == 5 I` and `5 == 2_BCC + 3_color` |
| quark CP phase | `atan(sqrt(5))` | from coin base, given `r=1` |
| BCC symmetric | `sqrt(2)` | two-path Clebsch |
| BCC antisymmetric | `1/sqrt(2)` | two-path Clebsch |
| charged-lepton mixing | `sqrt(3/2)` | `1/<a|e1>` from residual port |
| leptonic phase word | `-5 pi/12` | holonomy verdict `LEPTONIC_PHASE_WORD_DERIVED_PASS` |

**Free input (B2 — 4):**
1. quark depth embedding `{0,2,6}` — fit to the CKM hierarchy (even+additive
   checked and load-bearing).
2. charged-lepton two-step depth (= 2).
3. ergodicity prior `r=1` (Jaynes max-entropy → `physical_vacuum_order_parameter_exists`).
4. CP-phase branch (discrete sign).

**Count (B3):** `N_free = 4 < N_observables = 8` (CKM 3 angles + 1 phase; PMNS 3
angles + 1 phase) → surplus 4 → `TEXTURE_PREDICTIVE`. Genuine derived predictions:
the two CP phases, the PMNS angle structure, and the CKM additivity relation
`depth_13 = depth_12 + depth_23`.

## Remaining-input chain (terminates)

- A2 declares `unified_H_Q_on_chiral16_reproduces_all_Sigma_f` (the full numerical
  reproduction; can KILL but cannot confirm at A2).
- A3a splits it into `quark_family_depths_0_2_6_derived` +
  `clebsch_and_coin_factors_derived_from_chiral16`.
- A3b resolves: the Clebsch/coin/phase-word factors are **derived**; the depths are
  **not derivable** (free input), so the single surviving terminal input is
  `generation_depth_embedding_derived` — deriving `{0,2,6}` is a generation
  mechanism (`N=3` empirical per the closed `triality/broken_triality/exceptional/
  topology` kills), **not attempted** here.
