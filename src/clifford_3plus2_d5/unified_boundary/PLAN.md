# unified_boundary — design (roadmap gate A3a)

## Load-bearing question

> Are the lepton and quark flavor sectors Schur complements of **one** common
> boundary `H_Q`, with all sector differences carried by the coupling `V_f`?

This is the constructed form of the A2 universality claim (A2 checked only the
necessary conditions).

## Decisive reframe (from exploration)

The quark sector has **no Schur self-energy**: its CKM is assembled
combinatorially (flat Cl_5 coin phase `atan(sqrt(5))`, transfer depths `eps^k`,
color/BCC Clebsches `4/3, sqrt(2), 1/sqrt(2)`), with no quark `H_Q` and no
resolvent. Only the neutrino core `K_nu = eps^2 P_u + P_b` is a genuine
`Sigma = V^dagger (z - H_Q)^{-1} V`. So "one `H_Q` reproduces both `Sigma_f`" is
not well-posed as stated. A3a instead **puts the quark transfer sector on the
same Schur footing** using the *same* `H_Q` — feasible because the quark transfer
amplitudes `eps^2, eps^4, eps^6` are powers of the *same* sterile-chain Weyl
factor that drives the neutrino core.

## Scope

A3a unifies the **transfer boundary** `H_Q` and shows the sector-specific
structure lives in `V_f`. It does **not** derive the quark depths `{0,2,6}` or the
Clebsch/coin factors from the chiral-16 geometry (= A3b, deferred). The gate can
KILL but does not prove the full flavor pattern.

## Gates

- **A3-1** (`a31_common_hq.py`). Fix the common `H_Q` = semi-infinite sterile
  chain; its Weyl transfer factor is `sqrt(2)-1 = epsilon`; its normalized Schur
  self-energy is exactly `K_nu = eps^2 P_u + P_b`. Verdict
  `LEPTON_SIGMA_FROM_COMMON_HQ`.

- **A3-2** (`a32_quark_transfer_schur.py`). The quark transfer amplitudes equal
  `(common chain factor)^depth` for depths `{1<->2:2, 2<->3:4, 1<->3:6}`; the
  finite-shell chain Green ratios converge to them. Controls: an odd-depth power
  is not in the even hierarchy, and a different chain (K2/golden) changes the
  amplitudes. Verdict `QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR` /
  `TRANSFER_NOT_UNIFIABLE_KILL`.

- **A3-3** (`a33_coupling_structure.py`). The remaining quark structure is `V_f`
  quantum-number data: color `sum_A T^A T^A = (4/3) I`, BCC `sqrt(2)` and
  `1/sqrt(2)`, coin `atan(sqrt(5))`; leptons color-singlet, quarks color-triplet.
  Verdict `SECTOR_STRUCTURE_IN_COUPLINGS`.

- **A3-4** (`unified_boundary_audit.py`). Aggregate. Pass
  `UNIFIED_TRANSFER_BOUNDARY_PASS`; record the A3b deferrals
  (`quark_family_depths_0_2_6_derived`, `clebsch_and_coin_factors_derived_from_chiral16`).

## Acceptance standard

The gate **can kill but does not prove** the full flavor pattern. A pass means the
transfer boundary unifies and the sector structure is in `V_f`; deriving the
depths and Clebsch/coin factors (A3b) is the remaining work.

## Reuse (all via `reuse.py`)

- `boundary_response.weyl_sterile` — `semi_infinite_weyl_function`, `weyl_product_sterile_normalized_response`.
- `boundary_response.explicit_hq` — `green_transfer_amplitude`, `transfer_probe`.
- `boundary_response.residual_basis` — `k_nu_operator`.
- `boundary_response.quark_transfer_hierarchy` — `quark_transition_amplitude`, `quark_transition_depths`, `CKM_TRANSITIONS`.
- `boundary_response.quark_clebsch_factors` — `color_return_contraction`, `color_return_factor`, BCC factors.
- `boundary_response.quark_boundary_shell` — `quark_boundary_phase_angle`.
- `boundary_response.residual_graph_transfer` — `residual_graph_decaying_factor` (controls).
- `universality.u3_coupling_catalog` — `color_split_is_quark_triplet_lepton_singlet`.
