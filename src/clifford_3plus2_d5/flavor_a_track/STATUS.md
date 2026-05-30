# flavor_a_track — Status

**Status**: the whole flavor A-track (roadmap Track A), one block, three phases.
Phase verdicts:
- A2 — **UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS**
- A3a — **UNIFIED_TRANSFER_BOUNDARY_PASS**
- A3b — **TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT**

**The load-bearing claim under test:** the dimensionless SM flavor pattern is the
spectral data of *one* boundary operator `H_Q` read through four quantum-number
projections `Sigma_f(z) = V_f^dagger (z - H_Q)^{-1} V_f`. A2 checks the necessary
conditions, A3a builds the unification of the transfer sector, A3b is the honest
derive-or-count ledger for the texture factors.

**Net result:** the A-track predicts flavor *structure* — the texture's
group-theory / geometry factors and both CP phases (`atan(sqrt(5))`, `5 pi/12`),
read off one boundary through four `V_f` projections — but the magnitude
*hierarchy* rides on a free, fit depth embedding `{0,2,6}` that the closed
algebraic/topological kills say has no derivation (`N=3` empirical). One remaining
declared input: `generation_depth_embedding_derived`.

## Gate status

### A2 — universality (`universality_audit`)

| Gate | Verdict |
|---|---|
| U1 — shared transfer invariant | **SHARED_TRANSFER_INVARIANT** — all sectors are powers of the single residual K3 root `sqrt(2)-1`; K2/K4 graph-tracking + independent-epsilon negative controls |
| U2 — sector difference = color label | **SECTOR_DIFFERENCE_IS_COLOR_LABEL** — quark & lepton shells share the `1+2` non-color core; quarks add exactly 3 color ports |
| U3 — coupling catalog | **COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS** — each field's chiral-16 multiplicity factors as `color x weak`; quarks triplet, leptons singlet |
| U4 — combined | **UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS** (can KILL, cannot confirm) |

### A3a — unified transfer boundary (`unified_boundary_audit`)

| Gate | Verdict |
|---|---|
| A3-1 — one common `H_Q`; lepton `Σ` recovered | **LEPTON_SIGMA_FROM_COMMON_HQ** — sterile chain, factor `sqrt(2)-1`, `Σ = K_nu = eps^2 P_u + P_b` |
| A3-2 — quark transfer is same-chain Schur | **QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR** — amplitudes `eps^2, eps^4, eps^6`; finite-shell Green ratios converge; odd-depth + scaled-chain controls reject |
| A3-3 — sector structure in `V_f` | **SECTOR_STRUCTURE_IN_COUPLINGS** — `C_F = (4/3)I`, BCC `sqrt(2)`/`1/sqrt(2)`, coin `atan(sqrt(5))`; lepton singlet, quark triplet |
| A3-4 — combined | **UNIFIED_TRANSFER_BOUNDARY_PASS** |

### A3b — texture provenance / "derive or count" (`texture_provenance_audit`)

| Gate | Verdict |
|---|---|
| B1 — derived-factor ledger | **DERIVED_FACTORS_CATALOGUED** — 7 factors machine-checked against source |
| B2 — free-input enumeration | **FREE_INPUTS_ENUMERATED** — 4 inputs (even+additive load-bearing; can KILL) |
| B3 — parameter count | **TEXTURE_PREDICTIVE** — `N_free=4 < N_obs=8`, surplus 4 |
| B4 — combined | **TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT** |

## What this does and does not establish

- **Does:** the transfer invariant is genuinely shared; the quark/lepton boundaries
  differ by *exactly* the color quantum number; couplings are SM quantum-number
  projections; one sterile chain `H_Q` yields both the lepton `Σ` and the quark
  transfer hierarchy as Schur complements; and the texture's group-theory/geometry
  factors and both CP phases are derived predictions, with fewer free inputs than
  observables (predictive for structure, not numerology).
- **Does not:** prove one explicit `H_Q` on the chiral-16 carrier reproduces all
  four `Sigma_f` numerically (the sector shells live in different representation
  spaces), nor derive the magnitude hierarchy — the depth embedding `{0,2,6}` is a
  free input fit to the CKM hierarchy. Deriving it is a generation mechanism
  (`N=3` empirical), recorded as the one remaining input and not attempted.

## Remaining-input chain (terminates cleanly)

A2's `unified_H_Q_on_chiral16_reproduces_all_Sigma_f` splits in A3a into
`quark_family_depths_0_2_6_derived` + `clebsch_and_coin_factors_derived_from_chiral16`.
A3b resolves these: the Clebsch/coin/phase-word factors are **derived**; the depths
are **not derivable** (free input), renaming the surviving deferral to the single
terminal input `generation_depth_embedding_derived`.

## Cross-module dependency

Borrows from `boundary_response` (transfer invariant, sterile-chain `H_Q`, lepton
Schur `Σ`, sector shells, conserved labels, quark Clebsch/coin/transfer data, V10
holonomy, ergodicity-prior marker) and `lepton` (chiral-16 SM quantum numbers).
All external borrowings go through `reuse.py`; intra-A-track references (e.g. A3-3
reading U3's color split) use ordinary sibling imports, so `reuse` stays free of
any `flavor_a_track` import (no cycle).

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/flavor_a_track/tests -q
```

Expected: 57 passing. Each phase has decisive negative controls proving its gates
can KILL: U1's K4 independent-epsilon, A3-2's golden-root mismatch, B2's
non-even/additive depths, B3's `N_free >= N_obs`, and per-gate verdict-helper KILL
tests across U2/U3/A3-1/A3-3/B1/B4.
