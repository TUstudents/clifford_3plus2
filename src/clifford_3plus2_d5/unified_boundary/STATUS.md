# unified_boundary — Status

**Status**: A3a transfer-boundary unification audited.
Verdict: **UNIFIED_TRANSFER_BOUNDARY_PASS**.

Roadmap gate A3a. A2 (`universality/`) established the *necessary conditions* for
one-boundary universality; A3a constructs the unification of the **transfer
sector**: it puts the quark sector on the same Schur footing as the lepton sector
using the *same* `H_Q`.

**Decisive reframe:** the quark sector has no Schur self-energy — CKM is assembled
combinatorially (flat Cl_5 coin phase `atan(sqrt(5))` + transfer depths `eps^k` +
color/BCC Clebsches `4/3, sqrt(2), 1/sqrt(2)`). The neutrino core
`K_nu = eps^2 P_u + P_b` is a genuine Schur complement of an explicit sterile
chain `H_Q`. A3a shows the quark transfer amplitudes are powers of the *same*
sterile-chain Weyl factor that drives the neutrino core.

**Honest scope:** A3a unifies the transfer boundary `H_Q` and shows the
sector-specific structure lives in the coupling `V_f`. It does **not** derive the
quark depths `{0,2,6}` or the Clebsch/coin factors from the chiral-16 geometry —
those remain inputs (A3b). The gate can KILL but does not prove the full flavor
pattern.

## Gate status

| Gate | Status | Verdict |
|---|---|---|
| A3-1 — one common `H_Q`; lepton `Σ` recovered | done | **LEPTON_SIGMA_FROM_COMMON_HQ** — common sterile chain, factor `sqrt(2)-1`, Schur self-energy = `K_nu = eps^2 P_u + P_b` |
| A3-2 — quark transfer is same-chain Schur | done | **QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR** — amplitudes `eps^2, eps^4, eps^6` are powers of the same chain factor; finite-shell Green ratios converge; odd-depth + scaled-chain controls reject |
| A3-3 — sector structure in `V_f` | done | **SECTOR_STRUCTURE_IN_COUPLINGS** — `C_F = sum_A T^A T^A = (4/3)I`, BCC `sqrt(2)`/`1/sqrt(2)`, coin `atan(sqrt(5))`; lepton color-singlet, quark color-triplet |
| A3-4 — combined verdict | done | **UNIFIED_TRANSFER_BOUNDARY_PASS**; depths `{0,2,6}` and Clebsch/coin derivation deferred to A3b |

## What this does and does not establish

- **Does:** one `H_Q` (the sterile chain) yields both the lepton `Σ` and the quark
  transfer hierarchy as Schur complements, with all sector-specific structure
  (color, BCC, coin) carried by `V_f`. The transfer boundary is unified.
- **Does not:** derive the quark family depths `{0,2,6}` or the Clebsch/coin
  factors from the chiral-16 geometry — those are still inputs (A3b). It is not a
  proof of the full flavor pattern, and the quark CKM *magnitudes* are
  conditional on the depth/Clebsch inputs.

## Cross-module dependency

Bridges `boundary_response` (common sterile chain, lepton Schur `Σ`, quark
transfer/Clebsch/coin data), `universality` (the `V_f` quantum-number catalog),
and `lepton` (chiral-16 color structure). All imports via `reuse.py`.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/unified_boundary/tests -q
```

Expected: 15 passing. A decisive negative control
(`test_gate_can_fail_decisive_negative_control`) confirms the gate is sensitive:
the quark amplitudes match the common (K3) chain factor's powers but not a
different chain (K2/golden root), so a sector built from a different chain would
force `TRANSFER_NOT_UNIFIABLE_KILL`.
