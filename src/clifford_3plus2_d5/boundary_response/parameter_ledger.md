# boundary_response Parameter Ledger

## Exact Values

| Symbol | Value | Status |
|---|---:|---|
| `epsilon` | `sqrt(2) - 1` | derived from residual transfer recurrence |
| `epsilon^2` | `3 - 2 sqrt(2)` | derived |
| `epsilon^4` | `17 - 12 sqrt(2)` | derived |

## Operators

| Operator | Definition | Status |
|---|---|---|
| `P_u` | projector onto `(1,1,1)/sqrt(3)` | exact basis choice |
| `P_a` | projector onto `(2,-1,-1)/sqrt(6)` | exact basis choice |
| `P_b` | projector onto `(0,1,-1)/sqrt(2)` | exact basis choice |
| `K_nu` | `epsilon^2 P_u + P_b` | target, not derived by V1 |
| `H_Q^(N)` | finite `S_3`-equivariant residual `K_3` tail | audit candidate |
| `V` | identity coupling from projected residual channels to shell zero | audit candidate |
| `c_tail` | `(1,1,1)` | V2 local collective incidence |
| `c_edge` | `(0,1,-1)` | V2 local opposite-edge incidence |
| `amp(depth)` | `epsilon^depth` | V2 transfer-depth amplitude |
| `z_transfer` | `2 sqrt(2)` | V3 transfer-chain resolvent probe |
| `H_chain^(N)` | nearest-neighbor path adjacency | V3 finite transfer-chain model |
| `edge_energy_match` | `z_transfer - 1 / inferred_collective_return` | V4 tuned matched load |
| `H_product^(N)` | `H_chain^(N) ⊗ I_family` | V5 product sterile-tail model |
| `amp_N` | `G_chain(1,0) / G_chain(0,0)` | V5 derived finite transfer amplitude |
| `m_head_N` | `G_chain(0,0)` | V5 common sterile head return |
| `Sigma_product_N` | `m_head_N (amp_N^2 P_u + P_b)` | V5 normalized product response |
| `m(z)` | `(z - sqrt(z^2 - 4)) / 2` | V6 semi-infinite Weyl function |
| `m(z_transfer)` | `epsilon` | V6 exact transfer amplitude |
| `Sigma_weyl(z)` | `m(z) [m(z)^2 P_u + P_b]` | V6 exact product response |
| `e1` | `(1,0,0)` | V7 selected charged-lepton/Higgs port |
| `<a|e1>` | `sqrt(2/3)` | V7 selected-port Clebsch |
| `<u|e1>` | `1/sqrt(3)` | V7 selected-port component |
| `<b|e1>` | `0` | V7 no direct `b` leakage |
| `sin(theta_e)` | `sqrt(3/2) epsilon^2` | V7 two-step charged-lepton leakage |
| `q_A3` | `exp(i pi/4)` | V8 BCC/tetrahedral parent spin lift |
| `q_A2` | `exp(i pi/3)` | V8 residual triangle spin lift |
| `Schur sign` | `exp(i pi)` | V8 second-order Schur complement sign |
| `W_e` | `-q_A3 q_A2 = exp(-i 5 pi/12)` | V8 conditional leptonic phase word |

## Verdict Flags

| Flag | Meaning |
|---|---|
| `TRANSFER_PASS` | recurrence gives the advertised `epsilon` |
| `S3_KILL` | target requires `S_3 -> S_2` breaking |
| `K3_TAIL_KILL` | unbroken `K_3` tail self-energy cannot match target |
| `CORE_PASS` | reserved for a later explicit framed `H_Q,V` model |
| `FRAMED_INCIDENCE_PASS` | V2 incidence maps derive `u` and `b`, excluding `a` |
| `TRANSFER_DEPTH_PASS` | V2 derives `g_u/g_b = epsilon` from depth |
| `FRAMED_STERILE_EFFECTIVE_PASS` | V2 effective framed response yields target |
| `EXPLICIT_HQ_PASS` | raw finite `H_Q` Schur response matches target diagnostics |
| `EXPLICIT_HQ_CONVERGENCE_ONLY` | finite `H_Q` derives transfer amplitude but raw Schur still fails |
| `EXPLICIT_HQ_KILL` | explicit finite `H_Q` fails transfer or response diagnostics |
| `IMPEDANCE_MATCH_PASS` | untuned local endpoint load matches the required return |
| `IMPEDANCE_FREE_PARAMETER` | endpoint match requires a solved scalar load |
| `IMPEDANCE_KILL_MINIMAL_CATALOG` | no minimal endpoint catalog candidate matches |
| `PRODUCT_STERILE_CONVERGENCE_PASS` | product bath gives equal returns/cross-return cancellation and converges to `epsilon^2 P_u + P_b` |
| `PRODUCT_STERILE_KILL` | product bath fails return, cross-return, radial, negative-control, or convergence checks |
| `PRODUCT_STERILE_LIMIT_PASS` | exact semi-infinite Weyl-function theorem gives `epsilon^2 P_u + P_b` |
| `PRODUCT_STERILE_LIMIT_KILL` | Weyl branch, fixed-point, target, or negative-control check fails |
| `CHARGED_LEPTON_LEAKAGE_PASS` | selected port plus two Weyl-transfer steps derive `sin(theta_e) = sqrt(3/2) epsilon^2` |
| `CHARGED_LEPTON_LEAKAGE_KILL` | selected-port geometry, leakage depth, or controls fail |
| `LEPTONIC_PHASE_WORD_CONDITIONAL_PASS` | exact full-word phase arithmetic passes but word selection is not derived |
| `LEPTONIC_PHASE_WORD_KILL` | phase arithmetic or subword controls fail |
| `LEPTONIC_PHASE_WORD_DERIVED_PASS` | reserved for a future explicit boundary-loop holonomy theorem |

## Parked Items

PMNS, CKM, and quark boundary coins are intentionally outside this sidecar
until explicit PMNS assembly and quark boundary shells are derived.  V8
verifies only the conditional leptonic phase arithmetic, not the boundary-loop
selection theorem and not PMNS assembly.
