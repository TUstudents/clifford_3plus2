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
| `SCHUR_RETURN` | angle `1` | V10 boundary-loop Schur return factor |
| `PARENT_A3` | angle `1/4` | V10 parent BCC/tetrahedral spin-lift factor |
| `RESIDUAL_A2` | angle `1/3` | V10 residual triangle spin-lift factor |
| primitive leptonic loop | `SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2` | V10 selected boundary-loop word |
| `U_TBM` | residual basis matrix `[a u b]` | V9 neutrino mixing basis |
| `U_PMNS` | `R_e^dagger U_TBM` | V9 conditional PMNS assembly |
| `sin^2(theta13)` | `(3/4) epsilon^4` | V9 exact conditional PMNS output |
| `delta_CP` | `~261.6 degrees` | V9 default conditional phase branch |
| `delta_CP_conjugate` | `~98.4 degrees` | V9 CP-conjugate phase branch |
| `S_q` | `1_even + 5_odd = 1_direct + (2_BCC + 3_color)` | V11 primitive quark shell |
| `Gamma_q` | `sum_{A=1}^5 gamma_A` | V11 `Cl_5` odd-generator sum |
| `Gamma_q^2` | `5 I` | V11 Clifford closure |
| `B_q` | `(I + i Gamma_q) / sqrt(6)` | V11 flat primitive quark coin |
| `delta_q` | `atan(sqrt(5))` | V11 quark coin phase source |
| `depth(1), depth(2), depth(3)` | `0, 2, 6` | V12 quark boundary-depth embedding |
| `A_12` | `epsilon^2` | V12 raw quark transfer amplitude |
| `A_23` | `epsilon^4` | V12 raw quark transfer amplitude |
| `A_13` | `epsilon^6` | V12 raw quark transfer amplitude |
| `T^A` | `lambda^A / 2` | V13 SU(3) fundamental generators |
| `C_F` | `4/3` | V13 color-singlet return factor |
| `L_12` | `epsilon^2 / sqrt(1 + epsilon^4)` | V13 normalized Cabibbo leakage |
| `C_23` | `sqrt(2)` | V13 symmetric BCC two-path factor |
| `C_13` | `1/sqrt(2)` | V13 antisymmetric BCC projection factor |
| `s_12^q` | `(4/3) epsilon^2 / sqrt(1 + epsilon^4)` | V14 conditional CKM input |
| `s_23^q` | `sqrt(2) epsilon^4` | V14 conditional CKM input |
| `s_13^q` | `epsilon^6 / sqrt(2)` | V14 conditional CKM input |
| `V_CKM` | `R_23 R_13(delta_q) R_12` | V14 conditional CKM assembly |
| `J_q` | `~2.98e-5` | V14 conditional CKM output |

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
| `LEPTONIC_PHASE_WORD_DERIVED_PASS` | boundary-loop holonomy model uniquely selects the full leptonic phase word |
| `LEPTONIC_PHASE_WORD_DERIVED_KILL` | boundary-loop holonomy selection or controls fail |
| `PMNS_CONDITIONAL_ASSEMBLY_PASS` | V6, V7, and V8 assemble into the advertised conditional PMNS texture |
| `PMNS_CONDITIONAL_ASSEMBLY_KILL` | conditional PMNS assembly, exact theta13 relation, or CP-conjugate control fails |
| `QUARK_BOUNDARY_SHELL_Q1_PASS` | primitive quark shell, `Cl_5` closure, flat coin, phase source, and controls pass |
| `QUARK_BOUNDARY_SHELL_Q1_KILL` | quark shell count, `Cl_5` closure, flat coin, phase source, or controls fail |
| `QUARK_TRANSFER_HIERARCHY_Q2_PASS` | quark boundary-depth model derives raw `epsilon^2`, `epsilon^4`, `epsilon^6` transfer hierarchy |
| `QUARK_TRANSFER_HIERARCHY_Q2_KILL` | quark transfer depths, amplitudes, prerequisite, or controls fail |
| `QUARK_CLEBSCH_Q3_PASS` | SU(3) color return, normalized leakage, BCC Clebsches, and controls pass |
| `QUARK_CLEBSCH_Q3_KILL` | color return, leakage, BCC Clebsches, prerequisite, or controls fail |
| `CKM_CONDITIONAL_ASSEMBLY_PASS` | Q1, Q2, and Q3 assemble into the advertised conditional CKM texture |
| `CKM_CONDITIONAL_ASSEMBLY_KILL` | CKM prerequisite gates or assembled unitarity check fail |

## Parked Items

PMNS is implemented first as a V9 conditional assembly from V6, V7, and V8.
V10 derives the V8 phase-word selection within a minimal boundary-loop holonomy
model.  V11 derives the quark primitive shell and flat-coin phase source.  V12
derives the raw quark transfer-depth hierarchy.  V13 derives the color/BCC
Clebsches.  V14 assembles those Q1-Q3 ingredients into a conditional CKM
texture.  The microscopic derivation of the primitive quark boundary shell
remains open.
