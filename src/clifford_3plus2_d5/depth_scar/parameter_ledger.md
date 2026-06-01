# depth_scar Parameter Ledger

## Exact Values

| Symbol | Value | Status |
|---|---:|---|
| `epsilon` | `sqrt(2) - 1` | imported from `boundary_response.transfer` |
| `Spec(Delta(P3))` | `{0, 1, 3}` | derived |
| `Spec(D_scar)` | `{0, 2, 6}` | derived from `D_scar = 2 Delta(P3)` |

## Operators

| Operator | Definition | Status |
|---|---|---|
| `partial_scar` | `[[1,-1,0], [0,1,-1]]` | path incidence |
| `Delta(P3)` | `partial_scar.T partial_scar` | path repair Laplacian |
| `D_scar` | `2 Delta(P3)` | candidate depth operator |
| `Delta(K3)` | `3I - J` | unbroken control |
| `D_K3` | `2 Delta(K3)` | doubled unbroken control |
| `T_boundary` | `epsilon^D_scar` | transfer operator |
| `P0` | projector onto `(1,1,1)/sqrt(3)` | V2 family projector |
| `P2` | projector onto `(1,0,-1)/sqrt(2)` | V2 family projector |
| `P6` | projector onto `(1,-2,1)/sqrt(6)` | V2 family projector |
| `T_projector` | `P0 + epsilon^2 P2 + epsilon^6 P6` | V2 transfer kernel |
| `Delta(x,y)` | weighted `K3` with weights `(x,x,y)` | weighted-scar control |
| `S1` | `w_ua + w_ab + w_ub` | V3 edge-weight invariant |
| `S2` | `w_ua w_ab + w_ab w_ub + w_ub w_ua` | V3 edge-weight invariant |
| `S3` | `w_ua w_ab w_ub` | V3 edge-weight invariant |
| `V_edge(w)` | `(S1 - 2)^2 + (S2 - 1)^2 + S3` | V3 effective scar potential |
| `Delta(delta)` | weighted triangle with weights `(1,1,delta)` | V4 real loop healing |
| `Delta(delta,phi)` | Hermitian triangle with loop phase `phi` | V4 magnetic loop healing |

## Mode Ledger

| Mode | Vector | Depth |
|---|---|---:|
| uniform | `(1,1,1)/sqrt(3)` | `0` |
| endpoint antisymmetric | `(1,0,-1)/sqrt(2)` | `2` |
| middle compression | `(1,-2,1)/sqrt(6)` | `6` |

## Verdict Flags

| Flag | Meaning |
|---|---|
| `PATH_DEFECT_LAPLACIAN_DEPTH_PASS` | `D_scar = 2 Delta(P3)` gives the exact depth spectrum and transfer factors |
| `PATH_DEFECT_LAPLACIAN_SPECTRUM_KILL` | the path-defect operator misses `{0,2,6}` |
| `TRANSFER_OPERATOR_DEPTH_KILL` | `epsilon^D_scar` fails to produce `{1,epsilon^2,epsilon^6}` |
| `UNBROKEN_K3_CONTROL_FAILED` | the `K3` control did not remain degenerate |
| `SCAR_NOT_GRAPH_NATIVE_KILL` | the diagonal control was not separated from the graph-native scar |
| `DEPTH_SCAR_PREDICTION_LEDGER_PASS` | V2 prediction ledger passes without promoting the scar to a mass model or CP source |
| `DEPTH_SCAR_PROJECTOR_LEDGER_KILL` | V2 projectors fail orthogonality or completeness |
| `DEPTH_SCAR_TRANSFER_LEDGER_KILL` | V2 transfer kernel does not match V1 |
| `DEPTH_SCAR_ENDPOINT_PARITY_KILL` | V2 endpoint parity does not block even/odd mixing |
| `DEPTH_SCAR_CP_NO_GO_KILL` | V2 pure path unexpectedly has an intrinsic graph-holonomy CP cycle |
| `EDGE_WEIGHT_SCAR_POTENTIAL_PASS` | V3 effective edge-weight potential selects exactly the three path scars |
| `EDGE_WEIGHT_SCAR_MINIMA_KILL` | V3 potential does not select exactly the path-scar minima |
| `EDGE_WEIGHT_SCAR_SPECTRUM_KILL` | V3 selected minima do not give the target spectra |
| `EDGE_WEIGHT_SCAR_CONTROL_KILL` | V3 symmetry, load-bearing, or unbroken-triangle controls fail |
| `LOOP_HEALING_CP_DEFORMATION_PASS` | V4 loop healing separates real hierarchy deformation from one gauge-invariant CP holonomy |
| `PURE_PATH_CP_KILL` | V4 pure path unexpectedly carries an intrinsic CP holonomy |
| `LOOP_HOLONOMY_KILL` | V4 healed triangle does not produce exactly one gauge-invariant phase |
| `LOOP_HEALING_CONTROL_KILL` | V4 spectrum, Hermiticity, conjugation, or limit controls fail |

## Prediction Ledger

| Prediction | Value | Status |
|---|---|---|
| port relations | `T_uu = T_bb`, `T_ua = T_ab` | derived from `T_projector` |
| endpoint parities | `(even, odd, even)` for depths `(0,2,6)` | derived |
| leading response | rank-one democratic `P0` | derived |
| CKM lambda exponents | `(1,2,3)` for `(12,23,13)` | derived as transfer-depth pattern |
| two-sided mass depth semigroup | `{0,2,4,6,8,12}` | conditional diagnostic |
| two-sided lambda powers | `{0,1,2,3,4,6}` | conditional diagnostic |
| pure-path CP | no intrinsic graph holonomy | derived no-go |
| healed-loop CP | one gauge-invariant loop phase `phi` | derived graph role |

## Edge-Weight Scar Ledger

| Object | Value | Status |
|---|---|---|
| path-scar minima | `(1,1,0)`, `(1,0,1)`, `(0,1,1)` | V3 derived zero set |
| unbroken symmetric point | `(1,1,1)` | V3 rejected control |
| non-scar zero without `S3` term | `(4/3,1/3,1/3)` | V3 load-bearing control |
| spectra at minima | `Spec(Delta)={0,1,3}`, `Spec(2Delta)={0,2,6}` | V3 derived |

## Loop-Healing Ledger

| Object | Value | Status |
|---|---|---|
| real healed weights | `(1,1,delta)` | V4 deformation |
| real healed spectrum | `{0, 1+2 delta, 3}` | V4 derived |
| BCC-doubled healed spectrum | `{0, 2+4 delta, 6}` | V4 derived |
| pure path cycle rank | `0` | V4 tree CP no-go |
| healed triangle cycle rank | `1` | V4 loop holonomy |
| loop phase | `phi` on `u -> a -> b -> u` | V4 gauge invariant |

## Remaining Input

| Input | Status |
|---|---|
| microscopic origin of the effective edge-weight potential | open |
| microscopic boundary response returning `P3` normal modes | open |
| left/right Yukawa assignment for mass exponents | open |
| microscopic values of loop-healing `delta` and `phi` | open |
