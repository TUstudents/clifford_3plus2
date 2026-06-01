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
| `N_flag` | `|u><a| + |a><b|` | V5 nilpotent repair flag |
| `A_flag` | `N_flag + N_flag.T` | V5 path adjacency |
| `D_flag` | `N_flag N_flag.T + N_flag.T N_flag` | V5 path degree |
| `Delta_flag` | `D_flag - A_flag` | V5 equals `Delta(P3)` |
| `N_local` | `r e^{i alpha}|u><a| + s e^{i beta}|a><b|` | V6 generic local flag |
| `P_initial` | `N_local.H N_local = diag(0,r^2,s^2)` | V6 initial projection |
| `P_final` | `N_local N_local.H = diag(r^2,s^2,0)` | V6 final projection |
| `cost(N)` | `edge_count(N)` | V8 causal repair cost |
| feasible repair supports | `N^3=0`, `N^2!=0`, `rank(N)=2`, all ports active | V8 variational domain |
| `h` | `h(u)=0`, `h(a)=1`, `h(b)=2` | V9 defect filtration |
| residual boundary distance | `d(u,a)=1`, `d(a,b)=1`, `d(u,b)=2` | V9 one-tick geometry |
| bipartite parity | `p(u)=0`, `p(a)=1`, `p(b)=0` | V9 BCC-locality control |
| weighted path `Delta(w1,w2)` | two-edge path Laplacian with weights `w1,w2` | V9 normalization fallback |
| `A` | `span{|a>, |b>}` | V10 active repair domain |
| `R` | `span{|u>, |a>}` | V10 repaired range |
| `N` | `P_R U P_A` | V10 active repair block |
| `L` | `(I-P_R) U P_A` | V10 leakage block |
| `Omega(a)` | allowed one-tick successors of active state `a` | V11 selection set |
| `Omega(b)` | allowed one-tick successors of active state `b` | V11 selection set |
| transition certificate | `(source,target,verdict,vetoes,failure_class)` | V12 finite certificate row |

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
| `NILPOTENT_FLAG_SCAR_ORIGIN_PASS` | V5 length-3 nilpotent flag induces the path scar |
| `NILPOTENT_FLAG_ORDER_KILL` | V5 operator is not a genuine length-3 nilpotent |
| `NILPOTENT_FLAG_SPECTRUM_KILL` | V5 induced Laplacian misses `{0,1,3}` |
| `NILPOTENT_FLAG_CONTROL_KILL` | V5 transfer, rank-one, cyclic, or weighted controls fail |
| `LOCAL_FLAG_PARTIAL_ISOMETRY_PASS` | V6 local partial-isometry fixes unit magnitudes and tree-gauges phases |
| `LOCAL_FLAG_UNITARITY_MAGNITUDE_KILL` | V6 partial-isometry does not force unit nonzero magnitudes |
| `LOCAL_FLAG_SUPPORT_KILL` | V6 shorter support was not rejected |
| `LOCAL_FLAG_PHASE_GAUGE_KILL` | V6 tree phases were not removable |
| `LOCAL_FLAG_CONTROL_KILL` | V6 projection, transfer, contractive, unequal, or cyclic controls fail |
| `MINIMAL_NILPOTENT_SUPPORT_CLASSIFICATION_PASS` | V7 minimal binary support census has one path-flag orbit |
| `NILPOTENT_SUPPORT_NOT_UNIQUE_KILL` | V7 minimal length-3 nilpotent supports are not unique up to relabeling |
| `MINIMALITY_ASSUMPTION_LOAD_BEARING_KILL` | V7 did not show minimality as load-bearing |
| `SUPPORT_CLASSIFICATION_CONTROL_KILL` | V7 census, spectrum, rank-one, or cycle controls fail |
| `MINIMAL_CAUSAL_REPAIR_VARIATIONAL_PASS` | V8 edge-count minimization over feasible repairs selects exactly the path-flag orbit |
| `VARIATIONAL_MINIMIZER_NOT_PATH_KILL` | V8 minimizers are not uniquely the path-flag orbit |
| `VARIATIONAL_CONSTRAINT_CONTROL_KILL` | V8 relaxed-constraint controls fail |
| `VARIATIONAL_COST_CONTROL_KILL` | V8 edge-count cost does not separate path flags from shortcuts |
| `MICROSCOPIC_LOCALITY_MINIMALITY_CONDITIONAL_PASS` | V9 support locality passes and normalization is conditional on partial-isometry saturation |
| `V9A_MICROSCOPIC_SUPPORT_MINIMALITY_PASS` | V9 one-tick locality forbids the shortcut and uniquely forces path support |
| `V9B_MICROSCOPIC_NORMALIZATION_MINIMALITY_CONDITIONAL_PASS` | V9 unit weights follow only with partial-isometry saturation |
| `MICROSCOPIC_SUPPORT_LOCALITY_KILL` | V9 one-tick locality does not force path support |
| `MICROSCOPIC_NORMALIZATION_SATURATION_KILL` | V9 support passes but saturation does not fix unit weights |
| `V10_REPAIR_ISOMETRY_SATURATION_PASS` | V10 unitarity proves unit weights iff active repair has no leakage |
| `REPAIR_ISOMETRY_BALANCE_KILL` | V10 projected unitarity balance failed |
| `REPAIR_ISOMETRY_EQUIVALENCE_KILL` | V10 no-leakage was not equivalent to unit active repair weights |
| `REPAIR_ISOMETRY_CONTROL_KILL` | V10 leakage or target-weight controls failed |
| `V11_SELECTION_SIGNATURE_NO_LEAKAGE_PASS` | V11 unique allowed successors imply no leakage and hence unit path repair |
| `SELECTION_SIGNATURE_NOT_UNIQUE_KILL` | V11 allowed-successor data do not force no leakage |
| `SELECTION_NO_LEAKAGE_CONSEQUENCE_KILL` | V11 unique successors fail to imply the expected isometry/path consequences |
| `SELECTION_NO_LEAKAGE_CONTROL_KILL` | V11 leakage-bound or negative-control checks fail |
| `V12_UNIQUE_SUCCESSOR_ENUMERATION_CERTIFICATE_PASS` | V12 finite certificate table proves `Omega(a)={u}`, `Omega(b)={a}` for the modeled basis |
| `SUCCESSOR_CERTIFICATE_INCOMPLETE_KILL` | V12 certificate is missing rows or forbidden rows without vetoes |
| `UNIQUE_SUCCESSOR_ENUMERATION_KILL` | V12 certificate does not prove unique active successors |
| `SUCCESSOR_CERTIFICATE_CONTROL_KILL` | V12 shortcut, leakage, or V11 implication controls fail |

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

## Nilpotent Flag Ledger

| Object | Value | Status |
|---|---|---|
| canonical flag | `N=|u><a|+|a><b|` | V5 derived operator input |
| nilpotent order | `N^3=0`, `N^2!=0` | V5 verified |
| induced adjacency | `N+N.T` | V5 path adjacency |
| induced degree | `N N.T + N.T N = diag(1,2,1)` | V5 path degree |
| induced Laplacian | `Delta_flag=Delta(P3)` | V5 derived |
| rank-one control | spectrum `{0,0,2}` | V5 rejected |
| cyclic closure control | spectrum `{0,3,3}` | V5 rejected |
| weighted target condition | `x^2=y^2=1` | V5 derived |

## Local Flag Unitarity Ledger

| Object | Value | Status |
|---|---|---|
| generic flag | `r e^{i alpha}|u><a| + s e^{i beta}|a><b|` | V6 starting support |
| partial-isometry equations | `r^4-r^2=0`, `s^4-s^2=0` | V6 derived |
| nonzero nonnegative solution | `r=s=1` | V6 derived |
| phase gauge | `(eta_u,eta_a,eta_b)=(0,alpha,alpha+beta)` | V6 derived |
| canonical representative | `|u><a| + |a><b|` | V6 derived from support |
| rank-one control | partial isometry but `N^2=0` | V6 rejected |
| contractive control | `r=s=1/2` | V6 rejected |
| unequal control | `(r,s)=(1,2)` | V6 rejected |
| cyclic control | `N^3=I`, returns `K3` | V6 rejected |

## Support Classification Ledger

| Object | Value | Status |
|---|---|---|
| no-self-loop binary supports | `64` | V7 exhaustive census |
| minimal accepted supports | `6` | V7 derived |
| accepted `S3` orbits | `1` | V7 derived |
| accepted representative | `|u><a| + |a><b|` | V7 equivalent to V5 |
| minimal support spectra | `{0,1,3}`, doubled `{0,2,6}` | V7 verified |
| rank-one nilpotent controls | nonempty, `N^2=0` | V7 rejected |
| directed cycle controls | `2`, `N^3=I` | V7 rejected |
| nonminimal shortcut supports | `6` | V7 load-bearing control |

## Minimal Repair Variational Ledger

| Object | Value | Status |
|---|---|---|
| feasible supports | `12` | V8 derived domain |
| feasible support costs | `{2,3}` | V8 finite census |
| minimal causal repair cost | `2` | V8 derived |
| minimizers | `6` | V8 derived |
| minimizer `S3` orbits | `1` | V8 derived |
| minimizer representative | `|u><a| + |a><b|` | V8 equivalent to V5 |
| shortcut supports | feasible but cost `3` | V8 excluded by cost |
| relaxed nilpotent minimizers | one-edge rank-one repairs | V8 constraint control |
| constant-cost feasible minimizers | `12`, two support orbits | V8 cost control |

## Microscopic Locality Ledger

| Object | Value | Status |
|---|---|---|
| defect heights | `(0,1,2)` for `(u,a,b)` | V9 declared filtration |
| monotone edges | `a->u`, `b->a`, `b->u` | V9 derived from `h` |
| one-tick edges | `a->u`, `b->a` | V9 derived from path geometry |
| shortcut edge | `b->u` | V9 forbidden by distance 2 |
| BCC parity control | `p(u)=p(b)`, `p(a)` opposite | V9 shortcut rejected |
| local rank-complete supports | `1` | V9 path flag |
| relaxed monotone rank-complete supports | `2` | V9 path + shortcut |
| weighted path spectrum | `0`, `w1+w2 ± sqrt(w1^2-w1w2+w2^2)` | V9 derived |
| target-weight condition | `w1=w2=1` | V9 derived |

## Repair Isometry Ledger

| Object | Value | Status |
|---|---|---|
| projected unitarity balance | `N.H N + L.H L = I_A` | V10 derived |
| isometry condition | `N.H N = I_A iff L=0` | V10 derived |
| no-leakage weights | `|alpha|=|beta|=1` | V10 derived |
| symmetric leakage spectrum | `{0,w,3w}` | V10 rescaled control |
| unequal leakage | breaks `lambda_+ = 3 lambda_-` | V10 control |
| no-leakage microscopic origin | open | V10 not derived |

## Selection No-Leakage Ledger

| Object | Value | Status |
|---|---|---|
| target successors | `a -> u`, `b -> a` | V11 condition |
| unique successor sets | `Omega(a)={u}`, `Omega(b)={a}` | V11 abstract pass |
| leaky control | `Omega(a)={u, bulk_a}` | V11 rejected |
| unique successor block | `diag(e^{i theta_a}, e^{i theta_b})` into `(u,a)` | V11 isometry |
| approximate leakage bound | `1-eta^2 <= w_i <= 1` | V11 soft control |
| microscopic successor enumeration | not done | V12 target |

## Successor Certificate Ledger

| Object | Value | Status |
|---|---|---|
| finite candidate basis size | `12` | V12 modeled basis |
| certificate rows | `24` | V12 two active sources times basis |
| allowed successors | `Omega(a)={u}`, `Omega(b)={a}` | V12 certified |
| veto labels | `LOCALITY`, `HEIGHT`, `BCC_PARITY`, `COLOR`, `WEYL`, `BOUNDARY_SECTOR`, `SUPERSELECTION` | V12 exact vetoes |
| shortcut control | `b -> u` is `SHORTCUT_REPAIR` | V12 rejected |
| external leakage controls | bulk/spectator/wrong-sector candidates vetoed | V12 rejected |
| V11 implication | certificate satisfies unique-successor no-leakage | V12 derived |
| microscopic basis completeness | not derived | remaining input |

## Remaining Input

| Input | Status |
|---|---|
| microscopic origin of height filtration `h(u,a,b)` | open |
| microscopic origin of residual one-tick geometry `u-a-b` | open |
| completeness theorem for the V12 finite local boundary basis | open |
| microscopic boundary response returning `P3` normal modes | open |
| left/right Yukawa assignment for mass exponents | open |
| microscopic values of loop-healing `delta` and `phi` | open |
