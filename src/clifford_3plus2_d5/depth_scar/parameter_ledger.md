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
| `Delta(x,y)` | weighted `K3` with weights `(x,x,y)` | weighted-scar control |

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

## Remaining Input

| Input | Status |
|---|---|
| dynamic origin of the `S3 -> Z2` repair scar | open |

