# depth_scar

Prove-or-kill sidecar for the quark family depth spectrum.

The sidecar tests one narrow upgrade:

```text
{0, 2, 6} is not inserted as a diagonal spurion;
it is the spectrum of a graph-native boundary repair scar.
```

The proposed scar breaks the residual repair graph from `K3` to the path

```text
u -- a -- b
```

and defines

```text
D_scar = 2 Delta(P3).
```

## Result

V1 passes the operator-origin gate:

```text
PATH_DEFECT_LAPLACIAN_DEPTH_PASS
```

The exact spectrum is:

```text
Spec(D_scar) = {0, 2, 6}
```

and the transfer operator gives:

```text
epsilon^D_scar = diag(1, epsilon^2, epsilon^6)
```

in the defect normal-mode basis.

## What This Means

The generations are represented as normal modes of the repair graph:

| Mode | Vector | Depth |
|---|---|---:|
| uniform | `(1,1,1)/sqrt(3)` | `0` |
| endpoint antisymmetric | `(1,0,-1)/sqrt(2)` | `2` |
| middle compression | `(1,-2,1)/sqrt(6)` | `6` |

This upgrades the depth spectrum from a hand-written diagonal assignment to a
positive graph Laplacian, conditional on the existence of an `S3 -> Z2` repair
scar.

## What Is Still Open

- The sidecar does not dynamically derive the scar.
- It does not derive CKM magnitudes.
- It does not solve the generation problem.

The next real gate is deriving the path scar from a boundary condition, missing
repair channel, edge-weight potential, or monodromy.

## Test Command

```bash
uv run pytest src/clifford_3plus2_d5/depth_scar/tests -q
```

