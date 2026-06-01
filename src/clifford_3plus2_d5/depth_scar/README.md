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

V2 passes the prediction-ledger gate:

```text
DEPTH_SCAR_PREDICTION_LEDGER_PASS
```

V2 records what this operator actually predicts: fixed normal-mode families,
the port-space transfer kernel, endpoint parity, a democratic rank-one leading
response, CKM exponents `lambda:lambda^2:lambda^3`, and no intrinsic CP phase
from a pure path graph.

V3 passes the effective scar-potential gate:

```text
EDGE_WEIGHT_SCAR_POTENTIAL_PASS
```

The symmetric nonnegative edge-weight potential

```text
V = (S1 - 2)^2 + (S2 - 1)^2 + S3
```

has exactly the three missing-edge path scars `(1,1,0)` and permutations as
zero-energy minima.

V4 passes the loop-healing / CP-deformation gate:

```text
LOOP_HEALING_CP_DEFORMATION_PASS
```

The pure path has no intrinsic graph-holonomy CP phase because all tree phases
are removable.  Restoring the missing edge creates one loop.  The real healed
triangle has spectrum `{0, 1+2 delta, 3}` before BCC doubling, and the complex
healed triangle carries one gauge-invariant loop phase `phi`.

V5 passes the nilpotent boundary-flag gate:

```text
NILPOTENT_FLAG_SCAR_ORIGIN_PASS
```

The canonical length-3 repair flag `N=|u><a|+|a><b|` induces the path adjacency,
degree operator, and Laplacian:

```text
Delta_flag = N N.T + N.T N - N - N.T = Delta(P3).
```

The rank-one nilpotent and cyclic closure controls are rejected.

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

- The sidecar does not derive the equal unit length-3 flag from a microscopic QCA update.
- It does not derive CKM magnitudes.
- It does not solve the generation problem.
- It does not make `P3` a mass model without a left/right Yukawa assignment.
- It does not derive the microscopic loop-healing values `delta` or `phi`.

The next real gate is deriving the effective potential from a boundary
condition, missing repair channel, monodromy, or microscopic update.

See `THEORY.md` for the compact theory statement.

## Test Command

```bash
uv run pytest src/clifford_3plus2_d5/depth_scar/tests -q
```
