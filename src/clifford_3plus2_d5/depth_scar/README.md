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

V6 passes the local partial-isometry normalization gate:

```text
LOCAL_FLAG_PARTIAL_ISOMETRY_PASS
```

For the generic complex flag

```text
N = r exp(i alpha)|u><a| + s exp(i beta)|a><b|,
```

local partial-isometry forces the active nonzero magnitudes to be unit, and tree
rephasing removes both phases.  Thus the canonical V5 flag follows once the
length-3 support is supplied.

V7 passes the minimal support-classification gate:

```text
MINIMAL_NILPOTENT_SUPPORT_CLASSIFICATION_PASS
```

Among all `64` no-self-loop binary directed supports on three ports, the
supports with `N^3=0`, `N^2!=0`, rank `2`, and exactly two edges form one `S3`
orbit: the V5 path flag.  Dropping the two-edge condition admits shortcut
acyclic supports, so minimality is a real remaining assumption.

V8 passes the minimal causal-repair variational gate:

```text
MINIMAL_CAUSAL_REPAIR_VARIATIONAL_PASS
```

Over all rank-2, length-3 nilpotent, all-port-active supports, there are
twelve feasible supports: the six V7 path flags plus six acyclic shortcuts.
Minimizing the causal repair cost

```text
cost(N) = edge_count(N)
```

selects exactly the six path flags.  The minimum cost is `2`; the shortcuts have
cost `3`.  Relaxed controls show the constraints and the edge-count cost are
load-bearing.

V9 passes the microscopic locality/minimality split:

```text
MICROSCOPIC_LOCALITY_MINIMALITY_CONDITIONAL_PASS
```

V9a proves the support theorem:

```text
three-level filtration + monotone repair + one-tick geometry u-a-b
+ rank-complete repair
=> support(N) = {a -> u, b -> a}.
```

The shortcut `b -> u` is monotone but not one-tick local: it has residual
boundary distance `2` and preserves the BCC bipartite parity.  If one-tick
locality is relaxed, the shortcut support reappears.

V9b records the normalization theorem as conditional: partial-isometry
saturation fixes the two active repair weights to unity and removes tree phases.
Without saturation the result is a weighted path, and `{0,1,3}` forces
`w1=w2=1`.

V10 passes the repair-isometry saturation gate:

```text
V10_REPAIR_ISOMETRY_SATURATION_PASS
```

For the active repair block `N=P_R U P_A` and leakage block
`L=(I-P_R) U P_A`, full unitarity gives:

```text
N.H N + L.H L = I_A.
```

Thus `N` is an isometry exactly when `L=0`.  Under the V9 path support, this
forces unit edge weights up to removable tree phases.  If leakage remains, the
result is a weighted path: symmetric leakage rescales `{0,1,3}`, while unequal
leakage destroys the `1:3` ratio.

V11 passes the selection-signature no-leakage gate:

```text
V11_SELECTION_SIGNATURE_NO_LEAKAGE_PASS
```

If the microscopic one-tick selection rules leave exactly one allowed successor
for each active state,

```text
Omega(a) = {u}
Omega(b) = {a},
```

then no output lies outside the repaired range, so `L=0`.  V10 then supplies
unit weights.  V11 does not perform the actual microscopic enumeration; it
turns the remaining physical problem into the finite V12 successor count.

V12 passes the unique-successor enumeration certificate gate:

```text
V12_UNIQUE_SUCCESSOR_ENUMERATION_CERTIFICATE_PASS
```

For the current finite local boundary candidate basis, every transition
`source -> target` from `source in {a,b}` is certified as either `ALLOW` or
`FORBID` with an exact veto.  The resulting allowed successor sets are:

```text
Omega(a) = {u}
Omega(b) = {a}.
```

The shortcut `b -> u` is classified separately as topology-changing
`SHORTCUT_REPAIR`, while bulk, spectator, wrong-sector, and orthogonal-coin
candidates are classified as external leakage and vetoed by their corresponding
selection rules.

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

- The sidecar does not derive the height filtration or one-tick residual
  boundary geometry from the actual BCC-QCA update.
- It does not prove that the V12 finite candidate basis is the complete
  microscopic BCC-QCA local boundary basis.
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
