# depth_scar — Status

**Status**: V1-V5 implemented.

## Verdict

```text
PATH_DEFECT_LAPLACIAN_DEPTH_PASS
```

## Result

The path repair scar

```text
u -- a -- b
```

has incidence matrix

```text
partial = [[1, -1, 0],
           [0,  1, -1]]
```

and repair Laplacian

```text
Delta(P3) = partial.T partial
          = [[ 1, -1,  0],
             [-1,  2, -1],
             [ 0, -1,  1]].
```

With the BCC bipartite factor,

```text
D_scar = 2 Delta(P3)
```

has exact spectrum:

```text
{0, 2, 6}.
```

The defect normal modes are:

```text
(1,  1,  1) / sqrt(3)  -> depth 0
(1,  0, -1) / sqrt(2)  -> depth 2
(1, -2,  1) / sqrt(6)  -> depth 6
```

The transfer operator is:

```text
T_boundary = epsilon^D_scar
```

which is

```text
diag(1, epsilon^2, epsilon^6)
```

in the defect eigenbasis.

## Controls

- Unbroken `K3` gives spectrum `{0,3,3}` and doubled spectrum `{0,6,6}`.
- A hand-written diagonal `diag(0,2,6)` has the right spectrum but is not
  graph-native.
- The weighted triangle scar with edge weights `(1,1,0)` or `(1/3,1/3,4/3)`
  gives `{0,1,3}` before doubling; the symmetric triangle remains degenerate.
- The three missing-edge path scars are spectrum-equivalent.

## Meaning

This sidecar upgrades the depth spectrum from a diagonal declared assignment to
a positive graph Laplacian, conditional on the existence of an `S3 -> Z2` repair
scar.

It does **not** derive the scar dynamically.  The next gate is to prove or kill
the physical origin of the scar.

## V2 Prediction Ledger

V2 separates built-in outputs from genuine predictions.  Once `P3` is accepted,
`Spec(D_scar) = {0,2,6}` is not a prediction; it is the defining output.  The
derived predictions are:

```text
T = P0 + epsilon^2 P2 + epsilon^6 P6
```

with exact port-space relations:

```text
T_uu = T_bb
T_ua = T_ab
```

The leading response is democratic and rank one:

```text
T -> P0 = (1/3) all-ones.
```

The CKM exponent pattern is:

```text
lambda = epsilon^2
|V12| : |V23| : |V13| ~ lambda : lambda^2 : lambda^3.
```

The pure path has no graph cycle, hence no intrinsic graph-holonomy CP phase.
CP must come from a loop-restoring, holonomy, or non-normal deformation.

Mass exponents are not fixed until a left/right Yukawa assignment is supplied.
The possible two-sided depth exponents are:

```text
{0,2,4,6,8,12}.
```

V2 verdict:

```text
DEPTH_SCAR_PREDICTION_LEDGER_PASS
```

## V3 Effective Edge-Weight Scar Potential

V3 derives the path scar at the effective edge-weight level.  Let the repair
bond weights be:

```text
w = (w_ua, w_ab, w_ub),    w_i >= 0.
```

Define:

```text
S1 = w_ua + w_ab + w_ub
S2 = w_ua w_ab + w_ab w_ub + w_ub w_ua
S3 = w_ua w_ab w_ub
```

The effective potential is:

```text
V(w) = (S1 - 2)^2 + (S2 - 1)^2 + S3.
```

On the nonnegative domain, `V=0` iff:

```text
S1 = 2,  S2 = 1,  S3 = 0.
```

This gives exactly the three missing-edge path scars:

```text
(1,1,0), (1,0,1), (0,1,1).
```

Each minimum has:

```text
Spec(Delta) = {0,1,3}
Spec(2 Delta) = {0,2,6}.
```

The `S3` term is load-bearing: removing it admits non-scar zeroes such as
`(4/3, 1/3, 1/3)`.  The unbroken symmetric triangle `(1,1,1)` is rejected and
remains degenerate.

V3 verdict:

```text
EDGE_WEIGHT_SCAR_POTENTIAL_PASS
```

This derives the scar from an effective boundary repair potential.  It still
does **not** derive that potential from a microscopic QCA update.

## V4 Loop-Healing And CP Deformation

V4 keeps the hierarchy and CP roles separate.  The pure path scar is a tree:

```text
u -- a -- b
```

so arbitrary phases on its two edges can be removed by port rephasings.  It has
cycle rank `0` and no intrinsic graph-holonomy CP source.

Restoring the missing edge gives a healed triangle with weights:

```text
w_ua = 1,  w_ab = 1,  w_ub = delta.
```

The real healed Laplacian has exact spectrum:

```text
Spec(Delta(delta)) = {0, 1 + 2 delta, 3}
Spec(2 Delta(delta)) = {0, 2 + 4 delta, 6}.
```

The complex Hermitian healing assigns a phase `phi` to the oriented loop:

```text
u -> a -> b -> u.
```

The healed triangle has cycle rank `1`, so `phi` is gauge invariant and flips
under complex conjugation.  At `phi=0` the matrix reduces to the real healed
Laplacian; at `delta=0` it reduces to the original path scar.

V4 verdict:

```text
LOOP_HEALING_CP_DEFORMATION_PASS
```

This proves that a loop-healing deformation is the minimal graph-native place
for CP holonomy.  It does **not** derive the microscopic values of `delta` or
`phi`.

## V5 Nilpotent Boundary-Flag Origin

V5 derives the same path repair graph from a canonical length-3 nilpotent
boundary flag.  In port order `(u,a,b)`, define:

```text
N = |u><a| + |a><b|
  = [[0,1,0],
     [0,0,1],
     [0,0,0]].
```

Then:

```text
N^3 = 0,   N^2 != 0.
```

The Hermitian graph data induced by the flag are:

```text
A_flag     = N + N.T
D_flag     = N N.T + N.T N
Delta_flag = D_flag - A_flag.
```

For the equal unit flag:

```text
Delta_flag = Delta(P3)
Spec(Delta_flag) = {0,1,3}
Spec(2 Delta_flag) = {0,2,6}.
```

Controls:

- A rank-one nilpotent has spectrum `{0,0,2}` and is rejected.
- Closing the cycle gives the unbroken `K3` control `{0,3,3}` and is rejected.
- The weighted flag `N(x,y)=x|u><a|+y|a><b|` hits the target spectrum only for
  `x^2=y^2=1`; on nonnegative amplitudes this is the canonical `(1,1)` flag.

V5 verdict:

```text
NILPOTENT_FLAG_SCAR_ORIGIN_PASS
```

This upgrades the path scar from an effective edge-weight minimum to a
nilpotent repair-flag origin.  It still does **not** derive why the microscopic
QCA boundary must realize that equal unit length-3 flag.

## Test Command

```bash
uv run pytest src/clifford_3plus2_d5/depth_scar/tests -q
```
