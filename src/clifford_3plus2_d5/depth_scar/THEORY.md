# Depth Scar Theory

## What V1 Built In

Once the path repair scar is postulated,

```text
u -- a -- b
```

the spectrum is not a prediction.  It is the defining output:

```text
D_scar = 2 Delta(P3)
Spec(D_scar) = {0, 2, 6}.
```

The scientific question is therefore not whether `P3` gives `{0,2,6}`.  V1
already proves that.  The real questions are what follows from that operator,
and what must still be supplied by another mechanism.

## Genuine Predictions

The depth scar predicts fixed family directions:

```text
psi_0 = (1,  1,  1) / sqrt(3)
psi_2 = (1,  0, -1) / sqrt(2)
psi_6 = (1, -2,  1) / sqrt(6)
```

So a microscopic boundary response should return normal modes, not port-localized
states.

The transfer kernel is fixed:

```text
T = P0 + epsilon^2 P2 + epsilon^6 P6.
```

This implies exact port-space relations:

```text
T_uu = T_bb
T_ua = T_ab
```

and a leading democratic rank-one response:

```text
T -> P0 = (1/3) all-ones.
```

The CKM exponent prediction is:

```text
lambda = epsilon^2
|V12| : |V23| : |V13| ~ lambda : lambda^2 : lambda^3.
```

## Selection Rules And Limits

The path has one surviving endpoint reflection `u <-> b`.  Under this symmetry,
the depth-0 and depth-6 modes are even, while the depth-2 mode is odd.  Any
operator preserving the endpoint reflection is even/odd block diagonal.

The pure path has no loop, so it has no intrinsic graph-holonomy CP phase.  CP
must come from a loop-restoring deformation, holonomy, non-normal monodromy, or
sector-dependent complex seed matrices.

The path scar is not a mass model by default.  If a Yukawa sector factors through
left and right transfers, possible two-sided depth exponents are:

```text
{0, 2, 4, 6, 8, 12}
```

or, in powers of `lambda = epsilon^2`,

```text
{0, 1, 2, 3, 4, 6}.
```

Those are conditional diagnostics, not automatic mass predictions.

## Current Status

V2 records these consequences with verdict:

```text
DEPTH_SCAR_PREDICTION_LEDGER_PASS
```

This means the scar is a consistent quark transfer-depth operator with a fixed
prediction ledger.  It does not derive the scar dynamically, does not produce CP
by itself, and does not impose a universal lepton scar.

V3 derives the scar at the effective edge-weight level.  For nonnegative repair
weights `w=(w_ua,w_ab,w_ub)`, define:

```text
V(w) = (S1 - 2)^2 + (S2 - 1)^2 + S3,
```

where `S1`, `S2`, and `S3` are the elementary symmetric polynomials in the three
edge weights.  Since all terms are nonnegative on `w_i >= 0`, `V=0` forces:

```text
S1=2, S2=1, S3=0.
```

Thus one edge is missing and the other two have unit weight.  The zero set is
exactly:

```text
(1,1,0), (1,0,1), (0,1,1).
```

The verdict is:

```text
EDGE_WEIGHT_SCAR_POTENTIAL_PASS
```

This is not yet a microscopic QCA derivation.  It is an effective repair-weight
selection theorem.

## Loop Healing And CP

V4 proves the clean separation between hierarchy and CP.  A tree has no
gauge-invariant edge phase: for arbitrary path phases `theta_ua` and `theta_ab`,
port rephasings remove both.  Thus the pure `P3` scar can generate a real
hierarchy operator, but it cannot generate intrinsic graph-holonomy CP.

If the missing edge is healed with real weight `delta`, the triangle Laplacian
has exact spectrum:

```text
Spec(Delta(delta)) = {0, 1 + 2 delta, 3}
Spec(2 Delta(delta)) = {0, 2 + 4 delta, 6}.
```

So real loop healing moves the middle depth while leaving the top depth fixed in
this symmetric deformation.

If the healed edge carries an oriented phase `phi`, the triangle has one cycle
and one gauge-invariant holonomy:

```text
u -> a -> b -> u.
```

The phase cannot be removed by port rephasings and flips under complex
conjugation.  Therefore CP is not a property of the real `P3` depth graph; it is
the property of the loop-restoring deformation.

The verdict is:

```text
LOOP_HEALING_CP_DEFORMATION_PASS
```

This does not derive `delta` or `phi`.  It proves where they must live if the
depth-scar route is used for quark mixing: `P3` supplies hierarchy, the healed
loop supplies holonomy.

## Nilpotent Flag Origin

V5 gives the path scar an algebraic origin.  In port order `(u,a,b)`, define the
length-3 repair monodromy:

```text
N = |u><a| + |a><b|.
```

It obeys:

```text
N^3 = 0,   N^2 != 0.
```

The nilpotent induces the graph data:

```text
A_flag     = N + N.T
D_flag     = N N.T + N.T N
Delta_flag = D_flag - A_flag.
```

For equal unit adjacent flag steps:

```text
Delta_flag = Delta(P3).
```

So the hierarchy operator follows:

```text
Spec(2 Delta_flag) = {0,2,6}.
```

The rank-one flag fails because it does not have three layers.  The cyclic
closure fails because it restores the `K3` degeneracy.  A weighted flag
`N(x,y)` reaches the target spectrum only when `x^2=y^2=1`; with nonnegative
amplitudes this fixes the canonical unit flag.

The verdict is:

```text
NILPOTENT_FLAG_SCAR_ORIGIN_PASS
```

The remaining physical input is now sharper: derive the equal unit length-3
nilpotent flag from the microscopic boundary update, or declare it as the scar
axiom.
