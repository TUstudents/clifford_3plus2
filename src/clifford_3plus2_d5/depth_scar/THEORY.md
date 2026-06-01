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

## Local Unitarity Normalization

V6 removes the equal-amplitude part of that input.  On the same support, take the
most general complex local flag:

```text
N = r exp(i alpha)|u><a| + s exp(i beta)|a><b|.
```

If `N` is a subblock of a local unitary boundary update, it must preserve norm on
its active repair subspace.  Algebraically this is the partial-isometry
condition:

```text
(N.H N)^2 = N.H N
(N N.H)^2 = N N.H.
```

But:

```text
N.H N = diag(0, r^2, s^2)
N N.H = diag(r^2, s^2, 0).
```

Therefore every nonzero active singular value is forced to be one:

```text
r^2 = s^2 = 1.
```

With nonnegative magnitudes, `r=s=1`.  The remaining phases lie on a tree and are
removed by port rephasings.  Hence:

```text
N ~ |u><a| + |a><b|.
```

The verdict is:

```text
LOCAL_FLAG_PARTIAL_ISOMETRY_PASS
```

After V6, the remaining scar axiom is not an arbitrary normalization.  It is the
support statement: the microscopic boundary must provide the length-3 nilpotent
support `b -> a -> u`.

## Minimal Support Classification

V7 classifies the support statement itself at the finite binary level.  On three
ports there are six off-diagonal entries, hence:

```text
2^6 = 64
```

no-self-loop directed supports.  Imposing:

```text
N^3 = 0,
N^2 != 0,
rank(N) = 2,
edge_count(N) = 2
```

leaves six supports, all related by `S3` port relabeling.  The unique orbit is
represented by:

```text
N = |u><a| + |a><b|.
```

Thus minimal length-3 nilpotent repair implies the V5/V6 flag and therefore the
path Laplacian.  The verdict is:

```text
MINIMAL_NILPOTENT_SUPPORT_CLASSIFICATION_PASS
```

The two-edge minimality condition is load-bearing.  If it is dropped, additional
rank-2 length-3 acyclic shortcut supports survive.  Therefore the remaining
physical input is now sharply named:

```text
microscopic repair is minimal two-edge nilpotent repair.
```

## Minimal Causal-Repair Variational Principle

V8 turns that named minimality input into a finite constrained optimization.
Instead of imposing `edge_count(N)=2`, start with all supports satisfying:

```text
N^3 = 0,
N^2 != 0,
rank(N) = 2,
all three ports active.
```

There are twelve such supports: six path flags and six shortcut acyclic
supports.  Define the causal repair cost:

```text
cost(N) = edge_count(N).
```

Then the finite variational problem:

```text
minimize cost(N)
```

has:

```text
minimum cost = 2
minimizers = one S3 orbit.
```

That orbit is exactly the V5/V6/V7 path flag:

```text
N = |u><a| + |a><b|
```

up to port relabeling.  The shortcut supports remain feasible but have cost
`3`, so they are not shortest causal repairs.

The verdict is:

```text
MINIMAL_CAUSAL_REPAIR_VARIATIONAL_PASS
```

This is stronger than V7's support classification: the two-edge condition is no
longer a separate support predicate, but the minimizer of a declared finite
repair cost.  The remaining physics is the cost principle itself:

```text
microscopic QCA repair minimizes edge count / causal repair length.
```

That is still not derived by this sidecar.

## Microscopic Locality / Minimality Split

V9 replaces the abstract edge-count principle with an explicit locality theorem,
conditional on a residual defect filtration.  Assign:

```text
h(u)=0, h(a)=1, h(b)=2.
```

The repair block `N` is the strictly height-lowering part of the boundary
update, not the full unitary update.  Monotone repair permits:

```text
a -> u,
b -> a,
b -> u.
```

If the residual one-tick boundary geometry is:

```text
u -- a -- b,
```

then `b -> u` is not one-tick local.  Equivalently, under BCC bipartite parity:

```text
p(u)=0, p(a)=1, p(b)=0,
```

the allowed one-tick links change parity, while `b -> u` preserves parity.

Thus:

```text
three-level filtration
+ monotone repair
+ one-tick locality
+ rank-complete repair
=> N = alpha |u><a| + beta |a><b|,
```

with `alpha,beta` nonzero.  This is V9a:

```text
V9A_MICROSCOPIC_SUPPORT_MINIMALITY_PASS
```

The normalization is a separate statement.  Locality gives support, but a
generic unitary subblock can leak and need only be a contraction.  If the active
repair block saturates to a partial isometry, V6 applies:

```text
|alpha|=|beta|=1,
```

and tree rephasings remove both phases.  This is V9b:

```text
V9B_MICROSCOPIC_NORMALIZATION_MINIMALITY_CONDITIONAL_PASS
```

If saturation fails, the support theorem still gives a weighted path with
weights `w1,w2`.  Its exact spectrum is:

```text
0,
w1+w2 - sqrt(w1^2 - w1 w2 + w2^2),
w1+w2 + sqrt(w1^2 - w1 w2 + w2^2).
```

The target `{0,1,3}` forces:

```text
w1=w2=1.
```

So equal weights are not cosmetic; they are exactly the condition fixing the
nonzero eigenvalue ratio.

The combined V9 verdict is:

```text
MICROSCOPIC_LOCALITY_MINIMALITY_CONDITIONAL_PASS
```

The open physics is now highly localized:

```text
derive h(u,a,b),
derive residual one-tick geometry u-a-b,
prove or replace active repair isometry saturation.
```

## Active Repair Isometry Saturation

V10 proves the algebra behind the last item.  Let the active domain be:

```text
A = span{|a>, |b>},
```

and the repaired range be:

```text
R = span{|u>, |a>}.
```

For the full microscopic unitary update `U`, define:

```text
N = P_R U P_A,
L = (I - P_R) U P_A.
```

Restricting `U.H U = I` to the active domain gives:

```text
N.H N + L.H L = I_A.
```

Therefore the active repair block is a unit isometry if and only if there is no
leakage:

```text
N.H N = I_A  <=>  L = 0.
```

Under V9's path support:

```text
N = alpha |u><a| + beta |a><b|.
```

No leakage forces:

```text
|alpha| = |beta| = 1.
```

Tree rephasings then remove the two phases, giving the canonical flag:

```text
N ~ |u><a| + |a><b|.
```

If leakage is present, write:

```text
w1 = |alpha|^2,
w2 = |beta|^2.
```

The repair graph is a weighted path.  Equal leakage gives `w1=w2=w`, hence
spectrum:

```text
{0,w,3w}.
```

It preserves the `1:3` ratio but rescales the depth spectrum.  Unequal leakage
breaks the `1:3` ratio.  The exact target `{0,1,3}` requires:

```text
w1=w2=1.
```

The V10 verdict is:

```text
V10_REPAIR_ISOMETRY_SATURATION_PASS
```

This is still conditional physics.  V10 does not prove that the actual QCA has
`L=0`.  It proves that `L=0` is exactly the missing microscopic normalization
condition.

## Selection-Signature No-Leakage

V11 proves that no-leakage follows from unique microscopic successors.  For each
active residual state, define the allowed successor subspace:

```text
Omega(j) = span{one-tick outputs satisfying all selection rules from j}.
```

The relevant active states are `a` and `b`.  If:

```text
Omega(a) = {u},
Omega(b) = {a},
```

then unitarity forces:

```text
U|a> = exp(i theta_a)|u>,
U|b> = exp(i theta_b)|a>.
```

Both outputs lie in:

```text
R = span{|u>, |a>}.
```

Therefore:

```text
L = (I-P_R) U P_A = 0.
```

By V10:

```text
N.H N = I_A.
```

The tree phases are removable, giving:

```text
N ~ |u><a| + |a><b|.
```

The V11 verdict is:

```text
V11_SELECTION_SIGNATURE_NO_LEAKAGE_PASS.
```

This is still an abstract theorem, not a microscopic enumeration.  Its negative
control is immediate: if

```text
Omega(a) = {u, bulk_a},
```

then leakage is allowed and the theorem does not apply.

The next gate is therefore finite and concrete:

```text
V12_UNIQUE_SUCCESSOR_ENUMERATION_GATE.
```

It must enumerate the actual local boundary basis and apply the selection
filters.  The pass condition is exactly:

```text
Omega(a) = {u},
Omega(b) = {a}.
```

If V12 fails, the honest model becomes an effective weighted-path scar with
leakage corrections.  If V12 passes, the unit `P3` scar becomes a microscopic
consequence of the selection rules.

## Unique-Successor Enumeration Certificate

V12 implements the certificate format for the V11 condition.  A finite local
boundary candidate basis is supplied, and every transition from active source
`a` or `b` to every candidate target is assigned:

```text
ALLOW
```

or:

```text
FORBID + exact vetoes.
```

The veto labels are:

```text
LOCALITY,
HEIGHT,
BCC_PARITY,
COLOR,
WEYL,
BOUNDARY_SECTOR,
SUPERSELECTION.
```

The certificate table proves:

```text
Omega(a) = {u},
Omega(b) = {a}.
```

It also distinguishes two failure classes.  External leakage candidates, such
as bulk/spectator/wrong-sector states, are vetoed by internal or boundary-sector
selection rules.  The direct `b -> u` candidate is different: it is a shortcut
repair candidate, vetoed by one-tick locality, height, and BCC parity.  If that
shortcut ever becomes allowed, the graph topology changes toward loop healing,
not merely weighted-path leakage.

The V12 verdict is:

```text
V12_UNIQUE_SUCCESSOR_ENUMERATION_CERTIFICATE_PASS.
```

This is still not the final microscopic theorem.  V12 certifies the current
finite candidate table.  It does not prove that the table is the complete local
BCC-QCA boundary basis.  The remaining microscopic statement is therefore:

```text
the V12 candidate basis is complete.
```
