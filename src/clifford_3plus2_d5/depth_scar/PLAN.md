# depth_scar Plan

## Purpose

This sidecar tests whether the quark family depth spectrum

```text
{0, 2, 6}
```

can be represented by a graph-native boundary repair operator:

```text
D_scar = 2 Delta(P3),
```

rather than by a hand-written diagonal spurion.

## V1 Question

Does an `S3 -> Z2` residual repair scar whose graph is the path `P3` give the
exact depth operator needed by the quark transfer hierarchy?

## Implementation

- `repair_graphs.py`: exact `K3`, `P3`, weighted-scar, mode, and transfer
  operators.
- `audit.py`: aggregate verdict payload.
- tests: exact spectrum, mode, transfer, `K3` control, weighted-scar control,
  and aggregate verdict checks.

## V1 Verdict Standard

V1 reports `PATH_DEFECT_LAPLACIAN_DEPTH_PASS` only if:

- `Delta(P3) = partial.T partial`;
- `Spec(2 Delta(P3)) = {0, 2, 6}`;
- the normal modes have depths `0`, `2`, and `6`;
- `epsilon^D_scar` has transfer factors `1`, `epsilon^2`, and `epsilon^6`;
- unbroken `K3` remains degenerate with doubled spectrum `{0, 6, 6}`;
- the diagonal `diag(0,2,6)` control is identified as non-graph-native;
- the weighted-scar controls hit the same target only for actual scarred
  weights.

## Interpretation If V1 Passes

The path-defect Laplacian gives an exact operator origin for the depth spectrum,
conditional on an `S3 -> Z2` repair scar.  It does not derive the scar
dynamically.

## Next Gate

## V2 Question

What does the path-defect operator actually predict, beyond its built-in
spectrum?

## V2 Implementation

- `prediction_ledger.py`: exact projectors, transfer kernel, endpoint parity,
  rank-one leading kernel, CP no-go, CKM exponents, and mass-exponent semigroup.
- `role_separation.py`: aggregate no-overclaim verdict.
- `THEORY.md`: compact statement of the theory, predictions, and limits.

## V2 Verdict Standard

V2 reports `DEPTH_SCAR_PREDICTION_LEDGER_PASS` only if:

- the projectors `P0`, `P2`, and `P6` are rank-one, orthogonal, and complete;
- `T = P0 + epsilon^2 P2 + epsilon^6 P6` matches the V1 transfer operator;
- the port-space relations `T_uu = T_bb` and `T_ua = T_ab` hold;
- endpoint parity blocks even/odd mixing;
- the leading transfer kernel is democratic and rank one;
- the pure path has no intrinsic graph-holonomy CP phase;
- adding the missing edge restores one graph cycle;
- CKM exponents in `lambda = epsilon^2` units are `(1,2,3)`;
- two-sided mass exponents form `{0,2,4,6,8,12}`.

## Interpretation If V2 Passes

The path scar is a quark transfer-depth / mixing hierarchy operator with a fixed
prediction ledger.  It is not a mass model by default, not a CP source by
itself, and not a universal lepton scar.

## Next Gate After V2

## V3 Question

Can the `S3 -> Z2` path scar be selected by an exact symmetric effective
potential over nonnegative repair-bond weights?

## V3 Implementation

- `edge_weight_potential.py`: symmetric edge invariants, scar potential,
  zero-set logic, spectra at minima, controls, and aggregate verdict.
- tests: invariant symmetry, exact minima, spectra, load-bearing controls, and
  aggregate payload.

## V3 Verdict Standard

V3 reports `EDGE_WEIGHT_SCAR_POTENTIAL_PASS` only if:

- `V=(S1-2)^2+(S2-1)^2+S3` is symmetric under all edge permutations;
- on `w_i >= 0`, `V=0` gives exactly `(1,1,0)` and permutations;
- every minimum gives `Spec(Delta)={0,1,3}` and `Spec(2 Delta)={0,2,6}`;
- the unbroken symmetric triangle is not a minimum and remains degenerate;
- removing the `S3` term admits a non-scar zero, proving the defect term is
  load-bearing.

## Interpretation If V3 Passes

The path scar is derived from an effective boundary repair potential.  This is
stronger than declaring the path, but still weaker than deriving the potential
from a microscopic QCA update.

## Next Gate After V3

## V4 Question

Can the pure path's CP no-go be sharpened into an exact loop-healing statement:
tree phases are removable, but restoring the missing edge creates one
gauge-invariant CP holonomy?

## V4 Implementation

- `loop_healing.py`: real healed triangle, Hermitian magnetic triangle,
  rephasing/gauge phase algebra, cycle-rank controls, and aggregate verdict.
- tests: exact real spectrum formula, tree phase removal, loop phase invariance,
  Hermiticity, path/real limits, conjugation controls, and payload.

## V4 Verdict Standard

V4 reports `LOOP_HEALING_CP_DEFORMATION_PASS` only if:

- the real healed triangle with weights `(1,1,delta)` has
  `Spec(Delta)={0,1+2 delta,3}`;
- BCC doubling gives `Spec(2 Delta)={0,2+4 delta,6}`;
- the magnetic healed triangle is Hermitian;
- arbitrary path phases are removable by port rephasings;
- the path has cycle rank `0` and the healed triangle has cycle rank `1`;
- the loop phase is gauge invariant and flips under complex conjugation;
- the `phi=0` control reduces to the real healed graph;
- the `delta=0` control reduces to the original path scar.

## Interpretation If V4 Passes

The pure `P3` scar is a hierarchy operator but not a CP source.  The minimal
graph-native CP location is a loop-restoring deformation.  The sidecar now
separates real hierarchy (`delta`) from CP holonomy (`phi`), while keeping both
microscopic values open.

## Next Gate After V4

## V5 Question

Can the `P3` scar be generated by an algebraic boundary defect, rather than by
declaring either a path graph or an effective edge-weight potential?

## V5 Implementation

- `nilpotent_flag.py`: canonical length-3 nilpotent repair flag, induced
  adjacency/degree/Laplacian, weighted flag controls, rank-one and cyclic
  controls, and aggregate verdict.
- tests: nilpotent order, induced path Laplacian, target spectra, transfer
  matching, negative controls, weighted-unit constraint, and payload.

## V5 Verdict Standard

V5 reports `NILPOTENT_FLAG_SCAR_ORIGIN_PASS` only if:

- `N=|u><a|+|a><b|` satisfies `N^3=0` and `N^2 != 0`;
- `N+N.T` is the path adjacency;
- `N N.T + N.T N` is the path degree operator;
- `Delta_flag = N N.T + N.T N - N - N.T` equals `Delta(P3)`;
- `Spec(Delta_flag)={0,1,3}` and `Spec(2 Delta_flag)={0,2,6}`;
- the induced transfer kernel matches V1;
- a rank-one nilpotent and a cyclic closure are rejected controls;
- a weighted flag hits the target spectrum only for unit adjacent amplitudes up
  to sign, leaving `(1,1)` as the unique nonnegative solution.

## Interpretation If V5 Passes

The path scar has a nilpotent repair-flag origin.  This is stronger than the V3
effective edge-weight potential, but still leaves one named input: the
microscopic QCA boundary must realize an equal unit length-3 flag.

## Next Gate After V5

Derive or kill the scar itself from one of:

- a missing repair channel boundary condition;
- a microscopic boundary update.

Also derive or kill the microscopic loop-healing parameters `delta` and `phi`;
V4 proves their graph role, not their numerical values.
