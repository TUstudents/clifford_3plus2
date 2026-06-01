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

Derive or kill the scar itself from one of:

- a missing repair channel boundary condition;
- an edge-weight potential with minima at `(1,1,0)` permutations;
- a monodromy or nilpotent boundary flag;
- a microscopic boundary update.

