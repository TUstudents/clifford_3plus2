# depth_scar — Status

**Status**: V1 implemented.

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

## Test Command

```bash
uv run pytest src/clifford_3plus2_d5/depth_scar/tests -q
```

