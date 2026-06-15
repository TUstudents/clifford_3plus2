# FN Center-Power Structure

This note analyzes the successful center-CP power tables as `Z3` data.  The
question is whether the fitted powers look like an arbitrary discrete phase
table or like a small Wilson-center structure that a future FN/color-holonomy
path rule could plausibly derive.

The fitted powers are

```text
n_u = [[2, 1, 1],
       [1, 0, 0],
       [0, 2, 0]]

n_d = [[1, 1, 1],
       [2, 0, 0],
       [1, 2, 0]]
```

All arithmetic below is over `Z3`.  The helper
`sm_analyze_verdict_center_powers()` computes the decomposition.
The helper `sm_center_powers_from_wilson_flux_rule()` reconstructs the same
tables from the low-complexity Wilson-rule ansatz described below.

## Decomposition

For one matrix `n_ij`, split off the anchored row/column coboundary

```text
n_ij = base + r_i + c_j + curvature_ij.
```

A pure `base + r_i + c_j` table is removable by left/right family rephasings.
The invariant data are the curvature and the elementary plaquette fluxes

```text
F_ij = n_ij - n_(i+1)j - n_i(j+1) + n_(i+1)(j+1).
```

The up matrix is

```text
coboundary_u = [[2, 1, 1],
                [1, 0, 0],
                [0, 2, 2]]

curvature_u  = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 1]]

F_u          = [[0, 0],
                [0, 1]]
```

So `n_u` is a removable row/column phase field plus one lower-right Wilson
center flux.

The down plaquette fluxes are

```text
F_d = [[1, 0],
       [0, 1]]
```

So `n_d` is not a pure rephasing either; it carries two elementary Wilson-center
fluxes.

Both matrices have full rank over `Z3`, so the CP content cannot be collapsed to
a single trivial row or column phase.

## Relative Structure

The important simplification is the relative defect:

```text
n_d - n_u = [[2, 0, 0],
             [1, 0, 0],
             [1, 0, 0]]
```

This has rank one over `Z3` and is supported only on the first right-handed
column.  That is not generic.  It says the down texture is the up texture plus a
single column-local center insertion, rather than an unrelated second phase
table.

## Physics Verdict

The fitted center powers should not be labeled as a random discrete fit table.
They have a low-complexity structure:

- `n_u` = row/column coboundary + one invariant plaquette flux.
- `n_d` = `n_u` + a rank-one first-column center defect.
- Equivalently, the up/down pair is governed by one shared background and one
  simple relative Wilson-center insertion.

## Minimal Wilson Rule

The current constructive rule is:

```text
1. choose the row/column coboundary gauge background;
2. insert one up-sector Wilson plaquette flux at the lower-right plaquette;
3. insert one down-sector relative Wilson line on the first right-handed column.
```

In code this is

```text
sm_center_powers_from_wilson_flux_rule()
```

and it returns exactly `VERDICT_CENTER_HOLONOMY_POWERS`.  Nearby controls are
rejected by `sm_verdict_center_power_path_rule_diagnostics()`:

- pure row/column coboundary has no invariant flux;
- the same flux in the wrong plaquette does not reproduce the verdict;
- a row defect does not reproduce the first-column relative defect;
- the older CUSP-style flag/bilinear powers are a different structure;
- the all-zero control carries no CP structure.

This upgrades the status from "empirical table" to "minimal Wilson-center path
rule candidate."  It is still not a derivation from bare BCC dynamics.  The next
hard problem is deriving the lower-right plaquette flux and the first-column
down insertion from microscopic closed recirculation paths.

The honest status is now:

```text
minimal Z3 Wilson-rule candidate, not arbitrary fit data,
but still awaiting microscopic color-holonomy derivation.
```
