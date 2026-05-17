# Session 24b - BCC Plaquette Holonomy

Status: implemented as background-link plaquette geometry and holonomy
covariance.

## Target

Session 24 proved exact site-local covariance of the finite BCC Dirac step with
position-dependent background links.  Session 24b defines the elementary closed
BCC loops needed to talk about link curvature.

This is still background gauge geometry.  It is not a gauge-field action or a
dynamical gauge update rule.

## Plaquette Convention

The link convention remains pull-form:

```text
U[x <- y] transforms as G[x] U[x <- y] G[y]^-1.
```

An elementary BCC plaquette is a non-backtracking four-hop parallelogram:

```text
(a, b, -a, -b)
```

where `a` and `b` are distinct, non-opposite BCC body-diagonal displacements.
Modulo cyclic rotation and orientation reversal, there are six canonical
unoriented elementary plaquette shapes.

For base site `x_0`, the path is:

```text
x_0 <- x_1 <- x_2 <- x_3 <- x_0
```

and the holonomy is:

```text
H[x_0] = U[x_0 <- x_1] U[x_1 <- x_2] U[x_2 <- x_3] U[x_3 <- x_0].
```

## Gauge Covariance

Under site-local gauge transforms, the plaquette holonomy transforms by
conjugation at the base site:

```text
H[x_0] -> G[x_0] H[x_0] G[x_0]^-1.
```

This is the lattice curvature transformation law needed before any future
gauge action or gauge dynamics layer.

## Implemented Objects

- `canonical_bcc_plaquette_shapes()`.
- `is_elementary_bcc_plaquette_shape(...)`.
- `plaquette_vertices_unwrapped(...)`.
- `plaquette_path_sites(...)`.
- `plaquette_holonomy(...)`.
- `plaquette_holonomy_covariance_residual(...)`.

## Audit Results

- There are six canonical unoriented elementary BCC four-hop plaquette shapes.
- Each canonical shape is closed, non-backtracking, and has four distinct
  unwrapped vertices before returning to the base.
- Identity link fields have identity plaquette holonomy.
- Pure-gauge link fields have identity plaquette holonomy.
- General background-link plaquette holonomies transform by base-site
  conjugation under exact site-local gauge transforms.
- Degenerate backtracking loops are rejected.

## Interpretation

Session 24b supplies the missing BCC loop geometry for curvature.  Together
with Session 24, the module now has:

```text
finite BCC state + position-dependent links + local gauge covariance
+ plaquette holonomy covariance
```

Still open:

- plaquette action / Wilson observable normalization;
- dynamical gauge-field update rule;
- full all-momentum BCC unitarity proof;
- full fundamental-BZ no-doubling proof.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_plaquette.py -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

Current expected result after Session 24b: 57 tests green.
