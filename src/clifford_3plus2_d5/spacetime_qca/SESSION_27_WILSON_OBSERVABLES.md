# Session 27 - BCC Wilson Plaquette Observables

Status: implemented as background-link Wilson observables.

## Target

Session 24b built elementary BCC plaquette geometry and holonomies.  Session
27 adds Wilson-loop observables on top of those holonomies:

```text
H[x0, shape] = U[x0 <- x1] U[x1 <- x2] U[x2 <- x3] U[x3 <- x0]
W[x0, shape] = Tr(H[x0, shape])
W_norm[x0, shape] = Tr(H[x0, shape]) / internal_dim
```

This is still a background-gauge observable layer, not a dynamical gauge-field
update or Wilson action.

## Implemented

- Exact SymPy Wilson helpers:
  - plaquette trace;
  - normalized plaquette trace;
  - uniform site/shape average over the six canonical BCC plaquette shapes.
- JAX Wilson helpers:
  - plaquette holonomy from `(nx, ny, nz, 8, d, d)` link layout;
  - trace and normalized trace;
  - uniform average over all sites and canonical shapes.
- Public exports through `spacetime_qca.__init__`.

All helpers use the existing pull-link convention:

```text
U[target <- target + displacement]
```

and the same BCC displacement order as the JAX step backend.

## Audit Results

- Identity links give `Tr(H) = internal_dim`.
- Identity links give normalized Wilson loop `1`.
- Pure-gauge links give average normalized Wilson loop `1`.
- Exact Wilson traces are invariant under finite site-local gauge transforms.
- JAX plaquette holonomy matches exact SymPy holonomy on a tiny lattice.
- JAX normalized Wilson loop matches exact SymPy normalized Wilson loop.
- JAX Wilson traces are invariant under the exact transformed-link comparison.

## Interpretation

The module now has the curvature-observable layer needed before any Wilson
action or dynamical gauge update.  The exact SymPy implementation remains the
reference, and the JAX implementation is parity-tested against it.

What is still not implemented:

- plaquette action / gauge energy functional;
- gauge-field update rule;
- dynamical gauge-field Hilbert space;
- coupling to a dynamical Higgs field;
- long-time numerical gauge evolution.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_wilson.py -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

Current expected result after Session 27: 76 tests green.

## Still Open

- Wilson action normalization on BCC plaquette families.
- JAX batching/vectorization for Wilson averages on large lattices.
- Gauge-field dynamics.
- Observables for nontrivial non-pure-gauge fields.
