# Session 28 - Wilson Action Normalization

Status: implemented as a fixed-background gauge-energy audit.

## Target

Session 27 added Wilson plaquette traces.  Session 28 adds the corresponding
Wilson plaquette energy and total action:

```text
E_p = 1 - Re(Tr(H_p) / N)
S_W = beta * sum_p E_p
density = average_p E_p
```

where `N` is the internal link dimension and `H_p` is the BCC plaquette
holonomy.

This is still an observable/action layer.  It does not implement a dynamical
gauge-field update.

## Implemented

- Exact SymPy Wilson action helpers:
  - plaquette energy;
  - average action density;
  - total Wilson action with configurable `beta`.
- JAX Wilson action helpers with the same normalization.
- Defaults average uniformly over all lattice sites and the six canonical
  unoriented BCC plaquette shapes.
- Public exports through `spacetime_qca.__init__`.

## Audit Results

- Identity links have plaquette energy `0`.
- Identity links have action density `0`.
- Pure-gauge links have action density `0`.
- A nontrivial sign/flip link field has nonzero, nonnegative action density.
- Exact Wilson action density is invariant under finite site-local gauge
  transforms.
- JAX plaquette energy and average action density match exact SymPy values on
  tiny lattices.
- JAX total action satisfies
  `S = beta * density * site_count * shape_count`.
- JAX total action is invariant under transformed-link comparison.

## Interpretation

The module now has the fixed-background gauge-energy observable needed before
attempting gauge-field dynamics.  The Wilson action normalization is pinned,
parity-tested against exact SymPy calculations, and available in the JAX
backend.

Still not implemented:

- gauge-field update rule;
- canonical momentum / electric-field degrees of freedom;
- real-time Yang-Mills QCA;
- plaquette-force gradient tests;
- coupling to dynamical Higgs fields.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_wilson.py -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

Current expected result after Session 28: 81 tests green.

## Still Open

- Vectorized JAX action evaluation for large lattices.
- Wilson-action gradients with respect to link variables.
- Gauge-field dynamics.
- Coupling to position-dependent Higgs/scalar fields.
