# Session 24 - Position-Dependent Gauge Links

Status: implemented as a finite-lattice background-gauge covariance audit.

## Target

Session 22 added finite real-space BCC stepping, but only for ungauged and
constant-link cases.  Session 24 promotes the finite step to position-dependent
background internal links.

The pull-form convention is:

```text
out[x] = sum_h (W_h x U[x <- x+h]) psi[x+h]
```

where `h` ranges over the eight BCC body-diagonal displacements.  A link key is
therefore `(target_site, displacement)`, representing the oriented edge
`target_site <- target_site + displacement`.

## Gauge Convention

For a site-local internal gauge transform `G[x]`:

```text
psi[x]       -> (I_space x G[x]) psi[x]
U[x <- y]    -> G[x] U[x <- y] G[y]^-1
step(psi,U) -> G step(psi,U)
```

This is an exact lattice statement.  No continuum expansion is involved in
this audit.

## Implemented Objects

- `LinkField`: exact SymPy matrices keyed by `(target_site, displacement)`.
- `identity_link_field(...)` and `constant_link_field(...)`.
- `transform_internal_state(...)`.
- `transform_link_field(...)`.
- `pure_gauge_link_field(...)`.
- `dirac_step_with_link_field(...)`.

The implementation does not enforce that links are group-valued.  The
covariance identity is algebraic and only requires compatible, invertible gauge
matrices for the transformation operation.

## Audit Results

- Identity link fields reproduce the ungauged/identity-link Dirac step.
- Constant link fields reproduce the existing constant-link Dirac step.
- Exact site-local gauge covariance holds on a finite periodic BCC lattice:

```text
dirac_step_with_link_field(G psi, G U G^-1)
  = G dirac_step_with_link_field(psi, U)
```

- Pure-gauge links preserve the norm of the deterministic plane-wave test state
  because they are gauge-equivalent to identity links.
- Missing link entries are rejected explicitly.
- The pull convention is tested directly:
  `U[target <- source] -> G[target] U G[source]^-1`.

## Interpretation

This is the first finite-lattice gauge-covariance result in `spacetime_qca`.
It upgrades the module from constant-background Bloch and real-space checks to
a genuine position-dependent background-link rule.

It does not yet provide:

- plaquette holonomies or BCC curvature geometry;
- dynamical gauge-field evolution;
- group-membership projection for arbitrary link matrices;
- site-local gauge covariance for a dynamical Higgs/Yukawa field.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_gauge_links.py -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

Current expected result after Session 24: 51 tests green.

## Still Open

- BCC plaquette / holonomy geometry.
- Full symbolic all-momentum BCC Bloch unitarity proof.
- Fundamental-BCC-Brillouin-zone no-doubling proof.
- Hermitian/dynamical Higgs-Yukawa layer.
- Dynamical gauge fields.
