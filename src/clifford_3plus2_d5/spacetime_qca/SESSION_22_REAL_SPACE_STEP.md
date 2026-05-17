# Session 22 - Finite Real-Space BCC Step

Status: implemented as an exact finite periodic real-space step.

## Target

Sessions 20-21 worked primarily with Bloch symbols and Hamiltonian expansions.
Session 22 implements the microscopic pull-form update:

```text
psi(t + eps, x) = sum_h W(h) psi(t, x + h)
```

on a finite periodic 3D lattice.

## Lattice Convention

The finite lattice is a periodic cubic index lattice.  BCC hops are represented
by integer body-diagonal displacements:

```text
(+/-1, +/-1, +/-1)
```

The normalized BCC directions remain part of the continuum/Bloch geometry; the
real-space step uses integer displacements.

## Implemented Steps

- `weyl_step(state, lattice, helicity="right"|"left")`
- `dirac_step(state, lattice)`
- `dirac_step_with_constant_link(state, lattice, link)`

The Dirac step is the blockwise combination of the right- and left-helicity BCC
Weyl steps.

The constant-link step applies:

```text
sum_h (W_D(h) x link) psi(x + h)
```

with the internal link fixed across all sites and directions.

## Plane-Wave Consistency

The package uses the pull-form Bloch convention

```text
U(k) = sum_h exp(-i k.h) W(h).
```

The finite real-space plane-wave helper therefore uses:

```text
psi[x] = exp(-i k.x) spinor.
```

Tests verify that applying the finite real-space step to such a plane wave
matches multiplication by the corresponding Weyl or Dirac Bloch symbol.

## Norm Preservation

The real-space tests verify norm preservation on deterministic plane-wave
states for:

- Weyl step;
- Dirac step.

This is an exact finite-lattice check, not only a Bloch-symbol identity.

## Validation

The Session 22 tests cover:

- periodic wrapping and translation;
- Weyl state helpers;
- delta-state BCC support;
- Dirac step equals blockwise Weyl steps;
- plane-wave / Bloch-symbol consistency;
- norm preservation on deterministic states;
- constant identity-link equivalence to ungauged tensor stepping.

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

## Still Open

- Position-dependent links.
- Site-local gauge covariance on finite lattices.
- Exact norm preservation with finite orthogonal nontrivial gauge links.
- Dynamical gauge fields.
- Optimized numerical stepping for larger lattices.
