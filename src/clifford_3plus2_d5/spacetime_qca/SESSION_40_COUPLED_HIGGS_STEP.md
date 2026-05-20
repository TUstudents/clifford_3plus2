# Session 40 - Coupled Fermion + Gauge + Higgs Prototype

Status: implemented as a small-lattice JAX prototype with all field types
present.

## Target

Session 40 combines the existing pieces:

- BCC Dirac fermions on the real chiral16 internal carrier.
- Compact Pati-Salam/SM gauge links and link momenta.
- Session 37 matter-current gauge backreaction.
- Session 39 site-local Higgs field.
- Session 38 Hermitian Yukawa map, evaluated sitewise.

Per the chosen implementation direction, this session adds a Higgs conjugate
momentum and a leapfrog-style Higgs update.

This is still not a production Standard Model simulator.  It is a
small-lattice control that makes the interfaces explicit.

## Added API

New module:

```text
jax_coupled_higgs.py
```

Public helpers:

- `jax_transform_higgs_momentum`
- `jax_higgs_momentum_energy_density`
- `jax_higgs_total_energy`
- `jax_higgs_force`
- `jax_higgs_leapfrog_step`
- `jax_higgs_coordinates_from_patisalam_sector`
- `jax_higgs_link_field_from_patisalam_sector`
- `jax_higgs_site_gauge_from_patisalam_sector`
- `jax_apply_site_local_yukawa_kick`
- `jax_patisalam_fermion_gauge_higgs_step`
- `jax_patisalam_fermion_gauge_higgs_diagnostics`

The supported Higgs-coupled sectors are:

```text
u1_y, su2_l, sm
```

Color-only and full Pati-Salam sectors are rejected by the Higgs sector
adapter unless explicit fundamental Higgs links are supplied separately.

## Conventions

Higgs momentum has the same layout as the Higgs field:

```text
Pi.shape == Phi.shape == (nx, ny, nz, 2)
Pi[x] -> G[x] Pi[x]
```

The Higgs Hamiltonian used by the leapfrog update is:

```text
H_H = sum_x ( 0.5 |Pi[x]|^2 + sum_h |D_h Phi[x]|^2
              + lambda (|Phi[x]|^2 - v^2)^2 )
```

The Higgs force is computed by JAX autodiff over explicit real/imaginary
coordinates:

```text
F_Phi = - dH_H / dPhi
```

The coupled step ordering is:

```text
psi        <- half site-local Yukawa kick from Phi
(U, P, psi)<- Session 37 sourced gauge/fermion step
(Phi, Pi) <- fixed-link Higgs leapfrog step
psi        <- half site-local Yukawa kick from updated Phi
```

The site-local Yukawa kick is first-order explicit:

```text
psi -> psi - i dt (beta x Y(Phi[x])) psi
```

It is not yet an exact local unitary exponential.

## Audit Results

- Higgs momentum transforms covariantly and preserves norm.
- Higgs momentum energy is gauge invariant.
- Higgs force vanishes at the neutral VEV with identity Higgs links.
- Higgs force is gauge covariant under finite electroweak transforms.
- Higgs leapfrog is identity at zero step size.
- Neutral VEV with zero momentum is a fixed point for identity links.
- Neutral VEV with zero momentum and identity links has zero total Higgs
  energy for `v^2 = 1`.
- Sector adapters map `u1_y`, `su2_l`, and `sm` coordinates into the
  fundamental Higgs representation.
- Color-only sectors are rejected by the v1 coupled wrapper.
- Coupled diagnostics report finite fermion, gauge, Higgs, and Gauss residual
  quantities.
- Zero `Phi` and zero Yukawa step are no-ops for the Yukawa kick.
- With Higgs/Yukawa disabled, the coupled wrapper reduces to the Session 37
  gauge/fermion wrapper.
- Tiny-lattice smoke tests pass for `u1_y`, `su2_l`, and `sm` sectors.

## Interpretation

Session 40 is the first executable prototype with fermions, gauge fields, and
a Higgs field in one update path.  It closes the interface gap between the
Higgs/Yukawa layer and the gauge/fermion dynamics stack.

Important limitations remain:

- Higgs links are explicit fixed electroweak links in this v1 wrapper.
- Higgs current does not yet backreact on gauge momenta.
- The Yukawa kick is first-order explicit and can drift norm.
- There is no Gauss-law projection or long-time stability claim.
- Physical coupling constants and Yukawa hierarchies are not fitted.

## Validation

Focused commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_coupled_higgs.py -m "not slow" -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_coupled_higgs.py -m slow -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_coupled_higgs.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_coupled_higgs.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py
```

Focused result:

```text
9 passed, 6 deselected in 13.20s
6 passed, 9 deselected in 128.34s
111 passed, 125 deselected in 56.60s    # fast spacetime_qca suite
```

## Still Open

- Exact local unitary Yukawa exponential.
- Higgs current/backreaction into gauge momenta.
- Coupled Gauss-law projection.
- Long-time stability and energy-drift diagnostics.
- Physical Higgs/Yukawa parameter fitting.
