# Session 39 - Site-Local Higgs Field

Status: implemented as a JAX-first background Higgs-field layer.

## Target

Session 38 made the Hermitian Yukawa insertion depend on a fixed complex
doublet:

```text
Phi = (phi_plus, phi_zero)
Y_H(Phi) = beta x Y_internal(Phi)
```

Session 39 promotes `Phi` to a site-local lattice field with its own finite
gauge law, BCC covariant difference, kinetic energy, and Mexican-hat potential
diagnostics.

This is still not full Higgs dynamics.  There is no second-order field update,
no momentum conjugate to `Phi`, and no coupled fermion-Higgs time step.

## Field Convention

The numerical field is a fundamental electroweak doublet:

```text
phi.shape == (nx, ny, nz, 2)
phi[..., 0] = phi_plus
phi[..., 1] = phi_zero
```

Gauge coordinates are ordered:

```text
(su2_x, su2_y, su2_z, u1_y)
```

The anti-Hermitian generators are:

```text
T_a = -i sigma_a / 2        for SU(2)_L
T_Y = -i I_2 / 2            for hypercharge Y = +1/2
```

With this convention, `i(T_3 + T_Y)` has charges `(+1, 0)` on
`(phi_plus, phi_zero)`.

## BCC Link Convention

The Higgs layer uses the same pull-link convention as the fermion and gauge
layers:

```text
U[x,h] transports from source x+h to target x
Phi[x] -> G[x] Phi[x]
U[x,h] -> G[x] U[x,h] G[x+h]^dagger
D_h Phi[x] = U[x,h] Phi[x+h] - Phi[x]
```

The sitewise kinetic diagnostic is:

```text
K_H[x] = sum_h ||D_h Phi[x]||^2
```

The potential diagnostic is:

```text
V_H[x] = lambda (|Phi[x]|^2 - v^2)^2
```

## Added API

In `jax_higgs.py`:

- `jax_higgs_generators`
- `jax_constant_higgs_field`
- `jax_higgs_site_gauge_from_algebra`
- `jax_higgs_link_field_from_algebra`
- `jax_higgs_pure_gauge_links_from_site_algebra`
- `jax_transform_higgs_field`
- `jax_transform_higgs_links`
- `jax_higgs_covariant_difference`
- `jax_higgs_kinetic_energy_density`
- `jax_higgs_potential_density`
- `jax_higgs_energy_density`
- `jax_higgs_yukawa_internal_control_field`
- `jax_higgs_yukawa_hamiltonian_field`

The last two functions bridge the site-local Higgs field back to the Session
38 static `Y(Phi)` construction at each site.

## Audit Results

- The Higgs generators are anti-Hermitian.
- The pinned electromagnetic charge observable has charges `(+1, 0)`.
- Finite site and link gauge matrices are unitary.
- The BCC covariant difference transforms covariantly.
- Higgs norm, kinetic energy, and kinetic-plus-potential energy are gauge
  invariant on the tested tiny lattice.
- Pure-gauge links make a gauge-transformed constant Higgs field covariantly
  constant.
- The Mexican-hat potential depends only on `|Phi|^2`.
- The neutral VEV `Phi = (0, 1)` has zero potential for `v^2 = 1`.
- Site-local Yukawa matrices match the Session 38 neutral static helpers for a
  constant neutral field.
- A zero Higgs field gives zero site-local Yukawa matrices.

## Interpretation

Session 39 closes the first dynamical-Higgs infrastructure gap: `Phi(x)` now
has a gauge-covariant lattice representation and energy diagnostics.  The
physics-specific Yukawa slot remains the Session 38 one, evaluated sitewise.

The result should be read as a field-layer construction, not as a complete
Higgs sector.  A real dynamical Higgs update still needs a conjugate momentum
or equivalent first-order QCA state, a stable time-update rule, and coupling to
the fermion and gauge updates.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_higgs.py -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -m "not slow" -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_higgs.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_higgs.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py
```

Focused result:

```text
6 passed, 2 deselected in 5.16s       # fast Higgs tests
2 passed, 6 deselected in 108.73s     # slow sitewise Yukawa bridge tests
```

Fast `spacetime_qca` result:

```text
102 passed, 119 deselected in 47.16s
```

## Still Open

- Insert site-local `Y(Phi[x])` into the real-space fermion update.
- Add a Higgs field momentum or first-order update state.
- Couple Higgs dynamics to gauge links and fermion backreaction.
- Audit spontaneous breaking in a time-dependent field configuration.
