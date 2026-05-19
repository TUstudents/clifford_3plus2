# Session 37 Plan - Gauss Law And Fermion Backreaction

## Goal

Move beyond the Session 36 no-backreaction wrapper.

Session 36 evolves gauge links by pure-gauge leapfrog and then transports the
fermion field through the updated links.  Session 37 should add the first
constrained matter/gauge layer:

1. a gauge-covariant lattice Gauss residual;
2. a fermion charge density in the Pati-Salam/SM generator basis;
3. a small, explicit matter-current kick for link momenta;
4. covariance tests on tiny lattices.

This is still a prototype.  The goal is not production HMC, not a full
constraint solver, and not a final quantum-current prescription.  The goal is
to make the coupling conventions precise enough that later Higgs/Yukawa work
does not sit on top of an unconstrained gauge layer.

## Existing Inputs

Use only the current `spacetime_qca` and `sim` conventions:

- Fermion state:
  `(nx, ny, nz, 4, 32)`.
- Pati-Salam/SM link field:
  `(nx, ny, nz, 8, 32, 32)`.
- Link momenta:
  `(nx, ny, nz, 8, sector_dim)`.
- Pull-link convention:
  `U[x,h]` maps source `x+h` to target `x`.
- Site gauge transform:
  `psi[x] -> G[x] psi[x]`.
- Link gauge transform:
  `U[x,h] -> G[x] U[x,h] G[x+h]^dagger`.
- Left-trivialized momentum transform:
  `P[x,h] -> G[x] P[x,h] G[x]^dagger`.

The implementation should reuse:

- `jax_bcc_displacements`
- `jax_transform_link_field`
- `jax_transform_patisalam_dirac_state`
- `jax_patisalam_generators_chiral16`
- `jax_patisalam_algebra_matrix`
- `jax_patisalam_project_to_coordinates`
- `jax_patisalam_transform_momentum_field`
- `jax_patisalam_fermion_gauge_step`

## New Module

Add:

```text
src/clifford_3plus2_d5/spacetime_qca/jax_gauss.py
```

The module should be Pati-Salam/SM-sector generic, matching
`jax_patisalam.py`.

## Definitions

### Momentum Algebra Matrix

Convert link-momentum coordinates to anti-Hermitian matrices:

```text
P_matrix[x,h] = sum_a P_a[x,h] T_a
```

where `T_a` are the anti-Hermitian sector generators in the 32-dimensional
chiral16 representation.

### Gauge-Covariant Electric Divergence

For a site `x`, each BCC neighbor direction `h` contributes:

- incoming edge `x+h -> x`, stored as `U[x,h]`, with left momentum
  `P[x,h]` already transforming at `x`;
- outgoing edge `x -> x+h`, stored as `U[x+h,-h]`, whose left momentum lives
  at `x+h` and must be parallel-transported back to `x`.

Define the target-site algebra residual:

```text
divE[x] =
  sum_h ( P_matrix[x,h]
        - U[x+h,-h]^dagger P_matrix[x+h,-h] U[x+h,-h] )
```

This transforms covariantly:

```text
divE[x] -> G[x] divE[x] G[x]^dagger
```

Implementation detail:

- use `source_roll(field, h)` to access `field[x+h]`;
- find `minus_h_index` from `jax_bcc_displacements()`;
- project the final algebra matrices back to sector coordinates with
  `jax_patisalam_project_to_coordinates`.

### Fermion Charge Density

For anti-Hermitian generators `T_a`, the Hermitian charge observables are:

```text
Q_a = i T_a
```

Define site charge coordinates:

```text
rho_a[x] = sum_s psi[x,s]^dagger Q_a psi[x,s]
```

where `s` runs over the four Dirac components.

Because the generator bases are not guaranteed to be unit-normalized, the
first implementation should expose both:

- raw moments `rho_raw_a`;
- Gram-dual coordinates `rho_a`.

The default Gauss residual should use Gram-dual coordinates so it is compatible
with the basis-coordinate momentum convention from Sessions 34-35.

### Full Gauss Residual

Coordinate residual:

```text
G_a[x] = divE_a[x] - matter_coupling * rho_a[x]
```

The sign is a convention.  Pin it with `u1_y` tests:

- zero fermion state gives `G = divE`;
- zero momenta and nonzero fermion state give `G = -g rho`;
- flipping `matter_coupling` flips only the matter-source term.

### Fermion Link Current

A unique quantum current prescription is out of scope for Session 37.  The
prototype should define a covariant current as the compact left variation of a
one-step fermion overlap:

```text
F(psi, U) = Re <psi, D_U psi>

J_a[x,h] = d/domega_a F(psi, exp(omega_a T_a) U[x,h]) | omega=0
```

This has two useful properties:

- it is tied to the actual BCC Dirac transport operator;
- if `F` is gauge invariant, the current transforms in the same
  target-site adjoint convention as the left-trivialized momentum.

Use finite differences first.  An analytic bilinear form can be added later
after the convention is tested.

### Backreaction Momentum Kick

Add a small source kick:

```text
P_new[x,h] = P[x,h] + dt * matter_coupling * J[x,h]
```

Then call the existing pure-gauge leapfrog/update stack.

For v1, the sourced step order should be:

```text
current = current(psi, links)
sourced_momenta = momenta + dt * matter_coupling * current
(links2, momenta2) = pure_gauge_leapfrog(links, sourced_momenta)
psi2 = dirac_step(psi, links2)
```

This is explicit and easy to test.  It is not a symplectic full
fermion/gauge integrator.

## Proposed API

```python
def jax_patisalam_momentum_algebra(
    momenta,
    *,
    sector="su4",
) -> jnp.ndarray:
    """Return anti-Hermitian momentum matrices with shape (..., 8, 32, 32)."""


def jax_patisalam_electric_divergence(
    links,
    momenta,
    *,
    sector="su4",
) -> jnp.ndarray:
    """Return site algebra-coordinate divergence with shape (..., sector_dim)."""


def jax_patisalam_fermion_charge_density(
    state,
    *,
    sector="su4",
    coordinate_mode="gram_dual",
) -> jnp.ndarray:
    """Return site charge density with shape (..., sector_dim)."""


def jax_patisalam_gauss_residual(
    state,
    links,
    momenta,
    *,
    sector="su4",
    matter_coupling=1.0,
) -> jnp.ndarray:
    """Return divE - g rho in sector coordinates."""


def jax_patisalam_fermion_link_current(
    state,
    links,
    *,
    sector="su4",
    epsilon=1e-3,
) -> jnp.ndarray:
    """Return finite-difference current coordinates per link."""


def jax_patisalam_apply_fermion_backreaction(
    momenta,
    current,
    *,
    step_size,
    matter_coupling=1.0,
) -> jnp.ndarray:
    """Return momenta + dt * g * current."""


def jax_patisalam_fermion_gauge_step_with_backreaction(
    state,
    links,
    momenta,
    *,
    sector="su4",
    step_size,
    matter_coupling=1.0,
    beta=1.0,
    shapes=None,
    force_method="finite_difference",
    force_epsilon=1e-3,
    current_epsilon=1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one explicit source-kick plus pure-gauge leapfrog plus fermion step."""
```

## Tests

Create:

```text
src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_gauss_backreaction.py
```

Keep lattices tiny, usually `(1, 1, 1)` or `(2, 1, 1)`, and start with
`u1_y`, `su2_l`, and `su3_c`.  Avoid full `pati_salam`/`sm` current tests
until the finite-difference current is profiled.

### Validation Tests

1. Momentum algebra shape and anti-Hermiticity.
2. Electric divergence has shape `(nx, ny, nz, sector_dim)`.
3. Zero momenta give zero electric divergence.
4. Pure-gauge transformed momenta give covariantly transformed divergence.

### Charge Tests

5. Zero fermion state gives zero charge density.
6. Charge density is real and finite for deterministic states.
7. Charge density transforms covariantly under site-local gauge transforms.
8. `u1_y` charge sign/scale is pinned by a deterministic one-component state.

### Gauss Tests

9. With zero fermions, Gauss residual equals electric divergence.
10. With zero momenta, Gauss residual equals `-matter_coupling * rho`.
11. Gauge-transformed `(state, links, momenta)` gives covariantly transformed
    Gauss residual.

### Current / Backreaction Tests

12. Zero fermion state gives zero link current.
13. Current has shape `(nx, ny, nz, 8, sector_dim)` and is finite.
14. Current transforms covariantly under site-local gauge transforms.
15. Backreaction kick leaves momenta unchanged when current is zero.
16. Backreaction kick changes momenta linearly in `step_size` and
    `matter_coupling`.
17. With `matter_coupling=0`, the backreaction step exactly matches the
    Session 36 no-backreaction step.

### Resource Guardrails

18. No test should build larger than `(2, 1, 1, 8, 32, 32)` links by default.
19. Full `spacetime_qca` test suite must stay under the current memory profile.
20. Mark any finite-difference current tests that become slow with an explicit
    `pytest.mark.slow` before adding broader sectors.

## Pass Criteria

Session 37 passes if:

- Gauss residual is defined in the same pull-link convention as the existing
  Dirac and link-transform code.
- Fermion charge density is basis-aware and gauge covariant.
- The explicit matter-current kick is gauge covariant on tiny lattices.
- The sourced step reduces to Session 36 when `matter_coupling=0`.
- The implementation does not introduce heavy full-sector JAX memory spikes.

## Failure Modes

F1: Gauss residual is not covariant.

Diagnosis: outgoing-link transport sign/order is wrong, likely in
`U[x+h,-h]^dagger P[x+h,-h] U[x+h,-h]`.

F2: Charge density is not covariant.

Diagnosis: raw moments are being used where Gram-dual coordinates are needed,
or the sign of `Q_a = i T_a` is inconsistent.

F3: Current is not covariant.

Diagnosis: the finite-difference perturbation is not left-trivialized, or the
fermion overlap used to define current is not gauge invariant.

F4: Backreaction step breaks link compactness.

Diagnosis: momenta are being converted to matrices outside the compact
generator basis, or the existing compact momentum update is bypassed.

F5: Tests trigger JAX memory pressure.

Diagnosis: current finite differences are too broad.  Restrict to `u1_y` /
small sectors first, or defer analytic current to a follow-up.

## Scope Boundaries

Session 37 does not claim a complete dynamical Standard Model QCA.

Still out of scope:

- Gauss-law projection/constraint solving on large lattices.
- Quantum operator-ordering resolution for the fermion current.
- Dynamical Higgs field.
- Yukawa insertion in the fermion time step.
- Physical coupling fitting.
- Long-time stability and continuum scaling.

## Documentation Updates

If implemented successfully:

- Add `SESSION_37_GAUSS_BACKREACTION.md`.
- Update `STATUS.md` test count and Session 37 result section.
- Update `ROADMAP.md` decision point after Session 37.
- Keep the open-item list honest: even after Session 37, full constraint
  projection and production-grade dynamical gauge simulation remain open.
