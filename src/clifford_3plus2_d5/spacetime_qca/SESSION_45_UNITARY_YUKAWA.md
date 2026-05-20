# Session 45 - Exact Unitary Yukawa Insertion

Status: implemented as an optional norm-preserving site-local Yukawa substep.

## Target

Session 40 inserted the Higgs/Yukawa layer with an explicit first-order update:

```text
psi -> psi - i dt y (beta x Y(Phi[x])) psi
```

That path is useful as a regression control, but it is not exactly unitary and
can produce fermion norm drift in longer runs.  Session 45 adds an exact local
unitary option:

```text
psi -> exp(-i dt y beta x Y(Phi[x])) psi
```

The default coupled step remains `first_order` for backward compatibility.

## Added API

In `jax_coupled_higgs.py`:

- `YukawaUpdateMode = Literal["first_order", "unitary"]`
- `jax_apply_site_local_yukawa_unitary`
- `jax_apply_site_local_yukawa_update`

The coupled wrapper now accepts:

```text
yukawa_mode: "first_order" | "unitary" = "first_order"
```

In `jax_scaling.py`, `ScalingRunConfig` also accepts `yukawa_mode`, so Session
43 drift probes can compare the first-order and unitary updates.

## Implementation

The exact update uses the identity:

```text
exp(-i a beta x Y) = I_dirac x cos(aY) - i beta x sin(aY)
```

because `beta^2 = I` and the internal Yukawa matrix acts on the separate
internal factor.  This avoids a full `128 x 128` matrix exponential per site.
Instead, the implementation eigendecomposes the site-local Hermitian
`32 x 32` internal matrix `Y(Phi[x])` and evaluates `cos(aY)` and `sin(aY)`.

## Audit Results

- Zero step and zero coupling are no-ops.
- Zero Higgs field is a no-op for the exact unitary path.
- The exact unitary path preserves fermion norm for a deterministic neutral
  Higgs field.
- The first-order path has detectable norm drift for the same deterministic
  state, confirming the tests distinguish the modes.
- The exact unitary update agrees with the first-order update to first order
  in small step size.
- The coupled fermion/gauge/Higgs wrapper accepts `yukawa_mode="unitary"`.
- Session 43 scaling trials can compare `first_order` and `unitary` modes, and
  the unitary mode reduces Yukawa norm drift in the tiny deterministic setup.

## Validation

Scoped commands:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/jax_coupled_higgs.py \
  src/clifford_3plus2_d5/spacetime_qca/jax_scaling.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_unitary_yukawa.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_unitary_yukawa.py -m "not slow" -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_unitary_yukawa.py -q
```

Focused results:

```text
2 passed, 5 deselected
7 passed
```

## Still Open

- The exact unitary update is local at fixed `Phi`; the full coupled step can
  still drift through gauge, Higgs, and current approximations.
- Eigendecomposition of `32 x 32` matrices is too heavy for the fast suite, so
  nontrivial exact-unitary tests remain marked `slow`.
- Higgs current backreaction into gauge momenta is still open.
- Gauss-law projection and analytic matter current remain open.
