# Session 26 - JAX Numerical Backend

Status: implemented as a parity-checked numerical backend.

## Target

The symbolic `spacetime_qca` backend is exact but slow.  Session 26 adds a
JAX backend that can run the BCC Dirac QCA on finite lattices while preserving
the conventions established by the exact SymPy implementation.

The acceptance criterion is not a new physics claim.  It is backend parity:

```text
JAX step == exact SymPy step
```

on small lattices for ungauged, constant-link, and position-dependent-link
cases.

## Layout

State arrays use dense complex JAX tensors:

```text
ungauged Dirac:       (nx, ny, nz, 4)
Dirac x internal:     (nx, ny, nz, 4, internal_dim)
link field:           (nx, ny, nz, 8, internal_dim, internal_dim)
```

The link-field axis follows the exact backend's BCC displacement order and
pull convention:

```text
out[x] = sum_h (W_h x U[x <- x+h]) psi[x+h]
```

## Implemented

- `jax_state.py`
  - exact SymPy state to JAX array conversion;
  - flat and `(4, internal_dim)` tensor layouts;
  - zero-state helpers.
- `jax_links.py`
  - identity link fields;
  - constant link fields;
  - exact SymPy link-field to JAX conversion.
- `jax_step.py`
  - BCC displacement order;
  - JAX Dirac hop matrices;
  - ungauged BCC Dirac step;
  - position-dependent linked BCC Dirac step;
  - constant-link shortcut.

## Audit Results

- JAX runtime is available.
- Ungauged JAX Dirac delta step matches the exact SymPy `dirac_step`.
- JAX identity-link tensor step matches exact `dirac_step_with_constant_link`.
- JAX position-dependent link step matches exact `dirac_step_with_link_field`.
- Constant-link shortcut matches the explicit broadcasted link field.
- Linked JAX step is `jax.jit` compilable and runs on the default JAX device.

On the current workstation, `uv sync --extra cuda` gives:

```text
jax=0.10.0
jaxlib=0.10.0
default_backend=gpu
devices=[CudaDevice(id=0)]
```

The NVIDIA driver-version warning emitted by XLA is non-fatal in this
environment: GPU arrays allocate and matmul/step kernels execute on
`CudaDevice(id=0)`.

## Interpretation

The module now has an executable numerical QCA backend.  This unlocks repeated
finite-lattice stepping and later dynamical-field experiments without
abandoning the exact SymPy reference implementation.

This is still a kinematic/free-field backend:

- no dynamical gauge-field update;
- no plaquette action;
- no Higgs kinetic term;
- no Yukawa coupling fit;
- no long-time stability benchmark yet.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_backend.py -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

Current expected result after Session 26: 69 tests green.

## Still Open

- Performance benchmark for medium and large finite lattices.
- Batched trajectories and observables.
- JAX Wilson-loop evaluation from BCC plaquettes.
- Position-dependent Higgs/scalar fields.
- Dynamical gauge-field evolution.
