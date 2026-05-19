# Session 38 - Hermitian Yukawa `Y(Phi)`

Status: implemented as a static-background Hermitian Yukawa layer.

## Target

Sessions 23 and 25 found a Higgs-like internal map module:

```text
upper: Delta Y = +1/2, Delta T3_L = +1/2, dim_R = 4
lower: Delta Y = +1/2, Delta T3_L = -1/2, dim_R = 4
```

Session 38 promotes a deterministic two-complex slice of that module into a
static Hermitian operator:

```text
Y_internal(Phi) = A(Phi) + A(Phi)^T
Y_H(Phi)        = beta x Y_internal(Phi)
Phi    = (phi_plus, phi_zero)
```

This is still not a dynamical Higgs field.  `Phi` is a fixed background value.

## Coordinate Convention

The exact carrier is the real `R^32` chiral-16 basis supplied by `lepton`, so
the SymPy API uses explicit real/imaginary coordinates:

```python
phi_plus = (re_plus, im_plus)
phi_zero = (re_zero, im_zero)
```

The selected map slice is deterministic:

```text
(U0, U1, L0, L1) = (upper[0], upper[1], lower[0], lower[1])
A(Phi) = Re(phi_plus) U0
       + Im(phi_plus) U1
       + Re(phi_zero) L0
       + Im(phi_zero) L1
Y_internal(Phi) = A(Phi) + A(Phi)^T
Y_H(Phi) = beta x Y_internal(Phi)
```

The full eight-real-dimensional map module remains available through
`static_higgs_doublet_internal_control(...)`.  Session 38's `Phi` API is the
physics-facing two-complex slice, not a claim that the full module has been
complex-basis reduced.

The transpose is the real-form opposite-charge counterpart already audited in
Sessions 23 and 25.  It is the right Hermitian static-control construction for
this carrier, but it is not yet a full dynamical complex Higgs-field
conjugation law.

## Added API

- `selected_higgs_phi_basis()`
- `higgs_phi_raising_map(phi_plus, phi_zero)`
- `hermitian_yukawa_internal_control(phi_plus, phi_zero)`
- `hermitian_yukawa_hamiltonian(phi_plus, phi_zero)`
- `neutral_yukawa_internal_control(vev)`
- `neutral_yukawa_hamiltonian(vev)`
- `audit_hermitian_yukawa_phi()`
- JAX mirrors in `jax_yukawa.py` using native complex scalars.

## Audit Results

- `Phi = 0` gives zero internal control and zero Hamiltonian.
- `Y_internal(Phi)` is real symmetric for arbitrary exact real/imaginary
  coordinates in the selected slice.
- `beta x Y_internal(Phi)` is Hermitian.
- The construction is linear in all four real `Phi` coordinates.
- Neutral `Phi = (0, v)`:
  - preserves color;
  - preserves `Q_em = Y + T3_L`;
  - breaks `Y`;
  - breaks `T3_L`.
- Charged `Phi = (v, 0)` breaks `Q_em`.
- JAX outputs match the exact SymPy construction for representative complex
  `Phi` values.

## Interpretation

Session 38 supplies the missing static Yukawa insertion:

```text
fermion Hamiltonian = H_BCC + beta x Y(Phi)
```

It closes the algebraic gap between "there is a Higgs-like representation
slot" and "the fermion update can carry a Hermitian Yukawa term."  The result
is deliberately static.  Site-local Higgs fields, Higgs kinetic energy,
Mexican-hat potential, Yukawa hierarchy fitting, and full coupled time
evolution remain future sessions.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_yukawa.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_yukawa.py -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/yukawa.py \
  src/clifford_3plus2_d5/spacetime_qca/jax_yukawa.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_yukawa.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_yukawa.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py
```

Focused result:

```text
25 passed
```

These focused Session 38 tests are marked `slow` because they trigger the
exact Higgs-map nullspace construction and JAX/SymPy parity conversion.

Full `spacetime_qca` result:

```text
213 passed
```

## Still Open

- Promote `Phi` from static background to site-local dynamical Higgs field.
- Add gauge transformation, covariant finite differences, and Higgs potential.
- Insert site-local Yukawa into the real-space fermion time step.
- Fit or classify realistic Yukawa hierarchies.
