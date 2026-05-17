# Session 21 - Mass Layer

Status: implemented as a controlled Dirac mass audit.

## Target

Session 20 established the massless BCC Dirac Hamiltonian

```text
H_0(k) = alpha . k x I_internal.
```

Session 21 adds the standard Dirac mass layer

```text
H_m(k) = alpha . k x I_internal + beta x M_internal
```

with `beta = gamma^0` in chiral basis.

## Scalar Mass Control

For `M_internal = m I`, the code verifies:

```text
beta^2 = I
{alpha_i, beta} = 0
H_m(k)^2 = (k_x^2 + k_y^2 + k_z^2 + m^2) I
```

At `k = 0`, the scalar mass spectrum is `+m` and `-m` with the expected
Dirac/internal multiplicities.

This confirms that the spacetime QCA carrier supports the usual Dirac mass
mechanism.

## Gauge Compatibility

The scalar mass control commutes with:

- Pati-Salam generators from `lepton`:
  `SU(4) x SU(2)_L x SU(2)_R`;
- SM generators from `lepton`:
  `SU(3)_c x SU(2)_L x U(1)_Y`.

A non-scalar projector mass control breaks at least one SM generator, as
expected.  This verifies that the compatibility test is sensitive to internal
symmetry breaking rather than vacuously passing.

## Interpretation

The scalar mass is a universal Dirac control.  It is not an SM mass spectrum.

Real SM fermion masses require Higgs/Yukawa structure because the left- and
right-handed fields transform differently under `SU(2)_L x U(1)_Y`.  Session 21
therefore establishes the spacetime mass slot:

```text
beta x M_internal
```

but does not yet construct the Higgs/Yukawa operator that should occupy it.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

## Still Open

- Higgs/Yukawa representation audit.
- Position-dependent gauge links.
- Finite real-space BCC `step(state, links)`.
- Dynamical gauge fields.
- Full fundamental-BCC-Brillouin-zone no-doubling proof.
