# Session 42 - Lorentz Recovery Beyond `alpha . k`

Status: implemented as an exact finite-spacing free-dispersion audit.

## Target

Session 20 verified the first-order continuum Hamiltonian:

```text
H(k) = alpha . k
```

Session 42 asks the next finite-spacing question: where does rotational
anisotropy first appear in the BCC free dispersion, and how does that compare
with the naive hypercube control?

This is a free-symbol audit.  It does not prove full interacting Lorentz
invariance, boost covariance, or regulator-level continuum QFT recovery.
It does prove the leading free-dispersion anisotropy order for the pinned BCC
symbol and the naive hypercube control.

## Added API

In `lorentz_recovery.py`:

- `LorentzRecoveryAudit`
- `bcc_weyl_trace_cosine`
- `bcc_dirac_trace_cosine`
- `continuum_cosine_series`
- `bcc_weyl_cosine_residual_series`
- `bcc_dirac_cosine_residual_series`
- `hypercube_energy_squared`
- `hypercube_energy_squared_residual_series`
- `first_nonzero_epsilon_order`
- `bcc_dirac_leading_anisotropy_coefficients`
- `hypercube_leading_anisotropy_coefficients`
- `lorentz_recovery_audit_payload`

The BCC diagnostic uses the normalized trace cosine:

```text
cos_like(k) = tr(U(k)) / dim(U)
```

and compares it against:

```text
cos(epsilon |k|)
```

The hypercube control uses:

```text
E_cube(k)^2 = sum_i sin(epsilon k_i)^2 / epsilon^2
```

and compares it against `|k|^2`.

## Audit Results

Right Weyl residual through `O(epsilon^4)`:

```text
epsilon^3 * (epsilon*(kx^2 ky^2 + kx^2 kz^2 + ky^2 kz^2)
             - 6 kx ky kz) / 6
```

Left Weyl has the opposite cubic term:

```text
epsilon^3 * (epsilon*(kx^2 ky^2 + kx^2 kz^2 + ky^2 kz^2)
             + 6 kx ky kz) / 6
```

The Dirac pair cancels the Weyl cubic anisotropy:

```text
BCC Dirac residual =
epsilon^4 * (kx^2 ky^2 + kx^2 kz^2 + ky^2 kz^2) / 6
```

Thus the BCC Dirac trace-cosine diagnostic matches the continuum target
through `O(epsilon^3)`, with the first anisotropy at `O(epsilon^4)`.

For a momentum magnitude `q`, the leading BCC Dirac coefficients are:

```text
axis          0
face diagonal q^4 / 24
body diagonal q^4 / 18
```

The naive hypercube control has lower-order anisotropy:

```text
E_cube(k)^2 - |k|^2 =
-epsilon^2 * (kx^4 + ky^4 + kz^4) / 3 + O(epsilon^4)
```

with coefficients:

```text
axis          -q^4 / 3
face diagonal -q^4 / 6
body diagonal -q^4 / 9
```

## Interpretation

Session 42 gives a sharper free-dispersion statement than the original
`alpha . k` check:

- BCC Weyl blocks individually have a cubic anisotropic term.
- The chiral Dirac pairing cancels that cubic term.
- The remaining BCC Dirac anisotropy starts at quartic order in lattice
  spacing.
- The naive hypercube control shows anisotropy at quadratic order and still
  has the existing corner-doubler obstruction.

This supports BCC as the better finite-spacing kinematic carrier for the
spacetime side of the QCA.  It is not a substitute for a full Lorentz recovery
proof in the interacting theory.

## Validation

Scoped command:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_lorentz_recovery.py -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -m "not slow" -q
```

Focused result:

```text
7 passed
133 passed, 125 deselected
```

Lint:

```bash
uv run ruff check src/clifford_3plus2_d5/spacetime_qca/lorentz_recovery.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_lorentz_recovery.py \
  src/clifford_3plus2_d5/spacetime_qca/__init__.py
```

Result: passed.

## Still Open

- Full all-momentum BCC Bloch-symbol unitarity proof.
- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Boost covariance and interacting-field Lorentz recovery.
- Scaling/renormalization diagnostics for coupled gauge/Higgs/Yukawa fields.
