# Session 41 - Anomaly And Charge-Current Diagnostics

Status: implemented as an exact anomaly audit plus a hypercharge convention
correction for the JAX Pati-Salam/SM adapters.

## Target

Sessions 37-40 made charge density, current, Gauss residual, and coupled
Higgs/Yukawa updates executable.  Session 41 checks that the charge convention
feeding those dynamics is the physical one-generation SM convention:

```text
Y = T3_R_phys + (B-L)_phys / 2
  = T3_R_raw / 2 + (B-L)_raw / 3
```

The older raw Pati-Salam generator is algebraically valid, but it is not the
physical hypercharge normalization from Session 19b.

## Added API

In `anomaly.py`:

- `SMChiralField`
- `sm_chiral_field_table`
- `hypercharge_spectrum_from_fields`
- `sm_anomaly_sums`
- `matrix_charge_trace_diagnostics`
- `perturbative_anomalies_cancel`
- `sm_anomaly_audit_payload`

The JAX sector adapter now uses physical hypercharge by default:

```text
sector="u1_y"  -> physical Y
sector="sm"    -> SU(3)_c + SU(2)_L + physical Y
```

The old raw convention remains available explicitly:

```text
sector="u1_y_raw"
sector="sm_raw"
```

## Audit Results

Exact one-generation anomaly sums:

```text
Tr Y                 = 0
Tr Y^3               = 0
SU(3)^2-U(1)_Y       = 0
SU(2)^2-U(1)_Y       = 0
SU(3)^3              = 0
SU(2)_L doublets     = 4, even
```

The local field table matches the exact Session 19b hypercharge table:

```text
Q     : Y =  1/6, complex multiplicity 6
u^c   : Y = -2/3, complex multiplicity 3
d^c   : Y =  1/3, complex multiplicity 3
L     : Y = -1/2, complex multiplicity 2
e^c   : Y =  1,   complex multiplicity 1
nu^c  : Y =  0,   complex multiplicity 1
```

Matrix trace diagnostics over the real 32-dimensional charge observable agree
with the field-table sums after dividing by the real-form factor of 2.

## Interpretation

Session 41 closes a convention gap: `spacetime_qca` now treats `u1_y` and
`sm` as physical SM sectors, not raw Pati-Salam bookkeeping.  This matters for
Gauss-law sources, Higgs coupling, and future scaling diagnostics because the
link coordinate named `u1_y` now means the same physical hypercharge that
Session 19b used to derive the SM table.

This is still an anomaly diagnostic, not a lattice anomaly theorem.  It proves
that the one-generation representation used by the dynamics has the expected
continuum anomaly cancellations.  Regulator-level anomaly analysis for the
full interacting lattice theory remains future work.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_anomaly.py \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_anomaly_charge.py -q

uv run pytest \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_patisalam_subgroup_adapters.py \
  -m "not slow" -q

uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -m "not slow" -q

uv run pytest \
  src/clifford_3plus2_d5/spacetime_qca/tests/test_jax_patisalam_subgroup_adapters.py \
  -m slow -q
```

Focused result:

```text
7 passed
40 passed, 6 deselected
126 passed, 125 deselected
6 passed, 40 deselected
```

## Still Open

- Regulator-level anomaly analysis for the full lattice update.
- Long-time Gauss-law stability with physical sector couplings.
- Physical coupling constants for independent `SU(3)`, `SU(2)_L`, and
  `U(1)_Y` factors.
