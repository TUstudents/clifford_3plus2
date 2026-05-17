# Session 19b - Hypercharge Spectrum and Field Table

## Goal

Verify that the Standard Model generator extracted from the Pati-Salam
factorization has the one-generation chiral-16 hypercharge spectrum.

The real chiral-16 carrier has dimension 32. Since the gauge generators are
real-skew and commute with the chosen complex structure `J`, charges are read
from the real-symmetric observables

```text
Q_A = -J A.
```

Real eigenspace multiplicities are divided by 2 to get complex
multiplicities.

## Raw Normalization Diagnostic

The Session 19a raw generator

```text
Y_raw = T3_R_raw + (B-L)_raw / 2
```

does not match the SM hypercharge ratios by a single common scale.

Raw complex spectrum:

```text
  7/4 : 1
 -1/4 : 1
 -3/4 : 2
  3/4 : 3
 -5/4 : 3
  1/4 : 6
```

This is not a structural failure. It is a component-normalization issue:
the Clifford basis gives different raw normalizations for the two Pati-Salam
Cartans.

The physical normalization used by the audit is:

```text
T3_R       = (1/2) T3_R_raw
B-L        = (2/3) (B-L)_raw
Y          = T3_R + (B-L)/2
           = (1/2) T3_R_raw + (1/3) (B-L)_raw.
```

## Hypercharge Spectrum

After Pati-Salam component normalization, the complex hypercharge spectrum is
exactly:

```text
  1/6 : 6
 -2/3 : 3
  1/3 : 3
 -1/2 : 2
    1 : 1
    0 : 1
```

This matches one Standard Model generation in chiral-16:

```text
Q     : (3, 2,  1/6)  multiplicity 6
u^c   : (3*,1, -2/3)  multiplicity 3
d^c   : (3*,1,  1/3)  multiplicity 3
L     : (1, 2, -1/2)  multiplicity 2
e^c   : (1, 1,    1)  multiplicity 1
nu^c  : (1, 1,    0)  multiplicity 1
```

The `3` versus `3*` color distinction is not audited in this session; the
integer multiplicities identify the triplet-sized sectors. A future Casimir
or color-weight audit can label the representation orientation.

## Joint `(Y, T3_L)` Table

The joint charge decomposition also matches the weak-doublet structure:

```text
( 1/6,  1/2) : 3
( 1/6, -1/2) : 3
(-2/3,    0) : 3
( 1/3,    0) : 3
(-1/2,  1/2) : 1
(-1/2, -1/2) : 1
(   1,    0) : 1
(   0,    0) : 1
```

The complex multiplicities sum to 16.

## Audit Result

```text
real dimension:                  32
complex dimension:               16
Y_raw commutes with J:            True
Y_physical commutes with J:       True
observables symmetric:           True
charge observables commute:      True
raw common scale matches SM:      False
component normalization needed:   True
B-L normalization factor:         2/3
T3_R normalization factor:        1/2
normalized spectrum matches SM:   True
joint table matches SM:           True
```

## Interpretation

Session 19b closes the one-generation gauge-content table:

- Session 18 exposed `SU(4) x SU(2)_L x SU(2)_R` and a compatible `J`.
- Session 19a extracted `SU(3)_c x SU(2)_L x U(1)_Y`.
- Session 19b verifies the normalized hypercharge spectrum and weak-doublet
  table on the chiral-16 carrier.

The remaining open work is no longer the SM gauge-content extraction itself;
it is dynamics beyond the background 1D checkerboard:

- mass/Yukawa structure,
- dynamical gauge fields,
- 3+1D Lorentz recovery,
- three generations.

## Verification

Focused test:

```bash
uv run pytest src/clifford_3plus2_d5/lepton/tests/test_sm_hypercharge.py -q
```

Result:

```text
7 passed
```
