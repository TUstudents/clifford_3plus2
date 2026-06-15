# FN Center-CP Robustness Sweep

Status: stable local family, not a unique derivation.

Question:

> Is the successful FN center-CP quark texture stable under realistic input
> variations, or only an isolated fit?

Short answer:

> The exact frozen magnitudes are not fully stable, but the same discrete
> center-power texture remains viable after bounded order-one magnitude refits.
> Across seven PDG-scale input variations, the fixed center powers pass `7/7`
> with coefficient magnitudes staying in `[0.289, 4.075]`.  Nearby charge
> assignments show a constrained basin: `3/5` tested charge choices pass.

This supports the constructive ansatz as a robust local phenomenology mode.  It
does not derive the center powers or charges.

## Inputs

The sweep used PDG 2025 central values and one-sigma-style edge variations:

```text
CKM:
  lambda = sin(theta12) = 0.22501 +/- 0.00068
  sin(theta13) = 0.003732 +0.000090/-0.000085
  sin(theta23) = 0.04183 +0.00079/-0.00069
  delta = 1.147 +/- 0.026

Quark masses:
  m_u = 2.20 +/- 0.07 MeV
  m_d = 4.69 +/- 0.05 MeV
  m_s = 92.74 +/- 0.54 MeV
  m_c(m_c) = 1.275 +/- 0.009 GeV
  m_b(m_b) = 4.196 +/- 0.012 GeV
```

The top mass only sets the up-sector normalization; the run used
`m_t = 172.57 GeV`.

Sources:

```text
https://pdg.lbl.gov/2025/reviews/rpp2025-rev-ckm-matrix.pdf
https://pdg.lbl.gov/2025/reviews/rpp2025-rev-quark-masses.pdf
```

## Fixed Texture Test

First test: keep both the verdict center powers and the verdict magnitudes
frozen.  This asks whether one literal coefficient table survives input
variation without any refit.

Result:

```text
pass fraction = 3/7
```

| case | status | failures | up log RMS | down log RMS | CKM max | J rel |
|---|---:|---|---:|---:|---:|---:|
| baseline | pass | - | 0.01110 | 0.00870 | 0.00195 | 0.00042 |
| mass +1 sigma | fail | up_mass | 0.02951 | 0.00837 | 0.00195 | 0.00042 |
| mass -1 sigma | pass | - | 0.00846 | 0.01342 | 0.00195 | 0.00042 |
| CKM +1 sigma | fail | jarlskog | 0.01110 | 0.00870 | 0.00128 | 0.05472 |
| CKM -1 sigma | fail | jarlskog | 0.01110 | 0.00870 | 0.00261 | 0.05660 |
| lambda = sqrt(3/2)-1 | pass | - | 0.01677 | 0.00781 | 0.00185 | 0.00606 |
| combined edge | fail | up_mass | 0.03424 | 0.00511 | 0.00112 | 0.00759 |

Interpretation:

```text
The exact coefficient table is not a universal frozen point.
Small input shifts need small magnitude refits.
```

## Fixed Center Powers With Magnitude Refits

Second test: keep the same discrete center powers fixed, but refit bounded
order-one magnitudes in `[0.1, 10]`.

Result:

```text
pass fraction = 7/7
coefficient magnitude envelope = [0.289, 4.075]
```

| case | status | up log RMS | down log RMS | CKM max | J rel | min rho | max rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | pass | 0.00327 | 0.00324 | 0.00153 | 0.00026 | 0.290 | 3.961 |
| mass +1 sigma | pass | 0.00252 | 0.00297 | 0.00156 | 0.00006 | 0.291 | 3.993 |
| mass -1 sigma | pass | 0.00417 | 0.00288 | 0.00144 | 0.00001 | 0.291 | 3.941 |
| CKM +1 sigma | pass | 0.00669 | 0.00497 | 0.00208 | 0.00021 | 0.290 | 4.050 |
| CKM -1 sigma | pass | 0.00307 | 0.00361 | 0.00136 | 0.00027 | 0.289 | 3.939 |
| lambda = sqrt(3/2)-1 | pass | 0.00778 | 0.00678 | 0.00189 | 0.00018 | 0.290 | 4.075 |
| combined edge | pass | 0.00328 | 0.00237 | 0.00161 | 0.00027 | 0.291 | 3.994 |

Interpretation:

```text
The center-power texture is stable as a local family.
The fit is not a single accidental coefficient table.
```

## Nearby Charge Assignments

Third test: keep the center powers fixed and allow bounded magnitude refits,
but change the FN charge assignment.

| charge case | status | failures | up log RMS | down log RMS | CKM max | J rel | max rho |
|---|---:|---|---:|---:|---:|---:|---:|
| default | pass | - | 0.00325 | 0.00324 | 0.00153 | 0.00033 | 3.961 |
| `U_1` one less | pass | - | 0.00285 | 0.00129 | 0.00205 | 0.00024 | 3.646 |
| `D_1` one more | pass | - | 0.00758 | 0.00808 | 0.00226 | 0.00050 | 4.140 |
| shifted `Q` with same diagonals | fail | up_mass, down_mass | 0.03384 | 0.02803 | 0.00512 | 0.00010 | 5.555 |
| weak left gap | fail | up_mass, down_mass, ckm | 0.02117 | 0.08531 | 0.01136 | 0.00191 | 5.561 |

Interpretation:

```text
The charge choice is not unique, but it is constrained.
The default lives in a small basin; not every nearby-looking charge assignment works.
```

## Center-Power Seed Competition

Fourth test: start several center-power patterns from unit magnitudes and run
the same bounded optimizer for 300 steps.

| seed | status | failures | objective | up log RMS | down log RMS | CKM max | J rel |
|---|---:|---|---:|---:|---:|---:|---:|
| verdict powers, unit start | fail | up_mass, down_mass | 0.08476 | 0.02428 | 0.28680 | 0.00361 | 0.00502 |
| default powers, unit start | fail | up_mass, down_mass, ckm | 3.67832 | 1.00768 | 1.32520 | 0.11575 | 0.01435 |
| all-zero powers, unit start | fail | up_mass, down_mass, ckm, jarlskog | 7.59869 | 0.97694 | 0.47848 | 0.17475 | 1.00000 |
| conjugate verdict powers | fail | up_mass, down_mass, ckm | 0.28322 | 0.42097 | 0.21636 | 0.03569 | 0.00540 |
| transpose verdict powers | fail | up_mass, down_mass, ckm | 0.32999 | 0.19274 | 0.46430 | 0.03262 | 0.00040 |

Interpretation:

```text
The center powers are not generic.
The successful pattern is a special basin and is not rediscovered reliably by
a short unit-start local optimizer.
```

This is a limitation, not a contradiction.  The earlier broad search found the
verdict basin; this short competition shows that a production scan should keep
multiple seeds and not pretend the powers are derived.

## Final Verdict

The robustness result is:

```text
Stable local family: YES
Single frozen coefficient table: NO
Generic center-power phase pattern: NO
Charge assignment uniquely derived: NO
```

Physics reading:

> FN recirculation with discrete color-center CP phases is a viable
> constructive quark-flavor model.  The current center powers define a stable
> local phenomenology family under realistic mass, CKM, and lambda variations,
> but the center powers and charges remain theory inputs.  The next nontrivial
> physics problem is to explain why this center-power basin is selected.
