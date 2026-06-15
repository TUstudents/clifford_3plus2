# FN Recirculation + Center-CP Verdict

Status: first constructive yes-candidate.

Follow-up robustness sweep:

```text
FN_CENTER_CP_ROBUSTNESS.md
```

Question:

> Can quark flavor be modeled as Froggatt-Nielsen recirculation with order-one
> magnitudes and discrete color-center CP phases?

Short answer:

> Yes, at the constructive simulator level, for the default FN charges
> `Q=(3,2,0)`, `U=(5,2,0)`, `D=(1,0,0)`, `lambda=0.22501`, there is a
> center-phase texture with all coefficient magnitudes in `[0.295, 3.961]`
> that reproduces normalized quark masses, CKM moduli, and the Jarlskog
> invariant at percent-or-better accuracy.

This is not a derivation of the charges, of `lambda`, or of the center powers.
It is evidence that the effective QCA/FN simulator ansatz is viable:

```text
Y_u[i,j] = rho_u[i,j] * omega**n_u[i,j] * lambda**(Q_i + U_j)
Y_d[i,j] = rho_d[i,j] * omega**n_d[i,j] * lambda**(Q_i + D_j)
omega = exp(2 pi i / 3)
0.1 <= rho <= 10
```

The reproducible API entry points are:

```text
sm_center_cp_order_one_coefficients(...)
sm_center_cp_order_one_fit_residuals(...)
sm_fit_center_cp_order_one_magnitudes(...)
sm_center_cp_phenomenology_verdict(...)
sm_center_cp_robustness_scan(...)
```

The focused test coverage is in `tests/test_sm_cp.py`.

## Benchmark

Inputs used for the verdict run:

```text
lambda = sin(theta12) = 0.22501
sin(theta23) = 0.04183
sin(theta13) = 0.003732
delta_CKM = 1.147

up masses, normalized to top:
  (m_u/m_t, m_c/m_t, 1) = (1.2517e-5, 7.3767e-3, 1)

down masses, normalized to bottom:
  (m_d/m_b, m_s/m_b, 1) = (1.1164e-3, 2.2352e-2, 1)
```

The CKM values are PDG 2025 central values.  The light/heavy quark masses use
standard PDG-style central current-mass benchmarks at their conventional
scales; top is used only as an up-sector normalization.

## Important False Start

The exact-calibration report with the default basis gives a tiny nonzero
coefficient:

```text
coefficient_magnitude_min = 4.26e-5
coefficient_magnitude_max = 2.88
coefficients_are_order_one = false
```

That is not a physics verdict.  It is a basis/texture-gauge artifact of the
minimal calibration convention `Y_u=diag(m_u)` and `Y_d=CKM diag(m_d)`.  In that
gauge, several entries are forced tiny because unphysical right-frame freedom
has been fixed before the FN question is asked.

The real constructive question is whether a generated FN texture exists with:

1. fixed default charges,
2. coefficient magnitudes constrained to order-one size,
3. phases constrained to the color center `{1, omega, omega^2}`,
4. masses, CKM moduli, and Jarlskog fit simultaneously.

## Found Center Texture

Best center powers found in the first constrained search:

```text
n_u =
[[2, 1, 1],
 [1, 0, 0],
 [0, 2, 0]]

n_d =
[[1, 1, 1],
 [2, 0, 0],
 [1, 2, 0]]
```

Coefficient magnitudes:

```text
rho_u =
[[1.272654, 0.785760, 1.852192],
 [0.785760, 1.272654, 0.458693],
 [3.961016, 3.418301, 0.987706]]

rho_d =
[[0.351990, 0.316821, 0.601507],
 [0.574091, 0.565377, 0.295292],
 [1.404124, 0.883492, 0.344245]]
```

Magnitude quality:

```text
min rho  = 0.295292
max rho  = 3.961016
mean rho = 1.118426
```

So the coefficients are genuinely order-one by the simulator criterion
`0.1 <= rho <= 10`.

## Fit Quality

The found texture gives:

```text
up mass log RMS error   = 0.00195
down mass log RMS error = 0.00223
max CKM |V| error       = 0.00195
Jarlskog relative error = 0.000482

target J = 3.11698e-5
fit J    = 3.11848e-5
```

This is a constructive pass for the effective model:

```text
FN recirculation powers supply hierarchy.
Order-one magnitudes stay order-one.
CP can be carried by discrete color-center phases.
```

## What This Does Not Yet Prove

This result does not prove:

1. the default FN charges are derived from BCC geometry,
2. `lambda` is derived from a boundary matching rule,
3. the center-power matrices are uniquely selected by color holonomy,
4. the fit is stable under full RG-scale treatment,
5. the same mechanism automatically extends to leptons.

Those are the next theory questions.  But the simulator-level ansatz has now
passed the first serious existence test.

## Physics Takeaway

The viable picture is:

```text
quark flavor = FN recirculation path lengths
CP = discrete SU(3)_c center holonomy phases
O(1) coefficients = local return magnitudes
CKM = relative left-frame mismatch of the two generated Yukawa textures
```

The useful next step is not more infrastructure.  It is to turn this constrained
fit into a production API and then test robustness: nearby masses, lambda
choice, alternative charges, and whether the found center powers have a simple
closed-holonomy interpretation.
