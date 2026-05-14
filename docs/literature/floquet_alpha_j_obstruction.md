# Floquet Alpha J Obstruction

Status: theorem-standard decision point.

Floquet-α and α+ now separate two notions of "forced `J`".

## Spectral-Polarization `J`

Given the mandatory Floquet operator `U` with real minimal-polynomial factors:

```text
f_alpha(x) = x^2 + x + 1
f_eta(x) = x^2 + 1
```

Bezout gives:

```text
-x f_alpha + (x + 1) f_eta = 1
```

so the rank-`6` and rank-`4` spectral projectors are:

```text
P_alpha = (U + I)(U^2 + I)
P_eta = I - P_alpha
```

The oriented Floquet branches define:

```text
J_alpha = (2 / sqrt(3)) (U + I/2) P_alpha
J_eta = U P_eta
J = J_alpha + J_eta
```

This `J` is a polynomial expression in the rule-generated operator `U`.

It satisfies:

```text
J^2 = -I
J^T J = I
[J, P_alpha] = [J, P_eta] = 0
```

## Strict Compatible-Commutant `J`

The stricter test asks for uniqueness of compatible complex structures:

```text
J' in SO(10)
(J')^2 = -I
[J', U] = 0
```

and wants only:

```text
J' = ±J
```

The current exact solve does not certify that. Once the central
`P_alpha/P_eta` split exists, the compatible commutant retains block-sign
alternatives. The checker therefore reports:

```text
strict_compatible_j_forced = false
pass_strict_rule_to_bridge = false
```

## Decision

The project must choose which theorem standard it wants:

```text
spectral standard:
  J is accepted when it is a canonical oriented spectral polynomial in U.

strict commutant standard:
  J is accepted only when the full compatible-J variety collapses to ±J.
```

The current bridge remains non-load-bearing under the strict standard.
Under the spectral standard, Floquet-α solves the algebraic `J` and coarse
center problems but still needs a source-backed microscopic reason why the
oriented spectral branch is mandatory.
