# Up-Sector Clebsch Taylor Study

**Question.** Can a scalar / nilpotent Taylor coefficient theorem derive

```text
C_u = (1/4, 1/sqrt(2), 1)
```

instead of the older

```text
C_u = (1/2, sqrt(2), 1)?
```

This is a study note, not a new sidecar. It isolates the coefficient question
from the depth/exponent question.

## Setup

Keep the up-sector exponent skeleton fixed:

```text
u : eta^6
c : eta^3
t : eta^0
```

The remaining question is the dimensionless coefficient vector multiplying
those powers.

Let `N` be the length-3 nilpotent repair flag:

```text
N^3 = 0,   N^2 != 0.
```

For a scalar response, use the nilpotent Taylor kernel:

```text
exp(xN) = I + xN + (x^2/2)N^2.
```

The grade readout is:

```text
C_u(x) = (x^2/2, x, 1).
```

This readout is not a coherent path amplitude. It is a positive scalar grade
profile normalized to the top coefficient.

## The Normalized Taylor Point

Set the one-step scalar repair amplitude to:

```text
x = 1/sqrt(2).
```

Then:

```text
x^2/2 = (1/2)/2 = 1/4.
```

Therefore:

```text
C_u = (1/4, 1/sqrt(2), 1).
```

This is the preferred Taylor story. It turns the charm coefficient into the
one-step scalar repair amplitude and the up coefficient into the second-order
Taylor term.

## Relation To The Rational Control

The common-scale mass study found that the data also tolerates the nearby clean
rational vector:

```text
(1/4, 3/4, 1).
```

This is close to the Taylor value:

```text
(1/sqrt(2)) / (3/4) = 0.9428...
```

So the rational vector remains useful as an empirical control, but the cleaner
scalar theorem is:

```text
(x^2/2, x, 1),  x = 1/sqrt(2).
```

The old Bernstein/binomial cumulative story is now demoted to an alternative
route, not the preferred derivation.

## Why The Old Vector Is Not A Scalar Taylor Profile

The old vector:

```text
(1/2, sqrt(2), 1)
```

cannot be a positive scalar grade profile normalized to the top:

```text
sqrt(2) > 1.
```

Any positive top-normalized scalar grade profile must have every entry in
`[0,1]`. The old `sqrt(2)` coefficient is therefore a coherent current
amplitude, not a scalar mass-response coefficient.

This is the conceptual mistake in the older up-sector coefficient proposal: it
mixed a mass-scalar coefficient question with a coherent path-amplitude
Clebsch.

## Candidate Theorem

The viable theorem is:

```text
Length-3 nilpotent scalar response
+ normalized one-step scalar repair x = 1/sqrt(2)
+ Taylor grade readout
=> C_u = (1/4, 1/sqrt(2), 1).
```

Equivalently:

```text
C_u(r) = (x^2/2, x, 1)_r,
         x = 1/sqrt(2),
         r = 1,2,3.
```

## Verdict

```text
NILPOTENT_TAYLOR_UP_CLEBSCH_PASS
OLD_SQRT2_UP_CLEBSCH_NOT_SCALAR_KILL
BERNSTEIN_RATIONAL_CONTROL_RECLASSIFIED
```

The next microscopic burden is to derive:

```text
x = 1/sqrt(2)
```

from scalar boundary repair normalization. If that cannot be derived, then
`(1/4,1/sqrt(2),1)` remains a clean conditional theory candidate rather than a
closed theorem.

## Reproduction

```bash
uv run python docs/scripts/verify_up_clebsch_taylor_study.py
```
