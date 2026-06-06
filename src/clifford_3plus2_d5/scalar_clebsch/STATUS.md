# scalar_clebsch - Status

**Status**: V3 implemented.

## Verdict

```text
SCALAR_CLEBSCH_THEORY_CONDITIONAL_PASS
```

The up-sector Taylor gate passes. The down-sector now has a clean baseline plus
a candidate repair, not a closed derivation of the bottom coefficient.

## V2a Up-Sector Nilpotent Taylor Gate

The preferred up-sector scalar Clebsches are:

```text
C_u = (1/4, 1/sqrt(2), 1).
```

Let `N` be the length-3 nilpotent flag:

```text
N^3 = 0,  N^2 != 0.
```

The nilpotent Taylor kernel is:

```text
exp(xN) = I + xN + (x^2/2)N^2.
```

The scalar grade readout is:

```text
(x^2/2, x, 1).
```

With:

```text
x = 1/sqrt(2),
```

the vector is exactly:

```text
(1/4, 1/sqrt(2), 1).
```

The nearby rational vector:

```text
(1/4, 3/4, 1)
```

is retained as a data-oriented control. The old up vector:

```text
(1/2, sqrt(2), 1)
```

is rejected as a scalar positive grade profile because the charm coefficient is
larger than the top-normalized value.

Verdict:

```text
NILPOTENT_TAYLOR_UP_CLEBSCH_PASS
```

## V2b Down-Sector Baseline And Candidate Gate

The primitive quark shell from `boundary_response` has:

```text
1_direct + 2_BCC + 3_color = 6.
```

The natural S3/projector baseline counts are:

```text
(d,s,b) = (6,2,4).
```

The overlap rule:

```text
C_i = sqrt(n_i / 6)
```

therefore gives:

```text
C_d,baseline = (1, 1/sqrt(3), sqrt(2/3)).
```

The data-improved odd-shell candidate counts are:

```text
(d,s,b) = (6,2,5).
```

The candidate coefficient vector is:

```text
C_d,candidate = (1, 1/sqrt(3), sqrt(5/6)).
```

This candidate gives:

```text
m_d/m_s = sqrt(3) eta^2
m_s/m_b = sqrt(2/5) eta^2.
```

The bottom `4 -> 5` shift is not marked as derived. It is the next theory
burden.

Verdict:

```text
DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_PASS
```

## V3 S3 Projector Availability Audit

The regular S3 shell decomposes as:

```text
Reg(S3) = 1_triv + 1_sign + 2_std + 2_std.
```

The central isotypic ranks are:

```text
trivial = 1
sign = 1
standard_isotypic = 4
regular = 6
```

The audit proves:

```text
6 is available as the regular shell.
2 is available as a defect-polarized standard copy.
5 is available as regular minus a one-dimensional irrep.
```

It also proves the important negative facts:

```text
rank-2 standard is not a central S3 projector;
rank-5 is not unique because regular-minus-trivial and regular-minus-sign
both have rank 5.
```

Verdict:

```text
S3_PROJECTOR_COUNT_AVAILABILITY_PASS
```

Interpretation: `(6,2,5)` is available inside regular S3 representation theory,
but S3 alone does not force it. A defect-selection theorem must still choose the
standard copy and the excluded one-dimensional line.

## Open Theory Burdens

1. Derive `x = 1/sqrt(2)` from scalar repair normalization.
2. Derive or kill the defect-selection rule that chooses the standard copy and
   excluded one-dimensional line giving the down candidate `(6,2,5)`.
3. Attach the Clebsches to the depth-scar exponent skeleton and common-scale
   quark mass audit.
4. Keep CKM coherent-current Clebsches separate from scalar mass Clebsches.
