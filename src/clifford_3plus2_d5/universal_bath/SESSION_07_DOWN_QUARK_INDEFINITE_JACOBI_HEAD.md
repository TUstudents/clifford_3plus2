# Session 07 - Down-Quark Indefinite Jacobi Head

Session 07 compares the real/symmetric down-sector heads.  The source
dictionary still records `down_quark_boundary_source` as unresolved, so this is
not a full microscopic down-source derivation.  It is a count-level Jacobi head
audit.

## Three-Port Head

The residual three-port shell can carry ranks

$$
(d,s,b)=(3,1,2)
$$

over denominator $3$.  This gives

$$
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right).
$$

This is the clean baseline vector.  A $5/6$ bottom line would require

$$
\frac56\cdot3=\frac52
$$

ports, so the three-port shell cannot host the candidate
$\sqrt{5/6}$ bottom coefficient.

## Regular $S_3$ Head

The six-element regular shell carries the baseline ranks

$$
(6,2,4),
$$

again giving

$$
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right).
$$

It can also host the data-improved candidate

$$
(6,2,5),
\qquad
\left(1,\frac1{\sqrt3},\sqrt{\frac56}\right).
$$

But this candidate is not forced by $S_3$ alone.  The rank-$2$ standard copy
requires defect polarization, and rank $5$ is not unique: it can be the
complement of either one-dimensional line.

## Indefinite Policy

The down sector remains in the real/symmetric indefinite-Jacobi class.  The
session includes a signed-norm control and treats a negative norm as a
look-ahead signal, not a value to clamp into positivity.  The present count
heads are positive rank counts, but the reduction class is still
indefinite/look-ahead because the physical down bath has not been constructed.

## Verdict

The certificate verdict is:

```text
DOWN_HEAD_FORK_LOCALIZED_PASS
```

Meaning:

- the clean baseline $(6,2,4)$ is available in both the 3-port and regular
  $S_3$ shells;
- the candidate $(6,2,5)$ is available only in the regular shell;
- $S_3$ does not select the candidate rank-$5$ line;
- the bottom $4\to5$ shift remains an open selection theorem.

The down gate is therefore implemented as a conditional availability theorem,
not as a completed mass derivation.
