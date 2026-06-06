# Session 08B - Quark Color-Lift Audit

Session 08B tests the color part of the quark source problem.  The rule is:

$$
\text{visible color must be scalar, hidden color may count return channels.}
$$

Equivalently, the visible mass readout must be proportional to

$$
I_{\rm color}.
$$

## Fixed Color Control

A fixed color vector gives a rank-one visible projector:

$$
P_R=\operatorname{diag}(1,0,0).
$$

It does not commute with the full $SU(3)_c$ generator set, so it is rejected
before any mass head is read.

## Spectator Color

The color-scalar spectator embedding uses

$$
\frac13 I_{\rm color}.
$$

It preserves visible color covariance, but the hidden return graph remains the
three-port residual shell:

$$
(3,1,2)/3
\longrightarrow
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right).
$$

This reproduces the clean baseline but cannot host the candidate
$\sqrt{5/6}$ bottom line.

## Active Hidden Color Return

The active lift also preserves visible color:

$$
\frac13 I_{\rm color}.
$$

But the hidden return graph is enlarged to the six-channel quark shell:

$$
1_{\rm direct}+2_{\rm BCC}+3_{\rm color}.
$$

The shell breakdown is:

```text
even_direct = 1
bcc_odd     = 2
color_odd   = 3
odd_total   = 5
total       = 6
```

Thus the regular $S_3$ head gives the baseline

$$
(6,2,4)/6
\longrightarrow
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right),
$$

and can host the rank-five candidate

$$
(6,2,5)/6
\longrightarrow
\left(1,\frac1{\sqrt3},\sqrt{\frac56}\right).
$$

The candidate is available, not forced.

## Color Covariance Does Not Select Active

Both spectator and active embeddings are visible color scalars.  Therefore
gauge covariance alone rejects the fixed color vector but does not choose
between spectator and active hidden return.  The active lift needs a microscopic
boundary-return selection rule.

## Verdict

The certificate verdict is:

```text
QUARK_COLOR_LIFT_AUDIT_CONDITIONAL_PASS
```

Meaning:

- `C:9` fixed visible color vectors are forbidden;
- `C:9` spectator and active embeddings can both preserve visible color;
- `C:7` active hidden color return reaches the regular six-channel shell;
- `C:4` physical selection of active over spectator remains open.

This session reduces the color ambiguity to a concrete microscopic question:
why does the quark boundary excursion sum over active hidden color-return
channels rather than leaving color as a spectator?
