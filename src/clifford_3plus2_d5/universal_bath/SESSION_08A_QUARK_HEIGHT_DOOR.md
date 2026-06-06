# Session 08A - Quark Height-Door Audit

Session 08A checks the precondition for freezing quark sources:

$$
V_u=\Pi_{\tilde H}V_Q,
\qquad
V_d=\Pi_HV_Q.
$$

The electroweak part is exact.  For Dirac Yukawa hypercharge conservation,

$$
-Y(Q_L)+Y(\tilde H)+Y(u_R)=0,
$$

with

$$
Y(Q_L)=\frac16,\qquad
Y(\tilde H)=-\frac12,\qquad
Y(u_R)=\frac23.
$$

For the down door,

$$
-Y(Q_L)+Y(H)+Y(d_R)=0,
$$

with

$$
Y(H)=\frac12,\qquad
Y(d_R)=-\frac13.
$$

Both neutral Higgs components satisfy

$$
Q_{\rm em}=Y+T_3=0.
$$

## Height Assignment

The declared height-door assignment is:

$$
\tilde H \longmapsto N,
\qquad
H \longmapsto \Delta_N,
$$

where

$$
N=|u\rangle\langle a|+|a\rangle\langle b|,
\qquad
N^3=0,\quad N^2\ne0,
$$

and

$$
\Delta_N
=NN^\dagger+N^\dagger N-(N+N^\dagger).
$$

The up operator is oriented and non-Hermitian.  The down operator is Hermitian
and has the path spectrum

$$
\operatorname{Spec}(\Delta_N)=\{0,1,3\}.
$$

## Negative Control

Swapping the repair modes still satisfies hypercharge.  Therefore electroweak
charge conservation alone does not derive the coherent-up / Hermitian-down
split.

The split is a height-dynamics premise:

```text
up   -> oriented height-lowering nilpotent
down -> Hermitian path closure
```

## Verdict

The certificate verdict is:

```text
QUARK_HEIGHT_DOOR_AUDIT_CONDITIONAL_PASS
```

Meaning:

- `C:9` for the SM hypercharge doors $H$ versus $\tilde H$;
- `C:9` for the nilpotent and Hermitian closure algebra;
- `C:4` for the physical statement that the Higgs doors select those repair
  modes.

The up/down BCC source vectors remain unfrozen.  This session isolates the
height-door premise that a later microscopic source freeze must derive or
replace.
