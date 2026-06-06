# Session 15 - Quark Source Assembly Audit

The quark sector is the first place where the source vector lives in three
spaces at once:

$$
V_Q\in \mathbb C^3_{\rm color}\otimes \mathbb C^2_{\rm weak}
\otimes \operatorname{span}(u,a,b).
$$

Sessions 06 and 07 built the conditional up/down finite heads.  Sessions 08A
and 08B audited the two missing quark-source inputs.  Session 15 assembles
those certificates into one source-freeze ledger and asks a stricter question:

```text
Can V_u and V_d be frozen as microscopic BCC family-port boundary sources
without inserting flavor data?
```

The answer is no, and the obstruction is now localized exactly.

## What Is Available

The common residual family incidence basis is available from Session 11:

$$
e_1=\sqrt{\frac23}a+\frac1{\sqrt3}u,
\qquad
P_{\rm act}=P_u+P_b,
\qquad
P_{\rm rad}=P_a.
$$

The SM charge doors are available from Session 08A:

$$
Q_L\to u_R\quad\text{uses}\quad \widetilde H,
\qquad
Q_L\to d_R\quad\text{uses}\quad H.
$$

The conditional up head is available from Session 06:

$$
x=\frac1{\sqrt2},
\qquad
\left(\frac{x^2}{2},x,1\right)
=
\left(\frac14,\frac1{\sqrt2},1\right).
$$

The conditional down heads are available from Session 07 and 08B:

$$
(3,1,2)/3
\quad\text{and}\quad
(6,2,4)/6
\quad\longrightarrow\quad
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right),
$$

while the regular $S_3$ shell can also host

$$
(6,2,5)/6
\quad\longrightarrow\quad
\left(1,\frac1{\sqrt3},\sqrt{\frac56}\right).
$$

Thus the conditional head algebra is assembled.

## What Is Not Derived

The write-once source dictionary still has

```text
up_quark_boundary_source:
  port_vector = None
  residual_components = {}
  normal_depth = None

down_quark_boundary_source:
  port_vector = None
  residual_components = {}
  normal_depth = None
```

Four microscopic inputs remain unresolved:

```text
height_dynamics_selects_up_nilpotent_down_hermitian
microscopic_active_hidden_color_return_selects_regular_s3_shell
boundary_dynamics_selects_or_kills_down_rank_five_line
quark_normal_depth_placements_on_bcc_scar_are_frozen
```

The first is not an electroweak charge theorem: Session 08A shows that swapping
the repair modes is still hypercharge-allowed.

The second is not a color covariance theorem: Session 08B shows that both the
spectator shell and the active hidden color-return shell preserve visible color.

The third is not an $S_3$ theorem: the rank-five bottom line is available in
the regular shell but not unique.

The fourth is not supplied by the current dictionary: both quark anchors still
lack normal-depth placements.

## Verdict

```text
QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT
```

Meaning:

```text
The quark source dependency graph is assembled.  The conditional heads are
available, but V_u and V_d are not microscopic BCC family-port sources yet.
Promoting the conditional quark heads to completed boundary graphs would hide
four unresolved inputs.
```

Certainty: `C:9` for the exact dependency checks and conditional head algebra;
`C:4` for the current quark source theory, because the source freeze still
depends on the four named microscopic inputs.
