# Session 16 - Quark Normal-Depth Placement Audit

Session 15 left one source-freeze blocker as

```text
quark_normal_depth_placements_on_bcc_scar_are_frozen
```

Session 16 checks whether the existing `depth_scar` theorem already closes
that blocker.

It does not.

## What Depth Scar Gives

The depth-scar sidecar supplies the conditional residual filtration

$$
h(u,a,b)=(0,1,2)
$$

and the length-3 nilpotent repair flag

$$
N=|u\rangle\langle a|+|a\rangle\langle b|
=
\begin{pmatrix}
0&1&0\\
0&0&1\\
0&0&0
\end{pmatrix}.
$$

This induces the path-scar depth operator

$$
D_{\rm scar}=2\Delta(P_3)
=
\begin{pmatrix}
2&-2&0\\
-2&4&-2\\
0&-2&2
\end{pmatrix}
$$

with normal-mode spectrum

$$
\operatorname{Spec}(D_{\rm scar})=\{0,2,6\}.
$$

## What It Does Not Give

The universal-bath source dictionary needs a source placement:

```text
V_u.normal_depth
V_d.normal_depth
```

Those are not supplied by the graph spectrum.

The reason is structural.  The path-scar depth operator is not the hand-written
port diagonal

$$
\operatorname{diag}(0,2,6).
$$

It is a port-basis graph Laplacian whose eigenmodes carry depths.  Also, the
BCC-doubled port heights are

$$
2h(u,a,b)=(0,2,4),
$$

not $\{0,2,6\}$.  Therefore neither the port filtration nor the normal-mode
spectrum can be copied into the source dictionary as `normal_depth` for
`V_u,V_d`.

The current dictionary still has

```text
up_quark_boundary_source.normal_depth = None
down_quark_boundary_source.normal_depth = None
```

## Verdict

```text
QUARK_NORMAL_DEPTH_PLACEMENT_NOT_DERIVED_AUDIT
```

Meaning:

```text
The depth-scar algebra is exact, but it does not freeze quark source
normal-depth placements.  The blocker is now sharpened: we need a
source-placement theorem, not another proof of the {0,2,6} graph spectrum.
```

Certainty: `C:9` for the exact depth-scar algebra and non-diagonal control;
`C:4` for the microscopic quark source theory until a source-placement theorem
exists.
