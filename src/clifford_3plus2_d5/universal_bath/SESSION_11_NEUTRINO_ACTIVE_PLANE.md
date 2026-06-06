# Session 11 - Neutrino Active-Plane Incidence

Session 10 used the selected family-port split

$$
P_{\rm act}=P_u+P_b,\qquad P_{\rm rad}=P_a.
$$

Session 11 asks what part of that split is forced by the selected BCC residual
port geometry.

## Incidence Derivation

Use the selected residual port

$$
e_1=(1,0,0).
$$

In the residual basis

$$
u=\frac{(1,1,1)}{\sqrt3},\qquad
a=\frac{(2,-1,-1)}{\sqrt6},\qquad
b=\frac{(0,1,-1)}{\sqrt2},
$$

the selected port decomposes as

$$
e_1=\sqrt{\frac23}\,a+\frac1{\sqrt3}\,u.
$$

There is no $b$ component.  Removing the collective trace gives

$$
e_1-\langle u,e_1\rangle u
=
\sqrt{\frac23}\,a.
$$

Therefore the detraced selected port fixes the radial line $a$.  The active
plane is then the orthogonal complement:

$$
P_{\rm act}=I-P_a=P_u+P_b.
$$

Equivalently, the active incidence channels are the collective tail channel
$u$ and the unique opposite-edge current orthogonal to both $u$ and $a$:

$$
b=\frac{(0,1,-1)}{\sqrt2}.
$$

Thus the Session 10 active plane is not an arbitrary product-family ansatz.  It
is the selected-port incidence plane after detracing the collective mode.

## Symmetry Control

Selected $S_2$ symmetry alone is too weak.  The raw selected-port projector

$$
P_{e_1}=|e_1\rangle\langle e_1|
$$

is invariant under the selected-port stabilizer that swaps ports $2$ and $3$,
but it mixes the radial line $a$ and the collective active line $u$.  Hence
selected $S_2$ does not itself select the active plane.  The extra ingredient is
the incidence/detracing rule:

$$
\text{radial line}=
\frac{e_1-\langle u,e_1\rangle u}{\|e_1-\langle u,e_1\rangle u\|}.
$$

The raw selected-port line control is also rejected:

$$
I-P_{e_1}\ne P_u+P_b.
$$

## Verdict

```text
NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS
```

Meaning:

```text
The selected residual port plus the collective tail fixes the radial a line by
detracing.  Its orthogonal complement is exactly the active u/b incidence
plane used by the Session 10 family graph.
```

## Honest Boundary

This is stronger than Session 10 but still not the final microscopic BB/QCA
derivation.  The remaining physical inputs are:

```text
bb_q_mismatch_penalizes_the_detraced_selected_port_radial_line
retarded_outgoing_boundary_condition_closes_only_the_active_incidence_plane
```

So the status is:

$$
\text{selected exit incidence}\Rightarrow P_{\rm act}=P_u+P_b
$$

provided the microscopic edge dynamics identifies the detraced selected-port
line $a$ with the penalized/outgoing radial mode.

## Certainty Ledger

| Statement | Status |
|---|---|
| $e_1=\sqrt{2/3}\,a+u/\sqrt3$ | `C:9` |
| detracing $e_1$ against $u$ gives the $a$ line | `C:9` |
| the unique current perpendicular to $u$ and $a$ is the $b$ line | `C:9` |
| $I-P_a=P_u+P_b$ matches the Session 10 active plane | `C:9` |
| selected $S_2$ symmetry alone does not force the split | `C:9` |
| BB q-mismatch dynamically penalizes $a$ | `C:4` |
| retarded boundary condition closes only the active incidence plane | `C:4` |
