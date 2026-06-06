# Session 13 - Boundary-Material Origin Audit

Session 12 closed the neutrino family-port graph inside the
single-clock/outgoing-boundary model:

$$
H_{\rm fam}
=
H_{\rm chain}\otimes(P_u+P_b)+I\otimes\Lambda P_a.
$$

Session 13 asks whether the two remaining physical premises are already forced
by bare BB block algebra:

```text
single_clock_locking_field_is_realized_by_boundary_material
mixed_normal_clock_error_ports_are_outgoing_asymptotic_leads
```

The answer is no.

## What Is Derived

At a two-face BCC edge a local linear mismatch has the form

$$
K=a r_1+b r_2.
$$

If it vanishes on synchronous edge events $r_1=r_2$, then

$$
a+b=0.
$$

Thus the solution space is one-dimensional:

$$
K\propto r_1-r_2=q.
$$

So the local mismatch coordinate is unique.  If the boundary admits this
constraint field, then the positive penalty is forced in form:

$$
K^\dagger K=q^2.
$$

Certainty: `C:9`.

## What Is Not Derived

The bare BB blocks contain no stiffness parameter $g$ or $\Lambda$.  They give
the exact local emission norm split:

$$
B_+^\dagger B_+ + B_-^\dagger B_-=\frac12I,
\qquad
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}=\frac12I,
$$

but they do not say whether the mixed-normal channel is an outgoing asymptotic
lead or a recurrent wedge state.

Both completions use the same local mixed-normal emission blocks.  They differ
only in the boundary asymptotic condition.

## Control

The retarded completion is block triangular:

$$
T_R=
\begin{pmatrix}
A_{\mathbb N}&0\\
E&S_\chi
\end{pmatrix},
$$

so visible powers remain $A_{\mathbb N}^t$.

The recurrent wedge completion allows hidden-to-visible return.  Its local
two-step correction is

$$
M_{-2}M_{+2}+M_{+2}M_{-2}
=
\begin{pmatrix}
-(1+i)/4&0\\
0&-(1-i)/4
\end{pmatrix}
\ne0.
$$

Therefore the retarded and recurrent completions are physically different, but
the bare local emission norm does not select between them.

## Verdict

```text
NEUTRINO_BOUNDARY_MATERIAL_ORIGIN_NOT_DERIVED_AUDIT
```

Meaning:

```text
Bare BB edge algebra derives the q coordinate and mixed-normal emission blocks,
but it does not derive the positive locking stiffness or the outgoing
clock-error asymptotic condition.
```

## Consequence

The neutrino family-port boundary graph is now microscopic inside a precise
conditional model:

```text
selected-port incidence
+ single-clock q^2 locking
+ outgoing clock-error leads
```

The deeper boundary-material origin remains a real premise, not a theorem.

## Certainty Ledger

| Statement | Status |
|---|---|
| local linear mismatch is uniquely $q=r_1-r_2$ | `C:9` |
| $K^\dagger K$ gives positive $q^2$ if $K$ is admitted | `C:9` |
| bare BB blocks contain no stiffness/gap parameter | `C:9` |
| retarded and recurrent completions share local emission blocks | `C:9` |
| recurrent wedge has nonzero two-step return | `C:9` |
| outgoing asymptotics forced by bare BB blocks | `C:2` |
| positive locking stiffness forced by bare BB blocks | `C:2` |
