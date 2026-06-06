# BCC $q=0$ Superselection Derivation

This note makes the mixed-normal leakage explicit and derives the $q=0$ scar as
a hard-gap Feshbach limit. It does not claim the bare BB automaton selects
$q=0$. The new input is a local boundary penalty, or equivalently an outgoing
bath, for relative-depth modes $q\ne0$.

The theorem target is:

$$
\boxed{
\text{relative-depth boundary gap}
\quad\Longrightarrow\quad
q=0\text{ visible compression}
\quad\Longrightarrow\quad
\epsilon=\sqrt2-1.
}
$$

## 1. Edge Coordinates and Relative Charge

Use a codimension-two BCC edge with inward normal depths $(r_1,r_2)$ and
tangential coordinate $s$. Define the relative-depth charge

$$
q=r_1-r_2.
$$

A body-diagonal hop

$$
\sigma=(\sigma_1,\sigma_2,\sigma_3),\qquad \sigma_i=\pm1,
$$

sends

$$
q\mapsto q+\sigma_1-\sigma_2.
$$

Thus the hop set splits exactly:

$$
\sigma_1=\sigma_2\quad\Rightarrow\quad \Delta q=0,
$$

$$
(\sigma_1,\sigma_2)=(+,-)\quad\Rightarrow\quad \Delta q=+2,
$$

$$
(\sigma_1,\sigma_2)=(-,+)\quad\Rightarrow\quad \Delta q=-2.
$$

This is the finite mixed-normal model. The visible scar is $q=0$; the adjacent
leakage sectors are $q=\pm2$. `C:9`.

## 2. Exact BB Blocks

At zero tangential momentum $k_s=0$, sum over $\sigma_3=\pm1$. In the pinned
BB convention $q_\pm=(1\pm i)/4$, the same-normal blocks are

$$
B_+=W_{+++}+W_{++-}
=\frac14
\begin{pmatrix}
1+i&1-i\\
1+i&1-i
\end{pmatrix},
$$

$$
B_-=W_{--+}+W_{---}
=\frac14
\begin{pmatrix}
1+i&-1+i\\
-1-i&1-i
\end{pmatrix}.
$$

They are the visible $q=0\to q=0$ radial blocks.

The mixed-normal blocks are

$$
M_{+2}=W_{+-+}+W_{+--}
=\frac14
\begin{pmatrix}
1-i&1+i\\
1-i&1+i
\end{pmatrix},
$$

$$
M_{-2}=W_{-++}+W_{-+-}
=\frac14
\begin{pmatrix}
1-i&-1-i\\
-1+i&1+i
\end{pmatrix}.
$$

They are the leakage maps

$$
M_{+2}:q=0\to q=+2,\qquad M_{-2}:q=0\to q=-2.
$$

The norm split is exact:

$$
B_+^\dagger B_+ + B_-^\dagger B_-=\frac12 I,
$$

$$
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}=\frac12 I.
$$

So the $q=0$ visible branch and the mixed-normal leakage branch each carry one
half of the BB probability at the edge. This is why the visible branch survival
norm is $1/\sqrt2$. `C:9`.

## 3. Boundary Penalty and Schur Limit

Introduce a local boundary penalty for relative-depth excitation:

$$
H_{\rm bd}=\Lambda q^2.
$$

On the adjacent leakage sectors $q=\pm2$ this gives

$$
H_{\rm leak}=4\Lambda I.
$$

Let the mixed-normal coupling out of $q=0$ be the vertical map

$$
M=\begin{pmatrix}M_{+2}\\ M_{-2}\end{pmatrix}.
$$

Schur-eliminating the $q=\pm2$ leakage sectors gives the mixed-normal
self-energy

$$
\Sigma_{\rm mix}(z,\Lambda)
=M^\dagger(z-H_{\rm leak})^{-1}M.
$$

In the degenerate local penalty model this is exact:

$$
\Sigma_{\rm mix}(z,\Lambda)
=\frac{1}{z-4\Lambda}
\left(
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}
\right)
=\frac{1}{2(z-4\Lambda)}I.
$$

Therefore

$$
\boxed{
\lim_{\Lambda\to\infty}\Sigma_{\rm mix}(z,\Lambda)=0.
}
$$

In the hard-gap limit, recurrent mixed-normal feedback into $q=0$ vanishes and
the effective visible branch is exactly the same-normal compression

$$
A_{\mathbb N}=S\otimes B_+ + S^\dagger\otimes B_-
$$

with the half-line head correction described in
[LOCAL_UNITARY_BCC_Q0_DILATION.md](LOCAL_UNITARY_BCC_Q0_DILATION.md).

`C:9` for the Schur limit under the stated penalty model. `C:6` for the
single-clock microscopic locking model of the penalty developed in
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md);
`C:3` for deriving it from a more primitive boundary-material dynamics.

## 4. Finite-$\Lambda$ Corrections

For finite $\Lambda$, the leakage feedback is not zero:

$$
\Sigma_{\rm mix}(z,\Lambda)=\frac{1}{2(z-4\Lambda)}I.
$$

At large $\Lambda$,

$$
\Sigma_{\rm mix}(z,\Lambda)
=-\frac{1}{8\Lambda}I-\frac{z}{32\Lambda^2}I+O(\Lambda^{-3}).
$$

In this degenerate model the correction is scalar in Weyl spinor space. It
therefore shifts the visible edge impedance but does not split the two scar
spinor channels. A nondegenerate leakage bath would replace the scalar by a
matrix self-energy; those deviations are not hidden tuning, but measurable
deformations of the silver branch.

`C:9` for the expansion in the degenerate penalty model; `C:5` as a deformation
language for finite boundary stiffness.

## 5. Survival Point and Silver

The hard-gap limit leaves the visible same-normal branch with survival norm

$$
c=\frac1{\sqrt2}.
$$

The stationary scar equation in the interior is

$$
\zeta\psi_r=B_+\psi_{r-1}+B_-\psi_{r+1}.
$$

Its radial transfer has

$$
\det T(\zeta)=1,\qquad {\rm tr}\,T(\zeta)=2\zeta+\frac1\zeta.
$$

The trace is minimized at

$$
\frac{d}{d\zeta}\left(2\zeta+\frac1\zeta\right)=0
\quad\Longrightarrow\quad
\zeta=\frac1{\sqrt2}=c.
$$

At that point,

$$
{\rm tr}\,T(c)=2\sqrt2,
\qquad
\lambda_\pm=\sqrt2\pm1,
\qquad
\epsilon=\sqrt2-1.
$$

The equality

$$
\boxed{
\text{branch survival norm}=\text{trace-minimum point}
}
$$

is exact. The physical statement that the mass readout probes this open-boundary
threshold is still an interpretation of the boundary scattering problem, not a
new algebraic theorem. `C:9` for the trace-minimum identity; `C:5`-`C:6` for the
mass-readout interpretation.

## 6. What Has Been Closed

The earlier obstruction was:

$$
\text{raw BCC edge count }2\quad\text{but scalar edge quotient leaks}.
$$

This note closes the mathematical leakage problem conditionally:

$$
\boxed{
\text{mixed-normal leakage explicit}
+\text{relative-depth hard gap}
\quad\Rightarrow\quad
\text{exact }q=0\text{ visible scar}.
}
$$

Together with the local unitary dilation, the projected scar is no longer a
probability deletion. It is the survival compression of an open boundary
colligation.

What remains open at the full microscopic level is the physical origin of the
relative-depth gap:

$$
\boxed{
\text{why does the bare microscopic BCC boundary generate the single-clock
locking field for }q\ne0?
}
$$

The effective order-parameter answer is developed in
[BCC_RELATIVE_DEPTH_ORDER_PARAMETER.md](BCC_RELATIVE_DEPTH_ORDER_PARAMETER.md):
edge reflection forces the lowest local scalar penalty to have the form
$\Lambda q^2$. That reduces the open microscopic question to the sign and origin
of the stiffness $\Lambda>0$. The positive sign and retarded readout are then
modeled by
[BCC_POSITIVE_STIFFNESS_BACKREACTION.md](BCC_POSITIVE_STIFFNESS_BACKREACTION.md)
and
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md).

## 7. Certainty Ledger

| Claim | Status |
|---|---|
| Hop split by $\Delta q$ | `C:9` |
| Exact $B_\pm$ and $M_{\pm2}$ matrices | `C:9` |
| Same-normal/mixed-normal norm split $1/2+1/2$ | `C:9` |
| Hard-gap Schur limit $\Sigma_{\rm mix}\to0$ | `C:9` under stated model |
| Finite-$\Lambda$ scalar correction | `C:9` under degenerate leakage penalty |
| Survival point $\zeta=1/\sqrt2$ equals transfer trace minimum | `C:9` |
| Relative-depth gap form $\Lambda q^2$ | `C:9` as local even scalar form |
| Positive stiffness $\Lambda>0$ / outgoing-bath origin | `C:6` in single-clock model; `C:3` deeper boundary-material dynamics |
| Mass readout uses open survival compression | `C:5`-`C:6` |
