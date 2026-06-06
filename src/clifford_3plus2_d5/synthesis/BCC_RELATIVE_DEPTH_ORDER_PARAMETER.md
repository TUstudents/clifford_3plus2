# BCC Relative-Depth Order Parameter

This note lowers the remaining $q=0$ input one level. The previous result
[BCC_Q0_SUPERSELECTION_DERIVATION.md](BCC_Q0_SUPERSELECTION_DERIVATION.md)
showed that a hard relative-depth gap gives the exact $q=0$ visible scar. The
question left open was:

$$
\boxed{
\text{why should the boundary penalize }q\ne0?
}
$$

Here $q=r_1-r_2$ is not a fitted flavor variable. It is the local asynchronous
normal-depth coordinate at a codimension-two BCC edge. The new claim is modest:
symmetry and locality force the *form* of the first boundary penalty to be
quadratic in $q$. The positive-stiffness refinement
[BCC_POSITIVE_STIFFNESS_BACKREACTION.md](BCC_POSITIVE_STIFFNESS_BACKREACTION.md)
then shows that the sign is fixed if the boundary admits the mismatch
constraint $K_{\rm rel}=q$. The microscopic single-clock model in
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md)
then derives $q$ as the unique local antisymmetric edge-clock error and models
the mixed-normal sectors as outgoing retarded channels.

## 1. The Local Edge Scalar

At the BCC edge the two inward normal depths are $(r_1,r_2)$. Define

$$
q=r_1-r_2.
$$

The residual edge reflection exchanges the two faces:

$$
(r_1,r_2)\mapsto(r_2,r_1),
\qquad
q\mapsto -q.
$$

Tangential translation along the edge leaves $q$ invariant. Therefore a local
scalar boundary energy depending only on asynchronous normal depth must satisfy

$$
V(q)=V(-q).
$$

Near the synchronous scar this gives the even expansion

$$
V(q)=V_0+\Lambda q^2+\kappa q^4+O(q^6).
$$

The additive constant $V_0$ is physically irrelevant. The first nontrivial local
reflection-even penalty is therefore

$$
\boxed{
V_{\rm rel}(q)=\Lambda q^2.
}
$$

`C:9` for the reflection-even classification; `C:5` for using it as an
effective boundary order parameter. `C:9` that $K_{\rm rel}^\dagger K_{\rm rel}$
is positive once the mismatch constraint is admitted.

## 2. Adjacent Leakage Gap

The same-normal scar has $q=0$. The first mixed-normal leakage sectors are

$$
q=\pm2.
$$

For the quadratic boundary potential,

$$
V_{\rm rel}(0)=0,\qquad V_{\rm rel}(\pm2)=4\Lambda.
$$

Thus the finite mixed-normal derivation's leakage Hamiltonian

$$
H_{\rm leak}=4\Lambda I
$$

is not an arbitrary insertion. It is the adjacent-sector value of the lowest
reflection-even local boundary penalty.

If $\Lambda>0$, the synchronous scar $q=0$ is the unique local minimum of the
relative-depth potential. If $\Lambda<0$, the boundary destabilizes the
synchronous scar and the silver route fails. If $\Lambda=0$, the full mixed
edge remains recurrent and the $q=0$ compression is not selected.

`C:9` for the evaluation on $q=0,\pm2$; `C:3` for a microscopic derivation that
the physical boundary has $\Lambda>0$.

## 3. Schur Feedback Revisited

With the mixed-normal BB blocks $M_{+2}$ and $M_{-2}$,

$$
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}=\frac12I.
$$

The reflection-even quadratic gap gives

$$
\Sigma_{\rm mix}(z,\Lambda)
=
\frac{1}{z-4\Lambda}
\left(
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}
\right)
=\frac{1}{2(z-4\Lambda)}I.
$$

The hard-stiffness limit is

$$
\lim_{\Lambda\to\infty}\Sigma_{\rm mix}(z,\Lambda)=0.
$$

Therefore

$$
\boxed{
\Lambda q^2\text{ with }\Lambda>0
\quad\Longrightarrow\quad
q=0\text{ hard-gap Feshbach compression}.
}
$$

The mathematical chain is now:

$$
\text{edge reflection}
\Rightarrow
V(q)\text{ even}
\Rightarrow
V(q)=\Lambda q^2+\cdots
\Rightarrow
H_{\rm leak}=4\Lambda I
\Rightarrow
\Sigma_{\rm mix}\to0.
$$

`C:9` for the chain after assuming $\Lambda>0$ and the hard-gap limit; `C:3`
for deriving $\Lambda>0$ microscopically.

## 4. Open-Boundary Readout

The local unitary dilation shows that the $q=0$ branch is not probability
deletion. The relative-depth gap explains why mixed-normal feedback can be
suppressed. The final interpretation is that flavor reads the **open
survival compression**:

$$
P_{q=0}UP_{q=0}=A_{\mathbb N},
$$

not the closed recurrent spectrum of the enlarged unitary. In retarded
language,

$$
\Sigma_{\rm mix}(z+i0,\Lambda)\to0
$$

at the hard-gap boundary threshold.

The survival norm of the visible branch is

$$
c=\frac1{\sqrt2}.
$$

The radial transfer trace

$$
{\rm tr}\,T(\zeta)=2\zeta+\frac1\zeta
$$

is minimized at

$$
\zeta=c=\frac1{\sqrt2}.
$$

At this point the orientation-preserving transfer has

$$
\lambda_\pm=\sqrt2\pm1,
\qquad
\epsilon=\sqrt2-1.
$$

So the silver point is not a chosen probe once the open-boundary survival
reading is accepted: it is the equality of branch survival norm and transfer
trace minimum. `C:9` for the equality; `C:5`-`C:6` for the physical mass-readout
interpretation.

## 5. What This Closes and What It Does Not

Closed in this note:

$$
\boxed{
\text{the relative-depth gap has the unique lowest local even form }
\Lambda q^2.
}
$$

Still open:

$$
\boxed{
\text{derive the single-clock locking field and outgoing/survival readout from
the full bare microscopic boundary dynamics.}
}
$$

This is progress because the missing object is no longer a vague scar
projection. It is a local mismatch constraint for one explicitly geometric
coordinate, with a minimal conditional microscopic realization in
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md).

## 6. Certainty Ledger

| Claim | Status |
|---|---|
| $q=r_1-r_2$ is odd under edge reflection | `C:9` |
| local scalar $V(q)$ must be reflection-even | `C:9` |
| lowest nontrivial term is $\Lambda q^2$ | `C:9` |
| $q=\pm2$ leakage gap is $4\Lambda$ | `C:9` |
| hard-gap Feshbach compression follows for $\Lambda>0$ | `C:9` under stated model |
| $\Lambda>0$ from $K_{\rm rel}^\dagger K_{\rm rel}$ | `C:9` if $K_{\rm rel}$ is admitted |
| uniqueness of $K_{\rm rel}=q$ as local mismatch coordinate | `C:9` |
| single-clock microscopic locking model | `C:6` |
| deeper boundary-material origin of locking/readout | `C:3` |
| open survival compression as mass readout | `C:5`-`C:6` |
