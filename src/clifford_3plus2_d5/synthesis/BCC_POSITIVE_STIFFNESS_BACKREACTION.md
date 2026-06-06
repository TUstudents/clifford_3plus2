# BCC Positive Stiffness Backreaction

This note lowers the remaining sign question in the relative-depth model. The
previous note showed that locality and edge reflection force the lowest
relative-depth penalty to have the form

$$
V_{\rm rel}(q)=\Lambda q^2,\qquad q=r_1-r_2.
$$

The open problem was the sign:

$$
\boxed{
\text{why should }\Lambda>0?
}
$$

Here the answer is conditional but sharper: if the boundary contains a local
mismatch constraint whose violation is the asynchronous normal depth $q$, then
the backreaction energy is automatically positive because it is a square.
The follow-up note
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md)
gives the corresponding microscopic edge model: a two-face BCC edge has one
local edge clock, so $q$ is the unique local antisymmetric clock error, and the
mixed-normal sectors are read as outgoing retarded channels.

## 1. Local Mismatch Constraint

The synchronous-normal scar is the constraint

$$
r_1=r_2.
$$

Equivalently, define the local mismatch operator

$$
K_{\rm rel}=q=r_1-r_2.
$$

Under the residual edge reflection,

$$
r_1\leftrightarrow r_2,\qquad K_{\rm rel}\mapsto-K_{\rm rel}.
$$

The sign of the mismatch is orientation data; the physical penalty must not
prefer one face over the other. The local positive backreaction is therefore

$$
\boxed{
H_{\rm lock}=g\,K_{\rm rel}^\dagger K_{\rm rel}=gq^2,\qquad g>0.
}
$$

This is the same $\Lambda q^2$ term, but now the sign is not arbitrary once
$K_{\rm rel}$ is admitted as a boundary constraint. `C:9` for positivity of
$K^\dagger K$; `C:5` for the effective boundary-locking model; `C:3` for
deriving $K_{\rm rel}$ from microscopic boundary dynamics.

## 2. Adjacent Leakage Gap

The visible synchronous branch has

$$
q=0,\qquad H_{\rm lock}=0.
$$

The first mixed-normal leakage sectors have

$$
q=\pm2,
$$

so

$$
H_{\rm leak}=g(\pm2)^2=4g.
$$

Thus the leakage gap used in the Feshbach derivation is exactly the adjacent
violation energy of the positive mismatch constraint:

$$
\Lambda=g,\qquad H_{\rm leak}=4\Lambda I.
$$

If $g>0$, $q=0$ is the unique local minimum. If $g=0$, no relative-depth locking
exists. A negative stiffness cannot arise from $K^\dagger K$ with positive
coupling; it would require a different, destabilizing boundary term. `C:9`.

## 3. Schur Feedback

The mixed-normal BB blocks satisfy

$$
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}=\frac12I.
$$

With the positive leakage gap $4g$,

$$
\Sigma_{\rm mix}(z,g)
=
\frac{1}{z-4g}
\left(
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}
\right)
=\frac{1}{2(z-4g)}I.
$$

Hence

$$
\lim_{g\to\infty}\Sigma_{\rm mix}(z,g)=0.
$$

The chain is now:

$$
K_{\rm rel}=q
\quad\Rightarrow\quad
H_{\rm lock}=gK_{\rm rel}^\dagger K_{\rm rel}
\quad\Rightarrow\quad
H_{\rm leak}=4gI
\quad\Rightarrow\quad
q=0\text{ hard-gap compression}.
$$

`C:9` under the stated locking model.

## 4. Open-Boundary Readout

The positive stiffness suppresses recurrent mixed-normal feedback. The flavor
readout still uses the open survival compression:

$$
P_{q=0}UP_{q=0}=A_{\mathbb N},
$$

not the closed spectrum of the full unitary dilation. In retarded language, the
mixed-normal self-energy is treated as an outgoing boundary channel:

$$
\Sigma_{\rm mix}(z+i0,g)\to0.
$$

At the same time the visible branch has survival norm

$$
c=\frac1{\sqrt2},
$$

and the radial transfer trace

$$
{\rm tr}\,T(\zeta)=2\zeta+\frac1\zeta
$$

is minimized at $\zeta=c$. Therefore the silver point remains

$$
\lambda_\pm=\sqrt2\pm1,\qquad \epsilon=\sqrt2-1.
$$

`C:9` for the transfer algebra; `C:5`-`C:6` for the physical claim that mass
reads the open survival branch.

## 5. What This Closes and What Remains

Closed conditionally:

$$
\boxed{
K_{\rm rel}=q
\quad\Longrightarrow\quad
\Lambda=g>0.
}
$$

Still open:

$$
\boxed{
\text{derive the single-clock locking field and outgoing bath from deeper
boundary-material dynamics.}
}
$$

This is a real narrowing. The theory no longer needs an arbitrary positive
quadratic coefficient. It needs a local boundary constraint whose violation is
the asynchronous normal-depth coordinate. The minimal local realization is now
spelled out in
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md);
the remaining gap is deriving that realization from a more primitive physical
boundary material rather than accepting it as an edge-locking model.

## 6. Certainty Ledger

| Claim | Status |
|---|---|
| $K_{\rm rel}=q$ is odd under edge reflection | `C:9` |
| $K_{\rm rel}^\dagger K_{\rm rel}=q^2$ is positive | `C:9` |
| $q=\pm2$ leakage gap is $4g$ | `C:9` |
| hard-gap compression follows for $g\to\infty$ | `C:9` under stated model |
| uniqueness of microscopic mismatch coordinate $K_{\rm rel}=q$ | `C:9` |
| single-edge-clock locking field | `C:6` |
| deeper boundary-material derivation of the locking field | `C:3` |
| open survival compression as mass readout | `C:5`-`C:6` |
