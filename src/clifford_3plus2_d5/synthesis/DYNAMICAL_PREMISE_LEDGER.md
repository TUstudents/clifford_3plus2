# Dynamical Premise Ledger

This note records the current boundary of the radial theory. It is deliberately
not a rescue attempt. The keystone was tested in two symmetry channels, flavor
$S_3$ and minimal electroweak charge bookkeeping, and neither derives the
needed up/down split.

The honest conclusion is:

$$
\boxed{
\text{up = holomorphic, down = Hermitian}
\text{ is a dynamical premise, not a symmetry theorem.}
}
$$

The radial theory is therefore a conditional theorem: one algebraic transfer
constant plus one dynamical bit generate the observed quark hierarchy structure
to useful accuracy. The bit remains load-bearing.

## 1. What Minimal Electroweak Symmetry Forces

The Standard Model Yukawa terms are

$$
\mathcal L_Y
=
y_u\,\bar Q_L\tilde H u_R
+
y_d\,\bar Q_L H d_R
+
\text{h.c.}
$$

with

$$
Y(Q_L)=\frac16,\qquad
Y(H)=\frac12,\qquad
Y(\tilde H)=-\frac12,
$$

$$
Y(u_R)=\frac23,\qquad
Y(d_R)=-\frac13.
$$

Hypercharge conservation gives

$$
-Y(Q_L)+Y(\tilde H)+Y(u_R)
=
-\frac16-\frac12+\frac23
=0,
$$

and

$$
-Y(Q_L)+Y(H)+Y(d_R)
=
-\frac16+\frac12-\frac13
=0.
$$

The crossed terms fail:

$$
-Y(Q_L)+Y(H)+Y(u_R)=1,
$$

$$
-Y(Q_L)+Y(\tilde H)+Y(d_R)=-1.
$$

Thus minimal electroweak symmetry really does force an up/down asymmetry:

$$
\boxed{
u\text{-type uses }\tilde H,\qquad d\text{-type uses }H.
}
$$

Certainty: `C:9`.

## 2. What Electroweak Symmetry Does Not Force

The asymmetry forced by the Standard Model is conjugation:

$$
H\quad\hbox{versus}\quad\tilde H.
$$

This is not the same as

$$
\hbox{holomorphic coherent repair}
\quad\hbox{versus}\quad
\hbox{Hermitian standing-wave repair}.
$$

At the weak-ladder level, $H$ and $\tilde H$ select opposite orientations in
the $SU(2)_L$ doublet. This gives holomorphic versus anti-holomorphic
orientation, not holomorphic versus Hermitian dynamics. At the neutral real VEV
level, both up and down masses are ordinary left-right Yukawa masses.

Both Yukawas flip chirality:

$$
Q_L\leftrightarrow u_R,
\qquad
Q_L\leftrightarrow d_R.
$$

Therefore chirality cannot distinguish up from down. It distinguishes mass
couplings from kinetic propagation, not up-type mass from down-type mass.

Certainty: `C:9` that both up and down Yukawas are chirality-flipping. `C:8`
that minimal electroweak charge bookkeeping gives conjugation, not the
holomorphic/Hermitian repair split.

## 3. Why Flavor $S_3$ Cannot Supply The Missing Bit

The residual flavor group has one nontrivial normal subgroup:

$$
A_3\triangleleft S_3.
$$

The associated sign grading separates even 3-cycles from odd transpositions:

$$
S_3=A_3\sqcup (S_3\setminus A_3).
$$

This is enough to distinguish two repair grammars:

$$
Z_3\hbox{ forward cycle}
\quad\hbox{versus}\quad
Z_2\hbox{ involutive transposition}.
$$

But it is not enough to encode weak isospin. The up/down distinction lives in

$$
SU(2)_L\times U(1)_Y,
$$

while the $S_3$ sign grading is already used to distinguish even/odd flavor
repair. There is no second independent $S_3$ grading left that can force

$$
u\text{-type}\to Z_3\text{ coherent entry},
\qquad
d\text{-type}\to Z_2\text{ Hermitian residue}.
$$

Certainty: `C:8` as a group-theoretic obstruction for deriving the up/down
assignment from $S_3$ alone.

## 4. Conditional Radial Theorem

The radial hierarchy theorem should now be stated conditionally.

The exact Pell transfer gives

$$
T_{\rm Pell}=
\begin{pmatrix}
2&1\\
1&0
\end{pmatrix},
\qquad
\eta=
\left|\frac{\lambda_-}{\lambda_+}\right|
=(\sqrt2-1)^2.
$$

The mass form is

$$
m_g=A_g\,\eta^{k_g}v.
$$

Given the dynamical premise

$$
u\text{-type}=\hbox{coherent/holomorphic repair},
\qquad
d\text{-type}=\hbox{Hermitian/diffusive repair},
$$

the coefficient types are:

$$
A_u=
\left(\frac14,\frac1{\sqrt2},1\right)
$$

from coherent Taylor entry, and the clean down baseline is

$$
A_d=
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right)
$$

from Hermitian shell residues. The closure lengths give slopes $3$ for
forward $Z_3$ repair and $2$ for involutive $Z_2$ repair.

Certainty: `C:9` for the Pell eigenvalue theorem. `C:6` for the up coefficient
mechanism under the dynamical premise. `C:4-C:7` for the down baseline,
because the $4/6$ bottom residue is exact but the rank-$2$ strange selection
and down floor remain open.

## 5. The One Remaining Bit

The reduced theory has one central dynamical bit:

$$
\boxed{
\text{which repair mode is coherent, and why is it assigned to up-type?}
}
$$

Equivalently:

$$
\boxed{
\text{why is up-type boundary recirculation holomorphic/coherent,}
\quad
\text{while down-type recirculation is Hermitian/diffusive?}
}
$$

This is not a fitted real parameter. It is a binary dynamical assignment. It is
also falsifiable: if the microscopic boundary dynamics makes both sectors
coherent, both sectors should carry Taylor-entry coefficients; if it makes both
Hermitian, both should carry shell-residue coefficients; if it forces the
opposite assignment, the mass texture changes.

Certainty: `C:5` that this is the correct compressed remaining premise. `C:3`
that the current QCA dynamics will derive the observed assignment.

The rank-one alignment audit
[HIGGS_DEFECT_ALIGNMENT.md](HIGGS_DEFECT_ALIGNMENT.md) checks whether the
current Higgs/Yukawa representation layer already supplies the needed
zero-mode line. It does not. The exact result is:

| Space | Dimension | Basis ranks | Image span | Source span |
|---|---:|---:|---:|---:|
| upper Higgs-like maps | $4$ | all $1$ | $2$ | $2$ |
| lower Higgs-like maps | $4$ | all $1$ | $2$ | $2$ |

The neutral static Yukawa control has rank $2$ and nullity $30$. Thus the
current Higgs slot supplies rank-one building blocks, but not a unique
rank-one boundary alignment.

## 6. What Should Not Be Claimed

The synthesis should not claim:

$$
\hbox{minimal EW symmetry}\Rightarrow
u\text{ holomorphic},\ d\text{ Hermitian}.
$$

It should also not claim:

$$
S_3\Rightarrow
u\text{ holomorphic},\ d\text{ Hermitian}.
$$

Both routes have been checked and fail for structural reasons. Minimal EW gives
$H$ versus $\tilde H$. Flavor $S_3$ gives even versus odd repair. Neither gives
coherent versus Hermitian boundary dynamics assigned by up/down.

Certainty: `C:8`.

## 7. Next Microscopic Question

The next real calculation is dynamical, not representational. It should ask:

$$
\boxed{
\tilde H\text{ source coherent?}\qquad
H\text{ source Hermitian?}
}
$$

or, more neutrally,

$$
\boxed{
\text{What repair kernel does each Higgs orientation induce on the defect?}
}
$$

The output should be a kernel classification:

$$
K_{\tilde H}\in\{\hbox{coherent},\hbox{Hermitian},\hbox{mixed}\},
\qquad
K_H\in\{\hbox{coherent},\hbox{Hermitian},\hbox{mixed}\}.
$$

If the result is

$$
K_{\tilde H}=\hbox{coherent},
\qquad
K_H=\hbox{Hermitian},
$$

then the radial theorem closes one major crack. If not, the dynamical bit
remains an irreducible premise.

This is the honest next target.
