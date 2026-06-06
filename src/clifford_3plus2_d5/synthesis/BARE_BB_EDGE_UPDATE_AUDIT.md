# Bare BB Edge Update Audit

This note tests the strongest possible version of the boundary program:

$$
\boxed{
\text{does the bare Bialynicki-Birula BCC Weyl update alone close the }q=0
\text{ edge scar?}
}
$$

The answer is no. The failure is exact and useful. The bare BB update splits a
$q=0$ edge state into a same-normal visible branch and a mixed-normal branch
with equal norm. There is no nonzero Weyl spinor that kills both mixed-normal
leakage channels. Therefore a closed $q=0$ scar is not a theorem of the bare
BB coin alone. The microscopic completion must contain an additional boundary
principle: the single-clock locking field, an outgoing bath, or an equivalent
local boundary degree of freedom.
[BARE_BB_RELATIVE_CHANNEL_UPDATE.md](BARE_BB_RELATIVE_CHANNEL_UPDATE.md) writes
the corresponding unprojected walk: the mixed branch is the closed two-way
relative-depth channel of the bare update.

## 1. Bare Edge Blocks

Use the pinned BB convention

$$
q_\pm={1\pm i\over4}
$$

and the eight body-diagonal hops $W_{\sigma_1\sigma_2\sigma_3}$ with
$\sigma_i=\pm1$. At the codimension-two edge define

$$
q=r_1-r_2.
$$

Same-normal hops preserve $q$:

$$
\sigma_1=\sigma_2
\quad\Rightarrow\quad
\Delta q=0.
$$

Mixed-normal hops leave the diagonal scar:

$$
\sigma_1=-\sigma_2
\quad\Rightarrow\quad
\Delta q=\pm2.
$$

After summing over the tangential sign $\sigma_3$ at $k_s=0$, the same-normal
visible blocks are

$$
B_+=W_{+++}+W_{++-},
\qquad
B_-=W_{--+}+W_{---},
$$

and the mixed-normal blocks are

$$
M_{+2}=W_{+-+}+W_{+--},
\qquad
M_{-2}=W_{-++}+W_{-+-}.
$$

The exact BB identities are

$$
B_+^\dagger B_+ + B_-^\dagger B_-={1\over2}I,
$$

and

$$
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}={1\over2}I.
$$

Thus the full bare one-tick update is norm-balanced:

$$
\text{same-normal norm}+\text{mixed-normal norm}=I.
$$

The $q=0$ visible branch is a contraction because the other half of the
probability goes into real mixed-normal channels. `C:9`.

## 2. No Spinor Boundary Condition Kills Leakage

The mixed-normal matrices are rank one. Their kernels are

$$
\ker M_{+2}=\operatorname{span}\{(-i,1)^T\},
$$

and

$$
\ker M_{-2}=\operatorname{span}\{(i,1)^T\}.
$$

These two null lines are distinct:

$$
\ker M_{+2}\cap\ker M_{-2}=\{0\}.
$$

Therefore the condition

$$
M_{+2}\psi=0,\qquad M_{-2}\psi=0
$$

has only the trivial solution $\psi=0$. No nonzero Weyl spinor can remain in
the diagonal scar by the bare coin alone.

This is the load-bearing no-go. If one tries to obtain $q=0$ superselection by
choosing a local spinor boundary condition, the boundary condition kills the
state. If one keeps a nonzero state, the mixed-normal channels are present.
`C:9`.

## 3. Consequence For The Microscopic Program

The bare BB update does not delete leakage, and it does not secretly enforce
the mismatch constraint. The exact status is:

$$
\boxed{
\text{bare BB coin alone}
\;\not\Rightarrow\;
K_{\rm rel}=q\text{ locking}
\;\not\Rightarrow\;
q=0\text{ closed scar}.
}
$$

The previous effective notes are therefore not merely optional notation. They
are the missing physics:

1. [BARE_BB_RELATIVE_CHANNEL_UPDATE.md](BARE_BB_RELATIVE_CHANNEL_UPDATE.md)
   writes the full closed trace/relative BB edge walk before projection.
2. [BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md)
   supplies a local single-clock boundary principle whose violation is $q$.
3. [LOCAL_UNITARY_BCC_Q0_DILATION.md](LOCAL_UNITARY_BCC_Q0_DILATION.md)
   supplies a unitary completion of the visible contraction.
4. [BCC_Q0_SUPERSELECTION_DERIVATION.md](BCC_Q0_SUPERSELECTION_DERIVATION.md)
   shows that a hard relative-depth gap makes mixed-normal Schur feedback
   vanish.

The true microscopic completion cannot be "the pinned BB Weyl spinor on the
edge, with no other boundary structure." It must be:

$$
\boxed{
\text{bare BB bulk update}
+
\text{local boundary clock/constraint/outgoing channel}.
}
$$

This is not a retreat. It is a sharper target. The boundary degree must be
derived from the physical edge, but the exact matrix algebra proves it cannot
be removed.

## 4. Retarded Readout Status

The same audit also clarifies the retarded condition. The bare full-lattice BB
update is unitary and recurrent if every channel is kept as a closed system.
The silver root $\epsilon=\sqrt2-1$ is not an eigenvalue of that closed unitary.
It appears only in the open visible transfer.

Therefore the retarded boundary condition is not forced by the closed BB
spectrum. It is forced only after specifying the physical scattering problem:
mixed-normal modes are outgoing unresolved channels, not incoming coherent
return channels. In Green-function language, this selects the retarded Weyl
branch. In boundary-update language, it selects the visible compression as the
mass-return observable.

So the bare audit gives a falsifier:

$$
\boxed{
\text{if the physical boundary is closed and recurrent, the silver survival
readout is not protected.}
}
$$

## 5. Certainty Ledger

| Claim | Status |
|---|---|
| same-normal and mixed-normal split by $\Delta q=0,\pm2$ | `C:9` |
| same/mixed norm split is exactly $1/2+1/2$ | `C:9` |
| $\ker M_{+2}\cap\ker M_{-2}=\{0\}$ | `C:9` |
| bare BB spinor alone cannot close the $q=0$ scar | `C:9` |
| bare BB alone derives the mismatch constraint | `C:1` |
| retarded readout from closed BB spectrum alone | `C:1` |
| BB plus local boundary clock/outgoing channel | `C:6` effective model |
| full derivation of that boundary degree from microscopic edge physics | `C:3` |

The concise conclusion is:

$$
\boxed{
\text{The bare BB edge update is complete enough to falsify BB-only closure.
The missing microscopic object is a real boundary degree, not an algebraic
oversight.}
}
$$
