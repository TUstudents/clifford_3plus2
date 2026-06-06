# Local Unitary Dilation of the BCC $q=0$ Edge Scar

This note constructs the exact local unitary completion of the projected
BCC $q=0$ scar branch. It is the next theorem behind the boundary model:

$$
\boxed{
\text{subunitary BB scar branch}
\quad\leadsto\quad
\text{local unitary open-boundary colligation}.
}
$$

The result is strong but specific. The dilation proves that the scar branch can
be embedded in a local unitary boundary update without deleting probability.
It does **not** by itself derive the $q=0$ superselection, and it does **not**
make $\epsilon=\sqrt2-1$ an eigenvalue of the closed enlarged unitary. The
silver root is the visible survival/transfer eigenvalue of the compressed scar
branch.

## 1. Scar Blocks

Work in the symmetric/antisymmetric spinor basis $(e,o)$ on the synchronous
normal scar. The pinned BB $q=0$, $k_s=0$ branch has nearest-radial blocks

$$
B_+=
\begin{pmatrix}
\frac12&\frac i2\\
0&0
\end{pmatrix},
\qquad
B_-=
\begin{pmatrix}
0&0\\
\frac i2&\frac12
\end{pmatrix}.
$$

The exact identities are

$$
B_+^\dagger B_-=0,\qquad B_-^\dagger B_+=0,
$$

and

$$
B_+^\dagger B_+ + B_-^\dagger B_-=\frac12 I,
\qquad
B_+B_+^\dagger+B_-B_-^\dagger=\frac12 I.
$$

Thus the scar branch is not merely contractive on average. It is a normal
contraction with exact survival norm

$$
\|A\|=\frac1{\sqrt2}.
$$

`C:9`.

## 2. Bulk Radial Cover

First ignore the head and work on the radial cover $r\in\mathbb Z$. Let $S$ be
the unit shift, $(S\psi)_r=\psi_{r-1}$. Define the visible scar update

$$
A=S\otimes B_+ + S^\dagger\otimes B_- .
$$

Using the identities above and $S^\dagger S=SS^\dagger=I$,

$$
A^\dagger A=AA^\dagger=\frac12 I.
$$

So the bulk scar branch is exactly

$$
A=\frac1{\sqrt2}W
$$

for a unitary $W$ on the scar spinor line. The missing probability is exactly
one half at every site and every momentum.

The local Julia dilation is

$$
U_{\rm bulk}=
\begin{pmatrix}
A&cI\\
cI&-A^\dagger
\end{pmatrix},
\qquad c=\frac1{\sqrt2},
$$

acting on

$$
\mathcal H_{\rm scar}^{\rm vis}\oplus\mathcal H_{\rm scar}^{\rm leak}.
$$

Direct block multiplication gives

$$
U_{\rm bulk}^\dagger U_{\rm bulk}
=U_{\rm bulk}U_{\rm bulk}^\dagger=I.
$$

The real-space update is nearest-neighbor plus on-site leakage:

$$
v'_r=B_+v_{r-1}+B_-v_{r+1}+c\,\ell_r,
$$

$$
\ell'_r=c\,v_r-B_+^\dagger\ell_{r+1}-B_-^\dagger\ell_{r-1}.
$$

This is local on the radial cover. `C:9`.

## 3. Half-Line Head Correction

The physical edge is a half-line $r\in\mathbb N$, not the full cover. Let $S$ be
the unilateral shift with $(S\psi)_0=0$. The same formula

$$
A_{\mathbb N}=S\otimes B_+ + S^\dagger\otimes B_-
$$

is still local, but the head misses one incoming channel. Therefore

$$
A_{\mathbb N}^\dagger A_{\mathbb N}
=\frac12 I-P_0\otimes B_-^\dagger B_-,
$$

$$
A_{\mathbb N}A_{\mathbb N}^\dagger
=\frac12 I-P_0\otimes B_+B_+^\dagger,
$$

where $P_0=|0\rangle\langle0|$ is the head projector.

Define the rank-one spinor projectors

$$
P_{\rm in}=2B_-^\dagger B_-=
\begin{pmatrix}
\frac12&-\frac i2\\
\frac i2&\frac12
\end{pmatrix},
\qquad
P_{\rm out}=2B_+B_+^\dagger=
\begin{pmatrix}
1&0\\
0&0
\end{pmatrix}.
$$

Then the defect operators are local and explicit:

$$
D_A=(I-A_{\mathbb N}^\dagger A_{\mathbb N})^{1/2}
=cI+(1-c)P_0\otimes P_{\rm in},
$$

$$
D_{A^*}=(I-A_{\mathbb N}A_{\mathbb N}^\dagger)^{1/2}
=cI+(1-c)P_0\otimes P_{\rm out}.
$$

The half-line Julia operator is

$$
\boxed{
U_{\mathbb N}=
\begin{pmatrix}
A_{\mathbb N}&D_{A^*}\\
D_A&-A_{\mathbb N}^\dagger
\end{pmatrix}.
}
$$

By the Julia-operator theorem for contractions,

$$
U_{\mathbb N}^\dagger U_{\mathbb N}
=U_{\mathbb N}U_{\mathbb N}^\dagger=I.
$$

Here this is not abstract nonlocal functional calculus: $D_A$ and $D_{A^*}$ are
the scalar bulk leakage $cI$ plus one finite-rank head correction. The update is
therefore local at the BCC edge. `C:9` for the dilation once the projected
branch $A_{\mathbb N}$ is accepted.

At $r>0$ the update is the bulk rule of §2. At the head,

$$
v'_0=B_-v_1+D_{A^*}^{(0)}\ell_0,
$$

$$
\ell'_0=D_A^{(0)}v_0-B_+^\dagger\ell_1,
$$

with

$$
D_A^{(0)}=cI+(1-c)P_{\rm in},\qquad
D_{A^*}^{(0)}=cI+(1-c)P_{\rm out}.
$$

This is the exact local head reservoir required by unitarity.

## 4. The Silver Transfer Survives as the Compression

The visible-visible block of both dilations is exactly the scar branch:

$$
P_{\rm vis}U_{\rm bulk}P_{\rm vis}=A,
\qquad
P_{\rm vis}U_{\mathbb N}P_{\rm vis}=A_{\mathbb N}.
$$

Therefore the stationary visible survival equation remains

$$
\zeta\psi_r=B_+\psi_{r-1}+B_-\psi_{r+1}
$$

in the interior. This is the same equation whose transfer is

$$
T(\zeta)=
\begin{pmatrix}
\frac1{2\zeta}&\frac{i}{2\zeta}\\
-\frac{i}{2\zeta}&2\zeta+\frac1{2\zeta}
\end{pmatrix},
\qquad
\det T=1,\qquad {\rm tr}\,T=2\zeta+\frac1\zeta.
$$

At the survival point $c=1/\sqrt2$,

$$
{\rm tr}\,T(c)=2\sqrt2,
\qquad
\lambda_\pm=\sqrt2\pm1,
\qquad
\epsilon=\sqrt2-1.
$$

So the exact dilation preserves the silver theorem as a theorem about the
compressed visible branch. `C:9` algebraically; `C:5`-`C:6` physically until the
visible branch is shown to be the correct mass-return observable.

## 5. What the Dilation Does Not Prove

This is the main guardrail.

The enlarged operator $U_{\mathbb N}$ is unitary, so its closed-system spectrum
lies on the unit circle. Therefore $\epsilon=\sqrt2-1$ is **not** an eigenvalue
of the full closed unitary. It is the decaying eigenvalue of the open visible
transfer/compression.

If one eliminates the leakage copy coherently, the visible equation acquires an
energy-dependent Schur term:

$$
\Sigma_{\rm leak}(\zeta)
=D_{A^*}\bigl(\zeta+A_{\mathbb N}^\dagger\bigr)^{-1}D_A.
$$

That self-energy generally changes the closed visible poles. Thus there are two
different physical readings:

1. **Survival/scattering reading:** the flavor radial factor is the conditional
   visible branch $A_{\mathbb N}$; leakage is an unresolved outgoing bath. This
   preserves the silver transfer exactly.
2. **Closed recurrent-bath reading:** leakage returns coherently through the
   Julia copy; the observed poles are those of the full Schur problem, and the
   bare silver transfer is no longer automatically the mass factor.

The flavor synthesis uses the first reading. The exact dilation makes that
reading unitary at the boundary, but it does not yet derive why the leakage bath
is physically outgoing/non-returning for the mass readout. The minimal
retarded-readout model is developed in
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md):
mixed-normal modes are attached to an outgoing half-line Weyl branch rather
than a closed recurrent bath. `C:8` for the distinction; `C:6` for that
retarded effective model; `C:3` for deriving the outgoing-bath principle from a
more primitive boundary-material dynamics.

## 6. Relation to the Actual BCC Mixed-Normal Channels

The leakage copy in $U_{\mathbb N}$ is a minimal abstract unitary completion of
the projected $q=0$ branch. The physical local completion is now specified in
[MICROSCOPIC_BB_QCA_EDGE_UPDATE.md](MICROSCOPIC_BB_QCA_EDGE_UPDATE.md): the
real mixed-normal BCC channels $q\to q\pm2$ are the orientation-resolved
clock-error ports with blocks $M_{+2}$ and $M_{-2}$.

The finite mixed-normal model in
[BCC_Q0_SUPERSELECTION_DERIVATION.md](BCC_Q0_SUPERSELECTION_DERIVATION.md)
identifies those adjacent channels explicitly as $M_{+2}$ and $M_{-2}$ and
shows that a hard relative-depth penalty makes their Schur feedback vanish.
The positive-stiffness and microscopic-locking notes then sharpen the physical
origin of that gap: $K_{\rm rel}=q$ is the unique local antisymmetric
two-face mismatch, and a single-edge-clock locking field gives the positive
penalty. Thus the abstract leakage copy used here is mathematically compatible
with the real BB mixed-normal channels in the hard-gap limit; the edge-clock
scattering update supplies the exact local BB realization. The remaining
problem is deriving the single-clock locking field and outgoing asymptotics from
a more primitive boundary-material model.

The exact physical theorem still owed is:

$$
\boxed{
\text{the local BCC edge defect maps mixed-normal hops into an outgoing bath
whose compression on }q=0\text{ is }A_{\mathbb N}.
}
$$

This has three parts, two of which are now local-update theorems:

1. identify the minimal leakage copy with local mixed-normal boundary modes;
2. prove that the retarded mass readout sees the survival compression rather
   than a recurrent closed-bath pole;
3. derive the physical outgoing asymptotic condition from deeper boundary
   dynamics.

[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md)
gives the minimal conditional model for item 3: the $q=0$ constraint is the
single-clock edge-locking condition, and the mass readout is the retarded
half-line Weyl branch of the outgoing bath.

The present note closes the abstract dilation obstruction: probability need not
be deleted. The later edge-clock update closes the local BB port realization.

## 7. Certainty Ledger

| Claim | Status |
|---|---|
| $B_\pm$ identities and survival norm $1/\sqrt2$ | `C:9` |
| Bulk radial-cover dilation $U_{\rm bulk}$ is local unitary | `C:9` |
| Half-line head correction is finite-rank and local | `C:9` |
| Visible compression has the same silver transfer | `C:9` |
| Physical use of the compression as mass-return branch | `C:5`-`C:6` |
| Identification of the leak copy with real mixed-normal BCC modes | `C:9` in [MICROSCOPIC_BB_QCA_EDGE_UPDATE.md](MICROSCOPIC_BB_QCA_EDGE_UPDATE.md) |
| Single-clock effective derivation of $q=0$ locking | `C:6` |
| Deeper boundary-material origin of outgoing asymptotics | `C:3` |

The sharp one-line result is:

$$
\boxed{
\text{The }q=0\text{ BB scar is not a probability leak; it is the visible
compression of an exact local unitary edge colligation.}
}
$$
