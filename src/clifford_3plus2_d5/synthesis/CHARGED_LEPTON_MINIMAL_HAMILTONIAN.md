# Charged-Lepton Minimal Boundary Hamiltonian

This note builds the minimal colorless active boundary Hamiltonian whose Schur
residue realizes the charged-lepton BCC ansatz

$$
B_e=\rho\left(\sqrt2P_u+R_\theta P_\perp\right),
\qquad
\theta=-{2\pi\over3}-{2\over9}.
$$

It also records the main obstruction:

$$
\boxed{
\text{an ordinary one-sided Hermitian self-energy cannot produce }R_\theta.
}
$$

The charged-lepton mass kernel must be a chiral two-sided Yukawa Schur kernel,
not a positive self-energy:

$$
B_e(z)=V_R^\dagger(z-H_{Q,e})^{-1}V_L.
$$

Certainty: `C:9` for the algebraic realization; `C:8` for the obstruction to
one-sided Hermitian self-energy; `C:3` for the microscopic origin of
$\theta=-2\pi/3-2/9$.

## 1. Why A Two-Sided Kernel Is Necessary

For a Hermitian unresolved Hamiltonian $H_Q$, the pole residue of the usual
self-energy

$$
\Sigma(z)=V^\dagger(z-H_Q)^{-1}V
$$

at an isolated pole $\lambda$ is

$$
\operatorname*{Res}_{z=\lambda}\Sigma(z)=V^\dagger P_\lambda V.
$$

This matrix is Hermitian positive semidefinite. Therefore it cannot equal

$$
\rho R_\theta P_\perp
$$

for a nontrivial plane rotation. A real rotation is orthogonal, not positive,
unless $\theta=0$ or $\pi$.

This is not a problem. A charged-lepton Yukawa term connects left and right
chiral fields:

$$
\bar L_L H e_R.
$$

The correct effective object is therefore a two-sided chiral Schur kernel:

$$
B_e(z)=V_R^\dagger(z-H_{Q,e})^{-1}V_L,
$$

where $V_L$ couples the left active port into the boundary and $V_R$ couples
the boundary back to the right charged-lepton port. Such a kernel can carry a
real rotation because the left and right boundary frames need not coincide.

Certainty: `C:9`.

## 2. Boundary Spaces

Use the residual visible basis

$$
\mathcal H_P=\operatorname{span}\{u,a,b\}.
$$

The minimal rank needed for $B_e$ is three, because $B_e$ has rank three. But
to keep the coherent BCC trace enhancement explicit, use a four-dimensional
boundary pole space:

$$
\mathcal H_Q
=
\operatorname{span}\{t_+,t_-,p_a,p_b\}.
$$

The two states $t_\pm$ are the two coherent colorless BCC trace-return paths.
The states $p_a,p_b$ are the residual-plane boundary states.

Take the pole Hamiltonian

$$
H_{Q,e}=\lambda I_4.
$$

Gapped spectator states may be added, but they do not affect the pole residue.

Certainty: `C:8` for minimality with explicit trace paths.

## 3. Couplings

Define the left coupling

$$
V_L
=
{1\over\sqrt2}(|t_+\rangle+|t_-\rangle)\langle u|
+
|p_a\rangle\langle a|
+
|p_b\rangle\langle b|.
$$

This splits the trace input equally into the two BCC trace paths and sends the
plane basis directly into the plane boundary states.

Define the right coupling by

$$
V_R
=
\rho(|t_+\rangle+|t_-\rangle)\langle u|
+
\rho |p_a\rangle(\cos\theta\langle a|+\sin\theta\langle b|)
+
\rho |p_b\rangle(-\sin\theta\langle a|+\cos\theta\langle b|).
$$

In matrix form, with $Q$ ordered as $(t_+,t_-,p_a,p_b)$ and $P$ ordered as
$(u,a,b)$,

$$
V_L=
\begin{pmatrix}
1/\sqrt2&0&0\\
1/\sqrt2&0&0\\
0&1&0\\
0&0&1
\end{pmatrix},
$$

and

$$
V_R=
\rho
\begin{pmatrix}
1&0&0\\
1&0&0\\
0&\cos\theta&\sin\theta\\
0&-\sin\theta&\cos\theta
\end{pmatrix}.
$$

Then

$$
\operatorname*{Res}_{z=\lambda}B_e(z)
=
V_R^\dagger V_L
=
\rho
\begin{pmatrix}
\sqrt2&0&0\\
0&\cos\theta&-\sin\theta\\
0&\sin\theta&\cos\theta
\end{pmatrix}.
$$

That is exactly

$$
\boxed{
\operatorname*{Res}_{z=\lambda}B_e(z)
=
\rho(\sqrt2P_u+R_\theta P_\perp).
}
$$

Certainty: `C:9`.

## 4. Action On The Selected Higgs Port

The selected active Higgs/charged-lepton port is

$$
e_1={1\over\sqrt3}u+\sqrt{2\over3}a.
$$

Acting with the residue gives

$$
B_e e_1
=
\rho\left(
\sqrt2{1\over\sqrt3}u
+
\sqrt{2\over3}R_\theta a
\right).
$$

Since

$$
\sqrt2{1\over\sqrt3}=\sqrt{2\over3},
$$

the result is proportional to

$$
u+R_\theta a.
$$

After normalization,

$$
\widehat w_e
=
{1\over\sqrt2}
\left[
u+\cos\theta\,a+\sin\theta\,b
\right].
$$

Thus the minimal Hamiltonian realizes Koide equipartition automatically:

$$
|\widehat w_t|^2=|\widehat w_\perp|^2,
\qquad
K={2\over3}.
$$

Certainty: `C:9` inside the minimal model; `C:6` physically until the two
trace paths are derived from the microscopic active BCC/Higgs boundary.

## 5. The Charged-Lepton Angle

Take

$$
\theta=-{2\pi\over3}-{2\over9}.
$$

Then

$$
\widehat w_e
=
{1\over\sqrt2}
\left[
u+\cos\theta\,a+\sin\theta\,b
\right].
$$

With

$$
\mu_e=m_e+m_\mu+m_\tau,
\qquad
m_i=\mu_e|\widehat w_{e,i}|^2,
$$

the predicted charged-lepton masses are

| mass | prediction | observed |
|---|---:|---:|
| $m_e$ | $0.510965\ {\rm MeV}$ | $0.510999\ {\rm MeV}$ |
| $m_\mu$ | $105.652344\ {\rm MeV}$ | $105.658376\ {\rm MeV}$ |
| $m_\tau$ | $1776.866065\ {\rm MeV}$ | $1776.860000\ {\rm MeV}$ |

The maximum relative error is about $6.7\times10^{-5}$.

Certainty: `C:8` for the calculation from the stated angle and scale.

## 6. What Is Derived And What Is Not

Derived in the minimal Hamiltonian:

1. the need for a chiral two-sided Schur kernel;
2. the explicit four-state boundary pole realization;
3. the $\sqrt2$ trace enhancement from two coherent BCC trace paths;
4. Koide trace/traceless equipartition;
5. the exact charged-lepton residue form once $\theta$ is supplied.

Still not derived:

1. the microscopic origin of the two colorless trace paths from the full BCC
   Higgs boundary;
2. the active torsion value $2/9$ from the actual $H_{Q,e}$ dynamics;
3. the overall charged-lepton scale $\rho$ or $\mu_e$.

This is the honest status:

$$
\boxed{
\text{the minimal Hamiltonian realizes the target residue exactly;}
\quad
\text{the remaining theorem is the origin of }2/9.
}
$$

The next sidecar calculation should therefore not refit charged-lepton masses.
It should derive the active boundary connection

$$
R_{\rm active}=R_{-2/9}
$$

from the colorless Higgs/BCC return loop.
