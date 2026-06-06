# Koide geometry

The `koide` sidecar studies a geometric coincidence: the charged-lepton Koide
axis
$
(1,1,1)/\sqrt3
$
is also the BCC body-diagonal $Z_3$ axis. It treats three generations as a
phenomenological input. It does not derive the family number.

The verdict is:

$$
\text{Koide consistent, not Koide predicted.}
$$

The BCC $Z_3$-equivariant Yukawa locus intersects the Koide cone, but it does
not contain the physical charged-lepton spectrum. C:8 for the locus theorem;
C:1 for the claim that this pure equivariant construction derives the PDG
charged-lepton masses.

## Koide as a cone

For charged-lepton masses, define

$$
v=(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau}),\qquad
n={1\over\sqrt3}(1,1,1).
$$

Koide's relation is

$$
K={m_e+m_\mu+m_\tau\over
(\sqrt{m_e}+\sqrt{m_\mu}+\sqrt{m_\tau})^2}={2\over3}.
$$

Equivalently,

$$
{(v\cdot n)^2\over |v|^2}={1\over2},
$$

so $v$ lies on the $45^\circ$ cone around the trace direction $n$. In
trace/traceless language,

$$
v=v_t+v_o,\qquad
v_t=P_t v,\qquad v_o=(I-P_t)v,
$$

and Koide is

$$
|v_t|^2=|v_o|^2.
$$

This equivalence is exact. C:9.

Using the sidecar's PDG 2024 charged-lepton inputs gives
$
K_{\rm PDG}=0.666661
$
with deviation about $6\times10^{-6}$ from $2/3$. This is an empirical fact,
not a derivation. C:7.

## BCC $Z_3$ structure

The BCC body-diagonal rotation

$$
R=
\begin{pmatrix}
0&1&0\\
0&0&1\\
1&0&0
\end{pmatrix}
$$

fixes $n=(1,1,1)/\sqrt3$. The induced split is exactly

$$
\mathbb R^3=\mathbb R n\oplus n^\perp,
$$

the one-dimensional trivial $Z_3$ irrep plus the two-dimensional nontrivial
irrep. This is the same trace/traceless split that Koide uses. C:9.

The sidecar identifies the three axes with the Pauli labels in the CP
$T_{2g}$ correction,

$$
H^{(1)}_\chi
=\sigma^x k_yk_z-\sigma^y k_xk_z+\sigma^z k_xk_y,
$$

and then treats those axes as the three generation labels. This is a
phenomenological identification, because the previous topology/triality routes
did not derive three families. C:4.

## Equivariant Yukawa locus

For a starting vector $v_\*$ in the three-axis space, the sidecar builds the
$Z_3$-orbit overlap matrix

$$
Y_{ij}=\langle R^i v_\*,R^j v_\*\rangle.
$$

Writing
$
v_\*=v_t+v_o
$
with $v_t$ along the trace axis and $v_o$ in the orthogonal plane, the matrix
is circulant:

$$
Y_{ii}=|v_t|^2+|v_o|^2,\qquad
Y_{ij}=|v_t|^2-\frac12|v_o|^2\quad(i\ne j).
$$

Its eigenvalues are

$$
\lambda_1=3|v_t|^2,\qquad
\lambda_2=\lambda_3={3\over2}|v_o|^2.
$$

Thus every exactly $Z_3$-equivariant Yukawa has a degenerate pair. C:9.

The Koide condition holds on this locus if and only if

$$
{|v_t|\over |v_o|}=3+2\sqrt2.
$$

At that point,

$$
{m_1\over m_2}=2(3+2\sqrt2)^2=2(17+12\sqrt2)\approx67.9,
\qquad m_2=m_3.
$$

So the intersection $L_{Z_3}\cap C_{\rm Koide}$ is a one-parameter
scale family, but the locus is not contained in the cone. C:9.

## Why PDG is not in the locus

The physical charged-lepton masses are all distinct:

$$
m_e\ne m_\mu\ne m_\tau.
$$

The $Z_3$-equivariant locus always has $\lambda_2=\lambda_3$. Therefore
the PDG charged-lepton triple is not in the pure equivariant locus, even
though it lies very close to the Koide cone. C:9.

This matters for the flavor tree. Exact $Z_3$ equivariance is too rigid. A
real charged-lepton theory needs $Z_3$-breaking input: for example, a Higgs VEV
alignment or boundary perturbation that keeps the cone relation while splitting
the degenerate pair. C:3.

## Synthesis role

Koide gives a clue about trace/traceless equipartition:

$$
\text{charged-lepton square-root mass vector}
\quad\leftrightarrow\quad
\mathbb R n\oplus n^\perp.
$$

This matches the BCC body-diagonal $Z_3$ geometry and the trace/nontrivial
irrep decomposition. But it is not yet part of the mass derivation:

- it assumes three generations;
- it uses charged-lepton data as empirical input;
- the exact equivariant locus gives a degenerate pair;
- the needed $Z_3$ breaking is not derived.

The correct paper-level statement is:

> The BCC body-diagonal $Z_3$ supplies the same trace axis that appears in
> Koide's $45^\circ$ cone. The exact $Z_3$-equivariant Yukawa locus intersects
> the cone at $|v_t|/|v_o|=3+2\sqrt2$, but it predicts a twofold mass
> degeneracy and therefore does not reproduce the physical charged-lepton
> spectrum. Koide is geometrically compatible with the carrier, not derived by
> it.

Certainty: C:8 overall; C:9 for the cone/locus algebra; C:3 for any future
dynamical cone-selection mechanism.

