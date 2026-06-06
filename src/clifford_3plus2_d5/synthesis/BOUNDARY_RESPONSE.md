# Boundary Response Sidecar

The `boundary_response` sidecar is the first real flavor sidecar in the theory
tree. Its central idea is not a group-theoretic mass formula. It is a boundary
Green-function principle:

$$
\text{visible flavor data}=
\text{Schur response of unresolved boundary modes}.
$$

The sidecar contains many local gates, but the flesh is much smaller:

1. the Schur complement as the common response language;
2. the silver transfer root $\epsilon=\sqrt2-1$;
3. the necessity of framing, because unbroken $S_3$ is too symmetric;
4. conditional PMNS and CKM texture assemblies;
5. a separate vacuum-selector chain closed modulo one quartic axiom.

## 1. Primitive Object: Schur Response

Let the Hilbert space be split into a resolved sector $P$ and an unresolved
boundary sector $Q$. With

$$
H=
\begin{pmatrix}
H_P & V^\dagger\\
V & H_Q
\end{pmatrix},
$$

the $P$-block resolvent is governed by

$$
G_P(z)=\bigl[z-H_P-\Sigma(z)\bigr]^{-1},
$$

where

$$
\Sigma(z)=V^\dagger(z-H_Q)^{-1}V.
$$

This formula is the right language for the sidecar. Boundary shells, residual
graphs, sterile chains, Clebsches, and phases enter through $H_Q$ and $V$.
They do not become masses until read through $\Sigma(z)$.

Certainty: `C:9` as a Schur/Feshbach identity. `C:7` as the organizing
language of the flavor sidecars.

## 2. Residual Basis And The Framing Necessity

The residual three-port basis is

$$
u=\frac{1}{\sqrt3}(1,1,1),\qquad
a=\frac{1}{\sqrt6}(2,-1,-1),\qquad
b=\frac{1}{\sqrt2}(0,1,-1).
$$

Let $P_u,P_a,P_b$ be the corresponding rank-one projectors. The proposed
neutrino-core response is

$$
K_\nu=\epsilon^2P_u+P_b.
$$

This operator is not invariant under the full residual $S_3$. The reason is
representation-theoretic. The permutation representation decomposes as

$$
3=1\oplus2,
$$

so every full $S_3$-invariant operator has the form

$$
A=\alpha P_u+\beta(P_a+P_b).
$$

It cannot distinguish $a$ from $b$. But $K_\nu$ assigns different weights to
$a$ and $b$:

$$
K_\nu=\epsilon^2P_u+0\cdot P_a+1\cdot P_b.
$$

Therefore an unframed $K_3$ residual tail cannot produce $K_\nu$. Framing is
not cosmetic; it is mathematically necessary. The surviving symmetry is the
selected-port $S_2$, not the full $S_3$.

Certainty: `C:9` for the $S_3$ centralizer statement. `C:8` for the kill of
the unbroken $K_3$-tail route. `C:7` for the conclusion that physical
neutrino response needs a framed boundary.

## 3. Silver Transfer Root

The exact dimensionless flavor transfer root is

$$
\epsilon=\sqrt2-1.
$$

It appears first from the residual recurrence

$$
x_n=2x_{n+1}+x_{n+2}.
$$

If $x_{n+1}=\epsilon x_n$, then

$$
\epsilon^2+2\epsilon-1=0,
$$

and the decaying positive root is

$$
\epsilon=\sqrt2-1.
$$

The same root appears as a half-line Weyl function. For the unit
nearest-neighbor half-line chain, the head Green function satisfies

$$
m(z)=\frac{1}{z-m(z)},
$$

with decaying branch

$$
m(z)=\frac{z-\sqrt{z^2-4}}{2}.
$$

At the transfer probe

$$
z=2\sqrt2,
$$

one obtains

$$
m(2\sqrt2)=\sqrt2-1=\epsilon.
$$

Thus the transfer root has two equivalent descriptions: a residual recurrence
root and a sterile-chain Weyl-function value.

The non-sidecar fixed-point note
[SILVER_FIXED_POINT_NOTE.md](SILVER_FIXED_POINT_NOTE.md) adds a research
reframing: derive $\epsilon$ as the $N=2$ metallic fixed point

$$
\epsilon=\frac{1}{2+\epsilon}.
$$

That reframing is useful, but not yet a theorem. A raw BCC flat-face count gives
$N_{\rm raw}=4$, so a silver origin must come from a codimension-two scar or a
weighted effective self-consistency equation with $N_{\rm eff}=2$.
[BCC_SCAR_TRANSFER.md](BCC_SCAR_TRANSFER.md) performs the first concrete edge
calculation and finds that the raw edge count is $2$, but the scalar edge
quotient leaks and the bare BB Weyl coin does not close the scar.

Certainty: `C:9`.

## 4. Product Sterile Chain And The Neutrino Core

The clean framed model is the product sterile bath

$$
H_Q=H_{\rm chain}\otimes I_{\rm family}.
$$

The unresolved chain supplies the radial transfer; the family factor keeps the
$u$ and $b$ labels orthogonal. This gives equal head returns and zero
cross-return by tensor structure.

For the semi-infinite chain,

$$
\Sigma(z)=m(z)\bigl[m(z)^2P_u+P_b\bigr].
$$

After dividing by the common head return $m(z)$, the normalized response is

$$
\widehat\Sigma(z)=m(z)^2P_u+P_b.
$$

At $z=2\sqrt2$ this becomes

$$
\widehat\Sigma(2\sqrt2)=\epsilon^2P_u+P_b=K_\nu.
$$

This is the strongest theorem in the boundary-response sidecar: the framed
product half-line gives the advertised neutrino core exactly, without endpoint
impedance tuning.

As an effective light-neutrino mass operator,

$$
M_\nu=\mu_\nu K_\nu
$$

has the residual-basis spectrum

$$
M_\nu^{(a,u,b)}
=
\mu_\nu
\begin{pmatrix}
0&0&0\\
0&\epsilon^2&0\\
0&0&1
\end{pmatrix}.
$$

Thus

$$
(m_1,m_2,m_3)=\mu_\nu(0,\epsilon^2,1),
$$

and

$$
{\Delta m^2_{21}\over\Delta m^2_{31}}=\epsilon^4.
$$

The dimensionless spectrum is derived. The absolute eV scale $\mu_\nu$ remains
the separate sterile/Higgs boundary scale unless one uses a measured
oscillation splitting to set it. See [MASS_SECTOR.md](MASS_SECTOR.md).

Certainty: `C:9` as an exact theorem for the product sterile model. `C:7` as a
physical neutrino-core mechanism, because the product bath/framing must still
be justified as the microscopic boundary completion.

## 5. Leptonic Texture Layer

The selected charged-lepton/Higgs port is

$$
e_1=(1,0,0).
$$

In the residual basis,

$$
e_1=\sqrt{\frac23}\,a+\frac{1}{\sqrt3}\,u+0\cdot b.
$$

If the leading charged-lepton leakage is two Weyl-transfer steps, then

$$
\sqrt{\frac23}\,\sin\theta_e=\epsilon^2,
$$

hence

$$
\sin\theta_e=\sqrt{\frac32}\,\epsilon^2.
$$

The leptonic phase word combines a Schur minus sign, an $A_3$ parent
spin-lift contribution, and an $A_2$ residual spin-lift contribution:

$$
1+\frac14+\frac13=\frac{19}{12}.
$$

The principal phase is

$$
-\frac{5\pi}{12},
$$

so

$$
W_e=\exp\!\left(-\frac{5\pi i}{12}\right).
$$

Together with the residual basis matrix, this gives a conditional PMNS
assembly

$$
U_{\rm PMNS}=R_e^\dagger U_{\rm TBM},
$$

with the exact reactor-angle relation

$$
\sin^2\theta_{13}=\frac34\,\epsilon^4.
$$

The lesson is not that PMNS is fully derived. The clean statements are:
the neutrino core is exact in the product model; the charged-lepton leakage
angle follows from the selected port plus two transfer steps; the phase word is
arithmetic/holonomy data; the PMNS texture is conditional on those boundary
ingredients.

Certainty: `C:8` for the selected-port decomposition and two-step leakage
formula. `C:6` for the phase word as a selected boundary holonomy. `C:6` for
the assembled PMNS texture.

## 6. Quark Texture Layer

The quark primitive shell is

$$
S_q=1_{\rm even}+5_{\rm odd}
    =1_{\rm direct}+(2_{\rm BCC}+3_{\rm color}).
$$

The five odd channels are represented by a $\mathrm{Cl}_5$ system. With

$$
\Gamma_q=\sum_{A=1}^5\gamma_A,
$$

one has

$$
\Gamma_q^2=5I.
$$

The flat primitive quark coin is

$$
B_q=\frac{I+i\Gamma_q}{\sqrt6},
$$

and its scalar positive-branch phase is

$$
\delta_q=\arctan\sqrt5.
$$

The quark transfer hierarchy assumes family depths

$$
d_1=0,\qquad d_2=2,\qquad d_3=6,
$$

so the raw transition amplitudes are

$$
A_{12}=\epsilon^2,\qquad
A_{23}=\epsilon^4,\qquad
A_{13}=\epsilon^6.
$$

The Clebsch prefactors then dress these amplitudes:

$$
C_F=\frac43,\qquad
C_{23}^{\rm BCC}=\sqrt2,\qquad
C_{13}^{\rm BCC}=\frac{1}{\sqrt2}.
$$

The conditional CKM sines are therefore

$$
s_{12}=\frac43\frac{\epsilon^2}{\sqrt{1+\epsilon^4}},
\qquad
s_{23}=\sqrt2\,\epsilon^4,
\qquad
s_{13}=\frac{\epsilon^6}{\sqrt2}.
$$

The CKM matrix is assembled in the usual three-rotation form

$$
V_{\rm CKM}=R_{23}R_{13}(\delta_q)R_{12}.
$$

This is a conditional quark texture, not a microscopic mass derivation. Its
load-bearing assumptions are the primitive shell $S_q$, the depth embedding
$\{0,2,6\}$, and the Clebsch dressing rules. Later sidecars must decide which
of these are genuine theorems and which remain inputs.

Certainty: `C:7` for the internal algebra of the primitive shell and flat coin.
`C:6` for the conditional CKM texture. `C:4` for interpreting it as a mass
theory before the radial spectral layer is supplied.

## 7. Vacuum Selector Sector

The vacuum selector part of the sidecar is not the same as the flavor transfer
root. It belongs to the BCC/Higgs boundary-order sector.

The chain is:

$$
\text{single-Weyl BB walk}
\to
\text{real helicity-locked }A_{2u}\text{ selector}
\to
\text{accepted tetrahedral branch}
\to
\text{radial instability at }r=0
\to
\text{positive quartic stabilization}.
$$

The continuum-leading radial model is

$$
E(r)=-cr+m^2r^2+\lambda r^4.
$$

When $\lambda>0$, the quartic term can bound the selector radius. The sidecar
closes with exactly one named intermediate axiom:

$$
\texttt{positive\_quartic\_backreaction\_bounds\_selector\_radius}.
$$

This is good theory hygiene. The quartic coefficient is not microscopically
derived in this sidecar, and should not be treated as derived in the paper.

Certainty: `C:6` for selector closure under the quartic axiom. `C:3` for a
microscopic derivation of the positive quartic coefficient.

## 8. Synthesis Verdict

The sidecar's essential contribution is:

$$
\boxed{
\text{flavor starts as framed boundary response, not as direct finite-group
mass output}
}
$$

The strongest theorem is the exact framed sterile-chain response

$$
\widehat\Sigma(2\sqrt2)=\epsilon^2P_u+P_b.
$$

The strongest no-go is the $S_3$ centralizer obstruction:

$$
S_3\text{-invariant}\Rightarrow
A=\alpha P_u+\beta(P_a+P_b),
$$

so an unbroken residual $K_3$ tail cannot produce $K_\nu$.

The PMNS and CKM outputs are valuable conditional textures, but they are not
the final mass theory. They feed the later synthesis layers:

$$
\text{depth theory},\qquad
\text{scalar Clebsches},\qquad
\text{radial spectral response}.
$$

## Open Burdens Passed Forward

The boundary-response sidecar passes five real burdens to the rest of the
synthesis:

- derive the physical origin of boundary framing;
- derive or replace the quark depth embedding $\{0,2,6\}$;
- separate scalar mass Clebsches from CKM current-amplitude Clebsches;
- move actual mass values into a radial spectral-measure theory;
- derive the positive quartic selector backreaction microscopically.

These are the right open problems. The rest is bookkeeping.
