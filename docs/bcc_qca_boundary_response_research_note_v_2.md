# BCC-QCA Boundary-Response Flavor Theory

## A self-consistent theorem-driven research note

### Working status

This note is a **conditional mathematical-physics program**, not an established theory. Its purpose is to isolate a possible causal origin of flavor structure in a BCC Weyl quantum cellular automaton (QCA), while clearly separating:

1. principles and kinematic assumptions;
2. results proven from those assumptions;
3. sector-specific boundary ansätze;
4. open gaps required for a complete theory.

**Repository implementation status.** The `boundary_response` sidecar now
implements gates V1-V43.  The framed neutrino core is derived by the product
sterile / Weyl-function mechanism, and the vacuum-selector sector is closed for
polishing modulo one explicit intermediate axiom:

$$
\texttt{positive\_quartic\_backreaction\_bounds\_selector\_radius}.
$$

PMNS and CKM textures remain below closed-theorem status: they are assembled
behind explicit boundary-shell assumptions.  The current implementation summary
lives in `src/clifford_3plus2_d5/boundary_response/README.md`,
`STATUS.md`, and `parameter_ledger.md`.

The mature claim is not that BCC geometry alone predicts all Standard Model flavor data. The stronger and more honest claim is:

$$
\boxed{\text{Flavor is the sector-dependent boundary spectral response of a vacuum-framed BCC Weyl QCA.}}
$$

The proposed chain is:

$$
\text{BCC causal geometry}
\longrightarrow
\text{residual family channel space}
\longrightarrow
\text{boundary spectral tensor}
\longrightarrow
\text{masses, mixings, and CP phases}.
$$

The key mathematical object is not a list of numerical formulas, but a projected boundary self-energy:

$$
J_{ab}^{(f)}(\omega)
\longrightarrow
\Sigma_{ab}^{(f)}(z)
\longrightarrow
\text{poles and response eigenvectors}.
$$

The strongest compact invariant currently obtained is

$$
\boxed{\epsilon=\sqrt2-1.}
$$

The strongest phenomenological consequence is the neutrino mass-splitting relation

$$
\boxed{\frac{\Delta m^2_{21}}{\Delta m^2_{31}}=(\sqrt2-1)^4=17-12\sqrt2.}
$$

However, even this result is conditional on a specific neutrino boundary response. Exact residual $S_3$ symmetry alone is insufficient.

---

# 1. Guiding physical idea

The Standard Model flavor problem may be a **boundary-response problem** rather than a problem of arbitrary fundamental Yukawa constants.

The analogy is with Brownian motion and fluctuation-response theory. In Brownian motion, unresolved microscopic molecular collisions determine both diffusion and mobility. In the QCA flavor picture, unresolved microscopic boundary excursions determine the effective mass gaps, mixing angles, and CP phases.

The proposed principle is:

$$
\boxed{\text{The same boundary spectral tensor controls mass, mixing, and CP response.}}
$$

Mass is not interpreted as a primitive slowing of the microscopic causal speed. The universal microscopic speed $c$ is fixed by the local QCA shift. Acquired mass is an infrared pole effect caused by repeated local recirculation into unresolved internal or boundary modes: a QCA version of zitterbewegung.

Thus:

$$
\boxed{c=\text{causal propagation speed},}
$$

while

$$
\boxed{m=\text{effective self-energy/pole gap from boundary recirculation}.}
$$

Virtual particles are unresolved off-shell excursions into the projected-out boundary sector.

---

# 2. Kinematic principles

## Principle P1: Local unitary causal update

There is a finite-dimensional Hilbert space per cell and a homogeneous local unitary update

$$
\Psi(t+1)=U_{\rm QCA}\Psi(t).
$$

The update has a finite causal cone. The microscopic speed is

$$
c=\ell/\Delta t.
$$

All apparent subluminal behavior of massive particles is emergent from internal scattering and recirculation, not from a smaller microscopic propagation speed.

## Principle P2: BCC causal exits

A three-dimensional BCC Weyl-QCA step has four primitive unoriented body-diagonal causal exit classes, represented by the tetrahedral set

$$
d_0=(+,+,+),\qquad d_1=(+,-,-),\qquad d_2=(-,+,-),\qquad d_3=(-,-,+).
$$

Their indistinguishability under the local update is represented by $S_4$.

This is not a decorative symmetry. It means the local causal update cannot distinguish the four primitive exits before a vacuum or boundary chooses a frame.

## Principle P3: Vacuum framing

A stable vacuum, Higgs boundary, sterile defect, or equivalent order parameter selects one primitive exit, say $d_0$. The remaining residual causal space is

$$
\mathcal H_{\rm fam}=\operatorname{span}\{d_1,d_2,d_3\}.
$$

The residual permutation symmetry is

$$
S_4\to S_3.
$$

This gives a natural three-dimensional family channel space:

$$
\boxed{N_{\rm fam}=4-1=3.}
$$

This is currently a causal-channel counting argument, not yet a topological index theorem for exactly three chiral Standard Model families.

## Principle P4: Sector-dependent boundary response

The universal BCC geometry supplies the arena and transfer structure. It does not uniquely fix all sectoral response operators. Each sector has its own unresolved boundary space:

$$
\mathcal H_Q^{(\nu)},\quad \mathcal H_Q^{(e)},\quad \mathcal H_Q^{(u)},\quad \mathcal H_Q^{(d)}.
$$

Each sector has a boundary coupling map

$$
V_f:\mathcal H_P^{(f)}\to\mathcal H_Q^{(f)}.
$$

Flavor data are extracted from

$$
\Sigma_f(z)=V_f^\dagger(z-H_{Q,f})^{-1}V_f.
$$

## Principle P5: No ungrounded projectors

No projector, Clebsch factor, phase, or texture may be introduced unless it is the spectral footprint of an explicit boundary degree of freedom or an explicitly stated boundary ansatz.

---

# 3. Projection and boundary spectral formalism

Split the microscopic Hilbert space into observed low-energy propagating modes and unresolved boundary/tail/link modes:

$$
\mathcal H=\mathcal H_P\oplus\mathcal H_Q.
$$

Write the microscopic effective generator in block form:

$$
H=
\begin{pmatrix}
H_P & V^\dagger\\
V & H_Q
\end{pmatrix}.
$$

The projected Green function is

$$
G_P(z)=\left[z-H_P-\Sigma(z)\right]^{-1},
$$

with self-energy

$$
\boxed{\Sigma(z)=V^\dagger(z-H_Q)^{-1}V.}
$$

Physical pole positions are determined by

$$
\boxed{\det\left[z-H_P(\mathbf p)-\Sigma(z,\mathbf p)\right]=0.}
$$

For small momentum, the pole dispersion has the form

$$
E_i^2(\mathbf p)=c^2|\mathbf p|^2+m_i^2c^4+\cdots.
$$

The mass $m_i$ is extracted from this pole equation. Thus mass is not simply $\operatorname{Re}\Sigma$; it is a pole shift induced by the boundary response.

## Boundary spectral tensor

Let $V_a$ be the coupling of observed flavor channel $a$ into the unresolved sector. Define

$$
\boxed{J_{ab}(\omega)=V_a^\dagger\delta(\omega-H_Q)V_b.}
$$

Then

$$
\boxed{\Sigma_{ab}(z)=\int d\omega\,\frac{J_{ab}(\omega)}{z-\omega}.}
$$

This is the QCA analogue of a fluctuation-response relation:

$$
\boxed{J_{ab}(\omega)\text{ controls poles, mixings, CP phases, and virtual excursions.}}
$$

---

# 4. Residual family basis

In the residual family basis $(e_1,e_2,e_3)$, define

$$
u=\frac{1}{\sqrt3}(1,1,1),
$$

$$
a=\frac{1}{\sqrt6}(2,-1,-1),
$$

$$
b=\frac{1}{\sqrt2}(0,1,-1).
$$

To avoid notational confusion, the vector $u$ is the collective family vector, not a neutrino field.

The residual representation decomposes as

$$
\mathbf3=\mathbf1\oplus\mathbf2.
$$

The physical interpretation is:

$$
u=u=\text{collective residual mode},
$$

$$
a=\text{radial imbalance mode},
$$

$$
b=\text{antisymmetric opposite-edge mode}.
$$

The projectors are

$$
P_u=|u\rangle\langle u|,
\qquad
P_a=|a\rangle\langle a|,
\qquad
P_b=|b\rangle\langle b|.
$$

The doublet projector is

$$
P_D=P_a+P_b.
$$

---

# 5. The residual $K_3$ transfer theorem

After vacuum framing, the three residual channels form a triangle graph $K_3$. Its adjacency matrix is

$$
A_{K_3}=\begin{pmatrix}
0&1&1\\
1&0&1\\
1&1&0
\end{pmatrix},
$$

with graph degree

$$
d=2.
$$

## Theorem 5.1: Residual BCC transfer invariant

If the residual boundary tail is self-similar and has decaying shell amplitudes $x_n$ obeying

$$
x_n=2x_{n+1}+x_{n+2},
$$

then the decaying transfer factor is

$$
\boxed{\epsilon=\sqrt2-1.}
$$

### Proof

Seek a decaying solution

$$
x_{n+1}=\epsilon x_n,\qquad 0<\epsilon<1.
$$

Substitute into

$$
x_n=2x_{n+1}+x_{n+2}.
$$

This gives

$$
1=2\epsilon+\epsilon^2.
$$

Therefore

$$
\epsilon^2+2\epsilon-1=0.
$$

The positive solution is

$$
\epsilon=-1+\sqrt2.
$$

Hence

$$
\boxed{\epsilon=\sqrt2-1.}
$$

Equivalently, the transfer matrix

$$
M_{\rm tr}=\begin{pmatrix}2&1\\1&0\end{pmatrix}
$$

has eigenvalues

$$
\lambda_\pm=1\pm\sqrt2,
$$

and the decaying factor is

$$
\epsilon=\lambda_+^{-1}=(1+\sqrt2)^{-1}=\sqrt2-1.
$$

$\square$

### Status

This theorem is strong within the residual-tail construction. The remaining task is to derive the self-similar recurrence from an explicit microscopic boundary Hamiltonian $H_Q$, rather than treating it as an impedance model.

---

# 6. Exact residual $S_3$ and the neutrino obstruction

## Lemma 6.1: Exact residual $S_3$ forbids doublet splitting

If the residual $S_3$ symmetry remains exact on

$$
\mathcal H_{\rm fam}=\mathbf1\oplus\mathbf2,
$$

then any $S_3$-equivariant response operator has the form

$$
\boxed{\mathcal K^{S_3}=\alpha P_u+\beta P_D.}
$$

It cannot distinguish $a$ from $b$.

### Proof

The residual representation decomposes into inequivalent irreducible components:

$$
\mathbf3=\mathbf1\oplus\mathbf2.
$$

By Schur's lemma, an operator commuting with the full $S_3$ action is scalar on each irreducible component. Therefore it may assign one scalar to $u$ and one scalar to the whole doublet plane, but it cannot split the doublet:

$$
\mathcal K|_{\mathbf2}=\beta I_{\mathbf2}.
$$

Thus

$$
\mathcal K=\alpha P_u+\beta(P_a+P_b).
$$

$\square$

### Consequence

The operator

$$
\mathcal K_\nu=\epsilon^2P_u+P_b
$$

is not an $S_3$-invariant consequence. It requires an additional boundary datum that reduces

$$
S_3\to S_2
$$

or lower.

This is a central correction. The theory must not claim that residual $S_3$ alone gives the observed neutrino hierarchy.

---

# 7. Edge-framed sterile neutrino response

The purpose of this section is to show how the desired neutrino texture can arise from an explicit boundary datum rather than from hidden group-theory overclaiming.

## Boundary proposal N1: charged-lepton frame plus sterile edge

After vacuum framing, choose a charged-lepton/Higgs boundary frame in the residual family space. In a convenient basis, this frame selects a vertex direction, say

$$
w=e_1=(1,0,0).
$$

This should not be interpreted as the weak interaction alone choosing a generation. The weak charged current is flavor-universal before diagonalization. Rather, $w$ represents the charged-lepton/Higgs response frame relative to which the neutrino sterile boundary is oriented.

The subgroup preserving the opposite edge $(e_2,e_3)$ is

$$
S_2(e_2\leftrightarrow e_3).
$$

The $S_2$-adapted basis is $u,a,b$ as defined above. Here $a$ is the radial direction associated with the selected active frame, $b$ is the antisymmetric current on the opposite sterile edge, and $u$ is the collective residual mode.

## Boundary proposal N2: minimal sterile Hilbert space

Let the unresolved neutrino boundary sector be

$$
\mathcal H_{Q,\nu}=\mathcal H_{\rm tail}^{(u)}\oplus\mathcal H_{\rm edge}^{(b)}.
$$

Let

$$
|s_u\rangle\in\mathcal H_{\rm tail}^{(u)},
\qquad
|s_b\rangle\in\mathcal H_{\rm edge}^{(b)}.
$$

The minimal coupling map is

$$
\boxed{V_\nu=\lambda\left(\epsilon |s_u\rangle\langle u|+|s_b\rangle\langle b|\right).}
$$

The radial state $a$ is absent because it is a node mode of the selected active-frame/opposite-edge sterile boundary. It is orthogonal to both $u$ and $b$.

Assume the low-energy sterile returns obey

$$
\langle s_u|G_Q(z)|s_u\rangle
=
\langle s_b|G_Q(z)|s_b\rangle
=G_M(z),
$$

and

$$
\langle s_u|G_Q(z)|s_b\rangle=0
$$

at leading order.

## Theorem 7.1: Edge-framed sterile boundary implies the neutrino texture

Under boundary proposals N1 and N2, the projected neutrino self-energy is proportional to

$$
\boxed{\mathcal K_\nu=\epsilon^2P_u+P_b.}
$$

Consequently, the eigenvalue structure is

$$
0,\quad \epsilon^2,\quad 1.
$$

If neutrino masses are proportional to these eigenvalues, then

$$
\boxed{m_1=0,\qquad m_2=m_0\epsilon^2,\qquad m_3=m_0,}
$$

and

$$
\boxed{\frac{\Delta m^2_{21}}{\Delta m^2_{31}}=\epsilon^4=17-12\sqrt2.}
$$

### Proof

The Schur complement gives

$$
\Sigma_\nu(z)=V_\nu^\dagger G_Q(z)V_\nu.
$$

Substitute the definition of $V_\nu$. Using the diagonal low-energy return functions gives

$$
\Sigma_\nu(z)=\lambda^2G_M(z)
\left(\epsilon^2|u\rangle\langle u|+|b\rangle\langle b|\right).
$$

Thus, after absorbing the common scalar factor into the mass scale,

$$
\mathcal K_\nu=\epsilon^2P_u+P_b.
$$

Since $a\perp u$ and $a\perp b$,

$$
\mathcal K_\nu a=0.
$$

Also,

$$
\mathcal K_\nu u=\epsilon^2u,
\qquad
\mathcal K_\nu b=b.
$$

So the eigenvalues are $0,\epsilon^2,1$. For normal ordering,

$$
\frac{\Delta m^2_{21}}{\Delta m^2_{31}}=\epsilon^4.
$$

Using Theorem 5.1,

$$
\epsilon^4=(\sqrt2-1)^4=17-12\sqrt2.
$$

$\square$

### Status

This is a conditional theorem. The group-theoretic obstruction is resolved only by adding a physical edge-framed sterile boundary. The remaining assumptions are explicit:

1. the charged-lepton/Higgs response defines the active frame;
2. the sterile Majorana boundary lives on the opposite residual edge;
3. the collective channel reaches the sterile boundary through one $K_3$ transfer factor $\epsilon$;
4. the two sterile return Green functions share the same low-energy normalization;
5. leading cross-return between $|s_u\rangle$ and $|s_b\rangle$ is absent.

If these assumptions are not realized by an explicit boundary Hamiltonian, the neutrino texture is not derived.

---

# 8. Conditional PMNS boundary-scattering texture

The PMNS matrix comes from the mismatch between the charged-lepton boundary response and the neutrino boundary response:

$$
U_{\rm PMNS}=U_e^\dagger U_\nu.
$$

The neutrino eigenbasis from Theorem 7.1 is approximately

$$
U_\nu=(a,u,b),
$$

which is the tri-bimaximal basis up to column and phase conventions.

## Assumption L1: two-step charged-lepton leakage

The charged-lepton boundary frame is misaligned from the neutrino transfer eigenbasis by a two-step residual leakage of magnitude $\epsilon^2$.

The selected active port decomposes as

$$
e_1=\sqrt{\frac23}a+\frac1{\sqrt3}u.
$$

Assume the leakage condition

$$
\sqrt{\frac23}\sin\theta_e=\epsilon^2.
$$

Then

$$
\boxed{\sin\theta_e=\sqrt{\frac32}\epsilon^2.}
$$

## Assumption L2: spin-Coxeter-Schur boundary phase

Assume the charged-lepton boundary phase is the full primitive spin-Coxeter-Schur word

$$
W_e=-q_{A_3}q_{A_2}.
$$

The BCC/tetrahedral parent spin lift gives

$$
q_{A_3}=e^{i\pi/4},
$$

and the residual triangle spin lift gives

$$
q_{A_2}=e^{i\pi/3}.
$$

The second-order Schur complement contributes a minus sign. Therefore

$$
W_e=-e^{i(\pi/4+\pi/3)}=e^{i19\pi/12}=e^{-i5\pi/12}.
$$

So

$$
\boxed{\phi_e=-\frac{5\pi}{12}.}
$$

## Conditional proposition 8.1: PMNS texture

Under assumptions L1 and L2,

$$
\boxed{U_{\rm PMNS}=R_{12}^{e\dagger}\left(\theta_e,-\frac{5\pi}{12}\right)U_{\rm TBM}.}
$$

This gives

$$
\boxed{\sin^2\theta_{13}=\frac34\epsilon^4=0.0220779,}
$$

$$
\boxed{\sin^2\theta_{12}=0.304610,}
$$

$$
\boxed{\sin^2\theta_{23}=0.488712,}
$$

and, for one chirality branch,

$$
\boxed{\delta_{\rm CP}\simeq261.6^\circ.}
$$

The CP-conjugate branch gives approximately $\delta_{\rm CP}\simeq98.4^\circ$.

### Status

This texture is not derived from BCC geometry alone. It requires:

1. a charged-lepton boundary shell whose leading mixing is two-step;
2. a derivation of the Clebsch $\sqrt{3/2}$;
3. a proof that the full spin-Coxeter-Schur word is selected rather than a subword.

---

# 9. Conditional CKM boundary texture

The quark sector uses the same residual BCC transfer invariant $\epsilon$, but with color-dressed boundary response. This section is explicitly lower confidence than the neutrino mass-ratio theorem. The CKM numerics are striking, but the prefactors require an explicit quark boundary shell.

## Assumption Q1: quark primitive boundary shell

The quark primitive boundary shell decomposes as

$$
\mathcal S_q=\mathcal S_{\rm even}\oplus\mathcal S_{\rm odd},
$$

with

$$
\dim\mathcal S_{\rm even}=1,
\qquad
\dim\mathcal S_{\rm odd}=5=2+3.
$$

The five odd channels are interpreted as two residual BCC odd quadratures plus three color odd boundary ports.

Introduce five Clifford generators $\gamma_A$, $A=1,\ldots,5$, satisfying

$$
\{\gamma_A,\gamma_B\}=2\delta_{AB}I.
$$

Let

$$
\Gamma_q=\sum_{A=1}^{5}\gamma_A.
$$

Then

$$
\Gamma_q^2=5I.
$$

Assume the primitive quark boundary coin is flat over the one even and five odd channels:

$$
\boxed{B_q=\frac1{\sqrt6}(I+i\Gamma_q).}
$$

Then

$$
e^{i\delta_q}=\frac{1+i\sqrt5}{\sqrt6},
$$

and

$$
\boxed{\delta_q=\arctan\sqrt5.}
$$

## Assumption Q2: transfer-depth hierarchy

The quark mixing depths are assigned as

$$
1\leftrightarrow2:\epsilon^2,
\qquad
2\leftrightarrow3:\epsilon^4,
\qquad
1\leftrightarrow3:\epsilon^6.
$$

## Assumption Q3: color and BCC Clebsches

The Cabibbo transition is a color-singlet return process, producing the fundamental color contraction

$$
\sum_A T^AT^A=C_FI,
\qquad
C_F=\frac43.
$$

The $2\leftrightarrow3$ transition uses a symmetric two-path BCC channel, producing $\sqrt2$. The $1\leftrightarrow3$ transition uses an antisymmetric normalized two-path BCC channel, producing $1/\sqrt2$.

## Conditional proposition 9.1: CKM texture

Under Q1-Q3,

$$
\boxed{s_{12}^q=\frac43\frac{\epsilon^2}{\sqrt{1+\epsilon^4}},}
$$

$$
\boxed{s_{23}^q=\sqrt2\epsilon^4,}
$$

$$
\boxed{s_{13}^q=\frac{\epsilon^6}{\sqrt2},}
$$

and

$$
\boxed{\delta_q=\arctan\sqrt5.}
$$

Thus

$$
\boxed{V_{\rm CKM}=R_{23}(\theta_{23}^q)R_{13}(\theta_{13}^q,\delta_q)R_{12}(\theta_{12}^q).}
$$

Numerically:

$$
s_{12}^q=0.225469,
\qquad
s_{23}^q=0.041631,
\qquad
s_{13}^q=0.003571,
\qquad
\delta_q=65.905^\circ.
$$

The approximate magnitude matrix is

$$
|V_{\rm CKM}|\approx
\begin{pmatrix}
0.97425&0.22547&0.00357\\
0.22533&0.97340&0.04163\\
0.00858&0.04090&0.99913
\end{pmatrix}.
$$

The Jarlskog invariant is approximately

$$
J_q\simeq2.98\times10^{-5}.
$$

### Status

The CKM texture is a promising color-dressed boundary ansatz, not yet a derivation from BCC causality alone. It becomes a theorem only if the primitive quark boundary shell and the flat coin are obtained from an explicit local QCA boundary operator.

---

# 10. Gauge and Higgs sector

Gauge bosons are not family-transfer modes. They are link variables enforcing local internal-basis consistency.

A generic link has the form

$$
U_{x,h}=\exp\left[-ih^\mu\left(g_3G_\mu^AT^A+g_2W_\mu^a\frac{\tau_a}{2}+g_YB_\mu\frac{Y}{2}\right)\right].
$$

The Higgs order parameter gives the usual electroweak mass relations

$$
M_W=\frac{g_2v}{2},
\qquad
M_Z=\frac{v}{2}\sqrt{g_2^2+g_Y^2},
\qquad
M_\gamma=0.
$$

Therefore

$$
\boxed{M_W=M_Z\cos\theta_W,}
$$

and

$$
\boxed{\rho_0=1.}
$$

However,

$$
\boxed{g_Y/g_2\text{ is not predicted by the residual }K_3\text{ family-transfer mechanism}.}
$$

Predicting the Weinberg angle would require an additional gauge-unification principle, UV fixed point, or explicit gauge kinetic QCA coin.

---

# 11. Compatibility with universal $c$, zitterbewegung, and virtual particles

The boundary response mechanism is compatible with universal microscopic speed because the QCA update can be factorized schematically as

$$
U=SC,
$$

where $S$ is the causal shift and $C$ is a local internal or boundary coin. The shift fixes the light cone; the coin cannot enlarge it.

A massive excitation repeatedly alternates between propagation and internal recirculation:

$$
\text{move at }c\to\text{scatter internally}\to\text{move at }c\to\cdots.
$$

The observed subluminal group velocity is an infrared effect. The associated internal oscillation is the QCA version of zitterbewegung.

Virtual particles enter as unresolved off-shell excursions into $Q$:

$$
P\to Q\to P.
$$

These excursions are encoded by

$$
\Sigma(z)=V^\dagger(z-H_Q)^{-1}V.
$$

They are not acausal particles; they are internal terms in a causal Green function.

---

# 12. Consistency requirements

## 12.1 Electroweak gauge invariance

Charged-fermion masses must respect electroweak quantum numbers. Therefore the Yukawa response should have the schematic form

$$
\Sigma_f(E)=\langle H\rangle\,\mathcal Y_f(E),
$$

where

$$
\mathcal Y_f(E)=V_{f,L}^\dagger(E-H_{Q,f})^{-1}V_{f,R}.
$$

Then

$$
M_f=\frac{v}{\sqrt2}\mathcal Y_f(0).
$$

## 12.2 Chirality and anomaly consistency

The flavor construction assumes a chiral Standard Model boundary sector. A complete theory must explain why mirror fermions are absent or decoupled, and must reproduce the anomaly consistency of the Standard Model. This remains open.

## 12.3 Renormalization scale

Flavor relations must be assigned a matching scale, naturally

$$
\mu\sim m_Z.
$$

Yukawas and quark masses require renormalization-group running. CKM elements are less scale-sensitive at low energy but still need a specified matching convention.

## 12.4 Topological generation count

The relation

$$
S_4\to S_3,\qquad 4-1=3
$$

is currently a causal-channel counting argument. A stronger result would be an index theorem:

$$
\boxed{N_{\rm fam}=\operatorname{Ind}(\mathcal D_{\partial}^{\rm framed})=3.}
$$

This is a major open problem.

---

# 13. Compact relation set

Let

$$
r=\epsilon^4=17-12\sqrt2.
$$

The strongest relation is

$$
\boxed{\frac{\Delta m^2_{21}}{\Delta m^2_{31}}=r.}
$$

Conditional PMNS relations include

$$
\boxed{\sin^2\theta_{13}=\frac34r,}
$$

$$
\boxed{\sin^2\theta_{12}=0.304610,}
$$

$$
\boxed{\sin^2\theta_{23}=0.488712,}
$$

$$
\boxed{\delta_{\rm CP}\simeq261.6^\circ\quad\text{or CP-conjugate}.}
$$

Conditional CKM relations include

$$
\boxed{|V_{us}|=\frac43\frac{\sqrt r}{\sqrt{1+r}},}
$$

$$
\boxed{|V_{cb}|=\sqrt2\,r,}
$$

$$
\boxed{|V_{ub}|=\frac{r^{3/2}}{\sqrt2},}
$$

$$
\boxed{\delta_q=\arctan\sqrt5.}
$$

The word “conditional” is essential. These relations require boundary-shell assumptions beyond BCC kinematics.

---

# 14. Proven, conditional, and unproven statements

## Proven within stated mathematical assumptions

1. Exact residual $S_3$ permits only

$$
\alpha P_u+\beta(P_a+P_b).
$$

2. Exact residual $S_3$ cannot produce

$$
\epsilon^2P_u+P_b.
$$

3. The residual self-similar $K_3$ transfer recurrence gives

$$
\epsilon=\sqrt2-1.
$$

4. The edge-framed sterile coupling map

$$
V_\nu=\lambda(\epsilon |s_u\rangle\langle u|+|s_b\rangle\langle b|)
$$

with diagonal equal sterile returns gives

$$
\mathcal K_\nu=\epsilon^2P_u+P_b.
$$

## Conditional physical claims

1. Vacuum framing of a BCC Weyl QCA produces three residual family channels.
2. The sterile neutrino boundary is an opposite-edge Majorana defect.
3. The charged-lepton boundary leakage begins at two transfer steps.
4. The leptonic CP phase is the full spin-Coxeter-Schur boundary word.
5. The quark boundary shell is $1_{\rm even}\oplus5_{\rm odd}$ with a flat primitive coin.

## Not yet derived

1. A full microscopic $H_Q$ whose spectral tensor produces all flavor textures.
2. An index theorem for exactly three chiral families.
3. Absence or decoupling of mirror fermions.
4. Absolute charged-fermion masses.
5. Gauge coupling ratios such as $g_Y/g_2$.
6. Higgs mass and scalar potential parameters.

---

# 15. Roadmap to fill the gaps

## Task 1: Construct the explicit BCC boundary Hilbert space

Define the microscopic boundary shell $\mathcal H_Q$, its local basis, its residual symmetry action, and its coupling maps to observed fermion channels.

Deliverable:

$$
H_Q,\quad V_a,\quad G_Q(z)=(z-H_Q)^{-1}.
$$

## Task 2: Derive the $K_3$ recurrence from $H_Q$

Replace the impedance recurrence

$$
\epsilon=\frac1{2+\epsilon}
$$

with an explicit Green-function calculation.

Deliverable:

$$
G_{\rm tail}(z)\quad\Rightarrow\quad\epsilon=\sqrt2-1.
$$

## Task 3: Prove or falsify the edge-framed sterile boundary

Build a sterile Majorana edge defect on the residual triangle and compute the Schur complement.

Deliverable:

$$
\Sigma_\nu(z)\propto\epsilon^2P_u+P_b
$$

or a clear no-go.

## Task 4: Derive charged-lepton leakage

Construct the charged-lepton/Higgs boundary response and determine whether the leading misalignment is really two-step.

Deliverable:

$$
\sin\theta_e=\sqrt{\frac32}\epsilon^2
$$

or a corrected relation.

## Task 5: Derive the leptonic phase word

Compute the actual spinorial holonomy of the charged-lepton boundary loop.

Deliverable:

$$
\phi_e=-\frac{5\pi}{12}
$$

or a different phase selected by the explicit boundary.

## Task 6: Construct the quark boundary shell

Derive or reject the decomposition

$$
\mathcal S_q=1_{\rm even}\oplus5_{\rm odd}.
$$

Deliverable:

$$
B_q=\frac1{\sqrt6}(I+i\Gamma_q)
$$

or a different quark boundary coin.

## Task 7: Address chirality and anomalies

Show how the QCA produces the chiral Standard Model spectrum without unwanted mirror partners and with anomaly consistency.

Deliverable:

$$
\text{chiral boundary index/anomaly matching theorem.}
$$

## Task 8: Define matching scale and RG flow

Specify the scale at which the boundary response textures are matched to Standard Model parameters and evolve them consistently.

Deliverable:

$$
\text{matching at }\mu\sim m_Z\text{ plus RG evolution.}
$$

---

# 16. Falsifiability

The cleanest near-term falsifiers are in the neutrino sector:

$$
\boxed{\frac{\Delta m^2_{21}}{\Delta m^2_{31}}=17-12\sqrt2.}
$$

$$
\boxed{m_1=0,\qquad \Sigma m_\nu\approx0.058\text{--}0.059\,\mathrm{eV}.}
$$

Conditional PMNS falsifiers are:

$$
\boxed{\sin^2\theta_{13}=0.0220779,}
$$

$$
\boxed{\sin^2\theta_{12}=0.304610,}
$$

$$
\boxed{\sin^2\theta_{23}=0.488712,}
$$

$$
\boxed{\delta_{\rm CP}\simeq261.6^\circ\text{ or CP-conjugate}.}
$$

The CKM texture is already numerically close to current data, but because it was not obtained as a blind prediction, it should be treated as a lower-confidence consistency success until the quark boundary shell is explicitly derived.

---

# 17. Final statement

The mature version of the theory is:

$$
\boxed{\text{Causal geometry supplies the universal residual channel system.}}
$$

$$
\boxed{\text{Boundary spectral response supplies sector-specific masses and mixings.}}
$$

The core kinematic structure is:

$$
S_4\to S_3,
\qquad
N_{\rm fam}=3,
\qquad
K_3\Rightarrow\epsilon=\sqrt2-1.
$$

The most important caution is:

$$
\boxed{\text{BCC geometry alone does not determine the full flavor theory.}}
$$

It determines an arena and a transfer invariant. Realistic flavor requires explicit boundary response operators. The research program is to construct those operators, compute their spectral tensors, and then let the poles and eigenvectors produce masses, mixings, and CP phases.

That is the disciplined route from beautiful numerology toward a testable physical theory.
