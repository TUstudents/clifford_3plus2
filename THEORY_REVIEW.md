# A Boundary-Response Theory of Flavor and Mass on a BCC Quantum Cellular Automaton

**Status:** working prototype, compiled for external review.
**Scope:** the flavor and mass sector of one Standard Model generation lifted to a
Clifford/Pati–Salam carrier propagating on a body-centered-cubic (BCC) quantum
cellular automaton (QCA), with flavor and mass arising from a structured
boundary (Schur) response.

This document is self-contained. It states the construction, the theorems, the
conditional results, and the open problems, with an explicit certainty label on
every nontrivial claim. It is assembled from a set of internal "sidecar"
notebooks (proofs, obstruction audits, and simulators); those are referenced by
name but are not required to follow the argument here.

**Central claim.** Flavor is not an internal finite-group spectrum but
**boundary spectroscopy of the BCC Weyl automaton**. Conditional on a framed
three-port boundary (taken as a primitive axiom, §6.5, *not* derived), the
orientation-preserving BB $q=0$ scar fixes a single **silver contraction class**
$\epsilon=\sqrt2-1$ ($\det{+}1$, $\operatorname{tr}=2\sqrt2$), and each fermion
sector is a distinct **Schur readout sharing that class**: neutrinos the
one-sided sterile eigenresponse; charged leptons and quarks two-sided active
chiral kernels (Koide-equipartition holonomy; colored residues); CKM/PMNS the
relative left-frame rotations of paired kernels. The sterile branch is a
theorem-in-a-model; the active branches are conditional on the weak-framed repair
doublet and remain the open work. (Strength is stratified: the framing is `C:7`,
the silver-class universality `C:8`, the neutrino readout `C:9`-in-model; the
active-sector readouts are the conditional `C:3`–`C:6` claims.)

---

## 0. How to read this document

### 0.1 Certainty scale

Every load-bearing claim carries a label `C:k`:

| Level | Meaning |
|---|---|
| `C:9` | Proven theorem or exact symbolic identity under the stated definitions. |
| `C:8` | Exact finite certificate or exhaustive computation with decisive controls. |
| `C:7` | Derived result inside the accepted construction, ordinary assumptions explicit. |
| `C:6` | Conditional theorem depending on a named nontrivial hypothesis. |
| `C:5` | Strong organizing equivalence or duality, not yet a theorem. |
| `C:4` | Compatible reconstruction or inverse realization, not forward derivation. |
| `C:3` | Plausible conjectural mechanism needing a proof or falsifier. |
| `C:2` | Weak speculation or bookkeeping correlation. |
| `C:1` | Falsified or killed by a decisive control. |

The honest posture of the theory is that the **carrier and the boundary-response
formalism are strong (`C:7`–`C:9`)**, while the **microscopic selection of the
flavor data is mostly conditional (`C:3`–`C:6`)**. We do not hide this.

### 0.2 Two distinct constants both historically written ε

This is the single most important notational warning for a referee.

- **Silver transfer root** $\epsilon=\sqrt2-1\approx0.4142$. A dimensionless
  flavor/boundary quantity. Also written as the metallic fixed point
  $\epsilon=1/(2+\epsilon)$.
- **Lattice spacing / expansion parameter** $\epsilon_{\rm lat}$ used in the
  continuum expansion of the QCA walk and in CP/SME bounds
  ($\epsilon_{\rm lat}\lesssim10^2\,\ell_{\rm Planck}$).

These are physically unrelated. Throughout, $\epsilon$ means the silver root
unless explicitly subscripted. The flavor hierarchy parameter is

$$
\eta \equiv \epsilon^2 = 3-2\sqrt2 \approx 0.1716 .
$$

### 0.3 The residual three-port basis

Many results live on a 3-dimensional "residual family" space with the fixed
orthonormal basis

$$
u=\tfrac1{\sqrt3}(1,1,1),\quad
a=\tfrac1{\sqrt6}(2,-1,-1),\quad
b=\tfrac1{\sqrt2}(0,1,-1).
$$

Here $u$ is the collective/trace mode and $(a,b)$ span the traceless doublet.
$P_u,P_a,P_b$ are the rank-one projectors. The tetrahedral geometry (§9) fixes
this *basis* once a three-port boundary is granted; it does **not** derive the
*existence* of exactly three generations — that is the §6.5 axiom
$\dim\mathcal H_{\rm fam}=3$.

---

## 1. The central thesis

> **Flavor is not an internal finite-group spectrum but boundary spectroscopy of
> the BCC Weyl automaton.** Conditional on a framed three-port boundary (a
> primitive axiom, §6.5, not derived), the orientation-preserving BB $q=0$ scar
> fixes a single silver contraction class $\epsilon=\sqrt2-1$ ($\det{+}1$,
> $\operatorname{tr}=2\sqrt2$), and each fermion sector is a distinct Schur readout
> sharing that class: neutrinos the one-sided sterile eigenresponse (§9); charged
> leptons and quarks two-sided active chiral kernels — Koide-equipartition
> holonomy (§10), colored residues (§11); CKM/PMNS the relative left-frame
> rotations of paired kernels. The sterile branch is a theorem-in-a-model; the
> active branches are conditional on the weak-framed repair doublet (§11.5) and
> remain the open work.

The theory factorizes the physics of one fermion into three layers:

$$
\boxed{\;
\underbrace{\text{BCC QCA stage}}_{\text{relativistic propagation}}
\;+\;
\underbrace{\text{Clifford charge carrier}}_{\text{SM gauge charges}}
\;+\;
\underbrace{\text{boundary Brownian defect}}_{\text{flavor and mass}}
\;=\;
\text{observed fermion}
\;}
$$

A massless SM fermion propagates on the BCC QCA; its gauge charges live in a
chiral Clifford spinor; its **mass and flavor arise when the visible carrier
repeatedly enters an unresolved boundary layer, circulates, and returns**. The
boundary layer is a structured quantum bath with residual family symmetry,
defects, spectral modes, and holonomy.

**Key methodological claim (`C:7`):** flavor data are *not* direct finite-group
outputs. They are transfer factors, projectors, residues, pole shifts, and
holonomies of a boundary response. Finite algebra supplies the grammar; the mass
values require a spectral response. The organizing equation is the Schur
complement (§4).

The sharp version of the claim, suitable for an abstract, is:

> The QCA gives the relativistic stage; the Clifford spinor gives the SM
> charges; boundary Brownian defects give flavor.

We explicitly do **not** claim "the QCA explains everything."

---

## 2. The charge carrier

### 2.1 Construction

The internal carrier is the chiral spinor of $\mathrm{Cl}(0,10)$. The full real
module has dimension $64$; a real chirality involution built from the volume
element and a commuting complex structure $J$ has projectors of rank $(32,32)$.
Choosing one chirality and the compatible $J$ ($J^2=-I$, $J^\top J=I$) gives

$$
S_+,\quad \dim_{\mathbb R}S_+=32,\qquad (S_+,J)\simeq\mathbb C^{16}.
$$

`C:9` for the Clifford relations, chirality ranks, and complex dimension.

### 2.2 Pati–Salam factorization and the gauge chain

$$
\mathrm{Cl}(0,10)\simeq\mathrm{Cl}(0,6)\,\widehat\otimes\,\mathrm{Cl}(0,4)
\;\Rightarrow\;
\mathrm{Spin}(6)\times\mathrm{Spin}(4)\simeq SU(4)\times SU(2)_L\times SU(2)_R,
$$

with the conventional breaking

$$
\mathrm{Spin}(10)\supset SU(4)\times SU(2)_L\times SU(2)_R
\supset SU(3)_c\times SU(2)_L\times U(1)_Y .
$$

$SU(4)$ contains color and $B-L$ ($SU(3)_c\subset\mathrm{Cent}_{SU(4)}(B-L)$);
$SU(2)_R$ supplies $T_{3R}$. Hypercharge is the Pati–Salam remnant

$$
Y=T_{3R}+\tfrac12(B-L),\qquad Q_{\rm em}=T_{3L}+Y.
$$

`C:9` for the algebra extraction and hypercharge embedding (after the $B-L$ and
$T_{3R}$ directions are chosen).

### 2.3 One generation, and only one

In the code's exact real normalization the physical hypercharge is implemented
as $Y_{\rm phys}=\tfrac12 T_{3R}^{\rm raw}+\tfrac13(B-L)^{\rm raw}$, reproducing
the SM table exactly:

| Field | Rep | $Y$ | Complex mult. |
|---|---:|---:|---:|
| $Q$ | $(3,2)$ | $1/6$ | $6$ |
| $u^c$ | $(\bar3,1)$ | $-2/3$ | $3$ |
| $d^c$ | $(\bar3,1)$ | $1/3$ | $3$ |
| $L$ | $(1,2)$ | $-1/2$ | $2$ |
| $e^c$ | $(1,1)$ | $1$ | $1$ |
| $\nu^c$ | $(1,1)$ | $0$ | $1$ |

Total $=16$. `C:9`.

**What the carrier does not give:** $N_{\rm gen}=3$, the Yukawa matrices,
CKM/PMNS, or mass scales. `C:1` for any claim that the carrier alone derives
three generations.

### 2.4 Carrier honesty (obstruction audit)

The branch $\mathbb R^{10}\to\mathbb C^3\oplus\mathbb C^2\to
\Lambda^{\rm even}(\mathbb C^5)$ is a clean one-generation representation theorem
*once $J$ and the $3+2$ split are supplied* (`C:9`). But forcing $J$, the
$6+4$ split, and the "no-locking" gate algebra
$\operatorname{End}_{G_{\rm SM}}(\mathbb C^3\oplus\mathbb C^2)=
\mathbb C P_3\oplus\mathbb C P_2$ from microscopic QCA rule data is **not**
proven. Several primitive rule classes were exhaustively scanned (signed-twist
witness: 3840 candidates; monomial-hop Bloch: 2400 candidates) and none forces
the data without seeding it, generating forbidden lower-rank addressability, or
losing the hypercharge standard. `C:8` for the closed obstruction classes; `C:3`
for the microscopic bridge.

---

## 3. The spacetime QCA stage

The microscopic spacetime machine is a local unitary update
$\psi(t+1)=U\psi(t)$ with continuum Hamiltonian read from
$U(k)\simeq e^{-i\epsilon_{\rm lat}H_{\rm eff}(k)}$.

The relevant walk is the **Bialynicki–Birula (BB) BCC Weyl automaton** with the
pinned 1994 hop coefficients $q_\pm=(1\pm i)/4$ on the eight body-diagonal
directions $(\pm1,\pm1,\pm1)$. The two chiral blocks have continuum limits

$$
H_R(k)=\sigma\cdot k,\qquad H_L(k)=-\sigma\cdot k,
$$

and the chiral Dirac assembly gives $H_D(k)=\alpha\cdot k$ with
$\alpha_i=\mathrm{diag}(\sigma_i,-\sigma_i)$. `C:8` for the implemented free
carrier (all-momentum unitarity inherited from the source convention).

- **Mass slot.** $H_m(k)=\alpha\cdot k\otimes I_{\rm int}+\beta\otimes M_{\rm int}$
  satisfies $H_m^2=(|k|^2+m^2)I$ for scalar $M_{\rm int}=mI$. `C:8` that the
  carrier *can host* mass; `C:1` that the scalar control *is* the observed
  spectrum.
- **Lorentz recovery.** The two Weyl blocks have opposite-sign cubic anisotropy,
  which cancels in the Dirac assembly; the BCC-Dirac residual begins at quartic
  order $\Delta\sim\tfrac{\epsilon_{\rm lat}^4}{6}(k_x^2k_y^2+\dots)$. `C:8` for
  the free-dispersion diagnostic; `C:4` for full continuum QFT recovery (boost
  covariance, interacting renormalization, all-momentum no-doubling unproven).
- **Gauge/Higgs/Yukawa.** Background gauge coupling
  $H(k,A)=\alpha\cdot k\otimes I+I\otimes iA$ is correct (`C:8`); Wilson
  plaquettes, link dynamics, and a static Higgs/Yukawa slot
  $Y_{\rm internal}(\Phi)=A(\Phi)+A(\Phi)^\top$ exist as a **prototype** (`C:5`–`C:6`),
  not a derived interacting theory.

**Category guardrail (`C:8`):** asking the free BCC walk to produce three
generations, CKM angles, or mass ratios is a category error. Those are
boundary-response questions.

---

## 4. The boundary-response formalism

Split the Hilbert space into a resolved (visible) sector $P$ and an unresolved
boundary sector $Q$:

$$
H=\begin{pmatrix}H_P & V^\dagger\\ V & H_Q\end{pmatrix},
\qquad
G_P(z)=\big[z-H_P-\Sigma(z)\big]^{-1},
\qquad
\Sigma(z)=V^\dagger(z-H_Q)^{-1}V .
$$

The full visible fermion obeys
$\big[z-H_{\rm QCA}(k)-\Sigma_f(z)\big]\psi_f=0$, where the flavor label $f$
records *which internal charge sector and which family boundary channel* are
coupled through $V_f$ — it is **not** a new gauge group. `C:9` for the
Schur/Feshbach identity; `C:7` as the common flavor language.

**Two object classes, introduced from the start.** The boundary response appears
in *two* distinct forms, and conflating them is an error:

- a **one-sided self-energy** $\Sigma(z)=V^\dagger(z-H_Q)^{-1}V$ for
  *propagation / sterile* responses (same in- and out-coupling, $V_L=V_R$). Its
  pole residue $V^\dagger P_\lambda V\ge0$ is *positive semidefinite* — Hermitian.
  This governs the neutrino sterile sector (§9) and kinetic self-energies.
- a **two-sided chiral Yukawa kernel**
  $B(z)=V_R^\dagger(z-H_Q)^{-1}V_L$ for *active, chirality-flipping* (Dirac mass)
  responses, with generally $V_L\ne V_R$. Its residue need not be positive or even
  Hermitian, so it can carry the rotations/holonomies that masses-as-singular-
  values and CKM/PMNS require. This governs charged leptons (§10) and quarks
  (§11). The hidden bath $H_Q$ is Hermitian in both cases; only the kernel differs.

A one-sided Hermitian self-energy **cannot** produce the active rotations (§10.4),
so the active sectors *require* the two-sided kernel. Masses are then: poles/
residues of $\Sigma$ (sterile eigenvalues), or singular values $m_f\sim v\,s_i(B)$
(active amplitudes), with mixing $=$ relative left-frame holonomy of two kernels.

In time language this is a quantum memory kernel
$K(t)=V^\dagger e^{-iH_Qt}V$ — the "quantum Brownian" picture: the visible
particle makes hidden excursions into a structured bath and returns with memory,
phase, and spectral weight (still fully coherent).

Expanding the resolvent gives the recirculation series

$$
\Sigma(z)=\sum_{n\ge0}\frac{V^\dagger H_Q^{\,n}V}{z^{n+1}},
$$

each term meaning: enter the boundary, circulate $n$ hidden steps, return.
A mass is a pole/residue feature of $\Sigma_f$, not a finite-group count.

**Mass factorization (`C:5`):**

$$
m_f\sim A_f\,\eta^{k_f}\,R_f,
$$

with $\eta^{k_f}$ the Pell radial depth (§5), $A_f$ the entry/residue
coefficient (§7), and $R_f$ the remaining radial return response (§8). The same
$\eta$ — from the one BB $q=0$ silver transfer (§5.4) — runs through *every*
fermion sector; sectors differ only by the boundary readout that fixes $A_f$ and
$k_f$ (the unified backbone, §11.0).

---

## 5. The silver root and the Pell transfer theorem

### 5.1 The root

The dimensionless flavor transfer root is

$$
\epsilon=\sqrt2-1,\qquad \epsilon^2+2\epsilon-1=0,\qquad
\epsilon=\frac1{2+\epsilon}\ \ (N=2\ \text{metallic fixed point}).
$$

It also appears as a half-line Weyl function value. For the semi-infinite unit
chain, the head Green function obeys the Dyson fixed point
$m(z)=1/(z-m(z))$ with decaying branch
$m(z)=\tfrac{z-\sqrt{z^2-4}}{2}$; at the probe $z=2\sqrt2$,
$m(2\sqrt2)=\sqrt2-1=\epsilon$. `C:9`.

### 5.2 Pell transfer (sharper radial statement)

For the silver Pell matrix

$$
T_{\rm Pell}=\begin{pmatrix}2&1\\1&0\end{pmatrix},
\qquad \lambda_\pm=1\pm\sqrt2,
\qquad
\eta=\left|\frac{\lambda_-}{\lambda_+}\right|=(\sqrt2-1)^2 .
$$

Thus $\eta$ is the per-step contraction ratio between the subdominant and
dominant transfer channels, and radial masses are integer powers of $\eta$:
$m_g=A_g\,\eta^{k_g}v$. The irrational is the eigenvalue ratio of an integer
matrix, not a fit. `C:9` for the eigenvalue theorem; `C:7` for reading depth as
Pell contraction once $T_{\rm Pell}$ is accepted.

**Caveat on $T_{\rm Pell}$ (see §5.4).** $T_{\rm Pell}$ ($\det=-1$) is the
*arithmetic shadow* of the silver contraction, not the microscopic propagator.
The microscopic radial transfer derived from the pinned BB coin is
orientation-*preserving* ($\det=+1$, eigenvalues $\sqrt2\pm1$) and is conjugate
to the neutrino sterile chain; $T_{\rm Pell}$ shares only the eigenvalue
magnitude $\epsilon=\sqrt2-1$. Use $T_{\rm Pell}$ for bookkeeping integer depths,
not as the physical walk.

### 5.3 The microscopic origin question — current status

- A flat BCC coordinate face keeps **4** body-diagonal channels
  ($N_{\rm raw}=4$ ⇒ $x_4=\sqrt5-2$), **not** silver. `C:1` for the flat-face
  silver claim.
- A codimension-two coordinate edge keeps **2** raw channels — the right integer
  ($N_{\rm raw}=2$) — but the full edge graph **leaks**:
  $[n,n]\to 2[n-1,n-1]+4[n-1,n+1]+2[n+1,n+1]$, and the bare BB Weyl coin has no
  spinor subspace killing both leakage channels
  ($\ker L_1\cap\ker L_2=\{0\}$, exact). `C:9` for the leakage/kernel facts;
  `C:1` for "edge count $\Rightarrow\epsilon$" directly.

**Open (`C:3`):** derive the reduced scalar Dyson equation $x=1/(2+x)$ from a
physical no-leakage scar projection on the BCC edge. This is a sharp,
falsifiable target.

### 5.4 Candidate boundary rule: synchronous-normal ($q$) superselection

A candidate rule sharpens the no-leakage projection of §5.3 into a conserved
quantity. At a codimension-two BCC edge with inward normal depths $(r_1,r_2)$, a
body-diagonal hop $(\sigma_1,\sigma_2,\sigma_3)$ acts by
$(r_1,r_2)\mapsto(r_1+\sigma_1,r_2+\sigma_2)$. Define the relative-depth
coordinate $q=r_1-r_2$. Then

$$
q\mapsto q+(\sigma_1-\sigma_2)\quad\Rightarrow\quad
\sigma_1=\sigma_2\ \Longleftrightarrow\ \Delta q=0 .
$$

So the diagonal scar $q=0$ is preserved exactly by the same-normal-sign hops and
left by the mixed-sign hops. The proposed **synchronous-normal repair rule** is
that the boundary superselects (or energetically penalizes) the relative-depth
mode $q$, admitting only $\sigma_1=\sigma_2$ and projecting mixed hops into the
unresolved bath. This *names the conserved quantity* behind the no-leakage
projection — a strict upgrade over "two channels survive." `C:7` as the
geometric content; it identifies what must be conserved.

After this projection, the two tangential hops $\sigma_3=\pm1$ are summed
coherently at zero edge momentum ($k_3=0$). In the repository's pinned BB
convention ($q_\pm=(1\pm i)/4$, `bcc_weyl.py`) the two admitted normal channels
give the rank-one radial blocks

$$
B_+=\tfrac14\!\begin{pmatrix}1+i&1-i\\1+i&1-i\end{pmatrix},\quad
B_-=\tfrac14\!\begin{pmatrix}1+i&-1+i\\-1-i&1-i\end{pmatrix},\quad
B_+^\dagger B_+ + B_-^\dagger B_- = \tfrac12 I .
$$

So the $q=0$ scar is a **subunitary no-leakage branch with survival norm
$1/\sqrt2$** (the mixed-normal leakage carries the other half). `C:8` (verified
in the canonical convention).

**The microscopic transfer is the orientation-preserving BB walk ($\det=+1$),
not the Pell/impedance shadow ($\det=-1$).** The integer $2$ must *not* be read
as a Hermitian radial hopping (that reading hits $\epsilon$ only at the non-clean
$z=5\sqrt2-3\approx4.07$ — dead). The correct object is the BB *wave* transfer.
In the symmetric/antisymmetric ($e,o$) spinor basis the rank-one blocks are
bipartite-triangular, $B_+=\left(\begin{smallmatrix}\tfrac12&\tfrac i2\\0&0\end{smallmatrix}\right)$,
$B_-=\left(\begin{smallmatrix}0&0\\\tfrac i2&\tfrac12\end{smallmatrix}\right)$,
and the stationary scar equation
$\zeta\psi_r=B_+\psi_{r-1}+B_-\psi_{r+1}$ reduces (index-shifted) to a one-step
radial transfer

$$
T(\zeta)=\begin{pmatrix}\tfrac1{2\zeta}&\tfrac{i}{2\zeta}\\[2pt]
-\tfrac{i}{2\zeta}&2\zeta+\tfrac1{2\zeta}\end{pmatrix},
\qquad
\det T(\zeta)=1,\qquad \operatorname{tr}T(\zeta)=2\zeta+\tfrac1\zeta .
$$

The determinant is $+1$ identically — orientation-*preserving*, forced by the
bipartite-triangular structure. `C:9` for the algebra, `C:8` for the BB-scar
derivation in the pinned convention.

**Silver sits at the trace minimum, which is the survival norm.** By AM–GM,
$\operatorname{tr}T(\zeta)=2\zeta+1/\zeta$ is minimized at $\zeta=1/\sqrt2$ — the
*same* $1/\sqrt2$ as the branch survival norm — with minimum value *exactly*
$2\sqrt2$. At that point

$$
T(1/\sqrt2):\quad \det=1,\ \operatorname{tr}=2\sqrt2,\quad
\text{eigenvalues }\sqrt2\pm1,\quad \boxed{\epsilon=\sqrt2-1.}
$$

(The gauged real form is $\sqrt2\,T=\left(\begin{smallmatrix}1&-1\\-1&3\end{smallmatrix}\right)$.)
So the evaluation point is not arbitrary: survival norm, trace minimum (band
edge), and silver eigenvalue coincide.

**Exact unification with the neutrino sector.** The neutrino sterile-chain
transfer at the probe $z=2\sqrt2$ is
$\left(\begin{smallmatrix}2\sqrt2&-1\\1&0\end{smallmatrix}\right)$: $\det=1$,
$\operatorname{tr}=2\sqrt2$, eigenvalues $\sqrt2\pm1$ — the *same characteristic
polynomial* as the BB scar transfer. The BCC $q=0$ scar and the neutrino chain
are therefore **conjugate**: one silver universality class,

$$
\boxed{\;\det M=+1,\quad \operatorname{tr}M=2\sqrt2,\quad \text{eigenvalues }\sqrt2\pm1.\;}
$$

`C:8` for the shared class.

**Status of the Pell/impedance forms.** The Pell matrix
$\left(\begin{smallmatrix}2&1\\1&0\end{smallmatrix}\right)$ ($\det=-1$,
$\operatorname{tr}=2$) and the impedance map $\Sigma=1/(2+\Sigma)$ ($\det=-1$)
share only the eigenvalue *magnitude* $\sqrt2-1$, with an orientation flip. They
are the **integer/continued-fraction arithmetic shadow** of the silver
contraction, useful for bookkeeping radial depths, *not* the microscopic
propagator. (This retires the earlier impedance-primary and chiral-staggering
proposals; the chiral coin reverts to a CP-only role, §12.) The silver ontology:

| Object | $\det$ | role |
|---|---:|---|
| BB $q=0$ scar transfer (pinned coin) | $+1$ | microscopic origin |
| neutrino sterile chain at $z=2\sqrt2$ | $+1$ | same class, clean model |
| Pell matrix / impedance map | $-1$ | arithmetic shadow, bookkeeping |

**Remaining load-bearing gaps.** (a) Derive the $q$-superselection
($\sigma_1=\sigma_2$) from the BB coin/defect Hamiltonian rather than imposing it
(`C:7` geometric, not yet dynamical). (b) Justify reading the radial mass factor
as the decaying eigenvalue of $T(\zeta)$ evaluated at the survival/trace-minimum
$\zeta=1/\sqrt2$ — well-motivated (band edge $=$ survival norm) but the
interpretive choice, `C:5`–`C:6`.

---

## 6. The family-depth skeleton {0,2,6}

### 6.1 The object

The quark depth spectrum is $D=\{0,2,6\}$, realized as the doubled path
Laplacian on the path $u$–$a$–$b$:

$$
D_{\rm scar}=2\Delta(P_3),\qquad
\Delta(P_3)=\begin{pmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{pmatrix},
\qquad
\operatorname{Spec}D_{\rm scar}=\{0,2,6\}.
$$

Normalized modes: $\psi_0=\tfrac1{\sqrt3}(1,1,1)$,
$\psi_2=\tfrac1{\sqrt2}(1,0,-1)$, $\psi_6=\tfrac1{\sqrt6}(1,-2,1)$. `C:9`.

### 6.2 Triple identity (anti-rediscovery lesson)

The same object has three equivalent faces:

$$
\text{path Laplacian}\;\Longleftrightarrow\;
\text{length-3 nilpotent flag } N=\!\begin{psmallmatrix}0&1&0\\0&0&1\\0&0&0\end{psmallmatrix}\!,\ N^3=0
\;\Longleftrightarrow\;
S_3\to Z_2\ \text{repair scar}.
$$

A later theorem that merely rebuilds the nilpotent flag is **not** a new depth
mechanism. `C:9`.

### 6.3 The hard no-go and why depth = the generation problem

By Schur's lemma, the residual $S_3$ representation decomposes $3=1\oplus2$, so
**any** $S_3$-invariant depth operator has spectrum $\{\alpha,\beta,\beta\}$ —
at most two distinct eigenvalues. The complete-graph control gives
$\operatorname{Spec}(2\Delta(K_3))=\{0,6,6\}$, not $\{0,2,6\}$. `C:9`.

Therefore deriving $\{0,2,6\}$ is **equivalent to deriving an $S_3$-breaking
family spurion**:

$$
D_{\rm depth}-\tfrac83 I\sim(-4,-1,5).
$$

The raw BB hop/Walsh route to the depth ladder is killed: the hop source
contains a forbidden quadrupole ($E$ in the covariant $O$ decomposition,
$T_{2g}$ in coefficient-Walsh), so it is $A_1\oplus A_2\oplus E$, not the wanted
$A_1\oplus T_1\oplus A_2$. `C:1` for raw BCC hop/Walsh depth selection; `C:8` for
the kill; `C:6` for the path scar as a conditional physical origin.

### 6.4 Conditional support selection

A finite census shows: among directed two-edge length-3 nilpotent supports on
three ports, exactly six survive ($N^3=0$, $N^2\ne0$, rank 2), all one $S_3$
orbit (the path). A height filtration $h(u)=0,h(a)=1,h(b)=2$ with one-tick
locality and a BCC parity veto forbids the shortcut $b\to u$ and selects the
path; unique active successors force no-leakage and unit weights. A V12
certificate verifies this on a 12-state modeled basis (24 active rows, one
allowed successor each). `C:8` for the finite certificate; `C:6` pending a proof
that the candidate basis is complete and that microscopic repair minimizes
directed edge count.

### 6.5 The generation number is a primitive boundary axiom, not a theorem

We state the program's honest posture on $N_{\rm gen}=3$ explicitly, because it is
the load-bearing methodological choice. **We do not claim to derive
$N_{\rm gen}=3$** — not from the Clifford carrier (which gives *one* generation,
§2.3), not from the free BCC walk, not from the three spatial dimensions, and not
from the three QCD colors. Those recurring trivalent structures —

$$
3\ \text{spatial axes},\qquad 3\ \text{colors},\qquad 3\ \text{residual family ports}
$$

— are **consistency clues, not proofs**. (The §9 count "$4$ tetrahedral exits
$-\,1$ vacuum-framed $=3$" is likewise a geometric *motivation*, conditional on
the framing dynamics, not a derivation of *exactly* three.) Treating any of them
as the proof of $N_{\rm gen}=3$ is too easy to attack.

The clean position is the axiom

$$
\boxed{\;\dim\mathcal H_{\rm fam}=3\quad\text{(primitive empirical boundary fact)}.\;}
$$

This is the same kind of move the Standard Model itself makes: take the gauge
group and matter content as input, then *derive relations* from symmetry and
dynamics. With the three-port flag $P_3$ accepted, the program's job becomes the
sharper, conditional, testable question:

$$
\boxed{\;\text{given three residual boundary ports, derive their hierarchy, residues, and mixings.}\;}
$$

Everything downstream (the path nilpotent $N^3=0$, the silver depths, the
weak-framed doublet $\mathcal B=(N,N^\dagger)$, the up/down split, the Clebsches,
the CKM hierarchy) is then a *conditional theorem* on this axiom — which is a
respectable and defensible posture. The attempt to go *beyond* the axiom and
actually derive the $3$ is isolated, speculative, and clearly graded `C:3`
(§11.8). Certainty of the axiom itself: `C:5` as a physical input — motivated,
not derived.

---

## 7. The scalar Clebsch layer

The scalar mass coefficients must be separated from coherent current amplitudes
(the same radical can mean different things: $\sqrt2$ as a two-path current
enhancement vs. $1/\sqrt2$ as a normalized scalar repair amplitude). `C:8` for
the separation.

### 7.1 Up sector (holomorphic Taylor entry)

Using the nilpotent flag, $\exp(xN)=I+xN+\tfrac{x^2}2N^2$, the top-normalized
grade readout is $C_u(x)=(\tfrac{x^2}2,x,1)$. The two-channel no-leakage isometry
theorem ($|\alpha_+|^2+|\alpha_-|^2=1$, symmetry $\Rightarrow|\alpha_\pm|=1/\sqrt2$)
gives $x=1/\sqrt2$, hence

$$
C_u=\left(\tfrac14,\tfrac1{\sqrt2},1\right)\quad(\text{light-to-heavy}).
$$

`C:9` for the Taylor algebra once $x$ is fixed; `C:6` for $x=1/\sqrt2$ from the
no-leakage repair. The old vector $(\tfrac12,\sqrt2,1)$ is **killed** as a scalar
profile (middle $>$ top), `C:1`.

### 7.2 Down sector (Hermitian shell residues)

The primitive quark shell is $S_q=1_{\rm direct}+2_{\rm BCC}+3_{\rm color}$,
$|S_q|=6$, with scalar overlap $C_i=\sqrt{n_i/6}$. The $S_3$ Cayley Laplacian
(transposition generators) has spectrum $\{0,3,3,3,3,6\}$ and central isotypic
port weights $w_{\rm triv}=w_{\rm sign}=\tfrac16$, $w_{\rm std}=\tfrac46$, giving
the exact bottom coefficient $C_b=\sqrt{4/6}=\sqrt{2/3}$. The natural baseline is

$$
C_{d,\rm base}=\left(1,\tfrac1{\sqrt3},\sqrt{\tfrac23}\right),\quad (n)=(6,2,4).
$$

`C:9` for the spectrum/weights; `C:7` for the clean bottom residue once bottom is
assigned to the standard isotypic block.

The data-improved rank-5 candidate $C_{d,\rm cand}=(1,\tfrac1{\sqrt3},\sqrt{5/6})$,
$(n)=(6,2,5)$, is **representation-theoretically available** ($I_{\rm reg}$ minus
one 1-dim line) but **not forced**: $S_3$ does not choose which line to exclude,
and the rank-2 strange copy needs a defect-polarized standard copy, not central
$S_3$. `C:9` for availability/non-uniqueness; `C:4` for it as a derivation.

### 7.3 Phenomenology constraint (one-scale rule)

A Yukawa matrix lives at one renormalization scale; a mixed-scale six-mass eye
test is illegitimate. With common-scale QCD running the old texture overshoots
non-top masses by a coherent $\sim1.5$–$2$. The robust surviving signal is the
**shared $\eta$**: $\eta_{\rm CKM}\approx0.17209$ vs. $\eta_{\rm silver}\approx0.17157$
(within $0.3\%$), plus light-sector ratios
$m_d/m_s=\sqrt3\,\eta^2$, $m_s/m_b=\eta^2/\sqrt2$, $m_u/m_c=\eta^3/(2\sqrt2)$.
The live up-Clebsch stress is $1/\sqrt2\approx0.707$ (theory) vs. $3/4$ (data
control) for the charm coefficient; the old $\sqrt2$ charm coefficient is killed
(`C:1`). The down overshoot may be one sector normalization $r_d\sim0.6$ rather
than a family-shape Clebsch. `C:5`–`C:6`.

---

## 8. Radial mass response and the spectral no-go

Finite symmetry specifies channels, projectors, couplings, and transfer grammar.
The actual mass values live in a spectral measure:

$$
\Sigma_f(z)=\sum_i\frac{w_i}{z-\lambda_i},\qquad w_i>0\ \text{(positive Stieltjes)}.
$$

The target up/down textures are positive measures and reconstruct unique finite
Jacobi baths by the inverse spectral problem (moments round-trip). `C:8` for the
reconstruction.

**Pole/residue rigidity no-go (`C:9`, one of the most valuable results).** Two
baths sharing the same triality-head data ($\alpha_+=\alpha_-=1/\sqrt2$) and the
same allowed successor labels have *different* self-energies:
$\Sigma_1(z)=1/z$ vs. $\Sigma_2(z)=(z-1)/(z^2-z-1)$ — different poles and
residues. Therefore finite $S_3$ grammar + two-channel no-leakage + inherited
silver transfer are **insufficient** to determine physical masses. This is
reconstruction, not forward derivation. `C:4` for the target measures as
physically derived.

The closure lengths that set the depth slopes are *not* Cayley word lengths (a
3-cycle has length 2, a transposition length 1, giving $2:1$, wrong). They come
from repair-rule closure: holomorphic $Z_3$ closes at $\sigma^3=e$ (slope 3),
Hermitian $Z_2$ at $\tau^2=e$ (slope 2). Current target exponents:
$k_u=(6,3,0)$, $k_d=(6,4,2)$. `C:9` for closure lengths under the stated rules;
`C:5` as physical depth increments.

---

## 9. The neutrino prototype (the cleanest theorem)

This is the place the philosophy works without stretching: **geometry selects
channels, symmetry constrains projectors, recirculation sets eigenvalues.**

### 9.1 Geometry → three residual ports

The BCC QCA has eight body-diagonal exits $(\pm1,\pm1,\pm1)$; antipodal pairing
gives four primitive exits forming a regular tetrahedron
($v_i\cdot v_j=-1/3$, $i\ne j$). Selecting one vacuum-framed exit leaves **three**
residual exits with adjacency $K_3$ and stabilizer $S_3$
($S_4\supset\mathrm{Stab}(v_{\rm sel})\simeq S_3$). `C:8` for the finite orbit
audit; `C:6` for the physical vacuum selector. **Scope (cf. §6.5):** this
*explains the three-port basis and its $S_3$ structure given a selected exit*; it
is a geometric motivation for $\dim\mathcal H_{\rm fam}=3$, **not** a derivation
that the universe has exactly three generations — the "$4-1=3$" count is itself
conditional on the framing dynamics. The existence of the three-port flag remains
the §6.5 axiom.

### 9.2 Symmetry → framing is necessary

$K_\nu=\epsilon^2P_u+P_b$ assigns different weights to $a$ and $b$, but every
full-$S_3$-invariant operator is $\alpha P_u+\beta(P_a+P_b)$ and cannot split
$a$ from $b$. So the unframed $K_3$ tail is killed; framing to the selected-port
$S_2$ is mathematically necessary. `C:9` for the centralizer obstruction.

### 9.3 Recirculation → the eigenvalues

Product sterile bath $H_Q=H_{\rm chain}\otimes I_{\rm family}$ keeps $u,b$
returns orthogonal (zero cross-return by tensor structure). The normalized Schur
response is $\widehat\Sigma(z)=m(z)^2P_u+P_b$, and at the probe $z=2\sqrt2$,

$$
\boxed{\;K_\nu=\widehat\Sigma(2\sqrt2)=\epsilon^2P_u+P_b\;}
$$

(the square: the collective $u$ channel is one transfer step deeper). The light
effective operator $M_\nu=\mu_\nu K_\nu$ has, in basis $(a,u,b)$,
$M_\nu=\mu_\nu\,\mathrm{diag}(0,\epsilon^2,1)$, hence

$$
\boxed{\;(m_1,m_2,m_3)=\mu_\nu(0,\,\eta,\,1),\qquad
\frac{\Delta m^2_{21}}{\Delta m^2_{31}}=\eta^2=\epsilon^4
=17-12\sqrt2\approx0.02944.\;}
$$

`C:9` for the exact product-model theorem; `C:8` for the mass-squared ratio
inside the effective operator; `C:7` physically until the product bath is
derived microscopically.

### 9.4 Predictions and posture

Normal ordering; one massless light neutrino; the zero is **non-coupling**
(stable), not cancellation. Setting the scale from one measured splitting
($\mu_\nu=\sqrt{\Delta m^2_{31}}$, since $m_1=0,m_3=\mu_\nu$): with
$\Delta m^2_{31}=2.52\times10^{-3}\,{\rm eV}^2$,

$$
m_1=0,\quad m_2\approx8.6\,{\rm meV},\quad m_3\approx50.2\,{\rm meV},\quad
\sum m_i\approx58.8\,{\rm meV},\quad
\Delta m^2_{21}\approx7.4\times10^{-5}\,{\rm eV}^2,
$$

all consistent with oscillation data. The absolute scale $\mu_\nu$ is **not**
internally derived (`C:3`).

### 9.5 PMNS (adjacent, conditional)

$U_{\rm PMNS}=R_e^\dagger U_{\rm TBM}$ with $U_{\rm TBM}=[a\,u\,b]$. Charged-lepton
leakage (two Weyl-transfer steps from the selected port
$e_1=\sqrt{2/3}\,a+\tfrac1{\sqrt3}u$) gives $\sin\theta_e=\sqrt{3/2}\,\epsilon^2$,
hence $\sin^2\theta_{13}=\tfrac34\epsilon^4$; a conditional phase word
$\phi_e=-5\pi/12$. `C:9` for TBM basis and the port decomposition; `C:6` for the
conditional PMNS texture.

**This is the prototype standard for all other mass sectors:** identify
$(H_Q,V)$ and the resolvent first; read masses as its spectral data; do not guess
a texture.

---

## 10. Charged leptons

### 10.1 Different readout from neutrinos

Charged leptons are colorless but **active** (Higgs-coupled, $\bar L_L H e_R$),
so masses are read as **squared active boundary amplitudes**, not sterile
eigenvalues:

$$
w_e=(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau}),\qquad m_i=|w_{e,i}|^2 .
$$

### 10.2 Koide as trace/plane equipartition (`C:9` equivalence)

Koide's relation $K=\dfrac{\sum m_i}{(\sum\sqrt{m_i})^2}=\dfrac23$ is **exactly**
$(v\cdot n)^2/|v|^2=1/2$ with $n=u$ — i.e. $w_e$ lies on the $45^\circ$ cone
around the trace axis, $|w_t|^2=|w_\perp|^2$. Empirically $K_{\rm PDG}=0.666661$
(deviation $\sim6\times10^{-6}$). The exact $Z_3$-equivariant locus gives a
degenerate pair ($\lambda_2=\lambda_3$) and so cannot reproduce the distinct
physical masses — Koide is **consistent, not predicted** by exact equivariance
(`C:8`); $Z_3$-breaking input is required.

### 10.3 The active residue construction

Write the residue-amplitude operator on the residual space as

$$
B_e=\rho\big(\sqrt2\,P_u+R_\theta P_\perp\big),\qquad P_\perp=P_a+P_b,
\qquad w_e\propto B_e e_1 .
$$

The selected port $e_1=\tfrac1{\sqrt3}u+\sqrt{\tfrac23}\,a$ has trace/plane
amplitude ratio $1/\sqrt2$. A coherent two-path BCC trace enhancement
($1+1\to\sqrt2$) makes the trace and plane amplitudes equal, giving
$\widehat w_e=\tfrac1{\sqrt2}u+\tfrac1{\sqrt2}R_\theta a$ — i.e. Koide
equipartition automatically (`C:6`; the load is the coherent two-path premise).

The plane angle is reduced to a base $Z_3$ direction plus a small torsion:

$$
\theta_0=\phi_{\rm loop}-\phi_{A3}=-\tfrac{5\pi}{12}-\tfrac{\pi}{4}=-\tfrac{2\pi}{3},
\qquad
\boxed{\;\theta_e=\theta_0-\delta_e=-\tfrac{2\pi}{3}-\tfrac29.\;}
$$

With $\mu_e=m_e+m_\mu+m_\tau$ as the one scale, $m_i=\mu_e|\widehat w_{e,i}|^2$
reproduces the masses at max relative error $6.7\times10^{-5}$, with Koide exact
by construction. Neighboring torsions ($0,\tfrac19,\tfrac13$) all fail badly.
`C:8` for the calculation; `C:3`–`C:4` for the torsion as a forward prediction.

### 10.4 Minimal boundary Hamiltonian and the two-sided-kernel obstruction

A one-sided Hermitian self-energy **cannot** produce the rotation $R_\theta$: its
pole residue $V^\dagger P_\lambda V\ge0$ is positive semidefinite, while $R_\theta$
is orthogonal. The charged-lepton object must be a **chiral two-sided Yukawa
Schur kernel** $B_e(z)=V_R^\dagger(z-H_{Q,e})^{-1}V_L$. A minimal 4-state
realization ($H_{Q,e}=\lambda I_4$, explicit $V_L,V_R$ with the $\sqrt2$ trace
paths and the plane rotation) gives the residue exactly
$\rho(\sqrt2P_u+R_\theta P_\perp)$. `C:9` algebra; `C:8` obstruction.

### 10.5 New result: the torsion is a coupling-frame holonomy, not a spectral phase

We checked the proposed mechanism "the $1/z^2$ moment $V_R^\dagger H_{Q,e}V_L$
produces the torsion $2/9$" by direct symbolic computation. Findings:

1. **Flat bath kills it.** With $H_{Q,e}=\lambda I_4$, every Laurent moment is
   $M_n=V_R^\dagger H_{Q,e}^n V_L=\lambda^n V_R^\dagger V_L$ (verified
   $M_n-\lambda^n M_0=0$ for $n=0,1,2$). The $1/z^2$ term carries the **same**
   hand-inserted angle $\theta$; it does **not** independently generate $2/9$.
2. **No Hermitian bath generates the real rotation.** A plane-block
   $\lambda_pI_2+\omega\sigma_y$ produces an *imaginary* (Hermitian-phase)
   off-diagonal, not the real orthogonal $R_\theta$. The real rotation requires
   $V_R\ne V_L$ — it is a **holonomy between the left (source) and right (return)
   coupling frames**, independent of the bath spectrum. Hence "derive $2/9$ from
   $H_{Q,e}$" is structurally impossible in this class.
3. **Where $2/9$ does live, exactly.** The selected port occupations are
   $p_u=|\langle u,e_1\rangle|^2=\tfrac13$, $p_a=|\langle a,e_1\rangle|^2=\tfrac23$,
   and the **joint second-order return weight is $p_u p_a=\tfrac29$ exactly**,
   with first-order matrix element $\sqrt{p_u p_a}=\sqrt2/3$. The factor
   $p_a=2/3$ is **shared with the reactor angle** ($\sqrt{2/3}\sin\theta_e=\epsilon^2$).

So the corrected open target is **not** "find $H_{Q,e}$" but: derive the
left/right frame twist $V_R=V_L e^{G}$ with $G$ a port-projected second-order
return generator of weight $p_u p_a=2/9$. `C:9` for the moment identity and the
$p_u p_a=2/9$ arithmetic; `C:3` for the holonomy as the physical mechanism.

---

## 11. Quarks and CKM

### 11.0 One backbone for all flavor

With the silver origin now the orientation-preserving BB $q=0$ transfer (§5.4),
the program no longer needs a separate quark silver mechanism. **Every flavor
hierarchy uses the same BB/Weyl $q=0$ transfer**; sectors differ only by the
*boundary readout*:

| Sector | Common silver source | Readout |
|---|---|---|
| neutrinos | BB/Weyl $q=0$ transfer | sterile one-sided eigenresponse (§9) |
| charged leptons | BB/Weyl $q=0$ transfer | active two-sided chiral Koide residue (§10) |
| up quarks | BB/Weyl $q=0$ transfer | colored active coherent/holomorphic Taylor residue |
| down quarks | BB/Weyl $q=0$ transfer | colored active Hermitian shell residue |
| CKM | BB/Weyl $q=0$ transfer | relative quark-frame holonomy |

The compressed statement: **$\epsilon=\sqrt2-1$ is universal; $\eta=\epsilon^2$
is the fermion mass-return unit** (a Dirac mass is a left–right boundary return,
hence two transfer steps); **sector physics lives in residues and holonomies.**
`C:6` as the organizing backbone.

### 11.1 Common spine and family depth

All quark masses take the form $m_{q_i}=\mu_q\,C_{q_i}\,\eta^{k_{q_i}}$ with the
same $\eta$. Order the three families by an ordinal scar depth $n=(2,1,0)$
(light→heavy) and impose the two closure laws from the repair type
(§8: holomorphic $Z_3$ closes at length 3, Hermitian $Z_2$ at length 2):

$$
k_u=3n=(6,3,0),\qquad k_d=2(n+1)=(6,4,2).
$$

This reproduces the manuscript's exponents and gives the down "$+2$ floor" a
reading: the irreducible $Z_2$ baseline return ($n{+}1$), even for the bottom.
`C:7` for the exponents given the repair-type closure lengths; `C:3` for the
ordinal scar $n$ itself (= the generation problem, §6).

### 11.2 Up sector — coherent Taylor residue

Holomorphic entry through the nilpotent flag ($N^3=0$),
$\exp(xN)=I+xN+\tfrac{x^2}2N^2$, with $x=1/\sqrt2$ gives the light→heavy profile
$C_u=(\tfrac14,\tfrac1{\sqrt2},1)$, so

$$
(m_u,m_c,m_t)=\mu_u\!\left(\tfrac14\eta^6,\ \tfrac1{\sqrt2}\eta^3,\ 1\right),
\qquad
\frac{m_u}{m_c}=\frac{\eta^3}{2\sqrt2},\quad \frac{m_c}{m_t}=\frac{\eta^3}{\sqrt2}.
$$

Numerically $m_u/m_c\approx1.79\times10^{-3}$, $m_c/m_t\approx3.57\times10^{-3}$ —
both in the observed band (inputs, scheme, scale, and RG-stability in
**Appendix D**; note $m_c/m_t$ is the scale-sensitive entry). **Plausible new
unification (`C:4`–`C:5`):** the entry
amplitude $x=1/\sqrt2$ may be the *same* BB scar survival norm $1/\sqrt2$ that
sets the silver transfer (§5.4) — both are "one of two coherent channels." Not
yet proven identical, but no longer an unrelated input.

### 11.3 Down sector — Hermitian shell residue

The primitive quark shell $S_q=1_{\rm direct}+2_{\rm BCC}+3_{\rm color}$,
$|S_q|=6$, gives scalar overlaps $C_i=\sqrt{n_i/6}$ and the baseline profile
$C_d=(1,\tfrac1{\sqrt3},\sqrt{2/3})$:

$$
(m_d,m_s,m_b)=\mu_d\!\left(\eta^6,\ \tfrac1{\sqrt3}\eta^4,\ \sqrt{\tfrac23}\eta^2\right),
\qquad
\frac{m_d}{m_s}=\sqrt3\,\eta^2,\quad \frac{m_s}{m_b}=\frac{\eta^2}{\sqrt2}.
$$

Numerically $m_d/m_s\approx0.051$, $m_s/m_b\approx0.021$ — both observed-band
(Appendix D).
These residues are Hermitian shell overlaps, *not* Taylor coefficients: the
coherent/Hermitian readout split is exactly the up/down distinction.

### 11.4 CKM as quark-frame holonomy

With both sectors on the same residual family space, CKM is not a third object;
it is the relative holonomy of the two quark Schur frames:

$$
V_{\rm CKM}=U_{u,L}^\dagger U_{d,L},\qquad
Y_{u,d}\sim\mu_{u,d}\,U_{u,d,L}^\dagger\,\mathrm{diag}(C_{u,d}\,\eta^{k_{u,d}})\,U_{u,d,R}.
$$

The same $\eta$ sets the angle hierarchy (verified):
$s_{12}=\tfrac43\tfrac{\eta}{\sqrt{1+\eta^2}}\approx0.225$,
$s_{23}=\sqrt2\,\eta^2\approx0.0416$, $s_{13}=\eta^3/\sqrt2\approx0.00357$,
conditional phase $\delta_q=\arctan\sqrt5\approx65.9^\circ$ — all on the observed
CKM values. `C:5`–`C:6` for the angle powers; the coefficients ($\tfrac43$,
$\sqrt2$, $\arctan\sqrt5$) still need a loop/coin derivation.

### 11.5 The single remaining dynamical premise

The premise is best stated not as a bare label but as a **weak-framed repair
doublet conjecture** (the finite form, certified in §11.7):

$$
\boxed{\;\text{a boundary doublet } \mathcal B=(N,\,N^\dagger)\ \text{(oriented path scar and its reverse)}\;}
$$

with the conjugate Higgs $\tilde H=i\sigma_2H^*$ selecting the oriented nilpotent
$N$ (coherent Taylor up profile) and the direct Higgs $H$ selecting $N^\dagger$,
whose paired Hermitian closure generates the regular $S_3$ shell (down residues).
The old "up = coherent, down = Hermitian" is then the *consequence* of which
component each Higgs door contracts, not an independent postulate. The existence
and gauge-covariance of $\mathcal B$ is the central remaining hypothesis (§11.6–7).

**Subclaim grading (canonical — supersedes any single label elsewhere).** The
up/down "bit" is not one claim but a stack; grading it as a single number is
misleading. The honest decomposition:

| Subclaim | Grade | Where |
|---|---|---|
| Gauge doors ($\tilde H\!\to\!u$, $H\!\to\!d$ by hypercharge) | `C:9` | §11.0, DPL |
| Path-locality ($[D,X]=X\Rightarrow X=aE_1+bE_2$) | `C:9` | §6.5, §11.7 |
| Equal-strength $|a|=|b|$ (chiral-reflection / cover homogeneity) | `C:7` (conditional) | §11.7 |
| Orientation doublet $\mathcal B=(N,N^\dagger)$ from $J_H=i\sigma_2K$ | `C:8` (conditional on weak-framed defect) | §11.6–7 |
| Coherent-vs-Hermitian *type* split | `C:4` | §11.6 |
| Down $(6,2,4)$ via $3\to6$ regular-rep promotion | `C:4` | §11.6 |
| **Full up/down mechanism** (all of the above, microscopic) | **`C:3`–`C:5`** | §11.5–8 |

Where any earlier or later passage prints a single bare label for "the up/down
bit," this table is the intended reading.

The premise (in either form) is **not** forced by $S_3$ (its one sign grading $A_3\triangleleft S_3$ is
already used for even/odd repair; no second grading encodes weak isospin) **nor**
by electroweak symmetry (hypercharge forces $u\leftrightarrow\tilde H$,
$d\leftrightarrow H$ — a conjugation, not coherent-vs-Hermitian dynamics, since
both Yukawas flip chirality). `C:8`–`C:9` for both obstructions. A rank audit of
the existing Higgs slot shows rank-one building blocks but a **two-dimensional**
active plane (neutral static Yukawa control has rank 2), so no unique zero-mode
line is forced (`C:8`). The premise is falsifiable: if microscopic dynamics makes
both sectors coherent, both carry Taylor coefficients; etc. This — together with
the ordinal scar $n$ (§6), the down shell→$(d,s,b)$ assignment, and the CKM
coefficient/phase derivation — is what remains genuinely open in the quark
sector.

### 11.6 Physical story and design principles for the dynamical bit

This subsection records the *target* for deriving the §11.5 premise, not a
derivation. Stating the physics first guards against building an operator that
merely encodes the desired answer.

**The picture (one paragraph).** A quark mass event is a closed boundary
excursion of a colored chiral fermion through a Higgs-framed defect:
$Q_L\to$ boundary $\to q_R$. The universal BCC $q=0$ scar (§5.4) supplies the
radial attenuation $\eta$; the family path scar (§6) supplies the three depths;
the Higgs orientation acts as a **boundary frame selector** that fixes the
*class of closure*; the quark returns with a residue and a frame rotation. Masses
are singular values $m_{q_i}\sim v\,s_i(Y_q)$; CKM is the relative left-frame
holonomy $U_{u,L}^\dagger U_{d,L}$; CP needs an extra loop (a tree has no
phase, §12).

**Design axioms**, tagged by what they currently are:

1. *Universal silver* — all fermion hierarchies use the one BB $q=0$ transfer
   ($\eta$). **Established** (§5.4), `C:8`.
2. *Chiral two-sided Schur* — $Y_q=V_R^\dagger(z-H_Q)^{-1}V_L$, Hermitian bath but
   non-Hermitian kernel, masses as singular values. **Established class** (§10),
   `C:8`. (CKM *angles* need only $U_{u,L}\ne U_{d,L}$; a non-rephasable CKM
   *phase* needs non-normality $[Y,Y^\dagger]\ne0$.)
3. *Gauge covariance* — $\tilde H$ opens the up door, $H$ the down door. Fixes
   *which* channels exist, not the hierarchy. `C:9`.
4. *Color neutrality* — $Y_q=I_{\rm color}\otimes Y_q^{\rm family}$; color may
   count hidden return channels (the down shell) but may not split visible color.
   `C:9` as a filter.
5. *Family spurion* — the $S_3\to Z_2$ path scar is mandatory (else $3=1\oplus2$,
   two eigenvalues). `C:9` (no-go); origin `C:3` (§6).
6. *Closure* — a mass is a closed return word; exponents $=$ depth $\times$
   closure length, $k_u=3n$, $k_d=2(n{+}1)$, $n=(2,1,0)$. `C:7` given the
   closure lengths.
7. *Orientation split* (**the bit, `C:3`**) — $\tilde H\Rightarrow$ oriented
   coherent closure $\Rightarrow$ nilpotent/Taylor residue $(\tfrac14,\tfrac1{\sqrt2},1)$;
   $H\Rightarrow$ paired Hermitian closure $\Rightarrow$ positive shell residue
   $(1,\tfrac1{\sqrt3},\sqrt{2/3})$.
8. *Tree/loop separation* — hierarchy from the path scar, CP phase from a loop or
   frame holonomy. `C:9` (no tree phase).
9. *Positivity contrast* — down residues are *positive shell weights*
   ($C_i=\sqrt{w_i}$, $w_i\ge0$); up residues are *coherent amplitudes* (may
   interfere before squaring). A genuine, checkable structural distinction, `C:5`.
10. *Minimality* — one bath $H_Q$, two projections $\Pi_{\tilde H},\Pi_H$, not
    two arbitrary baths: $Y_{u}=V_{u_R}^\dagger\Pi_{\tilde H}(z-H_Q)^{-1}\Pi_{\tilde H}V_{Q_L}$,
    $Y_d=V_{d_R}^\dagger\Pi_H(z-H_Q)^{-1}\Pi_H V_{Q_L}$. `C:5` as the target form.

**The mechanism to test (`C:3`, the one path from assertion to derivation).**
Axiom 7 is the entire bit. The only structural difference between the two doors
is the conjugate-Higgs operation $\tilde H=i\sigma_2 H^*$, i.e. the anti-linear
Higgs duality

$$
J_H=i\sigma_2 K,\qquad \epsilon=i\sigma_2,\ \epsilon^T=-\epsilon,\ \epsilon U^*\epsilon^{-1}=U\ (U\in SU(2)),
$$

with $K$ complex conjugation. The careful conjecture is **not** "$\tilde H$ is
coherent" (still too vague) but:

$$
\boxed{\;J_H \text{ intertwines weak-doublet duality with orientation reversal of the boundary repair word.}\;}
$$

The following structural facts are verified and make this concrete:

- **Transposition is the path scar's own $Z_2$.** With $N=|u\rangle\langle a|+|a\rangle\langle b|$
  and the endpoint reflection $R:u\leftrightarrow b$ (the residual $Z_2$ of
  $S_3\to Z_2$, §6), $RNR^{-1}=N^T$. So orientation reversal $N\leftrightarrow N^T$
  *is* the path $Z_2$. `C:9`.
- **Orientation is the $Z_2$-parity.** $S=N+N^T$ is $Z_2$-even (orientation lost);
  $N$ is $Z_2$-odd. Selecting $N$ over $N^T$ = **breaking the path $Z_2$**. `C:9`.
- **The flip cannot come from conjugation.** $N$ is real, $K(N)=N$; the twist must
  come from the $\epsilon$ part, not $K$. `C:9`.
- **No-go control (passes):** $i\sigma_2K$ acts in weak space, $N/N^T$ in family
  space; without a bridge they never meet. So $i\sigma_2K$ *alone* cannot
  distinguish $N$ from $N^T$ — a bridge $\Phi$ is required. `C:8`.

This yields a more economical picture than a separate chiral coin: the
orientation axis is *already* the $S_3\to Z_2$ reflection, so the bit reduces to

$$
H\text{ door}\to Z_2\text{-even }S=N+N^T\to\text{paired/Hermitian shell},\qquad
\tilde H\text{ door}\to Z_2\text{-odd }N\to\text{oriented nilpotent/Taylor},
$$

with $i\sigma_2$ (through the bridge) as what breaks the $Z_2$. **Test theorem:**
construct the gauge-covariant, color-singlet, family-covariant bridge
$\Phi:\mathbb C^2_H\to\mathcal A_{\rm repair}$ and check whether
$\Phi(J_H h)=\Theta\,\Phi(h)^T\Theta^{-1}$ (with $\Theta=R$); if so, the $\tilde H$
door selects the transpose-reversed (oriented) channel and the split is a
*consequence*.

**Outcome of the explicit seven-step construction (verified).** With
$\Phi(e_u)=N$, $\Phi(e_d)=N^T$ and $J_H e_d=e_u$, $J_H e_u=-e_d$, the intertwining
$\Phi(J_H h)=J_R\Phi(h)$ holds with $J_R(N^T)=N$, $J_R(N)=-N^T$, $J_R^2=-1$
(`C:9`). The up door then gives $N\to\exp(xN)\to(\tfrac14,\tfrac1{\sqrt2},1)$ at
$x=1/\sqrt2$. But two facts deflate the claim that this *is* the up/down lever:

1. **The split is symmetric.** The swapped assignment $\Phi'(e_u)=N^T$ also
   intertwines (a valid $J_R'$ exists), so the construction does not pick which
   door is $N$. "Up $=N$" is a label, not an output.
2. **Both orientations are coherent-capable.** $\exp(xN^T)=\exp(xN)^T$ has
   identical coefficients, so a down door reading $N^T$ the same way gives the
   *up* Taylor profile, not a shell. **Orientation alone does not separate
   coherent from Hermitian.**

So the orientation flip is a genuine algebraic fact adjacent to the bit, not the
bit itself. The missing lever is a **preferred repair direction**, which the
theory already has: the height filtration $h(u)=0,h(a)=1,h(b)=2$ (§6). Against
it, $N$ is strictly height-*lowering* (runs as coherent forward repair → up),
while $N^T$ is strictly height-*raising* (cannot run forward → closes only as a
paired return $NN^T/N^TN$ → Hermitian → down). This breaks the symmetry *and*
supplies the coherent-vs-Hermitian **type**, from pre-existing structure. So the
bit splits:

- **type** (coherent vs Hermitian): plausibly forced by Higgs duality ∘ height
  filtration. `C:4`.
- **specific down shell** $(6,2,4)/6$: the height-paired shell is
  $\mathrm{diag}(1,1,0)$, *not* the $S_3$ Cayley shell, so the
  $(1,\tfrac1{\sqrt3},\sqrt{2/3})$ counts still come from a separate $1+2+3$
  color/BCC count whose linkage is unproven. `C:3`.

**Net:** the *qualitative* up/down distinction is no longer a free postulate; the
*quantitative* down shell remains the genuine open piece. (Also still required:
$\Phi$ must be *forced* by covariance + minimality, not chosen — the flexibility
trap.)

**Coxeter-shell attempt at the down counts (verified, but the gap relocates).**
The $H$-door paired closure can be built from the adjacent transpositions
$\tau_1=(u,a)$, $\tau_2=(a,b)$: $\tau_i^2=I$, $(\tau_1\tau_2)^3=I$, generating
$S_3$. The *regular* representation $\mathbb C[S_3]$ has central isotypic ranks
$(1,1,4)$ summing to $6$, weights $(\tfrac16,\tfrac16,\tfrac46)$, giving exactly
the down baseline $(1,\tfrac1{\sqrt3},\sqrt{2/3})$ — a cleaner statement than the
old "$1+2+3$" shell. **But the $\tau_i$ act on the three ports, i.e. in the
3-dim *permutation* rep, which decomposes as $\mathbf1\oplus\mathbf2$ with central
ranks $(1,0,2)$ — no sign rep, no doubling, total $3$, *not* $(6,2,4)$.** The
$(6,2,4)$ lives only in the 6-dim regular rep, so it requires promoting the down
bath from the $3$ ports to the $6$-element group algebra. That $3\to6$ promotion
is the load-bearing step and is the *same content* as the old shell assumption
(the "$6$" is now $|S_3|$ instead of $1+2+3$). It is justified **iff** the
$H$-door diffusive readout sums over all paired closure *words*
($=\mathbb C[S_3]$) rather than over the $3$ ports — a plausible reading of
"diffusive," but an assumption, not a theorem. `C:9` for the regular-shell
algebra; `C:4`–`C:5` for the word-space promotion; `C:3` for the block→$(d,s,b)$
assignment. So the down counts move from "available" to "available with a cleaner
$S_3$ story," not to "derived."

**Falsifiers.** If microscopic dynamics makes both projections coherent, down
should follow the up Taylor pattern; if both Hermitian, up loses the Taylor
hierarchy; if the path scar is absent, three generations collapse to $1\oplus2$;
if no loop holonomy, CKM CP vanishes. These are clean tests.

### 11.7 Finite certificate (verified)

The full chain is a reproducible finite certificate,
`docs/scripts/verify_quark_weak_framed_repair.py` (every link asserted):

1. weak doublet $\mathcal B=(N,N^\dagger)$, unitary gauge $H\to N^\dagger$,
   $\tilde H\to N$;
2. $J_H=i\sigma_2K$ intertwines $J_R$ ($N^\dagger\!\to\!N$, $N\!\to\!-N^\dagger$),
   both squaring to $-1$;
3. $N^3=0$, $\exp(\tfrac1{\sqrt2}N)|0\rangle\Rightarrow C_u=(\tfrac14,\tfrac1{\sqrt2},1)$;
4. $\tau_1=(0,1),\tau_2=(1,2)$ generate $S_3$;
5. regular-rep grouped weights $\Rightarrow C_d=(1,\tfrac1{\sqrt3},\sqrt{2/3})$;
6. closure laws $k_u=3n=(6,3,0)$, $k_d=2(n{+}1)=(6,4,2)$.

The certificate **also asserts the gaps**, so it cannot read as closure: the
$\tau_i$ live in the 3-dim permutation rep $(1,0,2)$ — the script checks this
explicitly — so step 5's $(6,2,4)$ requires the $3\to6$ regular-rep promotion
(GAP A), and the orientation split is the unitary-gauge statement with $\mathcal B$
a doublet under the *anti-linear* $J_H$ (GAP B).

**Uniqueness audit — the decisive verdict: PERMITTED, not FORCED**
(`docs/scripts/verify_weak_framed_uniqueness.py`). The most general path-local
oriented operator is $X=aE_1+bE_2$ ($a,b\ne0$ for rank-2 length-3 nilpotency).
Diagonal family rephasing rescales the two edges by *independent* phases
($E_1\!\to\!e^{i(\theta_1-\theta_0)}E_1$, $E_2\!\to\!e^{i(\theta_2-\theta_1)}E_2$),
so $\arg a,\arg b$ are gauge and overall scale is free: the only invariant is
$|a/b|\in(0,\infty)$. Hence under {locality, no-shortcut, rank-2 nilpotency,
$J_H$–$J_R$ intertwining} the solution space modulo gauge is a **one-parameter
family**, $\dim=1$ — so $(N,N^\dagger)$ is *permitted, not forced*. Forcing the
single orbit requires **exactly one further axiom**: equal local repair strength
$|a|=|b|$ (then $\dim=0$). That axiom is *plausibly* sourced — the BB coin has
equal-magnitude hops $|q_+|=|q_-|=\sqrt2/4$ — but only if the weak$\to$family
bridge $\Phi$ transports the equality to the two path edges, which is unproven.
(Matching the observed up profile also fixes $a=b=1$, but that is fitting, with an
$x\!\leftrightarrow\!a$ degeneracy, not independent forcing.)

Grading (the "permitted" branch): $C_{\rm algebra}=9$ (certified),
$C_{\rm mechanism}=5$, $C_{\rm microscopic\ origin}=3$–$4$. The decisive theorem
is now one sharp, checkable condition: **does $\Phi$ transport the BB coin's
equal-magnitude hops to equal-strength path-edge repair, $|a|=|b|$?** If yes, the
quark up/down mechanism is derived at the level of the BB-scar silver result; if
no, it remains a controlled ansatz.

**Two follow-up audits** (`docs/scripts/verify_equal_strength_audit.py`):

- **Path-locality is now *derived*, not assumed (`C:9`).** A one-tick defect
  height grading $D=\mathrm{diag}(0,1,2)$ with $[D,X]=X$ forces $X_{ij}=0$ unless
  $D_i-D_j=1$, leaving exactly $X=aE_1+bE_2$ — the shortcut $0\to2$ and all
  reverse/diagonal terms are algebraically forbidden.
- **The "irreducibility $\Rightarrow$ Schur $\Rightarrow$ equal-strength" closure
  is invalid**, for three independent reasons: (a) Schur constrains the
  *commutant* (scalars), and $X^\dagger X=\mathrm{diag}(|a|^2,|b|^2)$ is generically
  not in it, so irreducibility cannot force it scalar; (b) the active *source*
  sites $|0\rangle,|1\rangle$ have *distinct heights*, so the height-respecting
  defect algebra is diagonal on them — the active domain is *reducible*, the
  premise fails; (c) the clock $Z_3$ that would equate the two edges
  ($C_3E_1C_3^{-1}=E_2$) is exactly the symmetry *broken by the cut*
  ($C_3E_2C_3^{-1}=|0\rangle\langle2|$, the closing edge). On the open path,
  source $0$ (endpoint) and $1$ (interior) are inequivalent, so no symmetry forces
  $|a|=|b|$.

So equal-strength remains a **defect-homogeneity assumption** ("the cut adds no
edge-weight profile"), and the gap is now understood as *structural*: the cut
that builds the generation hierarchy is the same cut that breaks the edge-equating
symmetry. Hierarchy and edge-uniformity are in tension at one defect.

**The fix — height-cover homogeneity** (`docs/scripts/verify_height_cover_equal_strength.py`).
The resolution is to move the symmetry *upstream of the cut*. On the height cover
$\{|r\rangle:r\in\mathbb Z\}$ with $D|r\rangle=r|r\rangle$, one-tick locality gives
$X_{\rm cov}=\sum_r a_r|r{+}1\rangle\langle r|$, and **translation-covariance**
$T^{-1}X_{\rm cov}T=X_{\rm cov}$ — the *definitional* homogeneity of the BB QCA
(same coin at every height) — forces $a_{r+1}=a_r$, a constant (verified). The
pinned BB scar fixes $|a|=x=1/\sqrt2$; compressing to the flag, $P_3X_{\rm cov}P_3=xN$
(verified). The cut is now *downstream* of the symmetry, so the
hierarchy/edge-uniformity tension is evaded. **Residual assumption:** the
truncation must be *sharp* (no near-boundary amplitude modulation) — non-trivial
because a 3-site flag is all boundary.

**The cleaner route — chiral-reflection covariance** (replaces the withdrawn
Schur argument). Form the Hermitian chiral defect block
$H_X=\bigl(\begin{smallmatrix}0&X^\dagger\\X&0\end{smallmatrix}\bigr)$ and the
operation $\mathcal C=\sigma_x^{\rm chir}\otimes R$ (chiral L/R swap times the
open-path reflection $R:0\leftrightarrow2$). Then $\mathcal C H_X\mathcal C^{-1}=H_X
\Leftrightarrow RXR=X^\dagger \Leftrightarrow b=\bar a \Leftrightarrow |a|=|b|$
(verified). This uses only the surviving symmetry of the open path: no Schur, no
irreducibility, no broken $C_3$. The two factors of $\mathcal C$: $R$ is the
residual $Z_2$ that defines the path scar (guaranteed once $P_3$ is granted);
$\sigma_x^{\rm chir}$ is the up/down Higgs/$J_R$ duality already present.
Caveat against overclaiming: $\mathcal C$ is parity-like on the family path, not
spacetime parity, so it is not inherited from the BB walk's exact spacetime $P$
(§12) — the resemblance is structural. Honest status: $\mathcal C$-covariance
$\iff$ equal-strength, so this relocates the assumption from ad hoc "no
edge-weight spurion" to a named symmetry ("the scar's residual $Z_2$ constrains
bond weights, not just support") — a real gain in quality, not a logical
elimination.

Grading: $C_{\rm theorem}=9$ ($\mathcal C$-covariance forces $|a|=|b|$);
$C_{\rm assumption}=4$–$6$ (does the defect weight the two bonds
$R$-symmetrically?). Equal-strength is `C:7` overall — doubly motivated
(height-cover homogeneity, or chiral-reflection $Z_2$), still not forced by bare
structure.

**Net relocation.** With this, the up/down bit is **downstream of the generation
problem**: grant the length-3 flag $P_3$ and $\mathcal B=(N,N^\dagger)$ follows
from the truncated homogeneous shift. The deepest open question reverts to the
correct place — *why is the repair flag length three?* (§6, the generation
problem) — not edge-uniformity.

### 11.8 A bold defect model (speculative, `C:3`)

This subsection is an explicit *speculative swing* at going **beyond** the §6.5
axiom — i.e. attempting to actually *derive* $N_{\rm gen}=3$ rather than accept it.
It is **not** the program's default posture (that is the §6.5 axiom
$\dim\mathcal H_{\rm fam}=3$) and **not** part of the certified structure. It is
graded `C:3` and flagged as such. It proposes that **one defect generates the
generation number, the path scar, the orientation doublet, the up/down split, the
CP sign, and the silver hierarchy**: the $S_3=Z_3\rtimes Z_2$ **chiral clock
defect** at the head of the BCC $q=0$ edge scar (§5.4), with internal space
$\mathbb C^3_{\rm port}\otimes\mathbb C^2_\chi$ (residual tetrahedral ports × BB
chiral coin). Per §6.5, its "$N_{\rm gen}=3$ from $Z_3$" claim is a *consistency
clue*, not a proof — the program does not stake its credibility on it.

*Verified structural backbone* (`C:9` algebra,
`docs/scripts/verify_bold_clock_defect.py`):

1. The body-diagonal $Z_3$ rotation acts on the three ports as the clock $C_3$
   ($C_3^3=I$); $N_{\rm gen}=3$ is the §9 geometric count $(4-1)$, read as $|Z_3|$.
2. Cutting the clock's closing edge gives the path nilpotent $N$ ($N^3=0$) with
   $\operatorname{Spec}2\Delta(P_3)=\{0,2,6\}$ (uncut clock: $\{0,6,6\}$).
3. The endpoint reflection $(0,2)$ is the *unique* transposition that both
   inverts the clock ($\rho C_3\rho=C_3^{-1}$) and transposes the path
   ($\rho N\rho=N^\dagger$): $S_3=Z_3\rtimes Z_2$, and the $Z_2$ is the orientation
   doublet.
4. The chiral spin-½ lift of the $2\pi/3$ rotation has $U_3^3=-I$ (the chiral
   orientation/CP sign).

*Bold inputs* (`C:3`, the swing): (5) **coin–isospin lock** — the defect coin
orientation is identified with weak isospin, making $\mathcal B=(N,N^\dagger)$ a
gauge-covariant doublet; (6) **Higgs supplies the $Z_2$** — the antisymmetric
$i\sigma_2$ in $\tilde H$ *is* the endpoint reflection, so $\tilde H\to N$ (up),
$H\to$ full $S_3$ word-space (down); (7) **silver rides the same coin** (§5.4).

Unification: $Z_3=$ generations+path+up-orientation; $Z_2=$ orientation
flip+Higgs duality+up/down; coin $=$ silver+CP sign.

*Why it is bold, and where it most likely dies:*

- It **evades rather than contradicts** the `TOPOLOGY_GENERATIONS` kill (spatial
  $Z_3$ acts trivially on the *bulk* chiral-16, $16=16\cdot1$) by placing the
  $Z_3$ on the *boundary* residual ports — a different space. Whether the
  boundary $Z_3$ is genuinely independent of the killed bulk one is the first
  attack point.
- The **coin–isospin lock (5)** identifies a spacetime/coin structure with
  internal weak isospin *at the defect* — defensible only as a boundary-localized
  locking, and the most likely failure mode (it flirts with the category error
  the sector map forbids).
- It does **not** close GAP A: the down shell still needs the $3\to6$
  regular-rep promotion, now phrased as "the $H$-door sees the full $S_3$
  word-space."

Net: the model concentrates the entire generation+doublet keystone into *one*
falsifiable hypothesis — the boundary coin–isospin lock — instead of several
vague ones. That is its value; its truth is unestablished.

---

## 12. CP, strong CP, and Lorentz-violation bounds

- **Tree scar has no intrinsic CP.** Phases on a tree are removable by port
  rephasings (`C:9`). CP needs a loop, non-normal deformation, or chiral coin.
  Healing the missing edge of the depth triangle creates exactly one
  gauge-invariant holonomy $\phi\mapsto-\phi$ under conjugation. Hierarchy
  (tree/Laplacian) and phase (loop/coin) separate cleanly.
- **Lattice CP slot.** The leading continuum correction of the BB walk is exactly
  in the CP-odd $T_{2g}$ irrep of $O_h$ ($\|H^{(1)}_{\rm CP\,odd,T_{2g}}\|^2=12$,
  zero CP-even part), `C:8`. Internal Higgs-like maps have a universal $1/2$
  $J$-anticommuting split (`C:8`); the old "CP-violating fraction" label is an
  overclaim (`C:1`).
- **Strong CP.** The $A_{2u}$ theta channel is absent at $O(\epsilon_{\rm lat})$
  and $O(\epsilon_{\rm lat}^2)$; the spatial-only plaquette density is
  dimensionally zero; chiral-anomaly traces vanish through the audited orders
  (`C:8`). A nonzero cross-trace $\mathrm{tr}(\gamma^5 H^{(1)}H^{(2)})\ne0$ warns
  against an all-order claim (`C:3`). Strong CP ≠ CKM CP.
- **SME.** The $T_{2g}$ slot maps to a dim-5 axial spin-tensor coefficient
  $d^{(5)}$; the representative bound gives
  $\epsilon_{\rm lat}\lesssim2\times10^{-33}\,{\rm m}\approx10^2\ell_P$ —
  observationally safe but not predictive (`C:4`).

CP phenomenology is a **constraint** layer, not a flavor generator: a phase slot
must be coupled to a rephasing-invariant three-generation flavor mechanism before
it becomes a CKM/PMNS prediction (`C:3`).

---

## 13. Killed generation routes (negative controls)

All tested algebraic/topological routes to three generations fail, and converge
on one lesson: **three generations require a genuine family-breaking spurion, not
an unbroken symmetry.**

| Route | Result | `C` |
|---|---|---|
| Spin(8) triality | does not preserve the SM-restricted Cartan (rank 5 vs 4) | `C:1` |
| Broken triality | rank-deficient, nearly flat Yukawa (ratio $\approx1.66$) | `C:1` |
| Exceptional Jordan $J_3(\mathbb O)$ | $27=16\oplus10\oplus1$, not $3\times16$ | `C:1` |
| Spatial $Z_3$ | trivial internal action $16=16\cdot1$ | `C:1` |
| Color-center $Z_3$ | $16=(8,4,4)$, not three equal copies | `C:1` |
| $\pi_3$ torsion of carrier cosets | no verified $\mathbb Z/3$ class | `C:5` |
| Anomaly cancellation | cancels for any $N$ ($0=0$) | `C:1` for forcing $N=3$ |
| Raw BB hop/Walsh depth | forbidden quadrupole content | `C:1` |

`C:8` for the individual finite kills; `C:5`–`C:6` for the synthesis lesson.

---

## 14. Load-bearing open problems

These are the real theorems the program still owes. They are kept visible
deliberately.

1. **Generation number.** Per §6.5, $N_{\rm gen}=3$ ($\dim\mathcal H_{\rm fam}=3$)
   is taken as a **primitive boundary axiom** (`C:5`), *not* something the program
   claims to derive — the trivalent resonances (3 space, 3 color, 3 ports) are
   consistency clues, not proofs. *Given* the three-port flag, deriving the
   $S_3\to Z_2$ path scar / $\{0,2,6\}$ from the microscopic boundary remains open
   (`C:3`–`C:6`); attempting to derive the $3$ itself is the explicitly speculative
   §11.8. The honest split: the *number* is axiomatized; the *hierarchy, residues,
   and mixings given the number* are the conditional theorems the rest of the
   document builds.
2. **Silver from BCC.** *Exact transfer-class theorem; microscopic selection
   still conditional (§5.4).* The pinned BB coin, restricted to the
   $q=r_1-r_2=0$ synchronous-normal scar at $k_3=0$, gives a subunitary no-leakage
   branch (survival $1/\sqrt2$) whose radial transfer is orientation-preserving
   ($\det=+1$) with eigenvalues $\sqrt2\pm1$ at the trace-minimum/survival point
   $\zeta=1/\sqrt2$ — yielding $\epsilon=\sqrt2-1$, conjugate to the neutrino
   sterile chain (one silver universality class, $\det=+1$,
   $\operatorname{tr}=2\sqrt2$). The *transfer-class algebra* is verified in the
   canonical convention (`C:8`). It is **not** "closed": two inputs remain
   conditional — (a) the $q$-superselection is *imposed*, not derived from the BB
   coin/defect Hamiltonian; (b) the mass-readout interpretation (decaying
   eigenvalue at the survival/trace-minimum $\zeta=1/\sqrt2$) is a physical choice
   (`C:5`–`C:6`). The Pell/impedance ($\det=-1$) forms are the arithmetic shadow.
3. **Up/down dynamical bit.** *Split into two by the §11.6 seven-step
   construction.* (i) **Type** (coherent vs Hermitian): the Higgs duality
   $J_H=i\sigma_2K$ gives an orientation flip $N\leftrightarrow N^T$, but that
   alone is symmetric and coherent-capable both ways; the lever is the
   **height filtration** (§6) — $N$ height-lowering → coherent (up), $N^T$
   height-raising → paired Hermitian (down). Plausibly forced by pre-existing
   structure, `C:4`. (ii) **Specific down shell** $(6,2,4)/6$: the
   adjacent transpositions $\tau_1=(u,a),\tau_2=(a,b)$ generate $S_3$ and the
   *regular* rep's central ranks $(1,1,4)$ give $(1,\tfrac1{\sqrt3},\sqrt{2/3})$
   cleanly — but the $\tau_i$ act in the 3-dim *permutation* rep $(1,0,2)$, so the
   $(6,2,4)$ requires promoting the down bath to the 6-dim group algebra
   $\mathbb C[S_3]$ (the $H$-door summing over paired closure *words*). That
   $3\to6$ promotion is the same content as the old shell assumption, `C:4`–`C:5`;
   block→$(d,s,b)$ assignment still `C:3`. **Largely resolved (§11.7)**: path-locality is
   *derived* from one-tick height grading ($[D,X]=X\Rightarrow X=aE_1+bE_2$,
   `C:9`); equal-strength $|a|=|b|$ follows from **height-cover translation
   homogeneity + sharp truncation** (the BB QCA is homogeneous on the cover; the
   flag is its truncation), equivalently from the scar's residual $Z_2$ realized
   as $\rho X\rho=X^\dagger$ — doubly motivated, `C:7` (not forced by bare
   structure; rests on sharp-truncation / scar-$Z_2$-respect). **Net:** the
   up/down bit is now *downstream of the generation problem* — grant the length-3
   flag $P_3$ and $\mathcal B=(N,N^\dagger)$ follows. The remaining deep question
   is item 1 (why length 3), not edge-uniformity. (Down counts still need the
   $3\to6$ regular-rep promotion, GAP A.)
4. **Higgs zero-mode selector.** Reduce the rank-2 Higgs active plane to a unique
   aligned zero-mode line by boundary dynamics.
5. **Charged-lepton torsion.** Derive the frame holonomy $G$ of weight
   $p_u p_a=2/9$ (the corrected §10.5 target), plus the scale $\mu_e$.
6. **Forward spectral principle.** Select the quark mass measures
   $(\lambda_{fr},w_{fr})$ rather than reconstructing them.
7. **Absolute neutrino scale $\mu_\nu$** from microscopic sterile/Higgs dynamics.
8. **Positive quartic backreaction axiom** in the vacuum-selector sector.

---

## 15. Certainty ledger (summary)

| Claim | `C` |
|---|---|
| Schur/Feshbach response as flavor language | 7–9 |
| $\mathrm{Cl}(0,10)\to$ one chiral-16 SM generation | 8–9 |
| Silver root $\epsilon=\sqrt2-1$; Pell ratio $\eta=\epsilon^2$ | 9 |
| Pinned BB $q{=}0$ scar $\Rightarrow$ $\det{+}1$ transfer, eig $\sqrt2\pm1$ at $\zeta{=}1/\sqrt2$; conjugate to neutrino chain | 8 |
| Pell/impedance ($\det{-}1$) demoted to arithmetic shadow of silver | 8 |
| $\operatorname{Spec}2\Delta(P_3)=\{0,2,6\}$; triple identity | 9 |
| $S_3$ Schur obstruction ⇒ depth needs a spurion | 9 |
| Neutrino $K_\nu=\epsilon^2P_u+P_b$, $(0,\eta,1)$, ratio $\epsilon^4$ | 8–9 |
| Charged-lepton Koide ⇔ trace/plane equipartition | 9 |
| Charged-lepton masses at $6.7\times10^{-5}$ from $\theta_e$ | 8 (calc), 3–4 (forward) |
| Two-sided kernel necessity; torsion = frame holonomy; $p_up_a=2/9$ | 8–9 |
| Spectral measure rigidity no-go | 9 |
| Up Clebsch $(\tfrac14,\tfrac1{\sqrt2},1)$; down baseline $(1,\tfrac1{\sqrt3},\sqrt{2/3})$ | 6–7 |
| Unified backbone: one BB $q{=}0$ silver source, sector-specific readouts (§11.0) | 6 |
| Quark ratios + CKM angles from shared $\eta$ (verified numerically) | 6–7 |
| Up/down mechanism (subclaim stack, see §11.5 table) | 9/9/7/8/4/4 → 3–5 full |
| Microscopic origin of silver, depth, $\mu_\nu$, mass measures | 3 |
| All algebraic/topological generation routes | 1 (killed) |
| $N_{\rm gen}=3$ as a primitive boundary axiom (not derived) (§6.5) | 5 (axiom) |

---

## 16. What a referee should focus on

**Strongest, check these for rigor:** (i) the neutrino product-sterile theorem
§9.3 — is the product framing $H_Q=H_{\rm chain}\otimes I_{\rm family}$ a
legitimate boundary completion or an assumption that builds in the answer?
(ii) the $S_3$ Schur obstruction §6.3 — is "depth = generation spurion" airtight?
(iii) the spectral rigidity no-go §8 — does it really forbid a finite-data mass
derivation?

**Most seductive, scrutinize for overfitting:** the charged-lepton
$6.7\times10^{-5}$ agreement §10.3 rests on a single counted torsion $2/9$ that is
currently a numerical identity ($p_up_a$), **not** a derived holonomy (§10.5). The
agreement is sharpest exactly where the derivation is weakest. A genuine
derivation of the frame generator $G$ would make or break this.

**Falsifiable predictions:** normal neutrino ordering, one massless light
neutrino, $\Delta m^2_{21}/\Delta m^2_{31}=\epsilon^4\approx0.0294$,
$\sum m_\nu\approx0.059\,{\rm eV}$, $\sin^2\theta_{13}=\tfrac34\epsilon^4$; shared
$\eta$ in CKM and light-quark mass ratios; charm coefficient near
$1/\sqrt2$–$3/4$; and a sector-specific torsion rule (a smaller torsion for
colored quarks if the $p_up_a/\dim\mathrm{End}$ reading is correct).

**Do not expect:** a derivation of $N_{\rm gen}=3$ — it is taken as a *primitive
boundary axiom* (§6.5), with the program's claims explicitly *conditional* on the
three-port flag (the trivalent resonances are clues, not proofs); nor absolute
mass scales, $W/Z$ or Higgs masses (ordinary electroweak symmetry breaking), or
hadron observables (downstream QCD). The right question to judge the program by is
the conditional one: *given three boundary ports, are the hierarchy, residues, and
mixings derived?*

---

## Appendix A. Glossary

- **BCC QCA** — body-centered-cubic quantum cellular automaton; the
  Bialynicki–Birula Weyl walk on the eight body-diagonal directions.
- **Carrier** — the internal Clifford spinor holding SM gauge charges.
- **Boundary / Schur response** — elimination of unresolved modes $Q$, producing
  the self-energy $\Sigma(z)=V^\dagger(z-H_Q)^{-1}V$.
- **Silver root** $\epsilon=\sqrt2-1$; **Pell ratio** $\eta=\epsilon^2$.
- **Depth scar** — the path $u$–$a$–$b$ with $D_{\rm scar}=2\Delta(P_3)$.
- **Residual basis** $(u,a,b)$ — the framed tetrahedral exit family space.
- **Framing** — selecting one BCC vacuum exit, breaking residual $S_3\to S_2$.

## Appendix B. Key identities

$$
\epsilon^2+2\epsilon-1=0,\quad m(2\sqrt2)=\epsilon,\quad
\eta=|\lambda_-/\lambda_+|_{T_{\rm Pell}}=(\sqrt2-1)^2,
$$
$$
\operatorname{Spec}2\Delta(P_3)=\{0,2,6\},\quad
\operatorname{Spec}2\Delta(K_3)=\{0,6,6\},\quad
3=1\oplus2\ \text{under}\ S_3,
$$
$$
K_\nu=\epsilon^2P_u+P_b,\quad (m_1,m_2,m_3)=\mu_\nu(0,\eta,1),\quad
\Delta m^2_{21}/\Delta m^2_{31}=\epsilon^4=17-12\sqrt2,
$$
$$
K_{\rm Koide}=\tfrac23\Leftrightarrow|w_t|^2=|w_\perp|^2,\quad
\theta_e=-\tfrac{2\pi}3-\tfrac29,\quad p_u p_a=\tfrac13\cdot\tfrac23=\tfrac29 .
$$

## Appendix C. Internal sidecar map (for repository readers)

Carrier: `lepton`, `obstruction_r10`. Stage: `spacetime_qca`, `sim`. Boundary:
`boundary_response`, `flavor_a_track`. Silver/Pell: `SILVER_FIXED_POINT_NOTE`,
`BCC_SCAR_TRANSFER`, `PELL_RADIAL_THEOREM`. Depth: `depth_scar`,
`depth_hop_walsh`. Clebsch: `scalar_clebsch`, `QUARK_MASS_RG_NOTE`. Radial:
`radial_response`. Neutrino/leptons: `NEUTRINO_PROTOTYPE`,
`CHARGED_LEPTON_BCC_ANSATZ`, `CHARGED_LEPTON_RESOLVENT_DERIVATION`,
`CHARGED_LEPTON_MINIMAL_HAMILTONIAN`, `KOIDE`. CP: `cp`, `strongcp`, `sme`.
Kills: `triality`/`broken_triality`/`exceptional`, `topology`. The narrative
layer is in `src/clifford_3plus2_d5/synthesis/` (`OVERVIEW.md`,
`COMMON_PICTURE.md`, `SECTOR_MAP.md`, `MASS_SECTOR.md`).

## Appendix D. Phenomenological inputs and RG-stability

Representative reference data (PDG 2024 / NuFIT 5.2, normal ordering). These are
*inputs against which the program's ratio predictions are compared*; they are
not a fit. The program predicts **ratios and angles**, not absolute scales, so
the table flags which comparisons are renormalization-group (RG) stable.

**Quark masses** ($\overline{\rm MS}$): light $u,d,s$ at $\mu=2$ GeV
($m_u=2.16^{+0.5}_{-0.3}$, $m_d=4.67^{+0.5}_{-0.2}$, $m_s=93.4^{+8.6}_{-3.4}$ MeV);
$m_c(m_c)=1.27(2)$ GeV; $m_b(m_b)=4.18(3)$ GeV; $m_t(m_t)\simeq163$ GeV
($\overline{\rm MS}$), pole $172.7(3)$ GeV. **Within-sector mass ratios are
RG-stable** to good approximation: the QCD mass anomalous dimension is
flavor-independent, so $m_i/m_j$ in one sector is nearly scale-invariant.
**Cross-sector** (up/down) ratios and absolute scales are **not** RG-stable and
require a common scale plus a sector normalization $r_d$ (§7.3).

| Observable | Theory (in $\eta=(\sqrt2-1)^2$) | Predicted | Reference | RG-stable? |
|---|---|---:|---:|---|
| $m_u/m_c$ | $\eta^3/2\sqrt2$ | $0.0018$ | $\sim0.0017$–$0.002$ | yes (within-sector) |
| $m_d/m_s$ | $\sqrt3\,\eta^2$ | $0.0510$ | $\sim0.050$ | yes |
| $m_s/m_b$ | $\eta^2/\sqrt2$ | $0.0208$ | $\sim0.020$–$0.023$ | yes |
| $m_c/m_t$ | $\eta^3/\sqrt2$ | $0.0036$ | $\sim0.0035$–$0.008$ | **scale-sensitive** (top) |
| $\lvert V_{us}\rvert$ | $\tfrac43\eta/\sqrt{1+\eta^2}$ | $0.2255$ | $0.2243(5)$ | mild running |
| $\lvert V_{cb}\rvert$ | $\sqrt2\,\eta^2$ | $0.0416$ | $0.0408(13)$ | mild |
| $\lvert V_{ub}\rvert$ | $\eta^3/\sqrt2$ | $0.00357$ | $0.00382(20)$ | mild |
| $\delta_{\rm CKM}$ | $\arctan\sqrt5$ | $65.9^\circ$ | $\sim66^\circ$ | mild |
| $\Delta m^2_{21}/\Delta m^2_{31}$ | $\epsilon^4=17-12\sqrt2$ | $0.0294$ | $0.0296$ | mild (mass-ratio) |
| $\sin^2\theta_{13}$ | $\tfrac34\epsilon^4$ | $0.0221$ | $0.0222(6)$ | mild |
| $K_{\rm Koide}$ | $2/3$ (exact, by construction) | $0.66667$ | $0.666661$ | tiny (QED) |

**Caveats for external review.** (i) $m_c/m_t$ is the single most scale-sensitive
entry: the $\overline{\rm MS}$-at-own-scale value $\sim0.0078$ versus the
common-scale value near the prediction differ by the running flagged in §7.3 —
this comparison must be quoted at a stated common scale, not at each quark's own
scale. (ii) The CKM phase $\delta_q=\arctan\sqrt5$ has the largest unverified
coefficient (the angle *powers* in $\eta$ are on firmer footing than the rational
prefactors $\tfrac43,\sqrt2$, §11.4). (iii) Charged-lepton masses use pole values;
Koide holds at the $10^{-5}$ level there but the program's $\theta_e=-2\pi/3-2/9$
torsion is `C:3`–`C:4` (§10). (iv) No absolute mass scale ($v$, $\mu_\nu$, $\mu_e$)
is predicted; only ratios and angles.
