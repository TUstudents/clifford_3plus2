# Exact BCC Boundary Defect Model

This note states the model we now need. It is not another texture ansatz. It is
the proposed exact boundary object from which the later flavor readouts must be
derived.

The guiding claim is:

$$
\boxed{
\text{one framed BCC edge defect}
\quad\Longrightarrow\quad
\text{one universal silver transfer}
\quad\Longrightarrow\quad
\text{sector-specific Schur readouts}.
}
$$

The model is exact only if the first arrow is a theorem. Everything else should
be read downstream.

## 1. Bulk Object

The bulk stage is the pinned Bialynicki-Birula BCC Weyl automaton on
$\mathbb Z^3$ with body-diagonal hops

$$
h=(\sigma_1,\sigma_2,\sigma_3),\qquad \sigma_i=\pm1,
$$

and two-component Weyl coin. In the repository convention the hop coefficients
are fixed by $q_\pm=(1\pm i)/4$. We do not vary this coin.

Let

$$
U_{\rm BB}\psi(x)=\sum_{\sigma\in\{\pm1\}^3}W_\sigma\psi(x-\sigma)
$$

be the one-step update. `C:8` for the implemented pinned BB walk and its
continuum Weyl limit; `C:9` for all finite matrix identities below once this
convention is fixed.

## 2. Defect Geometry: A Codimension-Two Edge

Choose two inward normal coordinates $(r_1,r_2)$ and one tangential coordinate
$s$. The edge half-space is

$$
r_1\ge0,\qquad r_2\ge0.
$$

Define the relative-depth charge

$$
q=r_1-r_2.
$$

A body-diagonal hop sends

$$
q\mapsto q+(\sigma_1-\sigma_2).
$$

Hence same-normal hops $\sigma_1=\sigma_2$ preserve $q$, while mixed-normal hops
$\sigma_1=-\sigma_2$ leave the diagonal scar. The exact diagonal scar is

$$
\mathcal H_{\rm scar}
=\ell^2(\mathbb N_r)\otimes\mathbb C^2_{\rm Weyl},
\qquad r=r_1=r_2,
$$

after setting tangential momentum $k_s=0$.

This is the first place where the model differs from the killed scalar edge
count. The raw edge has two entrance channels but leaks. The exact defect model
therefore does not say "edge count $2$ gives silver"; it says:

$$
\boxed{
\text{the physical boundary superselects }q=0,
\text{ and mixed-normal hops enter the unresolved bath.}
}
$$

This $q=0$ superselection is the load-bearing boundary axiom. `C:9` for the
geometry of $q$; `C:6` for the exact model conditional on this superselection;
`C:3` until a full microscopic defect Hamiltonian derives it. The minimal
conditional microscopic route is developed later as the single-clock locking
model:
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md).
The BB-only route has now been tested and fails:
[BARE_BB_EDGE_UPDATE_AUDIT.md](BARE_BB_EDGE_UPDATE_AUDIT.md) proves that the
bare Weyl spinor has no nonzero state that kills both mixed-normal leakage
channels. The unprojected update itself is written in
[BARE_BB_RELATIVE_CHANNEL_UPDATE.md](BARE_BB_RELATIVE_CHANNEL_UPDATE.md): it is
an exact closed unitary walk on trace/relative coordinates $(r,q)$, with $q$ a
two-way Bloch channel until extra boundary dynamics locks or opens it. The real
wedge domain $r_1,r_2\ge0$ is checked in
[BARE_BB_WEDGE_DOMAIN_AUDIT.md](BARE_BB_WEDGE_DOMAIN_AUDIT.md); it confines the
relative coordinate at fixed depth but still lets $q=0$ leak to $q=\pm2$ and
return. The return is dynamically nonzero:
[BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md](BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md)
computes the exact two-step feedback
$R_{\rm rel}^{(2)}=\operatorname{diag}(-(1+i)/4,-(1-i)/4)$.
[BARE_BB_FACE_EXTERIOR_INSUFFICIENCY.md](BARE_BB_FACE_EXTERIOR_INSUFFICIENCY.md)
then checks the ordinary outgoing-face completion and shows that it cannot
remove this first return; the return is inward from $q=\pm2$ to $q=0$, not an
outward hop through $q=\pm n$.
[EDGE_CLOCK_SCATTERING_BOUNDARY_UPDATE.md](EDGE_CLOCK_SCATTERING_BOUNDARY_UPDATE.md)
therefore gives the constructive local completion: the exact BB mixed-normal
blocks are routed into outgoing edge-clock error ports instead of recurrent
wedge states. Its retarded compression removes the first return while preserving
local unitarity by an explicit finite unitary completion.
[MICROSCOPIC_BB_QCA_EDGE_UPDATE.md](MICROSCOPIC_BB_QCA_EDGE_UPDATE.md) collects
this into the current complete local update specification: the same-normal BB
blocks are visible radial ports, the mixed-normal BB blocks are forced
clock-error ports, and a local unitary scattering completion plus retarded
source preparation gives the $q=0$ survival branch.
[BB_QCA_BOUNDARY_UPDATE_COMPLETION_AUDIT.md](BB_QCA_BOUNDARY_UPDATE_COMPLETION_AUDIT.md)
audits this against the full boundary-update plan and records the local update
as complete, while separating the deeper boundary-material origin of outgoing
asymptotics as the next layer.
[EDGE_CLOCK_SCATTERING_MINIMALITY.md](EDGE_CLOCK_SCATTERING_MINIMALITY.md)
then proves that a hidden output sector of rank at least two is forced by the
visible BB norm deficit; preserving the $q\mapsto-q$ orientation resolves it as
the two clock-error ports $\chi_\pm$.
[CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md](CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md)
then proves the causal compression theorem: with outgoing clock-error leads and
no incoming clock-error data, the visible powers are exactly $A_{\mathbb N}^t$,
whereas allowing a hidden return block immediately restores the bare two-step
feedback.

The finite mixed-normal model in
[BCC_Q0_SUPERSELECTION_DERIVATION.md](BCC_Q0_SUPERSELECTION_DERIVATION.md) makes
this conditional statement sharper. The mixed-normal blocks $q=0\to q=\pm2$
carry the complementary half of the BB norm, and a local relative-depth penalty
$H_{\rm bd}=\Lambda q^2$ gives a Schur feedback
$\Sigma_{\rm mix}=I/[2(z-4\Lambda)]\to0$. Thus a hard relative-depth gap implies
the exact $q=0$ visible compression. The physical origin of that gap is modeled
as a one-edge-clock constraint; deriving that constraint from a more primitive
boundary material model remains open. The retarded/no-incoming part of the
readout is exact inside the edge-clock scattering model, but its physical
asymptotic origin remains a boundary-dynamics premise.

## 3. Visible Scar Branch

Project the pinned BB update to the same-normal, $q=0$, $k_s=0$ branch. The two
radial blocks are

$$
B_+=\frac14
\begin{pmatrix}
1+i&1-i\\
1+i&1-i
\end{pmatrix},
\qquad
B_-=\frac14
\begin{pmatrix}
1+i&-1+i\\
-1-i&1-i
\end{pmatrix}.
$$

They obey

$$
B_+^\dagger B_+ + B_-^\dagger B_-=\frac12 I.
$$

Thus the scar branch is a contraction with survival norm $1/\sqrt2$. The other
half of the probability is not discarded; it belongs to the unresolved boundary
bath. `C:8`.

In the symmetric/antisymmetric spinor basis $(e,o)$,

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

The stationary scar equation

$$
\zeta\psi_r=B_+\psi_{r-1}+B_-\psi_{r+1}
$$

is equivalent to the one-step transfer

$$
T(\zeta)=
\begin{pmatrix}
\frac1{2\zeta}&\frac{i}{2\zeta}\\
-\frac{i}{2\zeta}&2\zeta+\frac1{2\zeta}
\end{pmatrix}.
$$

The determinant and trace are

$$
\det T(\zeta)=1,\qquad {\rm tr}\,T(\zeta)=2\zeta+\frac1\zeta.
$$

At the branch survival point

$$
\zeta=\frac1{\sqrt2},
$$

the trace is minimized:

$$
{\rm tr}\,T(1/\sqrt2)=2\sqrt2.
$$

Therefore

$$
T(1/\sqrt2):
\qquad
\lambda_\pm=\sqrt2\pm1,
\qquad
\epsilon=\lambda_-=\sqrt2-1.
$$

This is the exact silver transfer theorem of the model:

$$
\boxed{
\text{pinned BB coin}
+q=0\text{ scar}
+k_s=0
+\zeta=1/\sqrt2
\quad\Rightarrow\quad
\epsilon=\sqrt2-1.
}
$$

`C:9` for the transfer algebra; `C:8` for the derivation from the pinned BB
blocks; `C:5`-`C:6` for the physical reading of the survival/trace-minimum point
as the mass-return probe. The dedicated theorem target for upgrading this
physical reading is
[BAND_EDGE_SELECTION_THEOREM.md](BAND_EDGE_SELECTION_THEOREM.md).

## 4. Exact Unitary Completion

The projected scar branch is subunitary. A boundary theory cannot lose
probability, so the exact model must include an unresolved bath $Q_{\rm leak}$.

Let $A$ denote the projected scar contraction. On the radial cover,
$A^\dagger A=AA^\dagger=\frac12I$, so the local Julia dilation is

$$
U_{\rm def}=
\begin{pmatrix}
A&\frac1{\sqrt2}I\\
\frac1{\sqrt2}I&-A^\dagger
\end{pmatrix}.
$$

On the physical half-line there is one finite-rank head correction, because
the missing $r=-1$ channel changes $I-A^\dagger A$ and $I-AA^\dagger$ only at
the edge head. The exact formulas are given in
[LOCAL_UNITARY_BCC_Q0_DILATION.md](LOCAL_UNITARY_BCC_Q0_DILATION.md):

$$
U_{\mathbb N}=
\begin{pmatrix}
A_{\mathbb N}&D_{A^*}\\
D_A&-A_{\mathbb N}^\dagger
\end{pmatrix},
\qquad
D_A=\frac1{\sqrt2}I+\left(1-\frac1{\sqrt2}\right)P_0\otimes P_{\rm in},
$$

with the analogous $D_{A^*}$ using the outgoing head projector. This is an exact
unitary on

$$
\mathcal H_{\rm scar}\oplus\mathcal H_{\rm leak}.
$$

It gives a clean mathematical completion of the statement "mixed-normal hops
enter the unresolved bath." The visible flavor object is not $A$ alone; it is
the visible compression or Schur response obtained after specifying how
$\mathcal H_{\rm leak}$ is read.

`C:9` for the local dilation theorem. `C:4` as a physical boundary model until
the leak sector is identified with the actual mixed-normal BCC channels and the
mass readout is proven to see the survival compression.

## 5. Family Framing

The BCC body diagonals, after antipodal pairing, give four primitive exits
forming a tetrahedron. A vacuum-framed boundary selects one exit and leaves
three residual exits. These define the residual family space

$$
\mathcal H_{\rm fam}\simeq\mathbb C^3
$$

with basis

$$
u=\frac1{\sqrt3}(1,1,1),\qquad
a=\frac1{\sqrt6}(2,-1,-1),\qquad
b=\frac1{\sqrt2}(0,1,-1).
$$

The exact BCC boundary defect therefore has the universal carrier

$$
\mathcal H_{\rm bd}
=
\mathcal H_{\rm scar}\otimes\mathcal H_{\rm fam}\otimes\mathcal H_{\rm charge}.
$$

The silver transfer acts on $\mathcal H_{\rm scar}$. Family projectors and
sector readouts act on $\mathcal H_{\rm fam}$ and $\mathcal H_{\rm charge}$.

This separation is the category guardrail:

$$
\text{silver is radial},\qquad
\text{generations are framed ports},\qquad
\text{sector differences are readouts}.
$$

`C:8` for the tetrahedral finite geometry; `C:5` for taking the three residual
ports as the primitive boundary family axiom.

## 6. Schur Readouts

Once the universal boundary object is fixed, each sector is a coupling map into
it. The readout is not allowed to change the BB silver transfer.

For sterile or propagation-like sectors the object is one-sided:

$$
\Sigma_f(z)=V_f^\dagger(z-H_Q)^{-1}V_f.
$$

For active Dirac mass sectors the object is two-sided:

$$
B_f(z)=V_{f,R}^\dagger(z-H_Q)^{-1}V_{f,L}.
$$

The exact BCC model should therefore be judged by whether the maps $V_f$ are
forced by charge, Higgs, color, framing, and locality. The flavor numbers are
not permitted as inputs to $V_f$.

Prototype readouts:

| Sector | Readout type | Target |
|---|---|---|
| neutrino | sterile one-sided | $K_\nu=\epsilon^2P_u+P_b$ |
| charged lepton | active two-sided | Koide cone plus $\theta_e=-2\pi/3-2/9$ holonomy |
| up quark | colored active coherent | Taylor residue $(1/4,1/\sqrt2,1)$ |
| down quark | colored active diffusive | regular-shell residue $(1,1/\sqrt3,\sqrt{2/3})$ |
| CKM/PMNS | relative left-frame holonomy | mixing angles and phases |

`C:7` as the correct formal architecture; individual sector certainty remains
as in their notes.

## 7. What Must Be Derived Next

The exact boundary defect model is not finished by writing $q=0$. The finite
mixed-normal Feshbach derivation shows that a hard relative-depth gap is
sufficient, and
[BCC_RELATIVE_DEPTH_ORDER_PARAMETER.md](BCC_RELATIVE_DEPTH_ORDER_PARAMETER.md)
shows that the lowest local reflection-even gap has the form $\Lambda q^2$.
[BCC_POSITIVE_STIFFNESS_BACKREACTION.md](BCC_POSITIVE_STIFFNESS_BACKREACTION.md)
shows that $\Lambda>0$ follows if the boundary admits the mismatch constraint
$K_{\rm rel}=q$.
[BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md](BCC_MICROSCOPIC_LOCKING_AND_RETARDED_READOUT.md)
then gives the minimal microscopic route: the two-face edge has one local edge
clock, so $q$ is the unique antisymmetric clock error. The causal outgoing
compression is exact inside the scattering model, but the physical asymptotic
origin of that condition remains the load-bearing microscopic problem:

$$
\boxed{
\text{explain why the physical BCC boundary realizes forced clock-error
channels as outgoing leads rather than recurrent wedge states.}
}
$$

Concretely, a microscopic implementation must preserve:

1. **Locality:** the single-clock mismatch constraint is generated locally near
   the codimension-two edge.
2. **Unitarity:** mixed-normal leakage is completed by explicit hidden channels,
   matching the local dilation rather than deleting probability.
3. **Covariance:** the construction respects the residual edge symmetries and
   the Clifford charge carrier.
4. **No tuning:** the survival point $\zeta=1/\sqrt2$ comes from the BB branch
   norm and is the trace minimum of the visible transfer, not a chosen spectral
   probe.
5. **Readout separation:** the same scar transfer is used by neutrinos,
   charged leptons, and quarks; sector data enter only through $V_f$.

Until these are proven, the correct status is:

$$
\boxed{
\text{exact conditional BCC defect model: }C:6,\qquad
\text{microscopic physical derivation: }C:3.
}
$$

## 8. Falsifiers

The model is useful because it is fragile. It fails if any of the following are
true:

1. No local BCC boundary rule with an explicit boundary degree can superselect
   $q=0$ without inserting the answer.
2. The exact unitary completion necessarily feeds mixed-normal leakage back into
   the scar at the same order, destroying the silver transfer.
3. The physical mass-return probe is not the survival/trace-minimum point
   $\zeta=1/\sqrt2$.
4. The three residual ports cannot be tied to a framed BCC exit.
5. Sector readouts require changing the radial transfer instead of changing only
   $V_f$.

If these controls fail, then the silver/flavor synthesis is pattern matching. If
they pass, the theory has a real microscopic object:

$$
\boxed{
\text{a BCC edge waveguide whose boundary Green function is the Standard Model
flavor seed.}
}
$$

## 9. Feynman Picture

The BCC boundary defect is an impedance-matched edge waveguide. A visible
fermion reaches the edge, enters two synchronous body-diagonal corridors, and
survives in the scar branch with amplitude $1/\sqrt2$. The mixed corridors do
not vanish; they are the unresolved Brownian bath. At the survival band edge the
orientation-preserving radial transfer has eigenvalues $\sqrt2\pm1$, so the
decaying return is $\epsilon=\sqrt2-1$. Dirac masses are two-sided returns,
hence the common unit $\eta=\epsilon^2$.

The boundary does not know "electron" or "top quark" at this stage. It knows
only radial decay, framed ports, and coupling doors. Flavor is what different
SM charge sectors read from the same defect.
