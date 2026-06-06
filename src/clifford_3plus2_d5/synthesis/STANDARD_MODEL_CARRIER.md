# Standard Model Scaffold And The Carrier

This note fixes the target before we synthesize the sidecars. The Standard
Model is not a vague inspiration here. It is the known low-energy structure
that fifty years of particle physics has made rigid: gauge symmetry,
representations, anomaly cancellation, electroweak breaking, flavor mixing, and
the empirical fact of three generations. The QCA/BCC program must refine
selected aspects of this structure without wandering into known dead ends.

The synthesis therefore begins with three questions:

1. What is the Standard Model data we must reproduce?
2. Which part of that data is already carried by our Clifford/Pati-Salam
   carrier?
3. Which sectors remain genuinely open for the flavor and mass theory?

## The Standard Model Target

The gauge group of the Standard Model is

$$
G_{\rm SM}=SU(3)_c\times SU(2)_L\times U(1)_Y .
$$

For one generation, written as left-handed Weyl fields, the fermion
representations are

$$
Q=(3,2)_{1/6},\qquad
u^c=(\bar 3,1)_{-2/3},\qquad
d^c=(\bar 3,1)_{1/3},
$$

$$
L=(1,2)_{-1/2},\qquad
e^c=(1,1)_1,\qquad
\nu^c=(1,1)_0 .
$$

The sterile $\nu^c$ is optional in the minimal Standard Model, but natural in
the Spin(10)/Pati-Salam carrier. Including it gives $16$ complex chiral
fermion states per generation.

The hypercharge multiplicity table is

$$
\{1/6:6,\ -2/3:3,\ 1/3:3,\ -1/2:2,\ 1:1,\ 0:1\}.
$$

This table is the minimal representation-level target for the carrier.

Certainty: `C:9`. This is standard representation theory of one SM generation
with a right-handed neutrino.

## What The Standard Model Explains And What It Leaves Free

The Standard Model tightly fixes the gauge interactions once
$G_{\rm SM}$ and the representations are chosen. It also gives powerful
consistency checks:

$$
\text{gauge anomaly cancellation},
\qquad
\text{CPT/Lorentz structure},
\qquad
\text{unitary charged-current mixing}.
$$

But it does not explain the flavor data. The Yukawa matrices are free inputs:

$$
Y_u,\quad Y_d,\quad Y_e,\quad Y_\nu .
$$

After electroweak symmetry breaking these become masses and mixing matrices.
The quark sector contains the CKM matrix, the lepton sector contains PMNS
mixing if neutrino masses are included, and the observed hierarchy spans many
orders of magnitude.

Thus the flavor problem is not to rediscover $G_{\rm SM}$. The real problem is
to explain why the boundary data select the observed mass scales, mixings, and
phases.

Certainty: `C:9` for the Standard Model leaving Yukawa data free. `C:7` for the
programmatic claim that the sidecars should attack the flavor problem through
boundary response rather than through another gauge-group derivation.

## Symmetry As A Sector Guide

The synthesis uses symmetry as a map. A theorem belongs to the sector whose
symmetry it acts on:

| Sector | Symmetry / object | What belongs there |
|---|---|---|
| Gauge carrier | $SU(3)_c\times SU(2)_L\times U(1)_Y$ | SM charges, one-generation representation content |
| Pati-Salam lift | $SU(4)\times SU(2)_L\times SU(2)_R$ | $B-L$, $T_{3R}$, hypercharge origin |
| Spacetime QCA | BCC Weyl/Dirac walk | locality, chirality, lattice CP slots, simulator dynamics |
| Boundary transfer | Schur complement / Weyl function | $\epsilon=\sqrt2-1$, sterile-chain response |
| Family depth | residual $S_3$ and its breaking | $\{0,2,6\}$, path scar, generation spurion |
| Clebsch layer | projectors, nilpotent flags, shell counts | scalar mass coefficients |
| Radial response | spectral measures | pole shifts, residues, actual mass values |
| CP and phases | holonomy, chiral coin, $T_{2g}$ | CKM/PMNS phase locations and lattice CP constraints |

This prevents category errors. For example, a lattice harmonic theorem does not
automatically explain family depth, and an $S_3$ projector count does not
automatically select a mass pole.

Certainty: `C:6`. This is an organizing principle, not a theorem.

## Group Assignment Of The Main Mechanisms

The first cleanup is to say which group each mechanism belongs to. A statement
about one group is not automatically a statement about another.

| Mechanism or object | Group / space | Belongs to | Does not belong to |
|---|---|---|---|
| Color | $SU(3)_c\subset SU(4)$ | internal gauge carrier | family depth |
| Hypercharge | $U(1)_Y$, with $Y=T_{3R}+(B-L)/2$ | Pati-Salam-to-SM breaking | BCC spatial symmetry |
| Weak isospin | $SU(2)_L$ | chiral electroweak gauge sector | residual family ports |
| Right isospin before breaking | $SU(2)_R$ | Pati-Salam lift | low-energy unbroken gauge group |
| $B-L$ | Cartan direction in $SU(4)$ | Pati-Salam lift | BCC point group |
| Family depth | residual $S_3$ on boundary ports, broken to $Z_2$ by a path scar | boundary flavor sector | color $SU(3)_c$ |
| Scalar Clebsches | regular $S_3$ shell, nilpotent flags, defect projectors | scalar mass readout | CKM current amplitudes |
| Radial masses | spectral data of $H_Q$ through $\Sigma_f(z)$ | boundary Green-function sector | finite group theory alone |
| CKM/PMNS phases | holonomy / chiral coin / boundary phase word | phase sector | tree-level depth spectrum |
| Lattice CP slot | $T_{2g}$ irrep of the cubic point group | BCC/QCA continuum correction | flavor transfer root |

This table is load-bearing. It rules out several attractive but false moves:

1. A color $SU(3)_c$ theorem is not a family-generation theorem.
2. A BCC lattice harmonic theorem is not automatically a flavor-depth theorem.
3. An $S_3$ projector count can give a Clebsch candidate but not a radial mass
   pole.
4. The symbol $\epsilon$ in the silver transfer root is not the BCC lattice
   expansion parameter.

Certainty: `C:7` as a classification of the current construction; `C:9` for
the internal gauge assignments of color and hypercharge.

## Internal Gauge Chain

The internal gauge chain is:

$$
\mathrm{Spin}(10)
\supset
SU(4)\times SU(2)_L\times SU(2)_R
\supset
SU(3)_c\times SU(2)_L\times U(1)_Y .
$$

Color is not an accidental threefold family label. It is the
trace-orthogonal part of the centralizer of $B-L$ inside $SU(4)$:

$$
SU(3)_c \subset \mathrm{Cent}_{SU(4)}(B-L).
$$

Hypercharge is not a BCC or flavor symmetry. It is the one-dimensional
Pati-Salam remnant

$$
U(1)_Y,\qquad
Y=T_{3R}+\frac{B-L}{2}.
$$

Thus color and hypercharge are carrier/gauge data. They constrain flavor
mechanisms, but they are not themselves mechanisms for three generations.

Certainty: `C:9`.

## Boundary Flavor Groups

The active flavor mechanisms live in boundary residual spaces, not in the
Standard Model gauge group.

The family-depth skeleton uses a three-port residual space with labels such as
$(u,a,b)$. The unbroken residual symmetry is

$$
S_3,
$$

with decomposition

$$
3=1\oplus 2.
$$

An $S_3$-invariant operator on this space has at most two distinct eigenvalues.
Therefore a depth spectrum with three distinct values,

$$
\{0,2,6\},
$$

requires breaking the residual $S_3$. The path scar

$$
u-a-b
$$

keeps a $Z_2$ endpoint symmetry and gives

$$
D_{\rm scar}=2\Delta(P_3),\qquad
\operatorname{Spec}D_{\rm scar}=\{0,2,6\}.
$$

The scalar Clebsch layer also uses $S_3$, but differently: it uses the regular
shell

$$
\mathrm{Reg}(S_3)=1_{\rm triv}\oplus 1_{\rm sign}\oplus 2_{\rm std}\oplus
2_{\rm std}.
$$

This is why the down-sector rank-$5$ candidate is available but not selected:
regular minus a one-dimensional line can mean either regular-minus-trivial or
regular-minus-sign.

Certainty: `C:8` for the $S_3$ representation facts; `C:6` for the path scar as
a physical family-depth mechanism.

## QCA/BCC Spatial Groups

The BCC QCA belongs to spatial lattice symmetry, not internal gauge symmetry.
Its nearest-neighbor directions are the eight body diagonals

$$
(\pm1,\pm1,\pm1).
$$

The relevant point group is the cubic/octahedral group. We use:

$$
O_h
$$

for the full cubic point group including inversion, and

$$
O\subset SO(3)
$$

for the orientation-preserving rotational subgroup. Degree-$2$ momentum
polynomials decompose under $O_h$ as

$$
\mathrm{span}(k_x^2,k_y^2,k_z^2,k_yk_z,k_zk_x,k_xk_y)
=A_{1g}\oplus E_g\oplus T_{2g}.
$$

The CP sidecar's leading lattice CP slot lives in $T_{2g}$. The strong-CP
audits track the $A_{2u}$-type theta channel and find it absent in the audited
construction.

The body-diagonal rotation

$$
(x,y,z)\mapsto(y,z,x)
$$

is a $Z_3$ subgroup of the rotational cubic group. On the eight BCC body
diagonals it has cycle structure $(1,1,3,3)$. Its spin-$1/2$ lift is an
$SU(2)$ matrix $U_3$ with

$$
U_3^3=-I,
$$

as expected for the double cover of a $2\pi$ spinor rotation.

There is also a coefficient-Walsh alphabet on the eight hop directions, built
from characters of the cube vertices. It is useful as a diagnostic, but it is
not by itself a covariant physical group label. The depth-hop-Walsh sidecar
explicitly killed the attempt to identify family depth with a clean
parity-selected BCC coefficient tower.

Certainty: `C:9` for the BCC direction set and cubic decomposition; `C:8` for
the killed coefficient-Walsh depth mechanism.

## Our Carrier

The carrier used by the repo is the real chiral spinor carrier of
$\mathrm{Cl}(0,10)$. The full real module has dimension $64$. The chirality
projector has rank $32$:

$$
\dim_{\mathbb R} S_+=32 .
$$

A commuting complex structure $J$ turns this into a complex $16$:

$$
(S_+,J)\simeq \mathbb C^{16}.
$$

This is the carrier-level meaning of the Spin(10) chiral $16$ in the repo.

Certainty: `C:9` for the exact Clifford construction and rank statement.

## Pati-Salam Factorization

The useful factorization is

$$
\mathrm{Cl}(0,10)\simeq \mathrm{Cl}(0,6)\ \widehat{\otimes}\ \mathrm{Cl}(0,4),
$$

which exposes

$$
\mathrm{Spin}(6)\times \mathrm{Spin}(4)
\simeq
SU(4)\times SU(2)_L\times SU(2)_R .
$$

The $SU(4)$ factor contains color and $B-L$. The $SU(2)_R$ factor supplies
$T_{3R}$. Hypercharge is the Pati-Salam combination

$$
Y=T_{3R}+\frac{B-L}{2}.
$$

In the exact real Clifford normalization used in code, the physical observable
is implemented as

$$
Y_{\rm phys}=\frac{1}{2}T_{3R}^{\rm raw}
             +\frac{1}{3}(B-L)^{\rm raw}.
$$

With this normalization, the carrier reproduces the Standard Model
hypercharge table exactly.

Certainty: `C:9` for the algebraic extraction and hypercharge spectrum under
the chosen Pati-Salam factorization.

## Carrier Field Content

The complex $16$ decomposes under $G_{\rm SM}$ as:

| Field | Representation | Hypercharge | Complex multiplicity |
|---|---:|---:|---:|
| $Q$ | $(3,2)$ | $1/6$ | $6$ |
| $u^c$ | $(\bar 3,1)$ | $-2/3$ | $3$ |
| $d^c$ | $(\bar 3,1)$ | $1/3$ | $3$ |
| $L$ | $(1,2)$ | $-1/2$ | $2$ |
| $e^c$ | $(1,1)$ | $1$ | $1$ |
| $\nu^c$ | $(1,1)$ | $0$ | $1$ |

Thus

$$
6+3+3+2+1+1=16.
$$

This is the clean carrier result: one complete SM generation, including a
sterile neutrino, is present.

Certainty: `C:9`.

## What The Carrier Does Not Yet Give

The carrier does not derive:

$$
N_{\rm gen}=3,
$$

nor the observed Yukawa matrices, nor the CKM/PMNS angles and phases, nor the
radial mass scales. These are the flavor and mass theory tasks.

The killed sidecars are important here. Spin(8) triality, broken triality,
exceptional-algebra candidates, topology/anomaly forcing, and raw BCC
hop/Walsh depth selection do not currently derive three generations. Therefore
the synthesis should not treat generation multiplicity as already solved by
the carrier.

Certainty: `C:8` for the local no-go evidence from the killed sidecars. `C:5`
for the broad conclusion that a successful flavor theory must introduce a
genuine family-breaking or boundary-selection mechanism.

## The QCA/BCC Ansatz Around The Carrier

The spacetime ansatz is the Bialynicki-Birula BCC Weyl walk. It uses eight body
diagonal hops and has right/left Weyl continuum limits

$$
H_R(k)=\sigma\cdot k,\qquad
H_L(k)=-\sigma\cdot k.
$$

The Dirac assembly uses

$$
H_D(k)=\alpha\cdot k,
$$

with the internal carrier tensored in:

$$
\mathbb C^4_{\rm Dirac}\otimes \mathbb R^{32}_{\rm internal}.
$$

Equivalently, once $J$ is used, this is the physical
$\mathbb C^4\otimes\mathbb C^{16}$ carrier, but the code keeps the exact real
$\mathbb R^{32}$ internal representation.

The BCC/QCA layer is the proposed microscopic arena for locality, chirality,
gauge links, Higgs/Yukawa insertions, and eventually boundary spectral
selection. It is not yet a derivation of the full Standard Model flavor data.

Certainty: `C:7` for the implemented QCA arena. `C:3` for its future role as a
forward flavor selector until the spectral selection mechanism is proved.

## First Boundary Between Known Physics And New Theory

The known-physics scaffold ends here:

$$
G_{\rm SM},\quad \text{one chiral }16,\quad
\text{Pati-Salam hypercharge},\quad
\text{BCC Weyl/Dirac arena}.
$$

The new theory begins when we ask why the boundary response should select:

$$
\epsilon=\sqrt2-1,\qquad
D=\{0,2,6\},\qquad
C_u=\left(\frac14,\frac1{\sqrt2},1\right),
$$

the down-sector Clebsches, the CKM/PMNS phases, and finally the radial spectral
measures that become masses.

This is the scaffold for the rest of the synthesis. Every later theorem should
be placed in one of these sectors, and every claim that crosses sectors must
state the map that carries it across.
