# The Common Picture

This note states the synthesized mechanism in one language. The sidecars are
proof notebooks, obstruction notebooks, and simulator notebooks. The common
picture is shorter:

$$
\boxed{
\text{BCC QCA stage}
+
\text{Clifford charge carrier}
+
\text{boundary Brownian defect}
=
\text{flavor and mass response}.
}
$$

A massless Standard Model fermion propagates on the BCC QCA. Its internal
charges live in the chiral Clifford carrier. Its mass and flavor arise when the
visible carrier repeatedly enters an unresolved boundary layer, circulates, and
returns. The boundary layer is not generic noise. It is a structured quantum
bath with residual family symmetry, defects, spectral modes, and holonomy.
The sector placement map is [SECTOR_MAP.md](SECTOR_MAP.md); it separates
quark flavor, charged leptons, neutrinos, gauge bosons, Higgs physics, QCD
composites, and CP observables.
The cleanest realized mass theorem is the neutrino prototype
[NEUTRINO_PROTOTYPE.md](NEUTRINO_PROTOTYPE.md), where BCC residual geometry,
selected symmetry, and sterile Schur recirculation produce
$M_\nu=\mu_\nu(\epsilon^2P_u+P_b)$.

Certainty: `C:7` as the organizing synthesis of the active sidecars. The
carrier pieces are stronger, while the microscopic origin of the family defect
is still the central missing theorem.

## The Stage: BCC QCA

The QCA is the microscopic spacetime machine. It is a local unitary update

$$
\psi(t+1)=U\psi(t),
$$

and the continuum Hamiltonian is read from

$$
U(k)\simeq e^{-iH_{\rm eff}(k)}.
$$

The relevant walk is the Bialynicki-Birula BCC Weyl automaton. At long
wavelength the two chiral blocks become

$$
H_R(k)=\sigma\cdot k,\qquad
H_L(k)=-\sigma\cdot k,
$$

and the chiral assembly gives

$$
H_D(k)=\alpha\cdot k.
$$

Thus the QCA supplies physical spacetime spin and relativistic propagation.
It is the stage, not by itself the flavor mechanism. Asking the free BCC walk
alone to produce three generations, CKM angles, and mass ratios is a category
error. Those are boundary-response questions.

Certainty: `C:8` for the implemented free BCC Dirac carrier; `C:3` for any
claim that raw QCA hops alone derive flavor.

## The Charge Carrier: Clifford Spin

The internal carrier is the chiral spinor of the Clifford/Pati-Salam scaffold:

$$
\mathrm{Cl}(0,10)\Rightarrow S_+,\qquad
S_+\simeq \mathbb C^{16}
$$

after choosing the compatible complex structure. The useful group chain is

$$
\mathrm{Spin}(10)
\supset
SU(4)\times SU(2)_L\times SU(2)_R
\supset
SU(3)_c\times SU(2)_L\times U(1)_Y.
$$

The Pati-Salam coordinates are:

$$
SU(4)\to SU(3)_c\times U(1)_{B-L},
$$

with weak left isospin from $SU(2)_L$ and hidden right isospin from
$SU(2)_R$. Hypercharge is the diagonal remnant

$$
Y=T_{3R}+\frac{B-L}{2},
$$

or in the repository's raw normalization,

$$
Y_{\rm phys}
=
\frac12T_{3R}^{\rm raw}
+
\frac13(B-L)^{\rm raw}.
$$

Electric charge is then

$$
Q_{\rm em}=T_{3L}+Y.
$$

This separates two kinds of spin. The Pauli matrices in the QCA are spacetime
spin. The Clifford spinor is internal gauge-charge spin. Flavor mechanisms must
respect the internal gauge labels, but they do not originate from color or
hypercharge themselves.

Certainty: `C:9` for the hypercharge and electric-charge embeddings; `C:8`
for the one-generation carrier in the local construction.

## The Boundary Layer

The visible QCA carrier is coupled to unresolved boundary degrees of freedom.
For a resolved sector $P$ and a boundary sector $Q$,

$$
H=
\begin{pmatrix}
H_P & V^\dagger\\
V & H_Q
\end{pmatrix}.
$$

Eliminating $Q$ gives the Schur response

$$
G_P(z)=\left[z-H_P-\Sigma(z)\right]^{-1},
\qquad
\Sigma(z)=V^\dagger(z-H_Q)^{-1}V.
$$

For the full visible fermion this becomes the central equation

$$
\left[z-H_{\rm QCA}(k)-\Sigma_f(z)\right]\psi_f=0.
$$

The flavor label $f$ is not a new gauge group. It records which internal charge
sector and which family boundary channel are coupled through $V_f$.

In time language the same object is a quantum memory kernel:

$$
i\dot\psi_{\rm vis}(t)
=
H_{\rm vis}\psi_{\rm vis}(t)
+
\int_0^t K(t-s)\psi_{\rm vis}(s)\,ds,
$$

with

$$
K(t)=V^\dagger e^{-iH_Qt}V.
$$

This is the quantum Brownian picture. The visible particle makes hidden
excursions into a structured bath and returns with memory, phase, and spectral
weight. It is Brownian only in the sense of unresolved recirculation; the full
evolution is still quantum and coherent.

Certainty: `C:9` for the Schur/Feshbach identity; `C:7` for the boundary
response as the common flavor language.

## Defects And Family Depth

If the boundary family space had exact residual $S_3$ symmetry, Schur's lemma
would force too much degeneracy. The permutation representation decomposes as

$$
3=1\oplus2,
$$

so an $S_3$-invariant response has only the form

$$
\Sigma=\alpha P_1+\beta P_2.
$$

That cannot give three distinct family depths. Flavor requires a controlled
defect:

$$
S_3\longrightarrow Z_2
$$

or an equivalent scar selection. The clean depth skeleton is the path

$$
u-a-b,
$$

with graph Laplacian

$$
D_{\rm scar}=2\Delta(P_3),
$$

and spectrum

$$
\operatorname{Spec}D_{\rm scar}=\{0,2,6\}.
$$

This is the current heart of the generation problem. The repeated failures of
triality, topology, raw hop-Walsh selection, and exact algebraic symmetry point
to the same lesson: three generations are not coming from an unbroken exact
symmetry. They require a selected boundary defect or an equivalent
family-breaking spurion.

Certainty: `C:9` for the Laplacian spectrum; `C:6` for the path scar as a
conditional physical mechanism; `C:3` for its microscopic derivation from the
full QCA.

## The Silver Root

The silver root is the boundary transfer contraction

$$
\epsilon=\sqrt2-1=\frac{1}{1+\sqrt2}.
$$

It is not the QCA lattice spacing and not a bulk velocity. It is the stable
root of the boundary transfer equation

$$
\epsilon^2+2\epsilon-1=0.
$$

Equivalently, it is the $N=2$ metallic fixed point

$$
\epsilon=\frac{1}{2+\epsilon}.
$$

This is the right way to ask for a microscopic origin. The lattice should not
be expected to output the irrational directly. It should output an integer
self-consistency coefficient $N_{\rm eff}$, and the fixed point turns that
integer into the metallic root.

The boundary-response sidecar also realizes the same number as the decaying
transfer eigenvalue of a half-line boundary response at the selected probe. A
geometric boundary mode

$$
\psi_n\sim\epsilon^n
$$

then converts depth into hierarchy:

$$
d\in\{0,2,6\}
\quad\Rightarrow\quad
1,\ \epsilon^2,\ \epsilon^6.
$$

The radial Pell theorem refines this language. The silver Pell matrix

$$
T_{\rm Pell}=
\begin{pmatrix}
2&1\\
1&0
\end{pmatrix}
$$

has eigenvalues $1\pm\sqrt2$, so the radial eigenchannel contraction is

$$
\eta=\left|\frac{\lambda_-}{\lambda_+}\right|
=\epsilon^2.
$$

This is why the silver ratio is central. It gives a natural small parameter
without inserting an arbitrary Yukawa number, and radial mass powers should be
tracked as integer powers of $\eta$.

The current caution is sharp. A flat BCC coordinate face keeps four raw
body-diagonal channels, so the naive face-count gives $N_{\rm raw}=4$, not
silver. A coordinate edge keeps two raw channels, but that is only suggestive:
the actual theorem must derive the reduced Dyson equation and its weighted
$N_{\rm eff}=2$ from the scar graph. The explicit scalar edge quotient leaks
out of the diagonal scar, and the bare BB Weyl coin does not remove both
leakage channels.

Certainty: `C:9` for the root identity, boundary recurrence, and Pell
eigenvalue ratio; `C:5` for using the silver/Pell powers as the flavor
hierarchy skeleton; `C:1` for the flat-face BCC silver claim; `C:3` for the
edge-scar origin until the graph reduction is done.

## Mass As Recirculation

Mass couples left and right chirality:

$$
m\bar\psi\psi
\sim
\psi_L^\dagger m\psi_R+\psi_R^\dagger m^\dagger\psi_L.
$$

In the boundary picture, the effective chirality-flipping amplitude is a return
amplitude through the bath:

$$
\Sigma^{(f)}_{LR}(z)
=
V_{L,f}^\dagger(z-H_Q)^{-1}V_{R,f}.
$$

Expanding the resolvent,

$$
\Sigma(z)
=
V^\dagger
\left(
\frac1z+\frac{H_Q}{z^2}+\frac{H_Q^2}{z^3}+\cdots
\right)
V.
$$

Each term has the same physical meaning: enter the boundary, circulate for
some number of hidden steps, and return. The physical mass is a pole or residue
effect of this recirculating response, not a finite group count by itself.

The mass factorization used by the synthesis is

$$
m_f\sim A_f\,\eta^{k_f}R_f.
$$

Here $\eta^{k_f}$ is Pell radial depth, $A_f$ is the entry/residue coefficient,
and $R_f$ is the remaining radial Brownian return response. The radial piece has
spectral form

$$
\Sigma_f(z)=\sum_r\frac{w_{fr}}{z-\lambda_{fr}},
\qquad
w_{fr}>0.
$$

The $\lambda_{fr}$ are hidden radial energies and $w_{fr}$ are return weights.
The present theory has inverse reconstructions of such measures; it still
needs a forward principle that selects them.

The current radial theorem explains part of $A_f$. In the holomorphic up
sector, $A_f$ is a coherent Taylor entry coefficient

$$
\left(\frac14,\frac1{\sqrt2},1\right).
$$

In the Hermitian down sector, the clean bottom coefficient is the
$S_3$ transposition-shell residue

$$
\sqrt{\frac46}=\sqrt{\frac23}.
$$

But the up/down holomorphic/Hermitian assignment is not a theorem of $S_3$.
Both up and down Yukawas flip chirality, so their distinction must be supplied
by the Higgs/electroweak coupling to the defect. Minimal electroweak symmetry
does force

$$
u\text{-type}\leftrightarrow\tilde H,\qquad
d\text{-type}\leftrightarrow H,
$$

but that is a conjugation distinction, not a derivation of coherent versus
Hermitian repair.

The current Higgs/Yukawa representation slot has rank-one charge-shift
building blocks, but it does not yet give rank-one boundary alignment. Each
Higgs orientation spans a two-dimensional active plane, and the neutral static
Yukawa control is rank $2$. A boundary zero-mode line must still be selected by
actual defect dynamics.

Phenomenology must obey the one-scale rule. A Yukawa matrix has all eigenvalues
at one renormalization scale, so the theory should not claim success by
comparing different quark masses to observations quoted at different preferred
scales. The robust signal is instead the shared $\eta$ pattern in CKM and
light/intermediate mass ratios, plus the localized up-Clebsch stress on the
charm coefficient.

Certainty: `C:9` for mass as left-right coupling and for the spectral
Schur form; `C:5` for the factorization; `C:3` for the missing forward radial
selection principle.

## CP And Spin Safety

The boundary defect should mostly act on family and internal-charge channels.
If it couples directly and strongly to spacetime spin, it produces
Lorentz-violating or CP-sensitive operators. This is why the QCA CP and SME
sidecars are consistency constraints rather than flavor generators.

A pure depth tree has no intrinsic holonomy: phases on a tree can be removed by
rephasing. CP needs a loop, a non-normal response, a chiral coin contribution,
or another holonomy carrier. Thus hierarchy and phase are separated:

$$
\text{depth tree}\Rightarrow\text{hierarchy},
\qquad
\text{loop/coin holonomy}\Rightarrow\text{CP phase}.
$$

Certainty: `C:9` for no intrinsic tree holonomy; `C:6` for loop healing as a
minimal graph-native CP location; `C:7` for the BCC chiral coin as a structural
CP-sensitive slot.

## What Must Be Proved

The synthesized theory is strongest where the objects are exact: the BCC
carrier, the Clifford charge carrier, the Schur response, the silver root, and
the representation-theory obstructions. It is weakest at the points where
physics must select a boundary object.

The central open theorem is

$$
\boxed{
\text{derive the }S_3\to Z_2\text{ family defect giving }d=\{0,2,6\}.
}
$$

The radial theorem has become sharper. The small parameter is the Pell
contraction $\eta=(\sqrt2-1)^2$, but the next radial theorems are:

$$
\boxed{
\text{derive the reduced BCC/QCA scar propagator }T_{\rm Pell},
}
$$

$$
\boxed{
\text{derive the dynamical repair-kernel principle assigning up/down sectors,}
}
$$

$$
\boxed{
\text{derive the boundary zero-mode selector inside the Higgs active plane,}
}
$$

and

$$
\boxed{
\text{derive the spectral measures }(\lambda_{fr},w_{fr})
\text{ instead of reconstructing them.}
}
$$

The paper should therefore not present the theory as "QCA explains everything."
The sharper claim is:

$$
\boxed{
\text{QCA gives the relativistic stage,}
\quad
\text{Clifford spin gives the SM charges,}
\quad
\text{boundary Brownian defects give flavor.}
}
$$

That is the common picture.
