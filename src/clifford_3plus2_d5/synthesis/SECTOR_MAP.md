# Sector Map

This note maps which mechanism is allowed to explain which object. It is a
guardrail against category errors. The synthesis is not a single magic formula
for every particle. It is a layered theory:

$$
\boxed{
\text{QCA carrier}
+
\text{Clifford Standard Model charges}
+
\text{Higgs/gauge breaking}
+
\text{boundary spectral response}
}
$$

Different observed particles live in different layers of this tree.

## 1. Universal Carrier Layer

The BCC QCA supplies the spacetime carrier:

$$
H_R(k)=\sigma\cdot k,\qquad
H_L(k)=-\sigma\cdot k,
$$

and the Dirac assembly gives

$$
H_D(k)=\alpha\cdot k.
$$

The Clifford/Pati-Salam carrier supplies the internal one-generation charge
space:

$$
\mathrm{Cl}(0,10)\Rightarrow S_+\simeq\mathbb C^{16},
$$

with

$$
\mathrm{Spin}(10)
\supset
SU(4)\times SU(2)_L\times SU(2)_R
\supset
SU(3)_c\times SU(2)_L\times U(1)_Y.
$$

This layer explains the stage and labels. It does not by itself explain flavor,
mass ratios, confinement, or composite meson physics.

Certainty: `C:8` for the current carrier construction; `C:1` for claiming this
alone derives flavor.

## 2. Quark Sector

Quark masses are the natural home of the Pell/depth/boundary-response machinery.
The current radial form is

$$
m_q\sim A_q\,\eta^{k_q}v_q,
\qquad
\eta=(\sqrt2-1)^2.
$$

The proposed quark flesh is spectral:

$$
\tilde H\to\text{zero-mode pole},
\qquad
H\to\text{gapped Hermitian bath}.
$$

In this reading, the top is a marginal boundary pole, the up tower is distance
from that pole, and the down tower is a Hermitian shell with no marginal zero
mode.

Quark-specific ingredients:

| Object | Mechanism | Status |
|---|---|---|
| shared small parameter | Pell transfer ratio $\eta$ | `C:9` algebra, `C:3` BCC derivation |
| up coefficients | coherent Taylor entry $(1/4,1/\sqrt2,1)$ | `C:6` conditional |
| down clean coefficients | Hermitian shell residues $(1,1/\sqrt3,\sqrt{2/3})$ | `C:4-C:7` |
| up/down assignment | zero-mode pole versus gapped bath | `C:3` |
| CKM angles | boundary transfer/phase response | `C:5-C:6` |
| CKM phase | boundary holonomy/chiral coin | `C:3-C:6` |

This sector is where the flavor sidecars have the most structure. It is also
where the main dynamical premise remains:

$$
\boxed{
\tilde H\text{ coherently hits the zero mode; }H\text{ does not.}
}
$$

Certainty: `C:5` for the quark-sector synthesis as a conditional theory.

## 3. Charged Lepton Sector

Charged leptons are not quarks without color. They lack the color shell that
enters the quark down-sector count

$$
1_{\rm direct}+2_{\rm BCC}+3_{\rm color}.
$$

Therefore the quark Clebsches should not be copied into the lepton sector by
deleting the color term. Charged leptons need their own colorless boundary
response:

$$
\Sigma_e(z)=V_e^\dagger(z-H_{Q,e})^{-1}V_e.
$$

Koide geometry may be a clue. The Koide audit identifies the cone condition

$$
K=\frac23
$$

as a trace-axis angle relation. In the neutrino-prototype language, the better
charged-lepton ansatz is to read the square-root mass vector as an active
boundary residue:

$$
(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau})
=
\sqrt{\mu_e}
\left[
{1\over\sqrt2}u
+
{1\over\sqrt2}(\cos\theta_e a+\sin\theta_e b)
\right].
$$

This makes Koide the trace/traceless equipartition condition and leaves the
BCC residual-plane angle $\theta_e$ as the missing theorem. The detailed ansatz
is [CHARGED_LEPTON_BCC_ANSATZ.md](CHARGED_LEPTON_BCC_ANSATZ.md), and the first
active-resolvent derivation attempt is
[CHARGED_LEPTON_RESOLVENT_DERIVATION.md](CHARGED_LEPTON_RESOLVENT_DERIVATION.md).

Certainty: `C:6` for keeping charged leptons in a separate colorless response
sector; `C:4` for Koide as compatible geometry; `C:3` for the BCC active
residue ansatz; `C:1` for directly applying the quark color-shell Clebsches to
leptons.

## 4. Neutrino Sector

Neutrinos naturally belong to the sterile/boundary Schur sector. This is the
cleanest mass theorem in the current repository. The dimensionless object from
the boundary-response sidecar is

$$
K_\nu=\epsilon^2P_u+P_b.
$$

In the framed product sterile-chain model,

$$
\widehat\Sigma(2\sqrt2)=\epsilon^2P_u+P_b.
$$

The light effective mass operator is therefore

$$
M_\nu=\mu_\nu K_\nu,
$$

with eigenvalues

$$
(m_1,m_2,m_3)=\mu_\nu(0,\epsilon^2,1).
$$

This predicts normal ordering, one massless light neutrino, and

$$
{\Delta m^2_{21}\over\Delta m^2_{31}}=\epsilon^4.
$$

It is not the same mechanism as the quark Pell radial hierarchy. The remaining
neutrino question is the absolute scale $\mu_\nu$ and the microscopic
Dirac/Majorana/seesaw completion that realizes the effective operator.

Allowed neutrino mechanisms:

| Object | Mechanism | Status |
|---|---|---|
| neutrino eigenvalue ratios | framed sterile Schur response | `C:9` in product model |
| absolute mass scale $\mu_\nu$ | sterile/Higgs boundary scale | open |
| PMNS angles | residual basis plus charged-lepton leakage | conditional |
| Dirac/Majorana nature | microscopic sterile completion | open |
| CP phase | boundary holonomy plus charged-lepton phase word | conditional |

Certainty: `C:9` for the product sterile eigenvalue theorem; `C:7` for the
physical neutrino-core placement; `C:3` for the absolute scale and microscopic
completion.

## 5. Gauge Bosons: Photon, W, Z, Gluons

Gauge bosons are not boundary flavor modes. They belong to gauge symmetry and
Higgs breaking.

Before electroweak breaking:

$$
SU(3)_c\times SU(2)_L\times U(1)_Y.
$$

After the Higgs VEV:

$$
SU(2)_L\times U(1)_Y\to U(1)_{\rm em}.
$$

At tree level,

$$
m_W=\frac{gv}{2},
\qquad
m_Z=\frac{\sqrt{g^2+g'^2}\,v}{2},
$$

and the photon remains massless:

$$
m_\gamma=0.
$$

Gluons remain massless at the gauge-boson level and become confined through
QCD dynamics, not through flavor boundary response.

Therefore:

$$
\boxed{
W,Z\text{ masses belong to the Higgs/gauge sector, not to the flavor sidecar.}
}
$$

The QCA program may provide a lattice carrier and gauge-link dynamics, but the
Pell/depth machinery should not be used to derive $m_W$ or $m_Z$.

Certainty: `C:9` for the Standard Model placement of gauge-boson masses; `C:1`
for using quark flavor hierarchy formulas to explain $W/Z$ masses.

## 6. Higgs Boson

The Higgs field belongs to the electroweak scalar sector. In the current repo,
the Higgs/Yukawa layer is representation-level and static:

$$
Y_{\rm internal}(\Phi)=A(\Phi)+A(\Phi)^T.
$$

The Higgs potential sidecars show a minimal gauge-invariant quartic form once a
Higgs doublet norm is supplied:

$$
V(\Phi)=\lambda(|\Phi|^2-v^2)^2+\text{constant}.
$$

This explains a stabilizing radial form under stated Landau assumptions. It
does not derive the physical Higgs mass from QCA microscopics.

The Higgs also mediates the unresolved up/down boundary question by selecting
legal Yukawa orientations:

$$
u\leftrightarrow \tilde H,
\qquad
d\leftrightarrow H.
$$

But this conjugation does not by itself select coherent versus Hermitian repair.

Certainty: `C:7` for the representation-level Higgs slot and quartic Landau
form; `C:3` for a microscopic Higgs mass derivation.

## 7. Hadrons, Kaons, Pions, B Mesons

Hadrons are composite QCD states. They are downstream of the elementary quark
and gauge sectors.

Examples:

$$
\pi^+\sim u\bar d,
$$

$$
K^0\sim d\bar s,
$$

$$
B^0\sim d\bar b.
$$

Their masses and mixings require:

1. quark masses,
2. CKM structure,
3. QCD confinement,
4. chiral symmetry breaking,
5. weak effective operators for decays and oscillations.

Thus kaons are not primitive outputs of the flavor sidecar. They are precision
tests of the downstream theory. In particular, kaon mixing and CP violation
test the CKM/phase sector plus QCD matrix elements:

$$
K^0\leftrightarrow \bar K^0.
$$

The current synthesis should not claim to derive kaon masses or CP observables
directly. It can only supply upstream quark masses, CKM parameters, and phase
structure that later feed a QCD/effective-field-theory calculation.

Certainty: `C:9` for hadrons as QCD composites; `C:3` for current predictive
power on hadron observables.

## 8. CP And Strong CP

CP appears in several distinct places and should not be collapsed:

| CP object | Sector | Status |
|---|---|---|
| CKM phase | quark boundary holonomy | conditional |
| PMNS phase | sterile/charged-lepton boundary response | conditional |
| BCC lattice CP slot | cubic $T_{2g}$ correction | audited |
| QCD $\theta$ | strong gauge vacuum angle | separate |
| kaon CP violation | downstream CKM + QCD observable | not primitive |

The strong-CP sidecar should be read as a leading-order/audit constraint, not
as a complete all-order QCD solution.

Certainty: `C:7` for the sector separation; `C:3` for a complete CP theory.

## 9. What Belongs Where

| Observed object | Primary sector | Do not explain it by |
|---|---|---|
| quark mass ratios | boundary spectral response, Pell depth | gauge boson mass formulas |
| CKM angles | quark boundary transfer | color alone |
| charged lepton masses | colorless boundary response | quark color-shell Clebsches |
| Koide relation | charged-lepton geometry | exact quark-style depth theorem |
| neutrino masses | sterile Schur response | quark Pell hierarchy alone |
| PMNS | sterile response plus charged-lepton rotation | CKM copy-paste |
| $W,Z$ masses | Higgs/gauge breaking | flavor boundary depth |
| photon | unbroken $U(1)_{\rm em}$ | boundary zero mode |
| gluons | $SU(3)_c$ gauge sector, QCD | family $S_3$ |
| kaons/pions/hadrons | QCD composites | elementary flavor formulas directly |
| strong CP | QCD vacuum angle | CKM phase alone |

## 10. Great Picture

The synthesis tree should be read as:

$$
\boxed{
\text{QCA gives relativistic propagation.}
}
$$

$$
\boxed{
\mathrm{Cl}(0,10)\text{ gives one-generation SM charge labels.}
}
$$

$$
\boxed{
\text{Higgs/gauge breaking gives }W,Z,\gamma\text{ and legal Yukawa doors.}
}
$$

$$
\boxed{
\text{Boundary spectral response gives fermion flavor hierarchies.}
}
$$

$$
\boxed{
\text{QCD turns quarks and gluons into hadrons.}
}
$$

The quark theory is the most developed branch, but it should not be made into
the universal explanation for every particle. The right discipline is to keep
each theorem in its sector.
