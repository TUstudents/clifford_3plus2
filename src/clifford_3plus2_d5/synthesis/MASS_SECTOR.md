# Mass Sector

This note is the mass-sector map for the synthesis sidecar. It separates the
objects that share the word "mass" but do not share the same mechanism.

The basic discipline is:

$$
\boxed{
\text{mass is a sector response, not one universal formula.}
}
$$

The BCC/QCA carrier supplies the relativistic slot for mass. The Clifford
carrier supplies Standard Model labels. The Higgs/gauge sector supplies
gauge-boson masses and legal Yukawa doors. The boundary Schur sector supplies
fermion flavor spectra.

Certainty: `C:8` for the carrier mass slot; `C:7` for the sector separation as
the synthesis scaffold.

## 1. The Carrier Mass Slot

The spacetime QCA mass layer is the usual Dirac slot:

$$
H_m(k)=\alpha\cdot k\otimes I_{\rm int}+\beta\otimes M_{\rm int}.
$$

For a scalar internal mass $M_{\rm int}=mI$, the implemented audit verifies

$$
\beta^2=I,\qquad \{\alpha_i,\beta\}=0,
$$

and hence

$$
H_m(k)^2=(|k|^2+m^2)I.
$$

This proves that the BCC Dirac carrier can host mass. It does not derive a
Standard Model mass spectrum, because physical fermion masses require the
Higgs/Yukawa representation structure:

$$
\bar Q_L\tilde H u_R,\qquad \bar Q_L H d_R,\qquad \bar L_L H e_R,
$$

and some neutrino completion.

Certainty: `C:8` for the implemented carrier slot; `C:1` for treating the
scalar control as the observed spectrum.

## 2. Neutrino Mass Theorem

The neutrino sector is the cleanest mass branch in the repository. Its primitive
object is the framed sterile Schur response, not a quark-like Pell depth ladder.

After vacuum framing, the residual family basis is

$$
u={1\over\sqrt3}(1,1,1),\qquad
a={1\over\sqrt6}(2,-1,-1),\qquad
b={1\over\sqrt2}(0,1,-1).
$$

Let $P_u,P_a,P_b$ be the corresponding rank-one projectors. The product sterile
bath is

$$
H_Q=H_{\rm chain}\otimes I_{\rm family}.
$$

For the semi-infinite unit chain, the Weyl function is

$$
m(z)={z-\sqrt{z^2-4}\over2},
$$

with branch fixed by

$$
\lim_{z\to\infty}z\,m(z)=1.
$$

At the transfer probe $z=2\sqrt2$,

$$
m(2\sqrt2)=\sqrt2-1=\epsilon.
$$

The exact normalized product response is

$$
\widehat\Sigma(z)=m(z)^2P_u+P_b,
$$

so at the probe

$$
\boxed{
K_\nu=\widehat\Sigma(2\sqrt2)=\epsilon^2P_u+P_b.
}
$$

This is a forward theorem inside the product sterile model. The tensor family
factor gives equal $u/b$ sterile returns and zero cross-return; the half-line
Weyl function gives the transfer value. The negative control without the family
factor becomes rank one and develops $u/b$ cross terms, so the family product is
not bookkeeping.

Certainty: `C:9` for the exact Schur/Weyl theorem in the stated product model;
`C:7` as a physical neutrino-core mechanism until the product sterile bath is
derived from microscopic QCA boundary dynamics.

## 3. Light Neutrino Eigenvalues

The light effective neutrino mass operator is the dimensionful version

$$
M_\nu=\mu_\nu K_\nu
=\mu_\nu(\epsilon^2P_u+P_b).
$$

In the residual eigenbasis $(a,u,b)$,

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

Thus the mass eigenstates and eigenvalues are

| state | vector | mass |
|---|---|---|
| $\nu_1$ | $a=(2,-1,-1)/\sqrt6$ | $m_1=0$ |
| $\nu_2$ | $u=(1,1,1)/\sqrt3$ | $m_2=\mu_\nu\epsilon^2$ |
| $\nu_3$ | $b=(0,1,-1)/\sqrt2$ | $m_3=\mu_\nu$ |

Equivalently, with

$$
\eta=\epsilon^2=(\sqrt2-1)^2=3-2\sqrt2,
$$

the spectrum is

$$
\boxed{
(m_1,m_2,m_3)=\mu_\nu(0,\eta,1).
}
$$

The model therefore predicts normal ordering and one massless light neutrino.
It also predicts the oscillation-ratio invariant

$$
{\Delta m^2_{21}\over\Delta m^2_{31}}
=
{m_2^2-m_1^2\over m_3^2-m_1^2}
=
\eta^2
=
\epsilon^4
=17-12\sqrt2
\approx0.0294373.
$$

This is the main point: the neutrino masses are not merely fitted. Their
relative spectrum is the eigenvalue spectrum of the sterile Schur operator.

Certainty: `C:9` for the eigenvalues of $K_\nu$; `C:8` for the exact
mass-squared ratio inside the effective light-neutrino operator; `C:7` for the
normal-ordering physical interpretation.

## 4. Absolute Scale

The theorem above fixes the dimensionless spectrum. The remaining dimensionful
scale is

$$
\mu_\nu.
$$

If the atmospheric splitting is used to set the scale, then

$$
\mu_\nu=\sqrt{\Delta m^2_{31}},
$$

because $m_1=0$ and $m_3=\mu_\nu$. With the representative scale

$$
\Delta m^2_{31}=2.52\times10^{-3}\ {\rm eV}^2,
$$

one obtains

| mass | value |
|---|---|
| $m_1$ | $0$ |
| $m_2$ | $8.61\times10^{-3}\ {\rm eV}$ |
| $m_3$ | $5.02\times10^{-2}\ {\rm eV}$ |
| $m_1+m_2+m_3$ | $5.88\times10^{-2}\ {\rm eV}$ |

The corresponding solar splitting is then

$$
\Delta m^2_{21}
=
\epsilon^4\Delta m^2_{31}
\approx
7.42\times10^{-5}\ {\rm eV}^2.
$$

So the repository has derived all three light neutrino masses in the following
precise sense:

$$
\boxed{
\text{the spectrum is fixed up to one dimensionful scale, and one measured}
\ \Delta m^2\ \text{then determines all three masses.}
}
$$

It has not yet derived the eV scale $\mu_\nu$ from the microscopic QCA/Higgs
boundary dynamics. That is the remaining scale theorem.

Certainty: `C:8` for the scale inference from one measured splitting; `C:3` for
an internal derivation of $\mu_\nu$ until a microscopic sterile/Higgs scale
calculation is supplied.

## 5. PMNS Is Adjacent But Not The Mass Theorem

The residual basis matrix

$$
U_{\rm TBM}=[a\ u\ b]
$$

is the neutrino-side mixing frame. Charged-lepton leakage supplies a conditional
left rotation:

$$
U_{\rm PMNS}=R_e^\dagger U_{\rm TBM}.
$$

The sidecar derives the leakage sine under a two-step assumption:

$$
\sin\theta_e=\sqrt{3\over2}\,\epsilon^2,
$$

and the conditional phase word gives

$$
\phi_e=-{5\pi\over12}.
$$

Then

$$
\sin^2\theta_{13}={3\over4}\epsilon^4,
$$

with the numerical PMNS texture recorded in the boundary-response sidecar.

This should be kept separate from the mass eigenvalue theorem. $K_\nu$ gives
the neutrino mass eigenvalues. The charged-lepton rotation converts the
neutrino eigenbasis into PMNS observables.

Certainty: `C:9` for the TBM residual basis; `C:6` for the conditional PMNS
texture because the phase-word selection remains conditional.

## 6. Quark Masses

Quark masses are a different branch. Their working form is

$$
m_q\sim A_q\,\eta^{k_q}v_q,
\qquad
\eta=(\sqrt2-1)^2.
$$

The current quark picture is:

$$
\tilde H\to\text{zero-mode pole},
\qquad
H\to\text{gapped Hermitian bath}.
$$

The up-sector coefficients are coherent-entry coefficients,

$$
(1,1/\sqrt2,1/4)
$$

in heavy-to-light order, while the clean down-sector coefficients are Hermitian
shell residues,

$$
(1,1/\sqrt3,\sqrt{2/3})
$$

in light-to-heavy convention as used in the quark notes. The strongest open
issue is the dynamical assignment of coherent versus Hermitian repair to the
up/down Higgs doors.

Certainty: `C:6` for the conditional quark texture; `C:3` for the zero-mode
pole versus gapped-bath dynamical assignment.

## 7. Charged-Lepton Masses

Charged leptons are not covered by the quark color-shell Clebsches. Their mass
sector needs a colorless boundary response:

$$
\Sigma_e(z)=V_e^\dagger(z-H_{Q,e})^{-1}V_e.
$$

The Koide cone is a real geometric constraint:

$$
{m_e+m_\mu+m_\tau\over(\sqrt{m_e}+\sqrt{m_\mu}+\sqrt{m_\tau})^2}
={2\over3},
$$

but the repository has not derived the charged-lepton masses from a boundary
Hamiltonian.

The neutrino prototype suggests the better BCC ansatz. Charged-lepton masses
should be read as squared active boundary amplitudes, not as sterile
eigenvalues. Define

$$
w_e=(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau}).
$$

Koide is exactly the statement that the normalized $w_e$ has equal trace and
traceless norm:

$$
\widehat w_e
=
{1\over\sqrt2}u
+
{1\over\sqrt2}
(\cos\theta_e\,a+\sin\theta_e\,b).
$$

Thus the charged-lepton mass problem reduces to deriving one BCC residual-plane
angle and one electroweak scale. This extension is laid out in
[CHARGED_LEPTON_BCC_ANSATZ.md](CHARGED_LEPTON_BCC_ANSATZ.md), with the first
resolvent derivation attempt in
[CHARGED_LEPTON_RESOLVENT_DERIVATION.md](CHARGED_LEPTON_RESOLVENT_DERIVATION.md)
and the explicit minimal boundary Hamiltonian in
[CHARGED_LEPTON_MINIMAL_HAMILTONIAN.md](CHARGED_LEPTON_MINIMAL_HAMILTONIAN.md).

Certainty: `C:4` for Koide as compatible geometry; `C:1` for treating Koide
alone as a mass derivation; `C:3` for the charged-lepton BCC ansatz until the
active boundary bath is built.

## 8. Gauge-Boson And Higgs Masses

Gauge-boson masses belong to electroweak symmetry breaking:

$$
m_W={gv\over2},
\qquad
m_Z={\sqrt{g^2+g'^2}\over2}v,
\qquad
m_\gamma=0.
$$

These are not boundary flavor depths. The Higgs mass belongs to the scalar
potential/radial-stabilization branch, not to the sterile neutrino core or the
quark Pell hierarchy.

Certainty: `C:9` for the Standard Model placement; `C:3` for a microscopic QCA
derivation of the Higgs radial parameters.

## 9. Mass-Sector Ledger

| Sector | Mass object | Status |
|---|---|---|
| carrier | $\beta\otimes M_{\rm int}$ slot | `C:8` |
| neutrinos | $\mu_\nu(0,\epsilon^2,1)$ | `C:9` ratios, `C:3` absolute scale |
| PMNS | $R_e^\dagger U_{\rm TBM}$ | `C:6` conditional |
| quarks | $A_q\eta^{k_q}v_q$ | `C:6` conditional |
| charged leptons | colorless $\Sigma_e(z)$ | open |
| $W,Z,\gamma$ | Higgs/gauge breaking | `C:9` placement |
| Higgs | scalar radial potential | partial |

The first genuinely sharp mass theorem is the neutrino spectrum:

$$
\boxed{
M_\nu=\mu_\nu(\epsilon^2P_u+P_b),
\qquad
(m_1,m_2,m_3)=\mu_\nu(0,\epsilon^2,1).
}
$$

The geometric and physical reading of this result is expanded in
[NEUTRINO_PROTOTYPE.md](NEUTRINO_PROTOTYPE.md). That note should be treated as
the prototype standard for the other mass sectors: geometry first, symmetry
second, Schur recirculation third.

The next mass-sector task is therefore not to rediscover this theorem. It is to
derive the remaining scales and boundary Hamiltonians:

1. the microscopic origin of $\mu_\nu$;
2. the quark zero-mode/gapped-bath Higgs-defect assignment;
3. the charged-lepton colorless response Hamiltonian.
