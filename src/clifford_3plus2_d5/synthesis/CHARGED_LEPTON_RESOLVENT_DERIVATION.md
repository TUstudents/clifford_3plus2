# Charged-Lepton Resolvent Derivation

This note attempts the charged-lepton analog of the neutrino theorem. The goal
is to derive the Koide trace/traceless equipartition and the residual-plane
angle from a colorless active BCC/Higgs boundary resolvent.

The result is strong but not yet closed:

$$
\boxed{
\begin{gathered}
\text{equipartition is derived from the selected Higgs port}\\
\text{plus a coherent BCC two-path trace enhancement}
\end{gathered}
}
$$

The residual-plane angle is reduced to one sharp torsion rule:

$$
\boxed{
\theta_e=-{2\pi\over3}-{2\over9}.
}
$$

This predicts the charged-lepton mass ratios at the $10^{-4}$ level once the
overall scale is set. The remaining open theorem is to derive the active
torsion $2/9$ from the microscopic colorless boundary Hamiltonian, rather than
treat it as a counted boundary action.

Certainty: `C:6` for the trace/traceless equipartition under the stated active
BCC return; `C:4` for the full charged-lepton spectrum because the $2/9$
torsion is not yet derived from an explicit $H_{Q,e}$.

## 1. Residue-Amplitude Object

For neutrinos, the mass object is a sterile Schur eigenvalue:

$$
M_\nu=\mu_\nu(\epsilon^2P_u+P_b).
$$

For charged leptons, the natural object is instead a square-root mass amplitude
vector:

$$
w_e=(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau}).
$$

Thus the active boundary object should be a residue-amplitude operator

$$
B_e
=
\operatorname*{Res}_{z=z_e}
C_e^\dagger(z-H_{Q,e})^{-1}V_e,
$$

acting on the selected Higgs/charged-lepton port:

$$
w_e\propto B_e e_1.
$$

The charged-lepton masses are then

$$
m_i=|w_{e,i}|^2.
$$

This is not a full Yukawa-matrix theorem yet. It is a spectrum theorem: it
constructs the square-root mass vector whose squared components are the charged
lepton masses.

Certainty: `C:5` as the active-residue readout.

## 2. Selected Active Port

The selected charged-lepton/Higgs port is

$$
e_1=(1,0,0).
$$

In the residual BCC basis

$$
u={1\over\sqrt3}(1,1,1),\qquad
a={1\over\sqrt6}(2,-1,-1),\qquad
b={1\over\sqrt2}(0,1,-1),
$$

it decomposes as

$$
e_1={1\over\sqrt3}u+\sqrt{2\over3}a.
$$

Equivalently, the trace amplitude is smaller than the traceless amplitude by
$\sqrt2$:

$$
{|\langle u,e_1\rangle|\over|\langle a,e_1\rangle|}
=
{1/\sqrt3\over\sqrt{2/3}}
=
{1\over\sqrt2}.
$$

Certainty: `C:9`.

## 3. Coherent BCC Trace Return And Koide

The colorless active BCC/Higgs boundary can repair the collective trace channel
through two coherent symmetry-related BCC paths. Coherent addition gives the
standard two-path enhancement

$$
1+1\quad\longrightarrow\quad \sqrt2
$$

after normalized path-state projection.

Therefore the residue-amplitude operator on the compressed residual space has
the form

$$
B_e
=
\rho\left(\sqrt2 P_u+R_\theta P_\perp\right),
\qquad
P_\perp=P_a+P_b,
$$

where $R_\theta$ is an orthogonal rotation in the residual $(a,b)$ plane and
$\rho$ is an overall scale.

Acting on the selected port gives

$$
B_e e_1
=
\rho\left[
\sqrt2{1\over\sqrt3}u
+
R_\theta\left(\sqrt{2\over3}a\right)
\right].
$$

Since

$$
\sqrt2{1\over\sqrt3}=\sqrt{2\over3},
$$

the trace and traceless amplitudes become equal:

$$
B_e e_1
\propto
u+R_\theta a.
$$

After normalization,

$$
\widehat w_e
=
{1\over\sqrt2}u
+
{1\over\sqrt2}R_\theta a.
$$

This is exactly Koide:

$$
|\widehat w_t|^2=|\widehat w_\perp|^2,
\qquad
K={2\over3}.
$$

So the trace/traceless equipartition is not a new mass fit. It follows from:

1. the selected active port $e_1$;
2. the residual BCC trace/traceless split;
3. a coherent two-path BCC enhancement of the trace return.

Certainty: `C:6`. The algebra is exact; the physical load is the coherent
two-path trace-return premise.

## 4. Base Plane Angle From The Existing Holonomy

The leptonic boundary holonomy sidecar derives the primitive charged-lepton
loop

$$
\text{SCHUR\_RETURN}\to\text{PARENT\_A3}\to\text{RESIDUAL\_A2}.
$$

Its principal phase is

$$
\phi_{\rm loop}=-{5\pi\over12}.
$$

To read a real residual-plane direction, remove the parent tetrahedral spin
lift contribution

$$
\phi_{A3}={\pi\over4}.
$$

The remaining real residual-plane base angle is

$$
\theta_0
=
\phi_{\rm loop}-\phi_{A3}
=
-{5\pi\over12}-{\pi\over4}
=
-{2\pi\over3}.
$$

This is the exact $Z_3$-rotated base direction in the $(a,b)$ plane.

Certainty: `C:7` inside the existing holonomy model.

## 5. Active Schur Torsion

The empirical charged-lepton square-root vector is not exactly at
$-2\pi/3$. It is displaced by

$$
\delta_e\simeq {2\over9}.
$$

The minimal colorless active Schur interpretation is:

$$
\delta_e
=
{N_{\rm Schur}\over \dim{\rm End}(\mathbb R^3)}
=
{2\over 3^2}
=
{2\over9}.
$$

Here $N_{\rm Schur}=2$ is the second-order active return

$$
P\to Q\to P,
$$

and $\dim{\rm End}(\mathbb R^3)=9$ is the colorless residual source-return
matrix on the three BCC ports.

Thus the proposed active-plane angle is

$$
\boxed{
\theta_e
=
\theta_0-\delta_e
=
-{2\pi\over3}-{2\over9}.
}
$$

This is the one new ingredient. It is a clean count, and it has strong
phenomenology, but it is not yet a theorem until an explicit $H_{Q,e}$ produces
this torsion in the residue phase.

Certainty: `C:3` for the active torsion rule; `C:8` for the numerical
consequence once the rule is accepted.

## 6. Charged-Lepton Mass Formula

With

$$
\theta_e=-{2\pi\over3}-{2\over9},
$$

define

$$
\widehat w_e
=
{1\over\sqrt2}
\left[
u+\cos\theta_e\,a+\sin\theta_e\,b
\right].
$$

In components,

$$
\widehat w_{e,1}
=
{1\over\sqrt6}\left(1+\sqrt2\cos\theta_e\right),
$$

$$
\widehat w_{e,2}
=
{1\over\sqrt6}\left(1+\sqrt2\cos(\theta_e-2\pi/3)\right),
$$

$$
\widehat w_{e,3}
=
{1\over\sqrt6}\left(1+\sqrt2\cos(\theta_e+2\pi/3)\right).
$$

The spectrum is

$$
m_i=\mu_e|\widehat w_{e,i}|^2,
\qquad
\mu_e=m_e+m_\mu+m_\tau.
$$

Using the physical sum as the one scale, the predicted masses are:

| mass | prediction | observed |
|---|---:|---:|
| $m_e$ | $0.510965\ {\rm MeV}$ | $0.510999\ {\rm MeV}$ |
| $m_\mu$ | $105.652344\ {\rm MeV}$ | $105.658376\ {\rm MeV}$ |
| $m_\tau$ | $1776.866065\ {\rm MeV}$ | $1776.860000\ {\rm MeV}$ |

The maximum relative error is about

$$
6.7\times10^{-5}.
$$

The exact formula enforces

$$
K={2\over3}
$$

by construction through trace/traceless equipartition.

Certainty: `C:8` for the calculation from the stated angle and scale; `C:4`
as a forward mass prediction until the $2/9$ torsion is derived from a concrete
boundary Hamiltonian.

## 7. Controls

The torsion order is sharply selected. Keeping the same scale
$\mu_e=m_e+m_\mu+m_\tau$:

| active torsion | angle | predicted masses $(e,\mu,\tau)$ MeV | verdict |
|---|---:|---:|---|
| $0$ | $-120^\circ$ | $(26.923,\ 26.923,\ 1829.183)$ | killed |
| $1/9$ | $-126.366^\circ$ | $(8.181,\ 58.857,\ 1815.992)$ | killed |
| $2/9$ | $-132.732^\circ$ | $(0.510965,\ 105.652,\ 1776.866)$ | survives |
| $1/3$ | $-139.099^\circ$ | $(1.491,\ 168.412,\ 1713.126)$ | killed |

Thus the $2/9$ rule is not a soft fit across a broad valley. The neighboring
Schur-order controls fail badly.

Certainty: `C:8` for the controls as calculations; `C:3` for the physical
selection of second-order active torsion.

## 8. Status

The charged-lepton branch now has a much sharper shape:

$$
\boxed{
B_e
=
\rho\left(\sqrt2P_u+R_{-2\pi/3-2/9}P_\perp\right),
\qquad
w_e\propto B_e e_1.
}
$$

This derives Koide from the active BCC/Higgs port plus two-path trace
enhancement and reduces the mass ratios to a single counted torsion angle.

The honest boundary is:

1. derive the coherent $\sqrt2$ trace return from the actual colorless BCC
   Higgs boundary, not only from a two-path projection;
2. derive the active torsion $\delta_e=2/9$ from an explicit $H_{Q,e}$;
3. derive the overall scale $\mu_e$ from the electroweak/Higgs boundary
   strength.

If those three items close, the charged-lepton masses become the charged
partner of the neutrino theorem:

$$
\text{neutrino: sterile eigenvalues},
\qquad
\text{charged lepton: active residue amplitudes}.
$$

The explicit minimal Hamiltonian realizing the residue is written in
[CHARGED_LEPTON_MINIMAL_HAMILTONIAN.md](CHARGED_LEPTON_MINIMAL_HAMILTONIAN.md).
