# Band-Edge / Marginal-Stability Selection Theorem

This sidecar starts the theorem program that the boundary-update work exposed.
The problem is not how to get the silver number algebraically. That is already
done for the pinned BB edge scar. The problem is why the physical mass readout
selects the point where that silver number appears.

The guiding claim is:

$$
\boxed{
\text{mass readout is the retarded boundary response at marginal visible
stability, i.e. at critical impedance matching between survival and leakage.}
}
$$

This is the missing bridge between "boundary-response reconstruction" and
"boundary-response prediction." In the regular outgoing class proved below, the
spectral probe is no longer chosen after the fact. It is selected by locality,
unitarity, and outgoing boundary conditions.

## 1. The Exact BB Coincidence

For the completed microscopic BB-QCA edge update, the visible same-clock branch
has blocks

$$
B_+=
\begin{pmatrix}
{1\over2}&{i\over2}\\
0&0
\end{pmatrix},
\qquad
B_-=
\begin{pmatrix}
0&0\\
{i\over2}&{1\over2}
\end{pmatrix}
$$

in the symmetric/antisymmetric scar basis. The exact survival identity is

$$
B_+^\dagger B_+ + B_-^\dagger B_-={1\over2}I.
$$

Thus the visible contraction has survival amplitude

$$
c={1\over\sqrt2}.
$$

The stationary visible equation

$$
\zeta\psi_r=B_+\psi_{r-1}+B_-\psi_{r+1}
$$

has transfer

$$
T(\zeta)=
\begin{pmatrix}
{1\over2\zeta}&{i\over2\zeta}\\
-{i\over2\zeta}&2\zeta+{1\over2\zeta}
\end{pmatrix},
$$

so

$$
\det T(\zeta)=1,
\qquad
\tau(\zeta)={\rm tr}\,T(\zeta)=2\zeta+{1\over\zeta}.
$$

The trace is minimized at

$$
{d\tau\over d\zeta}=2-{1\over\zeta^2}=0
\quad\Longrightarrow\quad
\zeta_*={1\over\sqrt2}=c.
$$

At that same point,

$$
\tau(\zeta_*)=2\sqrt2,
\qquad
\lambda_\pm=\sqrt2\pm1,
\qquad
\epsilon=\lambda_-=\sqrt2-1.
$$

The core identity is therefore

$$
\boxed{
\text{BB survival amplitude}
=
\text{transfer trace-minimum point}
\quad\Longrightarrow\quad
\text{silver evanescent root}.
}
$$

Certainty: `C:9` for the finite BB algebra.

## 2. Physical Reading

The determinant-one transfer behaves like a one-dimensional scattering transfer.
When $|\tau|\le2$, transfer eigenvalues are phases and the mode is propagating.
When $\tau>2$, the eigenvalues are reciprocal real numbers:

$$
\lambda_+>1,\qquad 0<\lambda_-<1.
$$

The smaller eigenvalue is the radial evanescent attenuation. For

$$
\tau=2\sqrt2,
$$

one gets

$$
\lambda_-=\sqrt2-1.
$$

This point is outside the closed tight-binding band. It is not a closed
recurrent eigenmode. It is the least-attenuated evanescent response available
inside the BB visible-survival family, because the trace is minimal there.

The Feynman picture is:

$$
\boxed{
\text{the visible particle is read at the impedance point where survival and
outgoing leakage are matched.}
}
$$

If $\zeta<c$, the visible branch is over-attenuated relative to the BB survival
norm. If $\zeta>c$, the transfer trace grows again and the evanescent mode
decays faster. The selected point is the marginal point of the retarded
survival branch, not a free spectral knob.

Certainty: `C:7` as the correct physical interpretation inside the completed
edge-clock scattering update; the regular-retarded theorem below upgrades the
mathematical selection law to `C:9` under its stated assumptions.

## 3. Regular-Retarded Selection Theorem

The theorem is not a symmetry-only statement. It is a retarded asymptotic
statement: a regular outgoing depth response is dominated by the least
attenuated evanescent BB channel.

Define the scalar retarded Weyl attenuation by

$$
m_R(\tau)
=
{\tau-\sqrt{\tau^2-4}\over2},
\qquad \tau>2,
$$

the solution of

$$
m={1\over \tau-m}
$$

with $0<m<1$. For the BB visible transfer,

$$
\lambda(\zeta)
=
m_R(\tau(\zeta)),
\qquad
\tau(\zeta)=2\zeta+{1\over\zeta}.
$$

A regular retarded depth response is a response of the form

$$
G_k=\int_0^\infty w(\zeta)\,\lambda(\zeta)^k\,d\zeta,
$$

where $w$ is a regular outgoing spectral weight: locally smooth near the
maximal channel, integrable with enough decay for the integral to exist, and
not tuned to vanish at the selected point. In the positive scalar case this
means $w(\zeta_*)>0$. Matrix or two-sided sector responses obey the same local
statement componentwise unless the coupling map has an explicit zero at
$\zeta_*$; such a zero is sector data, not a failure of the radial theorem.

Then

$$
\boxed{
G_k
\sim
w(\zeta_*)\sqrt{{2\pi\over 2\sqrt2\,k}}\,
(\sqrt2-1)^k
\qquad (k\to\infty),
}
$$

with

$$
\zeta_*={1\over\sqrt2},
\qquad
\lambda(\zeta_*)=\sqrt2-1.
$$

Thus the silver ratio is selected as the asymptotic contraction of every
regular outgoing retarded depth response of the BB scar. The physical slogan is

$$
\boxed{
\text{the mass readout sees the marginal retarded channel because it is the
unique least-attenuated evanescent channel.}
}
$$

Certainty: `C:9` for the theorem in the stated regular-retarded class; `C:6`
that the mass readout in the flavor sectors belongs to this class.

## 4. Proof

First,

$$
{d\tau\over d\zeta}=2-{1\over\zeta^2}.
$$

The unique positive critical point is

$$
\zeta_*={1\over\sqrt2}.
$$

It is a strict minimum because

$$
{d^2\tau\over d\zeta^2}\bigg|_{\zeta_*}
=4\sqrt2>0.
$$

For the retarded Weyl branch,

$$
{dm_R\over d\tau}
=
{1\over2}
-{\tau\over2\sqrt{\tau^2-4}}
<0
\qquad(\tau>2).
$$

Therefore minimizing $\tau$ maximizes the decaying retarded root
$\lambda(\zeta)=m_R(\tau(\zeta))$. The maximum is unique and occurs at
$\zeta_*$. At that point

$$
\tau(\zeta_*)=2\sqrt2,
\qquad
\lambda_*=\lambda(\zeta_*)=\sqrt2-1.
$$

The maximum is nondegenerate:

$$
{d\lambda\over d\zeta}\bigg|_{\zeta_*}=0,
\qquad
{d^2\lambda\over d\zeta^2}\bigg|_{\zeta_*}
=-4+2\sqrt2<0.
$$

Equivalently,

$$
\beta
:=
-{d^2\over d\zeta^2}\log\lambda(\zeta)\bigg|_{\zeta_*}
=2\sqrt2.
$$

Near $\zeta_*$,

$$
\lambda(\zeta)^k
=
\lambda_*^k
\exp\!\left[-{k\beta\over2}(\zeta-\zeta_*)^2+O(k|\zeta-\zeta_*|^3)\right].
$$

Laplace's method gives

$$
G_k
=
w(\zeta_*)\lambda_*^k
\sqrt{{2\pi\over k\beta}}\,(1+o(1)),
$$

hence

$$
G_k
\sim
w(\zeta_*)\sqrt{{2\pi\over 2\sqrt2\,k}}\,
(\sqrt2-1)^k.
$$

This proves that the exponential depth law is not chosen after the fact. Within
the regular outgoing class, the BB boundary itself selects the silver
contraction as the unique marginal-stability saddle.

## 5. Why The Regularity Assumption Is Necessary

The unrestricted statement "local unitarity alone forces silver" is false.
Choose a singular or hand-filtered spectral weight

$$
w(\zeta)=\delta(\zeta-\zeta_0),
\qquad
\zeta_0\ne {1\over\sqrt2}.
$$

Then

$$
G_k=\lambda(\zeta_0)^k,
$$

so the selected attenuation is not silver. Likewise, if a sector coupling is
tuned so that $w(\zeta_*)=0$, the leading prefactor can be removed and the
next saddle or next Taylor order controls the response.

This is not a defect of the theorem. It is the sharp boundary between
prediction and reconstruction:

$$
\boxed{
\text{silver is forced by regular outgoing retarded depth response, not by
arbitrary bath engineering.}
}
$$

Certainty: `C:9` for the counterexamples to the unrestricted claim; `C:1` for
the unrestricted symmetry-only selection law.

## 6. Why This Is The Right Keystone

The synthesis has two horns:

1. **Symmetry-only horn:** finite symmetry grammar over-degenerates spectra.
   Exact $S_3$ invariance gives a $\mathbf1\oplus\mathbf2$ split, not three
   independent masses.
2. **Bath-reconstruction horn:** once the unresolved bath is arbitrary,
   spectral measures can be reconstructed to fit the desired poles and
   residues.

The band-edge theorem is the escape route. It selects the spectral probe
from the boundary dynamics itself:

$$
\boxed{
\text{the bath is not arbitrary; the physical readout is the marginal
retarded response of the completed local unitary boundary.}
}
$$

Then the silver constant becomes a consequence of the BB survival deficit,
rather than a fitted spectral evaluation point. This would turn the framework
from Froggatt-Nielsen with an algebraic parameter into a genuine boundary
selection theory.

The theorem above closes this for the pinned BB scar and for every sector whose
radial response is a regular outgoing depth response. It does not close the
stronger problem of deriving the sector coupling weights $w_f$ from the
microscopic Higgs/flavor carrier. That remains the sector-specific layer.

Certainty: `C:9` for the BB regular-retarded theorem; `C:6` that this is the
correct compressed physical principle for the mass sectors.

## 7. Relation To Existing Notes

This sidecar sits on the following completed pieces:

| Input | Role |
|---|---|
| [MICROSCOPIC_BB_QCA_EDGE_UPDATE.md](MICROSCOPIC_BB_QCA_EDGE_UPDATE.md) | exact local BB scattering update and retarded compression |
| [CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md](CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md) | no-incoming hidden sector gives visible powers $A^t$ |
| [BB_QCA_BOUNDARY_UPDATE_COMPLETION_AUDIT.md](BB_QCA_BOUNDARY_UPDATE_COMPLETION_AUDIT.md) | confirms the local update is complete |
| [BCC_Q0_SUPERSELECTION_DERIVATION.md](BCC_Q0_SUPERSELECTION_DERIVATION.md) | hard-gap Schur limit leaves $q=0$ visible scar |
| [NEUTRINO_PROTOTYPE.md](NEUTRINO_PROTOTYPE.md) | prototype sector where the same silver transfer gives a clean ratio |

The sidecar does not replace those notes. It asks why their shared spectral
point is selected.

## 8. Falsifiers

The selection theorem fails if any of the following is true:

1. A regular outgoing retarded probe of the same completed BB edge gives a
   different dominant attenuation without an explicit zero at $\zeta_*$.
2. The mass readout in a controlled QCA scattering model is a closed recurrent
   pole rather than an extremal retarded survival response.
3. The trace-minimum/survival equality is an accident of the chosen basis and
   disappears under an equivalent local unitary boundary representation.
4. The neutrino sector can keep the silver ratio while using a spectral point
   not tied to the BB survival norm.

These are useful failure modes. If the theorem dies, the synthesis should fall
back to the honest weaker statement: silver is an exact BB scar invariant, but
the mass readout point remains a physical premise.

## 9. Completion Ledger

| Claim | Status |
|---|---|
| BB same-normal scar has survival norm $1/2$ | `C:9`, exact matrix identity |
| BB transfer has $\det T=1$ and ${\rm tr}\,T=2\zeta+1/\zeta$ | `C:9`, exact matrix identity |
| survival point equals trace minimum, $\zeta_*=1/\sqrt2$ | `C:9`, exact calculus |
| retarded attenuation at the minimum is $\sqrt2-1$ | `C:9`, exact Weyl branch |
| regular outgoing depth responses are dominated by this point | `C:9`, Laplace theorem under stated assumptions |
| physical mass sectors use regular outgoing depth response | `C:6`, supported by the retarded boundary construction but still sector-coupling dependent |
| arbitrary baths or singular filters must select silver | `C:1`, false by explicit counterexample |

So the completed theorem is:

$$
\boxed{
\text{BCC gives the BB transfer family; retarded regularity selects its
marginal band-edge saddle; that saddle is silver.}
}
$$

## 10. Remaining Work

The next calculations should be:

1. **Impedance formulation:** express the regularity condition as boundary
   impedance matching between visible survival and outgoing clock-error
   self-energy.
2. **Sector weights:** derive the physical $w_f(\zeta)$ from the Higgs/flavor
   coupling maps and prove which sectors have nonzero weight at $\zeta_*$.
3. **Representation invariance:** show that $\zeta_*=c$ is invariant under
   unitary relabelings of the local boundary ports.
4. **Schur/Weyl formulation:** derive the same point from the retarded Weyl
   function of the outgoing hidden lead.
5. **Sector check:** verify that neutrino, charged-lepton, and quark readouts
   use this same radial spectral point and differ only by the coupling maps
   $V_f$.
