# Charged-Lepton BCC Ansatz

The next mass sector after neutrinos should be charged leptons. They are the
nearest dual branch: neutrinos provide the sterile mass frame, while charged
leptons provide the active Higgs/detection frame that turns the neutrino basis
into PMNS.

The right move is not to copy the quark mass ansatz. Charged leptons are
colorless and live in the same residual BCC three-port geometry as the neutrino
sector, but they are not sterile. They couple through the electroweak Higgs
door

$$
\bar L_L H e_R.
$$

Thus the charged-lepton mass branch should be built as a colorless active
boundary response:

$$
\Sigma_e(z)=V_e^\dagger(z-H_{Q,e})^{-1}V_e,
$$

using the neutrino theorem as the prototype.

Certainty: `C:6` for the sector placement; `C:3` for the full charged-lepton
mass ansatz until its boundary Hamiltonian is constructed.

## 1. What The Neutrino Prototype Teaches

The neutrino theorem has the structure

$$
\text{BCC geometry}
\to
\text{selected residual symmetry}
\to
\text{Schur response}
\to
\text{mass eigenvalues}.
$$

Explicitly,

$$
M_\nu=\mu_\nu(\epsilon^2P_u+P_b),
\qquad
(m_1,m_2,m_3)=\mu_\nu(0,\epsilon^2,1).
$$

For charged leptons we should demand the same kind of chain:

$$
\text{BCC geometry}
\to
\text{selected active Higgs port}
\to
\text{colorless Schur response}
\to
\text{charged-lepton masses}.
$$

The existing ingredients already point in this direction:

1. the same residual BCC basis $(a,u,b)$ is available;
2. the selected charged-lepton/Higgs port is known;
3. the leakage angle for PMNS is derived from two Weyl-transfer steps;
4. Koide is exactly a trace/traceless geometry in this basis.

The missing object is $H_{Q,e}$, the charged-lepton boundary bath.

Certainty: `C:7` for the structural analogy; `C:3` for the missing charged
lepton bath.

## 2. Same Geometry, Different Readout

Use the same residual basis as the neutrino theorem:

$$
u={1\over\sqrt3}(1,1,1),\qquad
a={1\over\sqrt6}(2,-1,-1),\qquad
b={1\over\sqrt2}(0,1,-1).
$$

For neutrinos, $K_\nu$ is diagonal in this basis:

$$
K_\nu^{(a,u,b)}
=
\begin{pmatrix}
0&0&0\\
0&\epsilon^2&0\\
0&0&1
\end{pmatrix}.
$$

For charged leptons, the natural object is not first a diagonal sterile
operator. The empirical clue is Koide, which is a statement about the vector of
square-root masses:

$$
w_e=(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau}).
$$

Therefore the charged-lepton analog should read masses as squared boundary
amplitudes:

$$
m_i=|w_i|^2.
$$

This is the first conceptual difference:

$$
\boxed{
\text{neutrino masses are Schur eigenvalues;}
\qquad
\text{charged-lepton masses are squared active boundary amplitudes.}
}
$$

That difference is physically sensible. Neutrinos use a sterile return channel.
Charged leptons are active Higgs-coupled states; their mass data naturally show
up as residues/amplitudes at the active port.

Certainty: `C:5` as the proposed dual readout.

## 3. Koide As BCC Trace/Plane Equipartition

Koide's relation is

$$
{m_e+m_\mu+m_\tau\over
(\sqrt{m_e}+\sqrt{m_\mu}+\sqrt{m_\tau})^2}
={2\over3}.
$$

Let

$$
n=u={1\over\sqrt3}(1,1,1),
\qquad
w_e=w_t+w_\perp,
$$

with $w_t=(w_e\cdot u)u$ and $w_\perp\perp u$. Koide is exactly

$$
|w_t|^2=|w_\perp|^2.
$$

Thus the normalized square-root mass vector has the form

$$
\widehat w_e
=
{1\over\sqrt2}u
+
{1\over\sqrt2}r_e,
\qquad
r_e\in{\rm span}(a,b),\quad |r_e|=1.
$$

This is the BCC meaning of Koide. The charged-lepton square-root mass vector is
at $45^\circ$ between the tetrahedral trace axis and the residual doublet
plane.

Using the physical charged-lepton masses, normalized in the residual basis,

$$
\widehat w_e
\approx
0.707110\,u
-0.479827\,a
-0.519386\,b.
$$

The $u$ coefficient is essentially $1/\sqrt2$. The remaining information is an
angle in the $(a,b)$ plane:

$$
r_e=\cos\theta_e^{\rm mass}\,a+\sin\theta_e^{\rm mass}\,b,
$$

with

$$
\theta_e^{\rm mass}\approx -132.733^\circ.
$$

So Koide reduces the charged-lepton mass problem to:

1. a scale $\mu_e=|w_e|^2=m_e+m_\mu+m_\tau$;
2. the trace/plane equipartition $|w_t|^2=|w_\perp|^2$;
3. one angle in the residual BCC plane.

Certainty: `C:9` for the Koide cone equivalence; `C:7` for the empirical
charged-lepton vector decomposition; `C:3` for the dynamical origin of the
equipartition and angle.

## 4. Existing Charged-Lepton Port

The boundary-response sidecar already identifies the selected active
charged-lepton/Higgs port:

$$
e_1=(1,0,0).
$$

In the residual basis,

$$
e_1=\sqrt{2\over3}\,a+{1\over\sqrt3}u+0\cdot b.
$$

This port has no $b$ component. With two Weyl-transfer leakage steps,

$$
\sqrt{2\over3}\,\sin\theta_{\rm leak}=\epsilon^2,
$$

so

$$
\sin\theta_{\rm leak}=\sqrt{3\over2}\,\epsilon^2.
$$

This already produces the PMNS reactor-angle relation

$$
\sin^2\theta_{13}={3\over4}\epsilon^4.
$$

This is not yet a charged-lepton mass theorem. It tells us how the active
charged-lepton frame leaks relative to the neutrino sterile basis. In the
charged-lepton mass program, it should be used as the boundary condition on
$V_e$: the active Higgs port starts in the $a/u$ plane, and $b$ enters only
through boundary recirculation or holonomy.

Certainty: `C:8` for the selected-port decomposition and leakage arithmetic;
`C:6` for its PMNS role; `C:3` for extending it into the mass bath.

## 5. Candidate BCC Charged-Lepton Ansatz

The minimal charged-lepton ansatz should therefore be:

$$
\boxed{
\widehat w_e
=
{1\over\sqrt2}u
+
{1\over\sqrt2}
(\cos\theta_e\,a+\sin\theta_e\,b).
}
$$

The physical masses are then read as

$$
m_i=\mu_e\,|\widehat w_{e,i}|^2,
\qquad
\mu_e=m_e+m_\mu+m_\tau.
$$

This automatically gives Koide:

$$
|w_t|^2=|w_\perp|^2
\quad\Longleftrightarrow\quad
K={2\over3}.
$$

The charged-lepton mass problem becomes a boundary-angle problem:

$$
\boxed{
\text{derive }\theta_e\text{ from a BCC/Higgs boundary resolvent or holonomy.}
}
$$

This is a good reduction. It is analogous to the neutrino result, but not the
same:

| Sector | Object | Output |
|---|---|---|
| neutrino | sterile Schur operator $K_\nu$ | eigenvalues $(0,\epsilon^2,1)$ |
| charged lepton | active residue amplitude $\widehat w_e$ | squared components $m_i/\mu_e$ |

Certainty: `C:4` for the ansatz as a reconstruction of charged-lepton masses;
`C:3` for a forward BCC derivation of $\theta_e$.

## 6. Relation To The Leptonic Holonomy Word

The existing leptonic boundary holonomy selects the primitive charged-lepton
word

$$
\text{SCHUR\_RETURN}\to\text{PARENT\_A3}\to\text{RESIDUAL\_A2}.
$$

Its phase is

$$
\exp\!\left(-{5\pi i\over12}\right).
$$

This has already been used for PMNS. The new possibility is that the same
primitive loop, or a closely related real projection of it, also selects the
mass-plane angle $\theta_e^{\rm mass}$ in the Koide circle.

That would close the charged-lepton branch in the same spirit as the neutrino
branch:

$$
\text{selected BCC port}
\to
\text{primitive loop}
\to
\text{trace/plane split}
\to
\theta_e
\to
(m_e,m_\mu,m_\tau).
$$

This should be treated carefully. The PMNS phase and the Koide-plane angle are
not automatically the same object. A real theorem must specify the map from
the complex boundary holonomy to the real $(a,b)$ mass-plane direction.

Certainty: `C:8` for the existing holonomy word inside its model; `C:3` for
using it to select the charged-lepton mass angle.

## 7. What Must Be Proved

To upgrade this ansatz to a theorem, we need the charged-lepton analog of the
neutrino proof.

### Geometry

Show that the same BCC vacuum framing gives the charged-lepton active boundary
space and selected Higgs port:

$$
e_1=\sqrt{2\over3}a+{1\over\sqrt3}u.
$$

This part is mostly present.

### Symmetry

Show that exact $Z_3$ or full $S_3$ is too symmetric for charged-lepton masses.
This is already supported by the Koide sidecar: exact $Z_3$ equivariance gives
a degenerate pair and cannot reproduce the physical spectrum.

### Recirculation

Construct a colorless active boundary bath

$$
H_{Q,e}
$$

and coupling

$$
V_e
$$

such that the boundary residue amplitude has the Koide form:

$$
\widehat w_e
=
{1\over\sqrt2}u
+
{1\over\sqrt2}r_e.
$$

The strongest version would derive both:

$$
|w_t|^2=|w_\perp|^2,
\qquad
r_e=\cos\theta_e\,a+\sin\theta_e\,b.
$$

### Scale

Derive or identify the charged-lepton scale

$$
\mu_e=m_e+m_\mu+m_\tau.
$$

This scale likely belongs to the electroweak Yukawa/Higgs boundary strength,
not to the dimensionless BCC geometry alone.

Certainty: `C:3` for the full proof program.

## 8. Immediate Research Program

The next detailed pass should do four things.

1. Re-express all charged-lepton sidecar results in the residual basis
   $(a,u,b)$, as the neutrino theorem does.
2. Treat Koide as a square-root residue geometry, not as a mass formula.
3. Build the minimal $2$-dimensional boundary resolvent on the $(a,b)$ plane
   that selects $\theta_e^{\rm mass}$, with controls that kill exact $Z_3$ and
   arbitrary angle fitting.
4. Check whether the existing leptonic boundary holonomy word can select that
   real angle without smuggling the observed masses into the construction.

The target theorem would be:

$$
\boxed{
\begin{gathered}
\text{The charged-lepton square-root mass vector is}\\
\text{the active BCC boundary residue at the selected Higgs port.}
\end{gathered}
}
$$

In formula form:

$$
\boxed{
(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau})
=
\sqrt{\mu_e}
\left[
{1\over\sqrt2}u
+
{1\over\sqrt2}
(\cos\theta_e\,a+\sin\theta_e\,b)
\right].
}
$$

The neutrino theorem tells us what good looks like. The charged-lepton branch
should now be forced to meet that standard.

The first concrete derivation attempt is
[CHARGED_LEPTON_RESOLVENT_DERIVATION.md](CHARGED_LEPTON_RESOLVENT_DERIVATION.md).
It derives trace/traceless equipartition from the selected port plus a coherent
$\sqrt2$ BCC trace return, and reduces the residual-plane angle to the active
torsion rule $\theta_e=-2\pi/3-2/9$.
The explicit minimal boundary Hamiltonian is
[CHARGED_LEPTON_MINIMAL_HAMILTONIAN.md](CHARGED_LEPTON_MINIMAL_HAMILTONIAN.md).
