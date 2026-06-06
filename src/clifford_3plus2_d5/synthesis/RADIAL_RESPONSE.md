# Radial Response Sidecar

The `radial_response` sidecar is the mass-value layer of the theory tree. The
previous layers supply carrier quantum numbers, transfer powers, family depth,
and scalar Clebsches. This sidecar asks what kind of object a mass actually is.

Its central answer is:

$$
\hbox{mass values are boundary Green-function pole shifts and residues},
$$

not direct finite-group outputs.

Finite $S_3$, BCC, and scalar-selection algebra still matter, but they specify
channels, projectors, couplings, and transfer grammar. The radial numbers live
in a spectral measure.

Certainty: `C:9` for the Schur/Feshbach Green-function identity. `C:5` for the
current full mass-value theory, because the sidecar proves compatibility and a
no-go, not a forward selection principle.

## 1. The Primitive Object

For a resolved/unresolved split

$$
H=
\begin{pmatrix}
H_P&V^T\\
V&H_Q
\end{pmatrix},
$$

the resolved Green function is

$$
G_P(z)
=
\left(z-H_P-\Sigma(z)\right)^{-1},
$$

where

$$
\Sigma(z)=V^T(z-H_Q)^{-1}V.
$$

Expanding the unresolved resolvent gives

$$
\Sigma(z)
=
\sum_{n\ge0}\frac{V^T H_Q^n V}{z^{n+1}}.
$$

This is the exact recirculation picture:

$$
P\to Q\to P,\qquad
P\to Q\to Q\to P,\qquad
P\to Q\to Q\to Q\to P,\ldots
$$

Thus a low-energy mass is not a Clebsch by itself. It is a pole/residue feature
of $\Sigma_f(z)$ in a sector-selected boundary bath:

$$
\Sigma_f(z)=V_f^\dagger(z-H_{Q,f})^{-1}V_f.
$$

Certainty: `C:9`.

## 2. Up Sector: Stacking Law, Not Yukawa Matrix

The scalar Clebsch sidecar used the nilpotent Taylor profile

$$
C_u(x)=\left(\frac{x^2}{2},x,1\right).
$$

The radial sidecar clarifies what this really means. The invariant

$$
I_u=\frac{C_u C_t}{C_c^2}
$$

distinguishes two stacking laws:

$$
\hbox{exponential/Poisson stack: }
\left(\frac{x^2}{2},x,1\right)
\Rightarrow I_u=\frac12,
$$

$$
\hbox{geometric/resolvent stack: }
\left(x^2,x,1\right)
\Rightarrow I_u=1.
$$

So the nilpotent Taylor story earns the factorial relation $1/2!$. It does not
derive $x$ by itself.

The sidecar also kills a tempting overinterpretation: literal $\exp(N)$ is not
the physical family-space Yukawa matrix. At $x=1$, the singular values are

$$
(2,1,1/2),
$$

and the left metric is not diagonal. Therefore $\exp(xN)$ may be a scalar
response jet, but it should not replace the boundary charged-current/CKM
holonomy story.

Certainty: `C:9` for the stacking invariant and literal-Yukawa kill. `C:6` for
the up scalar mass interpretation until the scalar boundary map is derived.

## 3. Conditional Origin Of $x=1/\sqrt2$

The sidecar gives a conditional derivation of the scalar repair amplitude used
in `scalar_clebsch`.

If an active scalar repair state has exactly two symmetry-related successors
and no leakage, then

$$
|\alpha_+|^2+|\alpha_-|^2=1,
$$

and symmetry gives

$$
|\alpha_+|=|\alpha_-|.
$$

Therefore

$$
|\alpha_+|=|\alpha_-|=\frac1{\sqrt2}.
$$

This is the right place for the up-sector amplitude $x=1/\sqrt2$. The sidecar
also shows the controls: one channel gives $1$, three channels give
$1/\sqrt3$, leakage gives $\sqrt{(1-\ell)/2}$, and asymmetric two-channel
weights are not fixed.

Certainty: `C:9` for the two-channel isometry theorem. `C:6` for using it as a
physical derivation of $x$, because the successor/no-leakage assumptions still
need microscopic completion.

## 4. Scalar Successor Pair And $S_3$

The finite successor certificate has one active source and exactly two allowed
outputs:

$$
\Omega_{\rm scalar}
=
\{\mathrm{triality}_+,\mathrm{triality}_-\}.
$$

The certificate rejects same-state, wrong-height, two-tick, external-leakage,
asymmetric-sector, and third-successor controls. This proves the two-channel
condition inside the modeled finite basis.

The stronger group statement is the $S_3$ census. The six elements split as

$$
S_3=\{e\}\sqcup (A_3\setminus\{e\})\sqcup \{\hbox{three transpositions}\}.
$$

The identity is same-state, the two non-identity $A_3$ cycles are the scalar
holomorphic repair successors, and the transpositions belong to the
Hermitian/$Z_2$ repair sector:

$$
A_3\setminus\{e\}
=
\{\mathrm{triality}_+,\mathrm{triality}_-\}.
$$

Each transposition conjugates the two triality elements into each other, so the
successors form one $Z_2$-conjugate pair.

Certainty: `C:8` for the finite successor certificate. `C:9` for the $S_3$
classification. `C:6` for full QCA boundary completeness.

## 5. BCC/Vacuum-Frame Reduction To $S_3$

The sidecar connects the scalar successor pair back to the BCC/vacuum frame via
tetrahedral exit automorphisms. There are

$$
24
$$

tetrahedral exit automorphisms. The selected-exit stabilizer has size

$$
6,
$$

and induces the full residual $S_3$ shell. The $18$ selected-exit-moving
automorphisms are rejected as vacuum-frame breaking. Non-automorphism controls
are also rejected.

After the scalar holomorphic restriction, the selected-preserving residual
$S_3$ again leaves exactly

$$
\mathrm{triality}_+,\qquad \mathrm{triality}_-.
$$

The next bridge narrows the premise: a declared one-tick scalar-local
deterministic exit map lands exactly in this vacuum-frame-preserving
automorphism census. The only accepted maps are

$$
\mathrm{triality}_+\hbox{-automorphism},
\qquad
\mathrm{triality}_-\hbox{-automorphism}.
$$

But this still does not derive the declared scalar-local map class from the
full BB/QCA update.

Certainty: `C:8` for the finite tetrahedral automorphism census. `C:6` for the
physical bridge from QCA boundary dynamics to the declared scalar-local map
class.

## 6. Down Dark Line

The data-improved down coefficient has

$$
C_b^2=\frac56.
$$

Radially, this can be read as the regular $S_3$ shell minus one forbidden
one-dimensional line:

$$
6-1=5.
$$

But the regular shell has two central rank-$5$ complements:

$$
I_{\rm reg}-P_{\rm triv},
\qquad
I_{\rm reg}-P_{\rm sign}.
$$

So $S_3$ does not choose the dark line. The middle rank-$2$ standard copy also
requires defect polarization rather than central $S_3$ alone.

This is a useful framing for the down candidate, not a derivation of it.

Certainty: `C:8` for availability. `C:4` for the down dark-line mechanism until
the excluded line is selected by a physical rule.

## 7. Minimal Unitary Defect Form

The sidecar constructs an exact unitary Floquet toy

$$
U=SC,
$$

where $S$ is an $S_3$ regular shift and $C$ is a Givens defect coin coupling the
resolved scalar port to an unresolved $S_3$ shell. The unitary self-energy is

$$
\Sigma_U(z)=U_{PQ}(zI-U_{QQ})^{-1}U_{QP}.
$$

The Schur $P$-block agrees with the full unitary resolvent $P$-block. This is a
valid exact QCA/Floquet form.

The controls are important: changing the coin angle or the defect vector
changes $\Sigma_U$. Therefore the form itself does not force phases, pole
locations, residues, or radial mass values.

Certainty: `C:9` for exact unitarity and Schur consistency. `C:1` for the claim
that the minimal $U=SC$ toy alone derives physical radial values.

## 8. Silver/Pell Transfer Is Inherited

The radial sidecar does not recompute or fit the small transfer parameter. It
inherits the established root from `boundary_response` and `flavor_a_track`:

$$
\epsilon=\sqrt2-1,
$$

$$
\eta=\epsilon^2=3-2\sqrt2,
$$

$$
r=\epsilon^4=17-12\sqrt2.
$$

The later synthesis note [PELL_RADIAL_THEOREM.md](PELL_RADIAL_THEOREM.md)
sharpens the radial interpretation. For

$$
T_{\rm Pell}=
\begin{pmatrix}
2&1\\
1&0
\end{pmatrix},
$$

the contraction ratio between transfer eigenchannels is

$$
\eta=\left|\frac{\lambda_-}{\lambda_+}\right|
=(\sqrt2-1)^2.
$$

Thus the radial mass powers should be read as integer powers of $\eta$, while
$\epsilon$ remains the one-step silver amplitude.

The residual graph source, sterile-chain Weyl function, shared-transfer audit,
and quark common-chain Schur audit agree. Independent fitted $\eta$, alternate
$K_2/K_4$ graph roots, and the claim that the minimal unitary toy forces radial
values are rejected.

Thus the radial open problem is no longer fitting the small parameter. It is
deriving the Pell boundary propagator from the BCC/QCA scar, deriving the
integer depths, and selecting pole/residue data.

Certainty: `C:8` for cross-sidecar inheritance. `C:9` for the exact identities
once $\epsilon$ is imported.

## 9. Pole/Residue Rigidity No-Go

The sidecar proves an important no-go: the current finite data do not force the
radial spectral measure.

Two baths preserve the same triality-head data:

$$
\alpha_+=\alpha_-=\frac1{\sqrt2},
\qquad
\|\alpha\|^2=1,
$$

and the same allowed successor labels. Yet their self-energies differ.

The one-level bath gives

$$
\Sigma_1(z)=\frac1z.
$$

The symmetric two-level tail bath gives

$$
\Sigma_2(z)=\frac{z-1}{z^2-z-1}.
$$

Their poles and residues are different. Therefore finite $S_3$ grammar,
two-channel no-leakage, and inherited silver transfer are insufficient to
determine physical quark masses.

Certainty: `C:9` for the no-go inside the stated bath class. This is one of
the sidecar's most valuable results.

## 10. Spectral Measures: Compatible, Not Selected

The target textures define positive finite Stieltjes measures

$$
\Sigma_f(z)=\sum_i\frac{w_i}{z-\lambda_i}.
$$

For the up target:

$$
\lambda_u=
\left(
\frac{\eta^6}{4},\ \frac{\eta^3}{\sqrt2},\ 1
\right),
\qquad
w_u=
\left(
\frac1{25},\frac8{25},\frac{16}{25}
\right).
$$

For the down $S_3$ baseline:

$$
\lambda_{d,\rm base}
=
\left(
\sqrt{\frac32}\eta^4,\ \frac{\eta^2}{\sqrt2},\ 1
\right),
\qquad
w_{d,\rm base}
=
\left(
\frac12,\frac16,\frac13
\right).
$$

For the down odd-shell candidate:

$$
\lambda_{d,\rm cand}
=
\left(
\sqrt{\frac65}\eta^4,\ \sqrt{\frac25}\eta^2,\ 1
\right),
\qquad
w_{d,\rm cand}
=
\left(
\frac6{13},\frac2{13},\frac5{13}
\right).
$$

Each measure is positive and reconstructs a unique finite Jacobi bath by the
inverse spectral problem. The reconstructed Jacobi moments round-trip against
the target moments.

But the negative controls do not select the target measures:

$$
\hbox{R12 one-level bath},
\quad
\hbox{R12 two-level tail bath},
\quad
P_3\hbox{ repair Jacobi bath},
$$

$$
\hbox{constant silver-tail Jacobi bath},
\quad
\hbox{minimal unitary }S_3\hbox{ toy at }z=2.
$$

So this is reconstruction, not derivation. The target mass textures are
compatible with a boundary Green-function realization, but the QCA has not yet
selected the corresponding spectral density.

Certainty: `C:8` for positive-measure/Jacobi reconstruction. `C:4` for any
claim that these target measures are physically derived.

## 11. Synthesis Verdict

This sidecar should be central to the paper because it prevents a major
category error. Finite symmetry can organize the channels, but it cannot by
itself be the radial mass-value theorem.

The true structure is:

$$
\hbox{finite grammar}
\Rightarrow
\hbox{allowed channels and couplings},
$$

$$
\hbox{silver transfer}
\Rightarrow
\hbox{shared hierarchy scale},
$$

$$
\hbox{spectral measure}
\Rightarrow
\hbox{actual pole/residue mass values}.
$$

The current theorem stack is:

$$
\Sigma_f(z)=V_f^\dagger(z-H_{Q,f})^{-1}V_f
\quad\hbox{is exact},
$$

$$
\epsilon,\eta,r
\quad\hbox{are inherited, not fitted},
$$

$$
\{\mathrm{triality}_+,\mathrm{triality}_-\}
\Rightarrow
x=\frac1{\sqrt2}
\quad\hbox{conditionally},
$$

$$
\hbox{finite grammar}
\not\Rightarrow
\hbox{unique radial poles/residues}.
$$

The open burdens are precise:

1. Derive the one-tick scalar-local deterministic exit-map class from the full
   BB/QCA scalar boundary update.
2. Find the forward spectral-density principle that selects the target
   Stieltjes measure instead of reconstructing it.
3. Derive or kill the down dark-line selection rule.
4. Build a simulator-backed boundary spectral density $J_f(\omega)$.
5. Keep $\exp(xN)$ as scalar response, not as the literal Yukawa matrix.

My synthesis judgement: `radial_response` is a pass as a framework and a no-go
sidecar, not as a completed mass derivation. It tells us exactly what remains
to prove: a forward principle for $H_{Q,f}$ or its spectral measure.
