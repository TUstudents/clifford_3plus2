# Neutrino Prototype

The neutrino result is the cleanest prediction in the synthesis. It is the
place where the whole philosophy works without being stretched:

$$
\boxed{
\begin{gathered}
\text{geometry selects channels}\\
\text{symmetry constrains projectors}\\
\text{recirculation sets eigenvalues}
\end{gathered}
}
$$

The output is

$$
M_\nu=\mu_\nu(\epsilon^2P_u+P_b),
\qquad
\epsilon=\sqrt2-1,
$$

and therefore

$$
\boxed{
(m_1,m_2,m_3)=\mu_\nu(0,\epsilon^2,1).
}
$$

The dimensionless prediction is sharp:

$$
{\Delta m^2_{21}\over\Delta m^2_{31}}=\epsilon^4
=17-12\sqrt2
\approx0.0294373.
$$

This is not in tension with oscillation data. It predicts normal ordering and
one massless light neutrino. If one measured splitting fixes $\mu_\nu$, the
other splitting and all three masses follow.

Certainty: `C:9` for the exact product sterile theorem; `C:8` for the
oscillation-ratio prediction inside the light-neutrino effective operator;
`C:7` for physical interpretation until the sterile product bath is derived
microscopically.

## 1. Why This Is The Prototype

The neutrino result has the right shape of a theory:

1. it starts from a primitive geometry;
2. it derives a residual basis;
3. it proves which symmetry is too large;
4. it introduces exactly the required framing;
5. it computes a Schur response;
6. it reads masses as eigenvalues of that response;
7. it has negative controls that fail for the right reasons.

This is the standard we should demand from every mass sector. A successful mass
theory should not begin by guessing a texture. It should identify the boundary
Hilbert space, the visible coupling map, and the resolvent:

$$
\Sigma(z)=V^\dagger(z-H_Q)^{-1}V.
$$

Masses are then pole shifts, residues, or eigenvalues of $\Sigma(z)$.

Certainty: `C:7` as a synthesis principle from the sidecar sequence.

## 2. BCC Geometry: From Body Diagonals To Three Residual Ports

The BCC QCA has eight oriented body-diagonal exits:

$$
(\pm1,\pm1,\pm1).
$$

Antipodal pairing gives four unoriented primitive exits. These four
representatives form a regular tetrahedron:

$$
v_1,v_2,v_3,v_4,
\qquad
v_i\cdot v_j=
\begin{cases}
1,&i=j,\\
-1/3,&i\ne j.
\end{cases}
$$

Selecting one vacuum-framed exit leaves three residual exits. Their adjacency
is the complete graph $K_3$, and the selected-exit stabilizer induces the full
residual permutation group:

$$
S_4\supset{\rm Stab}(v_{\rm sel})\simeq S_3.
$$

This is the geometric origin of the three neutrino residual ports. The three
ports are not postulated generations floating in abstract flavor space. They
are the residual tetrahedral exits left after one BCC vacuum direction is
selected.

Certainty: `C:8` for the finite BCC orbit/framing audit; `C:6` for the
physical vacuum selector, because the order parameter that chooses the framed
exit is modeled but not yet derived from the microscopic QCA dynamics.

## 3. Symmetry: Full $S_3$ Is Too Symmetric

The residual basis is

$$
u={1\over\sqrt3}(1,1,1),\qquad
a={1\over\sqrt6}(2,-1,-1),\qquad
b={1\over\sqrt2}(0,1,-1).
$$

Here $u$ is the collective mode, while $(a,b)$ span the residual doublet.
Full residual $S_3$ decomposes the permutation representation as

$$
3=1\oplus2.
$$

Therefore every full $S_3$-invariant operator has the form

$$
A=\alpha P_u+\beta(P_a+P_b).
$$

It cannot split $a$ from $b$. But the neutrino operator is

$$
K_\nu=\epsilon^2P_u+P_b,
$$

with eigenvalue zero on $a$. Thus the full unframed $S_3$ route is killed:

$$
K_\nu\notin{\rm Cent}_{S_3}(3).
$$

The correct symmetry is the selected-port $S_2$ that remains after framing. It
allows $P_a$, $P_u$, and $P_b$ to be separately visible while still preserving
the selected residual reflection. This is exactly the kind of symmetry breaking
we want: not arbitrary, but minimal and geometrically forced by the framed BCC
exit.

Certainty: `C:9` for the $S_3$ centralizer obstruction; `C:8` for selected
$S_2$ compatibility of $K_\nu$.

## 4. Recirculation: The Weyl Chain Gives The Number

The neutrino number is not produced by group theory. Group theory gives the
projectors. The number comes from recirculation.

The sterile unresolved bath is the product

$$
H_Q=H_{\rm chain}\otimes I_{\rm family}.
$$

The family factor keeps the $u$ and $b$ returns orthogonal. The chain factor is
the radial sterile return. For the semi-infinite unit chain,

$$
m(z)={z-\sqrt{z^2-4}\over2},
$$

where the physical branch is fixed by

$$
\lim_{z\to\infty}z\,m(z)=1.
$$

The same function obeys the Dyson fixed point

$$
m(z)=\frac{1}{z-m(z)}.
$$

At the transfer probe,

$$
z=2\sqrt2,
\qquad
m(z)=\sqrt2-1=\epsilon.
$$

The normalized Schur response is

$$
\widehat\Sigma(z)=m(z)^2P_u+P_b.
$$

Thus

$$
\widehat\Sigma(2\sqrt2)=\epsilon^2P_u+P_b.
$$

The square is important. The collective $u$ channel is one transfer step deeper
in amplitude, so its mass eigenvalue is suppressed by the square of the
amplitude:

$$
{m_2\over m_3}=\epsilon^2.
$$

The solar-to-atmospheric mass-squared ratio is then

$$
{\Delta m^2_{21}\over\Delta m^2_{31}}=\epsilon^4.
$$

Certainty: `C:9` for the Weyl function, fixed-point identity, and Schur
eigenvalues in the product sterile model.

## 5. What The Zero Means

The massless state is not a cancellation. It is an absent coupling.

In the residual basis,

$$
K_\nu^{(a,u,b)}
=
\begin{pmatrix}
0&0&0\\
0&\epsilon^2&0\\
0&0&1
\end{pmatrix}.
$$

The $a$ mode does not couple to the sterile return channel in the neutrino core.
That is a stronger statement than tuning an eigenvalue to zero. A zero from
non-coupling is stable under the symmetry assumptions; a zero from cancellation
would be fragile.

This is one reason the neutrino branch is cleaner than the quark branch. The
neutrino response is diagonal in the geometrically natural residual basis. The
quark branch still needs a theorem selecting which Higgs door hits the
zero-mode pole and which hits the gapped Hermitian bath.

Certainty: `C:9` for the eigenvalue zero inside $K_\nu$; `C:7` for the
stability interpretation.

## 6. Universal Speed $c$

The neutrino mass theorem does not modify the universal light-cone speed. The
BCC QCA carrier supplies the relativistic kinetic term:

$$
H_R(k)=c\,\sigma\cdot k,\qquad
H_L(k)=-c\,\sigma\cdot k.
$$

The Dirac assembly gives

$$
E^2=p^2c^2+m^2c^4.
$$

Mass changes the rest frequency and the pole structure. It does not give each
particle its own microscopic speed. In the QCA picture, $c$ is the carrier
speed of the local update, while mass is a boundary recirculation impedance:

$$
\text{same carrier }c
\quad+\quad
\text{different boundary return}
\quad\Longrightarrow\quad
\text{different rest masses}.
$$

The group velocity of a massive neutrino is

$$
v_g={\partial E\over\partial p}
=
{pc^2\over E}
<c,
$$

but this is a dispersion effect of the pole, not a change in the QCA causal
speed. At high momentum all sectors approach the same speed $c$:

$$
\lim_{p\gg mc}v_g=c.
$$

This distinction is crucial. The boundary response produces inertia, not a new
light cone.

Certainty: `C:9` for the relativistic dispersion statement; `C:7` for the QCA
interpretation as carrier speed plus boundary impedance.

## 7. Why It Is Not In Tension With Experiment

The prediction

$$
\epsilon^4=17-12\sqrt2\approx0.0294373
$$

lands on the observed order of

$$
{\Delta m^2_{21}\over\Delta m^2_{31}}\sim3\times10^{-2}.
$$

Using the representative atmospheric splitting

$$
\Delta m^2_{31}=2.52\times10^{-3}\ {\rm eV}^2,
$$

the theorem gives

$$
\Delta m^2_{21}
=
\epsilon^4\Delta m^2_{31}
\approx7.42\times10^{-5}\ {\rm eV}^2.
$$

The corresponding masses are

$$
m_1=0,\qquad
m_2\approx8.61\ {\rm meV},\qquad
m_3\approx50.2\ {\rm meV},
$$

with total mass

$$
\sum_i m_i\approx58.8\ {\rm meV}.
$$

This is the right phenomenological posture: the model predicts a normal
hierarchy and a very light total mass. It does not yet predict the absolute
scale $\mu_\nu$ internally; one measured splitting has been used to set it.

Certainty: `C:8` for the no-tension comparison at the level of mass splittings;
`C:3` for the internal scale $\mu_\nu$ until a microscopic sterile/Higgs scale
theorem is supplied.

## 8. Prototype Rules For Other Sectors

The neutrino theorem gives the template for the rest of the mass program.

### Geometry First

Find the finite boundary object before fitting a texture:

$$
\text{BCC exits}\to\text{residual ports}\to\text{projectors}.
$$

For quarks, this means the zero-mode/gapped-bath story must be expressed as a
boundary graph or boundary Hilbert space, not as a symbolic assignment. For
charged leptons, it means finding the colorless boundary response, not copying
quark color Clebsches.

### Symmetry Second

Compute the centralizer. It tells us what can and cannot split:

$$
{\rm Cent}_G(\mathcal H_{\rm sector}).
$$

The neutrino sector succeeded because full $S_3$ was honestly killed and the
weaker selected $S_2$ was used. We should do the same for quarks and charged
leptons: do not ask a symmetry to distinguish things it cannot distinguish.

### Recirculation Third

Only after geometry and symmetry are fixed do numbers enter:

$$
\Sigma(z)=V^\dagger(z-H_Q)^{-1}V.
$$

The numbers should be Green-function values, residues, pole shifts, or transfer
amplitudes. The sterile chain gives $\epsilon$ because its Weyl function gives
$m(2\sqrt2)=\epsilon$. Other sectors need their own $H_Q$ and $V$.

### Speed Is Universal

All sectors propagate on the same QCA light cone. Different masses are not
different microscopic speeds:

$$
\boxed{
\text{universal }c,\quad\text{sector-dependent boundary self-energy}.
}
$$

This is the physical reason the neutrino result is so valuable. It shows how a
tiny mass can be a boundary response while preserving the same relativistic
carrier for all particles.

Certainty: `C:7` as the methodological prototype.

## 9. The Next Questions

The neutrino result should now be used as the benchmark. The open tasks are:

1. derive the product sterile bath from the microscopic BCC/QCA boundary,
   including the scale $\mu_\nu$;
2. find the quark boundary Hamiltonian that makes $\tilde H$ hit a zero-mode
   pole while $H$ hits a gapped Hermitian bath;
3. find the charged-lepton colorless boundary Hamiltonian and test whether
   Koide geometry appears as a residue or eigenvector constraint;
4. keep $c$ universal throughout: masses are self-energies, not modified
   carrier speeds.

The central lesson is:

$$
\boxed{
\text{Do not search for mass formulas. Search for boundary resolvents.}
}
$$
