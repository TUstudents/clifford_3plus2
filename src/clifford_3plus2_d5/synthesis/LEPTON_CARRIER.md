# Lepton Carrier Sidecar

The `lepton` sidecar is the carrier construction. It does not derive flavor,
Yukawa values, three generations, or masses. Its job is to build the internal
one-generation Standard Model stage on which the later flavor mechanisms act.

The positive endpoint is:

$$
\mathrm{Cl}(0,10)
\quad\Longrightarrow\quad
S_+,\quad \dim_{\mathbb R}S_+=32,
$$

with a compatible complex structure $J$, so

$$
(S_+,J)\simeq\mathbb C^{16}.
$$

On this chiral-16 carrier, the Pati-Salam factorization exposes

$$
SU(4)\times SU(2)_L\times SU(2)_R,
$$

and the usual breaking gives

$$
SU(3)_c\times SU(2)_L\times U(1)_Y
$$

with exactly the one-generation hypercharge spectrum.

Certainty: `C:8` for the implemented exact finite carrier audit. `C:7` for the
physical identification as one Standard Model generation, because several
orientation and normalization choices are declared rather than dynamically
derived.

## 1. What This Sidecar Is

The carrier layer is not a flavor mechanism. It supplies:

1. the chiral spinor space;
2. a compatible complex structure;
3. the internal gauge algebra;
4. one-generation charge multiplicities;
5. a background-gauge checkerboard continuum test.

It does not supply:

1. three generations;
2. scalar mass values;
3. CKM/PMNS mixing;
4. dynamical Yang-Mills fields;
5. a Higgs condensate;
6. $3+1$D Lorentz recovery.

This separation is essential. Later sidecars should attach flavor and radial
response to this carrier; they should not pretend that the carrier already
contains the family-depth mechanism.

Certainty: `C:9` for this role separation.

## 2. The Chiral-16 Carrier

The sidecar constructs exact real gamma matrices satisfying

$$
\gamma_i\gamma_j+\gamma_j\gamma_i=-2\delta_{ij}I,
\qquad i,j=1,\ldots,10.
$$

The full real module has dimension

$$
64.
$$

A real chirality involution is built from the volume element and a commuting
complex structure. Its projectors have ranks

$$
(32,32).
$$

Choosing one chirality gives

$$
S_+,\qquad \dim_{\mathbb R}S_+=32.
$$

With the chosen compatible $J$,

$$
J^2=-I,
\qquad
J^TJ=I,
$$

and the real carrier becomes a complex $16$:

$$
(S_+,J)\simeq\mathbb C^{16}.
$$

Certainty: `C:9` for the Clifford relations, chirality ranks, and complex
dimension under the chosen $J$.

## 3. The Early Octonion Route

Earlier sessions construct a useful but incomplete route through
$\mathrm{Cl}(0,8)$ and octonions. One oriented Fano table is fixed, then the
octonion derivation algebra is computed:

$$
\dim\operatorname{Der}(\mathbb O)=14,
$$

which identifies $G_2$ after the octonion multiplication table is chosen.
Choosing an imaginary direction, called $e_7$ in the sidecar, gives the
stabilizer

$$
\dim\operatorname{Stab}_{G_2}(e_7)=8,
$$

the expected $SU(3)$ color-sized algebra.

This is a real structural signal:

$$
\mathrm{Spin}(8)\supset G_2\supset SU(3).
$$

But it depends on declared choices: the Fano table and the imaginary direction.
It also does not produce electroweak $SU(2)_L$. The later audit confirms the
problem: the $\mathrm{Cl}(0,2)$ factor gives only

$$
\mathrm{Spin}(0,2)\simeq U(1),
$$

not $SU(2)_L$.

Certainty: `C:8` for the finite octonion stabilizer computations. `C:6` for
their role as a color carrier origin because the Fano table and $e_7$ are
declared. `C:1` for the claim that the $\mathrm{Cl}(0,8)\otimes\mathrm{Cl}(0,2)$
split alone supplies the full electroweak group.

## 4. Pati-Salam Factorization

The carrier closes at the Pati-Salam level by changing the factorization:

$$
\mathrm{Cl}(0,10)
=
\mathrm{Cl}(0,6)\otimes\mathrm{Cl}(0,4).
$$

This exposes

$$
\mathrm{Spin}(0,6)\times\mathrm{Spin}(0,4)
\simeq
SU(4)\times SU(2)_L\times SU(2)_R.
$$

The exact dimensions are

$$
\dim SU(4)=15,
\qquad
\dim SU(2)_L=3,
\qquad
\dim SU(2)_R=3.
$$

The two $SU(2)$ factors commute:

$$
[SU(2)_L,SU(2)_R]=0.
$$

The compatible default $J$ is a right-quaternionic unit in the commutant of the
$\mathrm{Cl}(0,4)$ action. It commutes with all Pati-Salam generators:

$$
[J,\mathfrak{su}(4)]=0,\qquad
[J,\mathfrak{su}(2)_L]=0,\qquad
[J,\mathfrak{su}(2)_R]=0.
$$

The sidecar also checks a tempting alternative: a simple $\mathrm{Spin}(4)$
bivector squares to $-I$, but it commutes with only one generator in each
$SU(2)$ factor. It would break both $SU(2)$ factors to Cartans, so it is not the
right carrier complex structure.

Certainty: `C:9` for the algebra dimensions and compatibility checks under
the chosen factorization and $J$. `C:7` for the uniqueness/physical preference
of this $J$ until the quaternionic unit choice is dynamically derived.

## 5. Standard Model Extraction

From the Pati-Salam algebra, the sidecar chooses a $B-L$ direction inside
$SU(4)$. Its centralizer in $\mathfrak{su}(4)$ has dimension

$$
9.
$$

The trace-orthogonal part of that centralizer is

$$
\mathfrak{su}(3)_c,
\qquad
\dim\mathfrak{su}(3)_c=8.
$$

The extracted Standard Model algebra is

$$
\mathfrak{su}(3)_c\oplus\mathfrak{su}(2)_L\oplus\mathfrak{u}(1)_Y,
$$

with total dimension

$$
8+3+1=12.
$$

The commutators vanish between the factors:

$$
[\mathfrak{su}(3)_c,\mathfrak{su}(2)_L]=0,
$$

$$
[\mathfrak{su}(3)_c,Y]=0,
\qquad
[\mathfrak{su}(2)_L,Y]=0.
$$

The chosen $J$ commutes with the extracted SM algebra.

Certainty: `C:9` for the finite algebra extraction after the Pati-Salam
breaking directions are chosen.

## 6. Hypercharge Normalization

The raw Pati-Salam formula is

$$
Y_{\rm raw}=T_{3R}^{\rm raw}+\frac12(B-L)^{\rm raw}.
$$

The raw spectrum does not match the Standard Model spectrum by one common
scale. This is not a structural failure; it is a component-normalization
issue. The physical normalization used by the audit is

$$
T_{3R}=\frac12T_{3R}^{\rm raw},
$$

$$
B-L=\frac23(B-L)^{\rm raw},
$$

so

$$
Y
=T_{3R}+\frac12(B-L)
=\frac12T_{3R}^{\rm raw}
 +\frac13(B-L)^{\rm raw}.
$$

With this normalization, the complex hypercharge spectrum is exactly

$$
\left\{
\frac16:6,\quad
-\frac23:3,\quad
\frac13:3,\quad
-\frac12:2,\quad
1:1,\quad
0:1
\right\}.
$$

This is the one-generation chiral-16 table:

$$
Q:(3,2)_{1/6},
\qquad
u^c:(\bar3,1)_{-2/3},
\qquad
d^c:(\bar3,1)_{1/3},
$$

$$
L:(1,2)_{-1/2},
\qquad
e^c:(1,1)_1,
\qquad
\nu^c:(1,1)_0.
$$

The sidecar verifies the joint $(Y,T_{3L})$ table:

$$
\left(\frac16,\frac12\right):3,\qquad
\left(\frac16,-\frac12\right):3,
$$

$$
\left(-\frac23,0\right):3,\qquad
\left(\frac13,0\right):3,
$$

$$
\left(-\frac12,\frac12\right):1,\qquad
\left(-\frac12,-\frac12\right):1,
$$

$$
(1,0):1,\qquad (0,0):1.
$$

The complex multiplicities sum to

$$
16.
$$

Certainty: `C:9` for the normalized hypercharge spectrum and joint weak table.
`C:7` for the physical normalization convention, because the component factors
$1/2$ and $2/3$ are chosen to match the Pati-Salam textbook normalization.

## 7. Background Checkerboard Dynamics

The sidecar also checks a narrow dynamics statement. The 1D massless
checkerboard/Floquet walk on the chiral-16 carrier has continuum Hamiltonian

$$
H_0(k)=k\,\sigma_z\otimes I_{32}.
$$

With a background gauge generator $A$ from the extracted SM algebra, the first
order continuum form is

$$
H(k,A)
=
k\,\sigma_z\otimes I_{32}
 +I_2\otimes iA.
$$

The exact sampled Floquet spectrum is gapless only at

$$
k=0.
$$

This is a background-link covariance and massless continuum check. It is not
dynamical Yang-Mills, not a Higgs system, and not a $3+1$D Lorentz recovery
theorem.

Certainty: `C:8` for the implemented 1D background-gauge continuum check.
`C:3` for extrapolating this alone to full $3+1$D Standard Model dynamics.

## 8. What Is Derived And What Is Chosen

Derived under the declared construction:

1. exact $\mathrm{Cl}(0,10)$ relations;
2. chiral projectors of real rank $32$;
3. Pati-Salam algebra dimensions $15+3+3$;
4. a $J$ compatible with the full Pati-Salam algebra;
5. SM algebra dimensions $8+3+1$ after choosing $B-L$ and $T_{3R}$;
6. normalized one-generation hypercharge and weak-doublet table;
7. 1D massless background-gauge checkerboard form.

Chosen or conventional:

1. the $\mathrm{Cl}(0,6)\otimes\mathrm{Cl}(0,4)$ factorization;
2. the right-quaternionic $J$ unit in the $\mathrm{Cl}(0,4)$ commutant;
3. the $B-L$ direction in $SU(4)$;
4. the $T_{3R}$ Cartan in $SU(2)_R$;
5. the Pati-Salam normalization factors;
6. earlier octonion Fano table and imaginary direction, when using the
   octonion route.

Open:

1. color orientation $3$ versus $\bar3$ as an explicit weight/Casimir audit;
2. Higgs and Yukawa couplings;
3. dynamical gauge fields;
4. $3+1$D Lorentz recovery;
5. three generations;
6. family-depth and radial mass selection.

Certainty: `C:8` for this derived/chosen split as a synthesis of the sidecar.

## 9. Synthesis Verdict

This sidecar is a strong carrier construction. Its clean theorem is not
"flavor is derived." Its clean theorem is:

$$
\mathrm{Cl}(0,10)\ \hbox{with Pati-Salam factorization}
\Rightarrow
\hbox{one chiral-16 SM generation}.
$$

It supplies the internal gauge and charge scaffold required by the later flavor
sidecars:

$$
\hbox{carrier}
\to
\hbox{boundary transfer}
\to
\hbox{family depth}
\to
\hbox{scalar Clebsch}
\to
\hbox{radial spectral measure}.
$$

The sidecar also tells us what not to do. Do not claim the octonion route alone
supplies electroweak $SU(2)_L$. Do not claim the carrier explains three
generations. Do not hide the Pati-Salam normalization choices.

My synthesis judgement: this is the correct first sidecar for the paper
scaffold. It gives the representation-theoretic arena. The flavor theory begins
only after this arena is fixed.
