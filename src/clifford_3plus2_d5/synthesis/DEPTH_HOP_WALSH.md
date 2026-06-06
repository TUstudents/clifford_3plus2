# Depth hop-Walsh kill

The `depth_hop_walsh` sidecar tests whether the raw Bialynicki-Birula BCC hop
source derives the depth ladder
$
\{0,2,6\}.
$
It is a correction to the depth story, not a replacement for the boundary/scar
construction.

The verdict is negative: the raw hop shell does not produce the clean cube
tower needed for the depth ladder, and Schur's lemma shows that any
$S_3$-invariant three-port depth operator cannot have three distinct depths.

## The tested claim

For the eight BCC Weyl hop matrices $H_v$, $v\in\{\pm1\}^3$, define the
coefficient-Walsh modes

$$
\widehat H_S={1\over8}\sum_v \chi_S(v)H_v,
\qquad
\chi_S(v)=\prod_{i\in S}v_i.
$$

The proposed mechanism wanted the parity tower

$$
A_{1g}(0)\oplus T_{1u}(1)\oplus A_{2u}(3),
$$

with no degree-two even $T_{2g}$ quadrupole, and then the conversion
$
d=2|S|
$
would give depths
$
0,2,6.
$

The sidecar checks this object, not the Taylor expansion of $h(k)$ and not the
BCH effective Hamiltonian. C:9 for the distinction.

## Coefficient-Walsh result

The raw hop source contains

$$
A_{1g},\qquad T_{1u}[111],\qquad A_{2u},
$$

but it also contains a nonzero degree-two even
$
T_{2g}[111]
$
singlet. That is exactly the quadrupole the parity-selection story needs to
remove. The lattice symbol is also not $C_3$-covariant in this coefficient
Walsh sense. C:8.

Therefore the cube/parity source mechanism is killed in the coefficient-Walsh
language. C:1 for the claim that the genuine BCC hop source directly supplies
the clean $0,1,3$ cube ladder.

## Covariant escape hatch

The sidecar also checks the stronger covariant decomposition under the
octahedral rotation group $O$, where both hop directions and Pauli matrices
transform. This is the correct escape hatch because the hop coefficients are
matrix-valued.

The result is

$$
\|A_1\|^2=1,\qquad
\|A_2\|^2={1\over3},\qquad
\|E\|^2={2\over3},\qquad
\|T_1\|^2=\|T_2\|^2=0.
$$

So the apparent coefficient $T_{2g}$ reassembles covariantly to $T_2=0$, but a
forbidden $E$ quadrupole remains and the required $T_1$ vector is absent. C:8.

Thus the covariant escape hatch is also killed. The source is

$$
A_1\oplus A_2\oplus E,
$$

not

$$
A_1\oplus T_1\oplus A_2.
$$

## Schur obstruction

The deeper theorem is independent of the hop-shell calculation. The residual
three-port family space decomposes under $S_3$ as

$$
3=1\oplus2.
$$

By Schur's lemma, any $S_3$-invariant depth operator has the form

$$
D=\alpha P_1+\beta P_2,
$$

so its spectrum is

$$
\{\alpha,\beta,\beta\}.
$$

It has at most two distinct eigenvalues. C:9.

The residual complete graph $K_3$ Laplacian gives

$$
\operatorname{spec}L(K_3)=\{0,3,3\},
\qquad
\operatorname{spec}(2L(K_3))=\{0,6,6\}.
$$

This is not $\{0,2,6\}$. C:9.

The desired depth operator

$$
D_{\rm depth}=\operatorname{diag}(0,2,6)
$$

has three distinct eigenvalues and is not $S_3$ invariant. Its invariant part
is
$
(8/3)I
$
and its breaking spurion is

$$
D_{\rm depth}-{8\over3}I
=\operatorname{diag}\left(-{8\over3},-{2\over3},{10\over3}\right)
\sim(-4,-1,5).
$$

Therefore deriving $\{0,2,6\}$ is equivalent to deriving an $S_3$-breaking
family spurion. C:9.

## Relation to the depth-scar sidecar

The `depth_scar` sidecar gives a clean mathematical object:

$$
D_{\rm scar}=2\Delta(P_3),
\qquad
\operatorname{spec}D_{\rm scar}=\{0,2,6\}.
$$

That remains useful. The hop-Walsh sidecar says the raw BCC Weyl hop shell does
not derive the path scar or the family-symmetry-breaking spurion. Thus:

$$
\epsilon=\sqrt2-1\quad\text{is derived from the residual transfer graph,}
$$

but

$$
(0,2,6)\quad\text{is still a declared family-depth input unless the path scar
is dynamically selected.}
$$

C:8 for the kill; C:6 for the path-scar construction as a conditional depth
source; C:3 for a future dynamical selection of the spurion.

## Synthesis role

This sidecar prevents a false shortcut. The hierarchy theory should not say:

$$
\text{BCC hop parity}\Rightarrow \{0,2,6\}.
$$

The honest statement is:

$$
\text{BCC/residual transfer}\Rightarrow \epsilon,
\qquad
\text{family scar/spurion}\Rightarrow \{0,2,6\}.
$$

Paper-level statement:

> The genuine Bialynicki-Birula BCC hop source does not realize the proposed
> parity-selected $A_{1g}\oplus T_{1u}\oplus A_{2u}$ cube tower. In both
> coefficient-Walsh and covariant $O$ decompositions, forbidden quadrupole
> content kills the mechanism. Moreover, Schur's lemma shows that an unbroken
> residual $S_3$ family symmetry can give only $\{\alpha,\beta,\beta\}$, so the
> depth ladder $\{0,2,6\}$ requires an $S_3$-breaking spurion.

Certainty: C:8 for the hop-source kill; C:9 for the Schur obstruction.

