# Algebraic generation kills

The `triality`, `broken_triality`, and `exceptional` sidecars are negative
controls for the family problem. They all ask whether three Standard Model
generations can be obtained from algebraic carrier structure without declaring
three copies by hand.

The answer is no for the tested natural mechanisms.

## Exact Spin(8) triality

Spin(8) has a genuine $Z_3$ outer automorphism. Spin(10) does not. Therefore a
triality generation mechanism must use a Spin(8) subgroup of Spin(10).

The natural Pati-Salam-aligned embedding uses gamma indices $\{0,\ldots,7\}$.
The Spin(8) Cartan basis is

$$
H_k={1\over2}\gamma_{2k}\gamma_{2k+1},\qquad k=0,1,2,3.
$$

The triality Cartan action is

$$
T={1\over2}
\begin{pmatrix}
1&1&1&1\\
1&1&-1&-1\\
1&-1&1&-1\\
-1&1&1&-1
\end{pmatrix},
\qquad T^3=I.
$$

The restricted SM Cartan inside this Spin(8) is
$
SU(3)_c\oplus U(1)_{Y'}
$
rather than the full SM, because $SU(2)_L$ and $SU(2)_R$ need the missing
Spin(10) directions. The K1 necessary condition is that $T$ preserve this
restricted SM Cartan subspace. It does not:

$$
(-1,1,0,0)\mapsto(0,0,-1,1),
$$

$$
(-1,0,1,0)\mapsto(0,-1,0,1),
$$

$$
\left({1\over3},{1\over3},{1\over3},{1\over2}\right)
\mapsto
\left({3\over4},-{1\over12},-{1\over12},-{1\over12}\right).
$$

All three images leave the restricted SM Cartan. Thus triality does not produce
three equivalent SM generations. C:8.

The structural reason is rank and factorization: Pati-Salam has rank five,
while Spin(8) has rank four, and the Pati-Salam split
$
\mathrm{Cl}(0,10)=\mathrm{Cl}(0,6)\otimes\mathrm{Cl}(0,4)
$
makes one Cartan direction qualitatively different. Spin(8) triality treats the
four directions democratically. These are incompatible requirements. C:8.

The exact-triality generation mechanism is killed. C:1.

## Broken triality

The broken-triality sidecar tries a weaker idea: maybe the exact K1 failure
itself provides the symmetry-breaking pattern. It projects the triality orbit
of the restricted hypercharge vector $Y'$ back into the SM Cartan and builds

$$
Y_{ij}=\langle \Pi_{\rm SM}(\tau^iY'),\Pi_{\rm SM}(\tau^jY')\rangle.
$$

For the natural hypercharge-aligned choice
$
Y'=(1/3,1/3,1/3,1/2),
$
the projected orbit gives

$$
Y=
\begin{pmatrix}
7/12&11/72&11/72\\
11/72&169/336&-53/1008\\
11/72&-53/1008&59/1008
\end{pmatrix}.
$$

The eigenvalues are

$$
\left\{{5\over7},{31\over72},0\right\}.
$$

The nonzero hierarchy ratio is

$$
{5/7\over 31/72}={360\over217}\approx1.66.
$$

This is essentially flat and far below even a minimal hierarchy threshold. The
zero eigenvalue is a rank-deficit caused by residual $H_1\leftrightarrow H_2$
symmetry. C:8.

Therefore broken triality does not generate the SM mass hierarchy. CP and
parameter-count audits were correctly skipped. C:1 for the proposed
hypercharge-aligned broken-triality mechanism.

## Exceptional algebra

The `exceptional` sidecar tests broader octonionic and exceptional-Jordan
routes.

The bimultiplication route collapses to

$$
\dim \operatorname{Bi}(\mathbb O)=28=\dim\mathfrak{so}(8),
$$

so it inherits the Spin(8) triality failure. C:8.

The three Fano lines through a preferred unit, e.g.
$
\{1,6,7\},\{2,5,7\},\{3,4,7\},
$
do not produce three $SU(2)$ Lie algebras. Octonion non-associativity makes
commutators such as $[L_a,L_b]$ leave the span of the putative line generators.
The candidates also overlap through the shared $e_7$ direction. C:8.

The exceptional Jordan algebra gives the standard Spin(10) branching

$$
J_3(\mathbb O):\qquad 27=16\oplus10\oplus1.
$$

This is one chiral-16 plus extras, not three generations. The three
preferred-row spinor candidates overlap pairwise by one octonion, and three
independent chiral-16 copies would require $3\times16=48$ real dimensions, far
more than the 27-dimensional carrier. C:8.

The complexified extension doubles the one-generation content:

$$
J_3^{\mathbb C}(\mathbb O):\qquad
54=16\oplus16^\ast\oplus10\oplus10^\ast\oplus1\oplus1^\ast.
$$

This is particle plus antiparticle structure, not three independent chiral
generations. C:8.

Thus the tested exceptional-algebra routes are killed at carrier
representation level. C:1 for the claim that these bare exceptional carriers
derive three SM generations.

## Synthesis role

The common lesson is:

$$
\text{one SM generation is a carrier theorem;}
\qquad
\text{three generations are not an algebraic carrier theorem here.}
$$

The exact mechanisms fail for different local reasons, but the global reason is
the same: the SM carrier is organized by Pati-Salam and a chiral Spin(10)
spinor, while the proposed threefold algebraic symmetries act on structures
that do not preserve the SM content as three equivalent copies.

For the synthesis tree, this means:

- do not reintroduce Spin(8) triality as a generation theorem;
- do not use broken triality as a hidden mass hierarchy;
- do not use bare $J_3(\mathbb O)$ or $J_3^\mathbb C(\mathbb O)$ as three
  families;
- if three generations appear in a formula, mark whether they are empirical
  input, boundary-depth data, or a derived symmetry-breaking result.

Paper-level statement:

> The natural algebraic generation routes tested in this repository fail:
> Spin(8) triality does not preserve the SM-restricted Cartan, broken triality
> gives a rank-deficient and nearly flat Yukawa, and the exceptional Jordan
> carriers branch as one chiral-16 plus vector/singlet content rather than
> three chiral-16s. The generation problem must be carried by boundary/depth
> structure or new input, not by these algebraic carriers.

Certainty: C:8 overall; C:1 for the killed mechanisms.

