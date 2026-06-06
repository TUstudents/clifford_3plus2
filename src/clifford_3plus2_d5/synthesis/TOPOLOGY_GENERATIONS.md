# Topology and failed generation routes

The `topology` sidecar is a negative sidecar. It asks whether three Standard
Model generations can be obtained from a natural topological or discrete
subgroup mechanism on the existing
$
\text{BCC}\times S_+(\mathrm{Cl}(0,10))
$
carrier.

The answer is no for every tested mechanism. This is important for the
synthesis tree: topology should not be used as a hidden source of family
number unless a genuinely new invariant is introduced.

## Spatial $Z_3$

The BCC lattice has an order-three body-diagonal rotation

$$
R:(x,y,z)\mapsto(y,z,x).
$$

On the eight body-diagonal directions, its cycle structure is
$
(3,3,1,1).
$
The spin-$1/2$ lift cubes to $-I$, as it should for the double cover.
C:8.

This spatial $Z_3$ does not act on the chiral internal carrier. The
chiral-16 comes from internal $\mathrm{Cl}(0,10)$ labels, not spatial
coordinates, so the decomposition is simply

$$
16=16\cdot 1+0\cdot\omega+0\cdot\omega^2.
$$

Therefore spatial $Z_3$ cannot be three families. C:9 for the trivial
internal action; C:1 for the hypothesis "spatial $Z_3$ gives the three
generations" on this carrier.

The sidecar also finds that the Bialynicki-Birula hop convention is not
$Z_3$-equivariant under this rotation: all eight hop residuals are nonzero.
This is orthogonal to flavor but relevant to the QCA carrier. It says the BCC
lattice may have the spatial symmetry while the chosen walk convention fixes
extra orientation data. C:8.

## Color-center $Z_3$

The color center route tries to use

$$
g_3=\operatorname{diag}(1,\omega,\omega^2)\in SU(3)_c,
\qquad \omega=e^{2\pi i/3}.
$$

Counting one SM generation with $\nu^c$ gives

| field | count | $Z_3$ character split |
|---|---:|---|
| $Q_L$ | 6 | $(2,2,2)$ |
| $u^c$ | 3 | $(1,1,1)$ |
| $d^c$ | 3 | $(1,1,1)$ |
| $L$ | 2 | $(2,0,0)$ |
| $e^c$ | 1 | $(1,0,0)$ |
| $\nu^c$ | 1 | $(1,0,0)$ |

Thus

$$
16=(8,4,4)_{(1,\omega,\omega^2)}.
$$

This is not three equivalent copies. Leptons sit entirely in the trivial
character, while quarks distribute among the three. A symmetric
three-family interpretation would require equal multiplicities, which is
impossible already because $16/3$ is not an integer. C:9.

So the color-center family hypothesis is killed:

$$
Z(SU(3)_c)=Z_3\quad\not\Rightarrow\quad N_{\rm gen}=3.
$$

Certainty: C:1 for the proposed mechanism.

## $\pi_3$ torsion

The sidecar's literature note surveys carrier-relevant cosets such as

$$
\mathrm{Spin}(10)/SU(5),\quad
\mathrm{Spin}(10)/(SU(4)\times SU(2)\times SU(2)),\quad
G_2/SU(3),\quad
F_4/\mathrm{Spin}(9).
$$

The intended conclusion is that no natural carrier coset supplies a
$\mathbb Z/3$ torsion class in $\pi_3$ that could be read as three
generations. This is plausible and consistent with the surveyed examples.
C:5.

For the paper we should recheck the homotopy table before citing it as a
theorem. The note contains at least one internal inconsistency:
$
G_2/SU(3)\simeq S^6
$
implies
$
\pi_3(S^6)=0,
$
not $\mathbb Z$. The conclusion "not $\mathbb Z/3$" survives, but the row
should be cleaned before publication. C:9 for the correction; C:5 for the
overall literature-level kill until the table is source-verified.

What is safe to use now:

$$
\text{No verified carrier-relevant }\pi_3\text{ route currently derives }
N_{\rm gen}=3.
$$

What is not safe:

$$
\text{A universal theorem that all relevant }G/H\text{ cannot have }
\pi_3=\mathbb Z/3
$$

without checking the exact embedding indices.

## Anomaly route

The anomaly audit is the cleanest negative topology result. Per generation,
the Standard Model charges cancel the continuous gauge and mixed anomalies:

$$
\sum Y=0,\qquad
\sum Y^3=0,\qquad
SU(2)_L^2U(1)_Y=0,\qquad
SU(3)_c^2U(1)_Y=0.
$$

The Witten $SU(2)$ global anomaly also gives no family-number constraint. One
generation contains four $SU(2)_L$ doublets:

$$
3\cdot Q_L + L = 3+1=4,
$$

which is even. Therefore $4N$ is even for every nonnegative integer $N$.
C:9.

The generation constraint reduces to

$$
0=0,
$$

so anomalies allow $N=3$ but do not select it. C:9 for the standard anomaly
logic; C:1 for the hypothesis "anomaly cancellation forces exactly three
generations" in this carrier.

## Synthesis role

Topology currently contributes negative constraints, not constructive flavor
mechanisms:

$$
\begin{array}{ccl}
\text{spatial }Z_3 &\rightarrow& \text{internal action trivial},\\
Z(SU(3)_c) &\rightarrow& 16=(8,4,4),\\
\pi_3(G/H) &\rightarrow& \text{no verified } \mathbb Z/3\text{ carrier class},\\
\text{anomalies} &\rightarrow& \text{cancel for any }N.
\end{array}
$$

This tells us where not to look. The family mechanism must come from the
boundary/scar/nilpotent structure or from a new invariant not yet present in
the carrier. The topology branch should be drawn as a killed branch of the
flavor tree.

Paper-level statement:

> The existing BCC $\times$ chiral-16 carrier does not derive three
> generations from spatial $Z_3$, the color center, $\pi_3$ torsion of the
> surveyed carrier cosets, or anomaly cancellation. These mechanisms are
> negative controls; the surviving flavor route is the boundary/depth
> construction.

Certainty: C:8 overall; C:9 for the spatial/color/anomaly finite arguments;
C:5 for the unverified $\pi_3$ literature table.

