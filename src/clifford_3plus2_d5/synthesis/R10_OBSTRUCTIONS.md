# R10 carrier obstructions

The `obstruction_r10` sidecar is a carrier-first classification program. It
does not derive flavor masses. It asks a more primitive question:

> Can microscopic real QCA rule data force the $\mathbb R^{10}$ carrier,
> complex structure $J$, $6+4$ real split, and safe gate algebra needed before
> invoking Spin(10)?

The answer is mostly negative for the closed primitive classes. This is a
boundary document for the whole theory: it separates the textbook Spin(10)
branching from the still-open microscopic QCA bridge.

## The bridge target

The desired bridge is

$$
K_x\simeq\mathbb R^{10},\qquad
J\in SO(K_x),\qquad
J^2=-I,
$$

with a $J$-invariant split

$$
K_x=K_3\oplus K_2,\qquad
\dim_\mathbb R K_3=6,\qquad
\dim_\mathbb R K_2=4.
$$

Then

$$
W=(K_x,J)\simeq\mathbb C^5=\mathbb C^3\oplus\mathbb C^2,
$$

and the chiral spinor is the even exterior algebra

$$
S_+=\Lambda^{\rm even}(W),\qquad \dim_\mathbb C S_+=16.
$$

The hypercharge identity is

$$
Y=-{1\over3}N_3+{1\over2}N_2.
$$

This reproduces the one-generation SM table:

$$
(N_3,N_2)=(0,0),(0,2),(1,1),(2,0),(2,2),(3,1)
$$

with charges
$
0,1,1/6,-2/3,1/3,-1/2
$
and multiplicities
$
1,1,6,3,3,2.
$
C:9 as representation theory once $J$ and the $3+2$ split are given.

The load-bearing problem is not this branching. The problem is forcing
$J$, $K_3/K_2$, and the safe gate algebra from QCA data. C:3 for the current
microscopic bridge.

## No-locking

The safe one-particle geometric gate algebra is only

$$
\operatorname{End}_{SU(3)\times SU(2)\times U(1)}
(\mathbb C^3\oplus\mathbb C^2)
=\mathbb C P_3\oplus\mathbb C P_2.
$$

Thus the QCA may distinguish the whole color block from the whole weak block,
but it may not address individual color axes, individual weak axes, or mix the
blocks:

$$
\operatorname{Hom}(\mathbb C^3,\mathbb C^2),\quad
\operatorname{Hom}(\mathbb C^2,\mathbb C^3),\quad
\text{rank-one color/weak projectors}
$$

are forbidden. This no-locking rule is the central guardrail. C:9.

## Closed primitive classes

The sidecar freezes several negative propositions.

Block-blind primitives do not generate an unseeded complementary $6+4$ center.
If lower-rank distinguishers are added, the no-locking guardrail rejects them.
C:8 for the bounded primitive class explored.

Commuting non-scalar second layers fail because a non-scalar commuting layer
has spectral projectors inside a coarse block. Those become lower-rank central
idempotents, violating no-locking. C:9 as finite-dimensional linear algebra.

Floquet-$\alpha$ block-preserving noncommuting on-site rules close under a
similar obstruction: a rule-generated central compatible $J$ forces one of
three bad outcomes:

$$
\text{lower-rank central locking}
\quad\text{or}\quad
\text{full internal addressability}
\quad\text{or}\quad
\text{effective commutativity}.
$$

The exhaustive signed-twist witness scans $3840$ candidates and finds no
bridge candidate in the no-locking shape. C:8 for the census; C:7 for the
stated class theorem under its hypotheses.

Projector-free monomial-hop Bloch rules of the audited windings give no bridge
candidate in the full $2400$-candidate census. C:8 for that finite family.

The Route-1 four-$J$ orbits are not equivalent under the fixed
$SU(3)\times SU(2)_L\times U(1)_Y$ standard; hypercharge does not survive the
relevant Hodge flip. C:8.

## What remains open

The sidecar does not falsify the BCC/QCA program globally. It closes specific
primitive classes:

- single-site block-blind primitives;
- commuting second-layer locks;
- Floquet-$\alpha$ block-preserving noncommuting on-site rules under the
  one-quadratic-factor-per-block hypothesis;
- the audited monomial-hop Bloch families;
- the tested Route-1 $J$ equivalence relaxation.

Open routes include spatial/defect mechanisms, parameterized rule families,
higher-dimensional carriers, and any mechanism that produces the local
orientation data without seeding the $3+2$ split. C:5 for their viability.

## Synthesis role

This sidecar tells us how to speak honestly about the carrier:

$$
\text{Spin(10) chiral-16 from } \mathbb C^3\oplus\mathbb C^2
\quad\text{is proven;}
$$

$$
\text{QCA rule data forcing } \mathbb C^3\oplus\mathbb C^2
\quad\text{is not yet proven.}
$$

For the paper, the carrier can be used as the accepted ansatz when discussing
flavor, CP, and mass. But the microscopic derivation from real QCA primitives
must be labeled as open, with the no-go propositions listed as constraints.

Paper-level statement:

> The $\mathbb R^{10}\to\mathbb C^3\oplus\mathbb C^2\to
> \Lambda^{\rm even}(\mathbb C^5)$ branch is a clean one-generation
> representation theorem once $J$ and the $3+2$ split are supplied. The
> `obstruction_r10` sidecar shows that several natural QCA primitive classes
> cannot force this data without either seeding it, generating forbidden
> lower-rank addressability, or losing the SM hypercharge standard.

Certainty: C:8 for the closed primitive-class obstructions; C:3 for the
remaining microscopic bridge.

