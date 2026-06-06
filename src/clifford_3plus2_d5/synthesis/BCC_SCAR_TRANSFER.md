# BCC Edge Scar Transfer Calculation

This note performs the concrete calculation demanded by the silver fixed-point
reframing. The goal is to test whether the BCC codimension-two edge scar
actually reduces to the silver transfer problem, rather than merely suggesting
the right integer by eye.

The answer is mixed and useful:

$$
\boxed{
\text{the BCC edge gives the raw integer }2,
\text{ but the full edge graph leaks and the bare Weyl coin does not close it.}
}
$$

Thus BCC supplies the right primitive count for silver, but not yet the full
theorem. A physical scar projection or boundary repair mechanism is still
needed.

## 1. Scalar BCC Edge Quotient

Use integer BCC body-diagonal nearest-neighbor directions

$$
h=(\pm1,\pm1,\pm1).
$$

Take the coordinate edge half-space

$$
\mathcal E=\{(x,y,z)\in\mathbb Z^3:x\ge0,\ y\ge0\},
$$

with edge line

$$
\ell=\{(0,0,z):z\in\mathbb Z\}.
$$

Quotient by translation along $z$ at $k_z=0$. Each allowed $(\Delta x,\Delta y)$
move then has multiplicity $2$ from the two choices $\Delta z=\pm1$. Quotient
also by the edge reflection $x\leftrightarrow y$, so the scalar orbits are

$$
[x,y],\qquad 0\le x\le y.
$$

The edge head has exactly two surviving body-diagonal channels:

$$
[0,0]\longrightarrow 2[1,1].
$$

This is the raw $N=2$ count expected for a codimension-two edge.

Certainty: `C:9`.

## 2. The Edge Is Not Closed

The decisive point is what happens after the first step. The diagonal orbit
$[n,n]$ does not only connect to the next and previous diagonal orbits. For
$n\ge1$,

$$
[n,n]\longrightarrow
2[n-1,n-1]+4[n-1,n+1]+2[n+1,n+1].
$$

The middle term is leakage into the off-diagonal face orbit:

$$
4[n-1,n+1].
$$

Explicitly,

$$
[1,1]\longrightarrow
2[0,0]+4[0,2]+2[2,2].
$$

Therefore the diagonal edge scar is not an invariant scalar subgraph of the
full BCC edge half-space. The full scalar BCC edge quotient is not the silver
transfer matrix

$$
T_{\rm silver}=
\begin{pmatrix}
2&1\\
1&0
\end{pmatrix}.
$$

Certainty: `C:9` for the quotient adjacency. `C:1` for the claim that the full
scalar BCC edge graph directly closes to the silver transfer matrix.

## 3. What A No-Leakage Scar Would Give

If one imposes an additional no-leakage projection onto the diagonal edge
channel, the projected scalar adjacency becomes

$$
[n,n]\longrightarrow
2[n-1,n-1]+2[n+1,n+1].
$$

This retains the two equivalent body-diagonal corridors. But even this
projected adjacency is not automatically the metallic recurrence

$$
x_n=2x_{n+1}+x_{n+2}.
$$

It is a two-channel half-line. To obtain the silver root one still needs the
correct scalar normalization and spectral offset, or equivalently a reduced
Dyson equation of the form

$$
x=\frac{1}{2+x}.
$$

Thus the raw edge count is necessary but insufficient. It supplies the integer
that a silver theorem wants, not the theorem itself.

Certainty: `C:8` for the projected adjacency under the stated no-leakage
projection. `C:3` for identifying that projection with the physical flavor
scar.

## 4. Bare BB Weyl Coin Does Not Kill The Leakage

One possible rescue is that the Bialynicki-Birula Weyl coin might annihilate
the off-diagonal leakage channels in a special spinor subspace. This can be
checked exactly.

At $k_z=0$, combine the two $\Delta z=\pm1$ hops in each $(\Delta x,\Delta y)$
pair. Let

$$
D_+=W_{+++}+W_{++-},
$$

$$
D_-=W_{--+}+W_{---},
$$

be the diagonal forward/backward edge-channel matrices, and let

$$
L_1=W_{-++}+W_{-+-},
\qquad
L_2=W_{+-+}+W_{+--}
$$

be the two leakage-channel matrices. Using the implemented BB convention
$q_\pm=(1\pm i)/4$, one obtains

$$
D_+=
\frac14
\begin{pmatrix}
1+i&1-i\\
1+i&1-i
\end{pmatrix},
$$

with

$$
\ker D_+=\ker L_1=\operatorname{span}\{(i,1)^T\},
$$

and

$$
\ker D_-=\ker L_2=\operatorname{span}\{(-i,1)^T\}.
$$

The leakage kernels have trivial intersection:

$$
\ker L_1\cap\ker L_2=\{0\}.
$$

Therefore there is no nonzero Weyl spinor that kills both leakage channels.
The bare BB Weyl coin does not by itself produce a closed diagonal edge scar.

Certainty: `C:9` as an exact finite matrix statement in the pinned BB
convention.

## 5. Result Of The Calculation

The calculation gives a precise status:

| Claim | Status |
|---|---|
| BCC bulk gives silver | killed |
| BCC flat face gives silver | killed, raw count $4$ |
| BCC coordinate edge has raw two-channel entrance | true |
| Full scalar BCC edge closes on the diagonal scar | killed by leakage |
| Bare BB Weyl coin removes the leakage | killed by exact kernel check |
| BCC edge plus an additional no-leakage scar projector can carry the right integer | conditional |

The route to silver from BCC is therefore not:

$$
\text{BCC edge count }2\Rightarrow \epsilon.
$$

The only honest route is:

$$
\text{BCC edge count }2
+
\text{physical no-leakage scar projection}
+
\text{correct scalar Dyson normalization}
\Rightarrow
\epsilon=\sqrt2-1.
$$

Certainty: `C:5` for this as the surviving research architecture; `C:3` that
the missing no-leakage projection is supplied by the actual flavor boundary
scar.

## 6. The New Theorem To Prove

The calculation changes the target theorem. It is not enough to count two
body-diagonal channels. The theorem must prove all three statements:

1. The physical flavor boundary is a codimension-two BCC scar, not a face.
2. The scar projection eliminates the off-diagonal leakage orbit
   $[n-1,n+1]$.
3. The Schur-reduced scalar channel has effective self-consistency

$$
x=\frac{1}{2+x}
$$

or an equivalent half-line Weyl equation whose decaying solution is

$$
\epsilon=\sqrt2-1.
$$

Until those three steps are proven, silver-from-BCC is a strong geometric
program, not a theorem.
