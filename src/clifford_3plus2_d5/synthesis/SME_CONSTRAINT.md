# SME constraint on the CP slot

The `sme` sidecar asks whether the CP-odd $T_{2g}$ lattice correction found in
`cp` is already experimentally excluded by Lorentz-violation searches. It is
therefore a constraint sidecar: it does not create flavor structure, masses, or
phases. It tests whether one structural CP slot is dead.

The sidecar's verdict is:

$$
\epsilon_{\max}\sim 2\times 10^{-33}\ {\rm m}
\approx 10^2\,\ell_P,
$$

using a representative fermion-sector dim-5 SME bound
$
|d^{(5)}|\lesssim 10^{-17}\ {\rm GeV}^{-1}.
$
This is an unfalsifiable pass in the sidecar's own classification: above the
Planck length, but far below present observational reach. C:4, because the
numerical bound is representative and not entry-ID verified in the sidecar.

## Object being constrained

The input is the alpha CP slot from the BCC walk. Per chirality,

$$
H^{(1)}_\chi(k)
=\sigma^x k_yk_z-\sigma^y k_xk_z+\sigma^z k_xk_y.
$$

Equivalently, the nonzero tensor entries are

$$
T^{x,yz}=+1,\qquad
T^{y,xz}=-1,\qquad
T^{z,xy}=+1.
$$

The identity-Pauli component vanishes, so the correction is spin-tensorial,
not a scalar dispersion correction. The two chirality blocks are equal. C:8
for the finite symbolic decomposition.

The group location remains the cubic lattice sector:

$$
H^{(1)}\in T_{2g}\subset \mathcal P_2(O_h),
$$

not color, hypercharge, or family depth. The SME sidecar translates this
spacetime anisotropy into an effective-field-theory coefficient.

## SME identification

The structural identification is

$$
\Delta\mathcal L
\sim
\epsilon\,T^{aij}\,
\bar\psi\,\gamma^a\gamma^5\,\partial_i\partial_j\psi.
$$

This has the right signatures for a dim-5 non-minimal SME fermion-sector
coefficient:

- two derivatives, hence dimension five;
- axial-vector spin structure, hence a $d$-type coefficient;
- CP-odd and CPT-even, matching the CP sidecar's $CP$ broken but $CPT$ exact
  pattern;
- spatial off-diagonal $T_{2g}$ cubic anisotropy.

The sidecar labels the target as
$
d^{(5)}_{\alpha\beta\gamma}.
$
C:6 for this structural identification. It is not yet C:8 because the
Kostelecky-Mewes Hamiltonian-form normalization is not derived and
field-redefinition equivalence is not checked.

## Scale bound

The sidecar uses the representative bound

$$
|d^{(5)}|\lesssim 10^{-17}\ {\rm GeV}^{-1},
\qquad
1\ {\rm GeV}^{-1}\simeq 1.97\times 10^{-16}\ {\rm m}.
$$

Since all three populated tensor entries have coefficient magnitude one,

$$
\epsilon |T^{aij}|\lesssim |d^{(5)}|_{\rm bound}
\quad\Longrightarrow\quad
\epsilon\lesssim 1.97\times 10^{-33}\ {\rm m}.
$$

With
$
\ell_P\simeq 1.62\times 10^{-35}\ {\rm m},
$
this is

$$
\log_{10}(\epsilon_{\max}/\ell_P)\simeq 2.09.
$$

The scale is about $10^8$ below the sidecar's nominal observable threshold
$10^{-25}\ {\rm m}$. Therefore the BCC CP slot is not experimentally useful in
this channel, but it is not killed. C:4.

## Caveats

Three caveats are load-bearing:

1. The Kostelecky-Russell table entries behind the representative
   $10^{-17}\ {\rm GeV}^{-1}$ bound are not verified in the sidecar. A bound
   one or two orders tighter would move the verdict toward strict
   Planck-consistency; a much tighter bound could become dangerous. C:3 for
   the precise numerical placement.
2. The mapping uses symmetry and tensor structure, not the full
   Kostelecky-Mewes Hamiltonian normalization. This likely changes only
   normalization unless a convention-dependent field redefinition intervenes.
   C:6 for the channel, C:3 for exact coefficient normalization.
3. Field-redefinition triviality is unchecked. If this $d^{(5)}$ direction is
   removable or equivalent to another SME coefficient, then this particular
   bound on $\epsilon$ is vacuous rather than falsifying. C:3.

Photon-sector bounds, especially non-minimal $(k_F)^{(5)}$ limits, are not
applicable to this fermion-sector walk correction. Applying them here would be
a category error. C:7.

## Synthesis role

The result is a consistency gate:

$$
O_h:T_{2g}\ {\rm CP\ slot}
\longrightarrow
d^{(5)}_{\alpha\beta\gamma}\ {\rm candidate}
\longrightarrow
\epsilon\lesssim 2\times 10^{-33}\ {\rm m}.
$$

It says the CP slot can survive known Lorentz-violation logic if the lattice
scale is near the quantum-gravity scale. It does not tell us the CKM phase, it
does not select three generations, and it does not constrain the radial mass
response directly.

Paper-level statement:

> The leading BCC lattice CP correction maps structurally to a dim-5
> fermion-sector SME axial spin-tensor coefficient. Using the sidecar's
> representative bound gives $\epsilon_{\max}\sim 10^2\ell_P$, so this channel
> is observationally safe but not presently predictive.

Certainty: C:4 overall; C:8 for the symbolic $T_{2g}$ decomposition.

