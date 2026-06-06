# BCC Microscopic Locking and Retarded Readout

This note addresses the two assumptions left open by the positive-stiffness
boundary model:

$$
\boxed{
\text{where do }K_{\rm rel}=q\text{ and the retarded/open readout come from?}
}
$$

The result is not a full derivation from the bare BB update. It is the minimal
local BCC edge mechanism that would make the previous conditional theorem
microscopic: a codimension-two edge has one local edge clock, so asynchronous
arrival from its two boundary faces is the unique local mismatch; and mixed
normal modes are unresolved outgoing channels, so their Green function is the
retarded half-line Weyl function rather than a closed recurrent pole.
[BARE_BB_EDGE_UPDATE_AUDIT.md](BARE_BB_EDGE_UPDATE_AUDIT.md) proves why this
extra boundary principle is necessary: the bare BB Weyl spinor alone has no
nonzero state that removes both mixed-normal leakage channels.
[BARE_BB_RELATIVE_CHANNEL_UPDATE.md](BARE_BB_RELATIVE_CHANNEL_UPDATE.md) then
shows where those channels live before projection: they are the closed two-way
relative-depth direction of the unprojected bare edge walk.
[BARE_BB_WEDGE_DOMAIN_AUDIT.md](BARE_BB_WEDGE_DOMAIN_AUDIT.md) checks the
actual half-space wedge and finds the same obstruction in geometric form:
$q=0$ leaks to $q=\pm2$ for every nonzero common depth and can return.
[BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md](BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md)
computes that return:
$R_{\rm rel}^{(2)}=\operatorname{diag}(-(1+i)/4,-(1-i)/4)$.
[BARE_BB_FACE_EXTERIOR_INSUFFICIENCY.md](BARE_BB_FACE_EXTERIOR_INSUFFICIENCY.md)
then shows that ordinary face exterior channels cannot remove this first
return. The required outgoing condition is relative to the diagonal edge clock,
not merely to the exterior faces.
[EDGE_CLOCK_SCATTERING_BOUNDARY_UPDATE.md](EDGE_CLOCK_SCATTERING_BOUNDARY_UPDATE.md)
constructs that condition explicitly by routing mixed-normal BB amplitudes into
outgoing clock-error ports.
[EDGE_CLOCK_SCATTERING_MINIMALITY.md](EDGE_CLOCK_SCATTERING_MINIMALITY.md)
shows that this clock-error sector is forced by the visible BB norm deficit;
only the retarded/no-incoming asymptotic condition remains a physical boundary
principle. Its exact causal compression is proved in
[CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md](CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md):
with no incoming clock-error data, the visible powers are $A_{\mathbb N}^t$,
while a recurrent return block restores the bare two-step obstruction.

## 1. One Edge Clock

At a BCC edge two boundary faces meet. Let their inward normal depths be

$$
(r_1,r_2).
$$

It is useful to split them into trace and relative coordinates

$$
r={r_1+r_2\over2},\qquad q=r_1-r_2.
$$

The edge reflection exchanges the two faces:

$$
\mathcal R:\quad (r_1,r_2)\mapsto(r_2,r_1),
\qquad
r\mapsto r,\qquad q\mapsto -q.
$$

The trace $r$ is the common radial depth into the edge. The relative coordinate
$q$ is asynchronous normal arrival: one face is ahead of the other. A local
edge update has only one boundary event at a given edge tick. Therefore the
microscopic gluing condition for a synchronous edge event is

$$
r_1=r_2.
$$

Equivalently,

$$
\boxed{
K_{\rm rel}=r_1-r_2=q.
}
$$

This is the first point that becomes exact. Suppose a local linear mismatch
operator is built only from the two normal depths:

$$
K=a r_1+b r_2.
$$

It must vanish on the synchronous edge line $r_1=r_2$. Hence

$$
a+b=0,
$$

so

$$
K=a(r_1-r_2)=a q.
$$

Thus, up to normalization, $q$ is the unique local linear mismatch. Equivalently,
it is the unique reflection-odd scalar in the local two-face normal space after
the trace mode is removed. This does not use flavor fitting. It is local edge
kinematics. `C:9`.

## 2. Positive Locking From a Constraint Field

The edge does not need to prefer either face. The sign of $q$ is orientation
data, not energy data. A local boundary-locking field can therefore couple only
to the violation norm

$$
H_{\rm lock}=g\,K_{\rm rel}^\dagger K_{\rm rel}.
$$

Using the unique mismatch operator gives

$$
\boxed{
H_{\rm lock}=gq^2,\qquad g>0.
}
$$

The first mixed-normal leakage sectors produced by the BB edge blocks have

$$
q=\pm2,
$$

so their adjacent violation energy is

$$
H_{\rm leak}=4g.
$$

The microscopic picture is simple: the two boundary faces may have separate
normal coordinates, but the codimension-two edge has one local clock. A
same-normal event reaches the edge synchronously and remains in $q=0$. A
mixed-normal event reaches the two faces out of phase and is a violation of the
edge gluing constraint. The positive square is the backreaction of enforcing a
single edge clock.

This closes the sign of the stiffness once the locking field is accepted. It
does not yet prove that a deeper physical boundary material dynamically
generates that field.
`C:6` for the local edge-locking model; `C:3` for deriving the same field from
the deeper boundary-material dynamics.

## 3. Why the Readout Is Retarded

The local unitary dilation proved that probability is not deleted when the
visible $q=0$ scar is compressed. The unresolved channels exist. The physical
question is whether the mass readout closes them into a recurrent finite bath
or treats them as outgoing boundary modes.

The microscopic retarded model is a half-line channel attached to the edge. Let
$h_{\rm out}$ be the scalar nearest-neighbor half-line Hamiltonian for an
unresolved outgoing channel. Its boundary Weyl function is

$$
m(z)=\langle0|(z-h_{\rm out})^{-1}|0\rangle.
$$

Removing the first boundary site leaves an identical half-line. Therefore

$$
\boxed{
m(z)={1\over z-m(z)}.
}
$$

The two algebraic branches are

$$
m_\pm(z)={z\pm\sqrt{z^2-4}\over2}.
$$

The physical retarded branch is fixed by the outgoing normalization

$$
m(z)\sim {1\over z}\qquad |z|\to\infty,
$$

and by the sign condition

$$
\operatorname{Im}m(E+i0)<0
$$

inside the open channel. Thus

$$
\boxed{
m_R(z)={z-\sqrt{z^2-4}\over2},
}
$$

with the square-root branch chosen so $\sqrt{z^2-4}\sim z$ at infinity. The
other solution behaves as $m_+(z)\sim z$ and is the growing/recurrent branch. It
is not a retarded boundary response. `C:9` for the Weyl equation and branch
selection.

For the mixed-normal leakage sector with locking gap $4g$, the retarded
feedback has the form

$$
\Sigma_{\rm mix}^{R}(z,g)
=
M^\dagger m_R(z-4g) M,
$$

where $M$ is the total mixed-normal boundary coupling. Since

$$
m_R(z-4g)\sim-{1\over4g}
\qquad g\to\infty,
$$

the hard-locking limit gives

$$
\boxed{
\lim_{g\to\infty}\Sigma_{\rm mix}^{R}(z,g)=0.
}
$$

This is the retarded version of the previous Schur compression. The difference
is conceptual: leakage is not erased. It is completed by an outgoing channel
whose boundary Green function is retarded and absorptive. `C:9` under the
half-line outgoing model.

## 4. The Visible Compression

The visible mass-return branch is then the retarded open compression

$$
P_{q=0}UP_{q=0}=A_{\mathbb N},
$$

not the pole spectrum of a closed recurrent Julia bath. The branch norm remains

$$
B_+^\dagger B_+ + B_-^\dagger B_-={1\over2}I,
\qquad c={1\over\sqrt2}.
$$

The radial transfer trace is

$$
\operatorname{tr}T(\zeta)=2\zeta+{1\over\zeta},
$$

and its minimum is at

$$
\zeta=c={1\over\sqrt2}.
$$

Therefore the orientation-preserving transfer has

$$
\lambda_\pm=\sqrt2\pm1,\qquad
\epsilon=\sqrt2-1.
$$

The silver root is not an eigenvalue of the closed enlarged unitary. It is the
decaying transfer root of the retarded visible compression.

## 5. What Is Now Derived

The boundary chain is now:

$$
\text{two-face BCC edge}
\Rightarrow
\text{one edge clock}
\Rightarrow
K_{\rm rel}=q
\Rightarrow
H_{\rm lock}=gq^2
\Rightarrow
q=\pm2\text{ leakage gap }4g
\Rightarrow
\Sigma_{\rm mix}^{R}\to0
\Rightarrow
A_{\mathbb N}\text{ survival readout}.
$$

This is the flesh of the mismatch and retarded assumptions. The local edge
symmetry gives the form of $K_{\rm rel}$ exactly. The single-clock locking field
and outgoing half-line readout are the microscopic boundary principle being
proposed. They are not yet consequences of the full bare BB update.

## 6. Falsifiers

The mechanism fails if any of the following holds:

1. the BCC edge admits two independent local boundary clocks rather than one;
2. a local edge rule can generate another reflection-odd linear mismatch not
   proportional to $q$;
3. the true mixed-normal channels are recurrent at the mass readout and do not
   have the retarded half-line Weyl branch;
4. the mass observable is a closed finite-box pole rather than the boundary
   survival response.

These are sharp failures. In particular, a closed recurrent bath would replace
$A_{\mathbb N}$ by an energy-dependent Schur pole problem and the silver
transfer would no longer be protected.

## 7. Certainty Ledger

| Claim | Status |
|---|---|
| trace/relative split $(r,q)$ and reflection action | `C:9` |
| uniqueness of $K_{\rm rel}=q$ as local linear mismatch | `C:9` |
| $H_{\rm lock}=gK^\dagger K=gq^2$ is positive | `C:9` |
| mixed-normal adjacent gap is $4g$ | `C:9` |
| half-line Weyl equation and retarded branch | `C:9` |
| hard-locking retarded feedback $\Sigma_{\rm mix}^{R}\to0$ | `C:9` under outgoing model |
| single-edge-clock locking field as physical boundary dynamics | `C:6` |
| mass readout uses retarded survival compression | `C:5`-`C:6` |
| deeper boundary-material derivation of locking plus outgoing bath | `C:3` |

The concise statement is:

$$
\boxed{
\text{The mismatch is the antisymmetric normal clock error of a two-face BCC
edge, and the silver readout is the retarded boundary response after that
error is locked.}
}
$$
