# Bare BB Face-Exterior Insufficiency

The wedge audit showed that the physical BCC edge domain confines relative
depth. The relative-return obstruction then showed that $q=\pm2$ feeds back to
$q=0$ at second order. This note checks a natural rescue:

$$
\boxed{
\text{can ordinary outgoing exterior channels at the two wedge faces make the
relative channel retarded?}
}
$$

The answer is no for the $q=0$ scar. Face exterior channels can absorb outward
hops at $q=\pm n$, but the first return $q=\pm2\to0$ is an inward hop that stays
inside the wedge. Therefore a face-only outgoing boundary cannot protect the
diagonal scar.

## 1. Face Boundaries

In integer coordinates

$$
n=r_1+r_2,\qquad q=r_1-r_2,
$$

the physical wedge is

$$
n\ge |q|.
$$

The two faces are

$$
q=+n\quad(r_2=0),
\qquad
q=-n\quad(r_1=0).
$$

At the upper face $q=+n$, the mixed hop

$$
(+,-):\quad q\mapsto q+2
$$

leaves the wedge. The opposite mixed hop

$$
(-,+):\quad q\mapsto q-2
$$

goes inward. At the lower face the roles reverse. Thus a face-exterior channel
only catches outward relative motion from the two faces. `C:9`.

## 2. First Return Does Not Touch Exterior

Start at the diagonal scar:

$$
(n,0),\qquad n\ge2.
$$

The first mixed-normal leakage step is

$$
(n,0)\to(n,+2)
\quad\text{or}\quad
(n,0)\to(n,-2).
$$

The first return is

$$
(n,+2)\to(n,0),
\qquad
(n,-2)\to(n,0).
$$

Both return hops remain inside the wedge for all $n\ge2$. Even at the shallow
case $n=2$, where $(2,+2)$ and $(2,-2)$ lie on the faces, the return hop is the
inward face hop, not the outward exterior hop. Therefore face absorption does
not remove the two-step feedback.

The exact BB return matrix from this interior/inward process is

$$
R_{\rm rel}^{(2)}
=
\begin{pmatrix}
-{1+i\over4}&0\\
0&-{1-i\over4}
\end{pmatrix}.
$$

The face exterior has no opportunity to act before this return has already
occurred. `C:9`.

## 3. What A Face-Only Boundary Can Do

A face-only outgoing completion is still physically meaningful. It can make the
outer edges of the relative strip absorptive:

$$
q=+n\to q=n+2,
\qquad
q=-n\to q=-n-2.
$$

That changes long relative excursions and finite-depth spectra. But it cannot
produce the hard $q=0$ scar, because the first leakage-return loop is local near
the diagonal and does not require reaching either face.

Hence:

$$
\boxed{
\text{face exterior channels}
\;\not\Rightarrow\;
\Sigma_{\rm mix}^R\to0\text{ on }q=0.
}
$$

The boundary law required by flavor must act earlier:

1. directly on the relative coordinate $q$ through a locking term $gq^2$;
2. directly on the transition $q=0\leftrightarrow q=\pm2$ through an edge-clock
   constraint;
3. or by converting every $q\ne0$ relative state, not only face states, into an
   outgoing unresolved bath.

## 4. Microscopic Consequence

This separates two different notions of "outgoing":

$$
\text{outgoing through the physical faces}
\neq
\text{outgoing from the diagonal scar}.
$$

The silver scar needs the second one. It is a boundary condition on relative
depth with respect to the diagonal edge clock, not merely the usual exterior
condition at $r_1=0$ or $r_2=0$.

This is why the single-clock model is not cosmetic. It says that $q\ne0$ is
already outside the low-energy edge event, even when it remains inside the
geometric wedge.

## 5. Certainty Ledger

| Claim | Status |
|---|---|
| face exterior catches only outward hops at $q=\pm n$ | `C:9` |
| first return $q=\pm2\to0$ stays inside the wedge | `C:9` |
| shallow case $n=2$ return is inward, not exterior | `C:9` |
| face-only outgoing boundary removes $R_{\rm rel}^{(2)}$ | `C:1` |
| relative-depth outgoing/locking boundary removes first return | `C:6` target |
| full microscopic derivation of relative-depth boundary law | `C:3` |

The concise result is:

$$
\boxed{
\text{Exterior faces do not save the scar. The needed outgoing condition is
relative to the diagonal edge clock, not merely to the wedge faces.}
}
$$
