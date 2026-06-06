# Bare BB Wedge Domain Audit

The relative-cover audit used independent Bloch phases for trace and relative
depth. That is the correct local interior symbol, but the physical
codimension-two edge is not the full relative cover. It is the wedge

$$
r_1\ge0,\qquad r_2\ge0.
$$

This note checks whether the wedge geometry itself supplies the missing
microscopic boundary law.

The answer is again negative, but sharper:

$$
\boxed{
\text{the physical wedge confines }q\text{ to a finite strip at fixed depth,
but it does not lock }q=0\text{ and it does not make leakage retarded.}
}
$$

## 1. Integer Wedge Coordinates

Use

$$
n=r_1+r_2,\qquad q=r_1-r_2.
$$

Then

$$
r_1={n+q\over2},\qquad r_2={n-q\over2}.
$$

The wedge condition is exactly

$$
\boxed{
n\ge |q|,\qquad n\equiv q\pmod2.
}
$$

At fixed $n$, the allowed relative depths are

$$
q=-n,-n+2,\ldots,n-2,n.
$$

Thus the real edge does not have an infinite relative line at fixed common
depth. It has a finite relative strip with two face boundaries $q=\pm n$.
`C:9`.

## 2. Bare Hop Rules In The Wedge

A normal hop $(\sigma_1,\sigma_2)$ changes $(n,q)$ by

$$
\Delta n=\sigma_1+\sigma_2,
\qquad
\Delta q=\sigma_1-\sigma_2.
$$

Therefore:

| normal signs | $\Delta n$ | $\Delta q$ | condition to remain in wedge |
|---|---:|---:|---|
| $(+,+)$ | $+2$ | $0$ | always |
| $(-,-)$ | $-2$ | $0$ | $r_1,r_2\ge1$ |
| $(+,-)$ | $0$ | $+2$ | $r_2\ge1$ |
| $(-,+)$ | $0$ | $-2$ | $r_1\ge1$ |

For a diagonal state $q=0$ at depth $n=2r$, the mixed-normal hops are allowed
whenever

$$
r\ge1.
$$

Explicitly,

$$
(n,0)\xrightarrow{(+,-)}(n,+2),
\qquad
(n,0)\xrightarrow{(-,+)}(n,-2),
\qquad n\ge2.
$$

So the actual wedge geometry does not protect the diagonal scar. It blocks
mixed-normal leakage only at the single head point $n=0$. Everywhere in the
interior of the edge, leakage is a legal bare BB hop. `C:9`.

## 3. The Relative Channel Is Recurrent

The same rules also show that leakage can return:

$$
(n,+2)\xrightarrow{(-,+)}(n,0),
\qquad
(n,-2)\xrightarrow{(+,-)}(n,0),
\qquad n\ge2.
$$

Thus the physical wedge gives a two-way relative-depth channel. The face
boundaries at $q=\pm n$ remove the outward mixed hop, but they do not remove the
inward hop. Without an additional boundary completion, the truncated wedge is
not a unitary automaton; with a reflecting or recurrent completion, relative
depth can feed back into $q=0$.
[BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md](BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md)
computes the first such feedback explicitly and finds a full-rank nonzero
two-step return operator.

This is the key microscopic conclusion:

$$
\boxed{
\text{wedge geometry alone gives confinement, not retarded absorption.}
}
$$

The wedge can motivate an outgoing boundary interpretation if the exterior
faces are treated as vacuum channels, but that is a physical boundary condition.
It is not produced by the bare interior BB hop algebra alone. `C:9`.

## 4. Relation To The Single-Clock Model

The single-clock locking model says that the edge permits only synchronous
normal arrival as a low-energy boundary event:

$$
K_{\rm rel}=q,
\qquad
H_{\rm lock}=gq^2.
$$

The wedge audit shows why this is extra dynamics rather than a coordinate
identity. The bare wedge allows $q=\pm2,\pm4,\ldots$ at the same common depth.
The geometry supplies the coordinate $q$ and its reflection symmetry, but it
does not supply the positive stiffness.

The microscopic completion must therefore add one of the following:

1. a local edge-locking field that gives $gq^2$;
2. an outgoing face/vacuum channel that makes relative-depth excursions
   retarded and non-returning for the mass readout;
3. a unitary local boundary rule whose Schur limit is equivalent to 1 or 2.

The bare wedge does not decide among these. It kills the hope that geometry
alone closes the scar.

## 5. Status

The boundary-update stack now reads:

$$
\text{bare BB edge hop algebra}
\Rightarrow
\text{closed/recurrent relative channel}
\Rightarrow
\text{need boundary dynamics for }q.
$$

The remaining target is no longer vague:

$$
\boxed{
\text{derive a local face/edge boundary rule that turns the wedge's recurrent
relative channel into the locked or retarded response.}
}
$$

## 6. Certainty Ledger

| Claim | Status |
|---|---|
| wedge domain is $n\ge|q|$, $n\equiv q\pmod2$ | `C:9` |
| diagonal $q=0$ leaks to $q=\pm2$ for $n\ge2$ | `C:9` |
| $q=\pm2$ can return to $q=0$ for $n\ge2$ | `C:9` |
| wedge geometry alone superselects $q=0$ | `C:1` |
| wedge geometry alone gives retarded relative channel | `C:1` |
| wedge plus local locking/outgoing boundary rule | `C:6` effective target |
| full microscopic derivation of that boundary rule | `C:3` |

The concise result is:

$$
\boxed{
\text{The physical BCC wedge names the mismatch coordinate, but it does not
solve it. The missing physics is still the boundary law for }q.
}
$$
