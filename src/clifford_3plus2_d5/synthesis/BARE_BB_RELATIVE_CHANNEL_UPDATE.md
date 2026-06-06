# Bare BB Relative-Channel Update

The previous audit proved that the bare BB Weyl spinor cannot close the
$q=0$ scar by killing leakage. This note writes the next microscopic object:
the unprojected edge update before any relative-depth penalty or outgoing
condition is imposed.

The result is exact and diagnostic:

$$
\boxed{
\text{bare BB on the edge is a closed unitary walk on }(r,q),
\text{ not an open }q=0\text{ scar.}
}
$$

Thus the missing boundary principle is not probability conservation. The bare
walk already conserves probability. The missing principle is a physical rule
that turns the relative-depth direction into a locked or outgoing channel for
the mass readout.

## 1. Trace And Relative Coordinates

At the codimension-two edge use

$$
r={r_1+r_2\over2},\qquad q=r_1-r_2.
$$

A body-diagonal hop $(\sigma_1,\sigma_2,\sigma_3)$ changes these by

$$
\Delta r={\sigma_1+\sigma_2\over2},
\qquad
\Delta q=\sigma_1-\sigma_2.
$$

Therefore the four edge classes are:

| normal signs | $\Delta r$ | $\Delta q$ | block |
|---|---:|---:|---|
| $(+,+)$ | $+1$ | $0$ | $B_+$ |
| $(-,-)$ | $-1$ | $0$ | $B_-$ |
| $(+,-)$ | $0$ | $+2$ | $M_{+2}$ |
| $(-,+)$ | $0$ | $-2$ | $M_{-2}$ |

At zero tangential momentum, the BB edge update on the full trace/relative
cover is

$$
\boxed{
U_{\rm rel}
=
S_r B_+ + S_r^\dagger B_-
+ S_q M_{+2} + S_q^\dagger M_{-2}.
}
$$

Here $S_r$ shifts common depth and $S_q$ shifts relative depth by two units.
This is the bare object before the $q=0$ compression.
The physical half-space version of this relative channel is checked in
[BARE_BB_WEDGE_DOMAIN_AUDIT.md](BARE_BB_WEDGE_DOMAIN_AUDIT.md).

## 2. Exact Unitarity On The Relative Cover

Let $x$ be the Bloch phase for the trace shift and $y$ the Bloch phase for the
relative shift. The symbol is

$$
U_{\rm rel}(x,y)
=
xB_+ + x^{-1}B_- + yM_{+2}+y^{-1}M_{-2},
\qquad |x|=|y|=1.
$$

Using the pinned BB matrices,

$$
U_{\rm rel}(x,y)^\dagger U_{\rm rel}(x,y)=I,
\qquad
U_{\rm rel}(x,y)U_{\rm rel}(x,y)^\dagger=I.
$$

So the unprojected edge update is not subunitary. It is the original BB
unitarity expressed in the coordinates natural to the edge. `C:9`.

The visible branch alone is

$$
A_r=S_r B_+ + S_r^\dagger B_-,
$$

and satisfies

$$
A_r^\dagger A_r={1\over2}I.
$$

The relative branch alone is

$$
L_q=S_q M_{+2}+S_q^\dagger M_{-2},
$$

and also satisfies

$$
L_q^\dagger L_q={1\over2}I.
$$

The silver survival factor comes from reading only $A_r$. The full bare update
keeps both $A_r$ and $L_q$.

## 3. Why This Does Not Give Retarded Readout

The mixed-normal terms do not propagate away in the common radial coordinate.
They propagate sideways in relative depth:

$$
q\mapsto q\pm2,\qquad r\mapsto r.
$$

On the full relative cover this is a two-way unitary channel. There is no
retarded branch choice in the closed symbol $U_{\rm rel}(x,y)$; the $y$ phase is
a normal Bloch quantum number. Therefore:

$$
\boxed{
\text{bare relative-channel unitarity}
\;\not\Rightarrow\;
\text{outgoing retarded boundary condition}.
}
$$

To obtain the flavor readout, one must do additional physics to the relative
coordinate:

1. lock it by a positive mismatch constraint $gq^2$;
2. attach it to an outgoing half-line bath and choose the retarded Weyl branch;
3. or derive an equivalent local boundary degree whose hard limit produces the
   same $q=0$ compression.

This is the precise microscopic burden left by the BB update.

## 4. The Completed Boundary Update Target

The completed boundary update cannot be just $U_{\rm rel}$. It must be a local
edge update of the form

$$
\boxed{
U_{\rm boundary}
=
U_{\rm rel}
+
\text{local dynamics for the relative channel}
}
$$

such that:

1. it is unitary before projection;
2. the $q=0$ visible compression is $A_{\mathbb N}$;
3. mixed-normal channels are either gapped by $K_{\rm rel}^\dagger K_{\rm rel}$
   or retarded/outgoing at the mass readout;
4. the hard boundary limit gives

$$
\Sigma_{\rm mix}^R\to0.
$$

The bare calculation has therefore advanced the problem from "maybe the BB coin
already does it" to the exact target:

$$
\boxed{
\text{derive the local relative-channel dynamics that converts the closed
unitary }q\text{ walk into the retarded/locked boundary response.}
}
$$

## 5. Certainty Ledger

| Claim | Status |
|---|---|
| $(\Delta r,\Delta q)$ classification of BB edge hops | `C:9` |
| unprojected relative-cover symbol $U_{\rm rel}(x,y)$ | `C:9` |
| $U_{\rm rel}(x,y)$ is unitary for $|x|=|y|=1$ | `C:9` |
| visible branch norm is $1/2$ | `C:9` |
| mixed relative branch norm is $1/2$ | `C:9` |
| bare relative channel is closed/two-way, not retarded | `C:9` |
| retarded readout from bare relative cover alone | `C:1` |
| BB plus derived relative-channel boundary dynamics | `C:6` effective target |
| full microscopic derivation of that dynamics | `C:3` |

The concise statement is:

$$
\boxed{
\text{The bare BB edge update is a unitary two-coordinate walk. Silver appears
only after the relative coordinate is physically locked or opened.}
}
$$
