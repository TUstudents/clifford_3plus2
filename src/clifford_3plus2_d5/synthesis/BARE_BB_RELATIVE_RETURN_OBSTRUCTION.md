# Bare BB Relative-Return Obstruction

The wedge-domain audit showed that $q=0$ can leak to $q=\pm2$ and return. This
note computes the actual two-tick BB return amplitude.

The result is exact:

$$
\boxed{
\text{the bare mixed-normal channel feeds back into }q=0
\text{ at second order.}
}
$$

Therefore the retarded/open boundary condition cannot be derived by saying that
mixed-normal modes simply go away in the bare wedge. They do not. They return
with a nonzero BB matrix.

## 1. Mixed-Normal Blocks

At zero tangential momentum, the mixed-normal blocks are

$$
M_{+2}=W_{+-+}+W_{+--}
=
{1\over4}
\begin{pmatrix}
1-i&1+i\\
1-i&1+i
\end{pmatrix},
$$

and

$$
M_{-2}=W_{-++}+W_{-+-}
=
{1\over4}
\begin{pmatrix}
1-i&-1-i\\
-1+i&1+i
\end{pmatrix}.
$$

They implement

$$
q=0\to q=+2,
\qquad
q=0\to q=-2.
$$

Inside the physical wedge, the inverse mixed hops are allowed for every
nonzero diagonal depth. Thus the two shortest relative-return paths are

$$
q=0\xrightarrow{M_{+2}}q=+2\xrightarrow{M_{-2}}q=0,
$$

and

$$
q=0\xrightarrow{M_{-2}}q=-2\xrightarrow{M_{+2}}q=0.
$$

## 2. The Two-Step Return Operator

The corresponding second-order return operator is

$$
R_{\rm rel}^{(2)}
=
M_{-2}M_{+2}+M_{+2}M_{-2}.
$$

Direct multiplication gives

$$
\boxed{
R_{\rm rel}^{(2)}
=
\begin{pmatrix}
-{1+i\over4}&0\\
0&-{1-i\over4}
\end{pmatrix}.
}
$$

This is nonzero. Its determinant is

$$
\det R_{\rm rel}^{(2)}={1\over8},
$$

so the relative return is not a rank-deficient accident. It acts on both Weyl
spinor components. `C:9`.

The individual path matrices are rank one, but their sum is full rank:

$$
\operatorname{rank}(M_{-2}M_{+2})=1,\qquad
\operatorname{rank}(M_{+2}M_{-2})=1,
$$

while

$$
\operatorname{rank}R_{\rm rel}^{(2)}=2.
$$

So the two leakage sides combine into a complete recurrent feedback on the
diagonal scar.

## 3. Orthogonality Is Not Protection

The one-step leakage norms satisfy

$$
M_{+2}^\dagger M_{+2}
+M_{-2}^\dagger M_{-2}
={1\over2}I.
$$

Also,

$$
M_{+2}^\dagger M_{-2}=0,
\qquad
M_{-2}^\dagger M_{+2}=0.
$$

This orthogonality explains the clean $1/2+1/2$ norm split. But it does not
mean the leakage is gone. The sequential products $M_{-2}M_{+2}$ and
$M_{+2}M_{-2}$ are nonzero, because return is a two-tick process, not an
inner product of simultaneous one-tick channels.

This is the important physical distinction:

$$
\boxed{
\text{one-tick norm orthogonality does not imply no recurrent feedback.}
}
$$

## 4. Consequence For Retarded Readout

A retarded boundary condition would replace recurrent return by an outgoing
self-energy. The bare wedge does the opposite: it supplies a nonzero local
return operator at the first possible order.

Thus:

$$
\boxed{
\text{bare BB wedge dynamics}
\;\Rightarrow\;
R_{\rm rel}^{(2)}\ne0,
}
$$

whereas the flavor scar requires, in the hard boundary limit,

$$
\boxed{
\Sigma_{\rm mix}^{R}\to0.
}
$$

The missing microscopic boundary update must therefore modify the relative
channel before or at this two-step return. It may do so by a large positive
$gq^2$ penalty, by an outgoing face/vacuum channel, or by an equivalent local
unitary boundary degree. But it cannot be the unmodified bare wedge update.
[BARE_BB_FACE_EXTERIOR_INSUFFICIENCY.md](BARE_BB_FACE_EXTERIOR_INSUFFICIENCY.md)
checks the face/vacuum option more carefully and shows that a face-only
outgoing condition is insufficient: the first return is inward and occurs
before exterior absorption can act.
[EDGE_CLOCK_SCATTERING_BOUNDARY_UPDATE.md](EDGE_CLOCK_SCATTERING_BOUNDARY_UPDATE.md)
then gives the constructive replacement: route $M_{\pm2}$ into edge-clock error
ports so the retarded visible return is zero.

## 5. Certainty Ledger

| Claim | Status |
|---|---|
| two-step mixed return paths exist in the wedge for nonzero depth | `C:9` |
| $R_{\rm rel}^{(2)}=M_{-2}M_{+2}+M_{+2}M_{-2}$ | `C:9` |
| $R_{\rm rel}^{(2)}=\operatorname{diag}(-(1+i)/4,-(1-i)/4)$ | `C:9` |
| relative return is full rank and nonzero | `C:9` |
| one-tick leakage orthogonality removes two-tick feedback | `C:1` |
| bare wedge alone gives retarded/open readout | `C:1` |
| boundary law suppresses/replaces $R_{\rm rel}^{(2)}$ | `C:6` target |
| full microscopic derivation of that law | `C:3` |

The concise result is:

$$
\boxed{
\text{The bare BB relative channel returns immediately. The retarded boundary
condition is a real dynamical input, not a consequence of one-tick leakage.}
}
$$
