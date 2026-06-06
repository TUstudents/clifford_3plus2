# Microscopic BB-QCA Edge Update

This note is the current complete specification of the bare BB-QCA
codimension-two edge update used by the synthesis. "Bare" means that the local
amplitudes are the pinned Bialynicki-Birula BCC Weyl amplitudes. No flavor mass
number, silver number, or fitted coefficient is inserted into the update.

The conclusion is:

$$
\boxed{
\text{the local microscopic update is exact BB scattering into visible
same-clock ports plus forced clock-error ports.}
}
$$

The update is complete as a local unitary scattering rule. Its retarded mass
readout is a causal source condition on the clock-error leads. What is still
not derived from a deeper boundary material model is why the physical BCC
defect realizes those clock-error ports as outgoing asymptotic leads rather
than closing them into recurrent wedge states.

## 1. Boundary Coordinates

At a codimension-two BCC edge let $(r_1,r_2)$ be the two inward normal depths
and $s$ the tangential coordinate. Define

$$
r={r_1+r_2\over2},\qquad q=r_1-r_2.
$$

A body-diagonal BB hop has signs

$$
\sigma=(\sigma_1,\sigma_2,\sigma_3),\qquad \sigma_i=\pm1,
$$

and changes the relative depth by

$$
\Delta q=\sigma_1-\sigma_2.
$$

Therefore

$$
\sigma_1=\sigma_2 \Rightarrow \Delta q=0,
\qquad
\sigma_1=-\sigma_2 \Rightarrow \Delta q=\pm2.
$$

This is the microscopic origin of the split. Same-normal hops are synchronous
edge-clock events. Mixed-normal hops are asynchronous clock-error events.
`C:9`.

## 2. Exact BB Blocks

At tangential momentum $k_s=0$, sum over $\sigma_3=\pm1$. The visible
same-normal blocks are

$$
B_+=W_{+++}+W_{++-}
=
{1\over4}
\begin{pmatrix}
1+i&1-i\\
1+i&1-i
\end{pmatrix},
$$

$$
B_-=W_{--+}+W_{---}
=
{1\over4}
\begin{pmatrix}
1+i&-1+i\\
-1-i&1-i
\end{pmatrix}.
$$

The mixed-normal blocks are

$$
M_{+2}=W_{+-+}+W_{+--}
=
{1\over4}
\begin{pmatrix}
1-i&1+i\\
1-i&1+i
\end{pmatrix},
$$

$$
M_{-2}=W_{-++}+W_{-+-}
=
{1\over4}
\begin{pmatrix}
1-i&-1-i\\
-1+i&1+i
\end{pmatrix}.
$$

They obey the exact norm split

$$
B_+^\dagger B_+ + B_-^\dagger B_-={1\over2}I,
\qquad
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}={1\over2}I.
$$

Thus the visible same-clock branch is not unitary by itself. The missing norm is
not optional and not adjustable: it is exactly the mixed-normal BB branch.
`C:9`.

## 3. Local Scattering Rule

Introduce four outgoing local ports

$$
r_+,\quad r_-,\quad \chi_+,\quad \chi_-.
$$

The first two are visible synchronous radial ports. The last two are
orientation-resolved clock-error ports. The local edge scattering isometry is

$$
\boxed{
C_{\rm edge}\psi
=|r_+\rangle B_+\psi
+|r_-\rangle B_-\psi
+|\chi_+\rangle M_{+2}\psi
+|\chi_-\rangle M_{-2}\psi .
}
$$

The BB identities give

$$
C_{\rm edge}^\dagger C_{\rm edge}=I.
$$

Therefore this is a locally probability-preserving BB edge event. Since
$C_{\rm edge}$ is an isometry from a two-dimensional Weyl spinor into an
eight-dimensional port space, it can be completed to a finite local unitary

$$
U_{\rm loc}=
\begin{pmatrix}
C_{\rm edge}&C_\perp
\end{pmatrix},
\qquad
U_{\rm loc}^\dagger U_{\rm loc}=I_8.
$$

The complement $C_\perp$ is the incoming auxiliary boundary basis required by
unitarity. It is not a flavor datum. The exact visible and clock-error
amplitudes are the first two columns and are fixed by the BB coin. `C:9`.

## 4. Propagation

After the local collision, the ports propagate:

$$
r_+:\ r\mapsto r+1,\qquad
r_-:\ r\mapsto r-1,
$$

with the usual half-line head correction at $r=0$, and

$$
\chi_\pm(a)\mapsto\chi_\pm(a+1)
$$

on clock-error lead age $a\in\mathbb N$. Equivalently, the global microscopic
update is

$$
\mathcal U = \mathcal S_{\rm ports}\,U_{\rm loc},
$$

where $\mathcal S_{\rm ports}$ is the port shift/permutation. With the full
incoming and outgoing lead spaces included, $\mathcal U$ is unitary because it
is a product of local unitaries and shifts. `C:9` for the finite local rule and
unitary scattering construction.

The half-line visible compression is

$$
A_{\mathbb N}=S_rB_+ + S_r^\dagger B_-.
$$

The clock-error emission is

$$
E=L_+M_{+2}+L_-M_{-2}.
$$

For retarded mass response, no incoming clock-error source is prepared. The
effective causal update is therefore

$$
\boxed{
T_R=
\begin{pmatrix}
A_{\mathbb N}&0\\
E&S_\chi
\end{pmatrix}.
}
$$

By the block-triangular theorem,

$$
P_{\rm vis}T_R^t\iota_{\rm vis}=A_{\mathbb N}^t.
$$

So the visible mass readout is exactly the $q=0$ survival branch. `C:9` inside
the scattering update.

## 5. Control: The Recurrent Wedge Is Not This Update

If the clock-error ports are instead identified with recurrent wedge states,
there is a hidden-to-visible return block

$$
G=\begin{pmatrix}M_{-2}&M_{+2}\end{pmatrix}.
$$

Then the two-step visible correction is

$$
GE=M_{-2}M_{+2}+M_{+2}M_{-2}
=
\begin{pmatrix}
-{1+i\over4}&0\\
0&-{1-i\over4}
\end{pmatrix}
\ne0.
$$

That is the exact bare-wedge return obstruction. Therefore the complete
edge-clock update is not the recurrent wedge update in disguise. It is a
different boundary asymptotic condition for the forced mixed-normal output.
`C:9`.

## 6. Silver Transfer

In the symmetric/antisymmetric spinor basis, the visible blocks become

$$
B_+=
\begin{pmatrix}
{1\over2}&{i\over2}\\
0&0
\end{pmatrix},
\qquad
B_-=
\begin{pmatrix}
0&0\\
{i\over2}&{1\over2}
\end{pmatrix}.
$$

The stationary visible scar equation has transfer matrix

$$
T(\zeta)=
\begin{pmatrix}
{1\over2\zeta}&{i\over2\zeta}\\
-{i\over2\zeta}&2\zeta+{1\over2\zeta}
\end{pmatrix},
$$

with

$$
\det T=1,\qquad {\rm tr}\,T=2\zeta+{1\over\zeta}.
$$

The branch survival norm is

$$
c^2={1\over2},\qquad c={1\over\sqrt2}.
$$

At $\zeta=c$,

$$
{\rm tr}\,T=2\sqrt2,\qquad
\lambda_\pm=\sqrt2\pm1,\qquad
\epsilon=\sqrt2-1.
$$

So the silver root is not added to the boundary update. It is the decaying
visible transfer root of the exact BB survival branch selected by the retarded
edge-clock response. The theorem program that tries to promote this
survival-norm / trace-minimum coincidence into a general spectral-selection
principle is
[BAND_EDGE_SELECTION_THEOREM.md](BAND_EDGE_SELECTION_THEOREM.md). `C:9`
algebraically for the BB scar; `C:3` for the general selection theorem.

## 7. What Is Complete, And What Is Not

Complete in this note:

$$
\boxed{
\text{a local unitary BB edge scattering update with exact BB amplitudes and
exact retarded }q=0\text{ compression.}
}
$$

The completed microscopic rule is:

$$
\boxed{
\begin{array}{c}
\sigma_1=\sigma_2:\ \text{visible radial ports }r_\pm \text{ with }B_\pm,\\
\sigma_1=-\sigma_2:\ \text{clock-error ports }\chi_\pm\text{ with }M_{\pm2},\\
\text{full local unitary completion on incoming auxiliary ports},\\
\text{retarded readout with no incoming clock-error source.}
\end{array}
}
$$

Not complete at the deeper physical level:

$$
\boxed{
\text{derive from a more primitive BCC boundary material why clock-error
channels are outgoing asymptotic leads rather than recurrent wedge states.}
}
$$

This is now the only remaining boundary-dynamics premise. The bare BB amplitudes
and the local unitary update are no longer the problem; the problem is the
asymptotic interpretation of the forced mixed-normal sector.

## 8. Certainty Ledger

| Claim | Status |
|---|---|
| same-normal/mixed-normal split from $\Delta q=\sigma_1-\sigma_2$ | `C:9` |
| exact $B_\pm,M_{\pm2}$ BB blocks | `C:9` |
| visible and mixed branches split norm as ${1\over2}I+{1\over2}I$ | `C:9` |
| $C_{\rm edge}$ is a local isometry | `C:9` |
| finite local unitary completion exists | `C:9` |
| retarded compression gives visible powers $A_{\mathbb N}^t$ | `C:9` |
| recurrent wedge return is nonzero and changes the visible branch | `C:9` |
| silver transfer follows from the visible survival branch | `C:9` |
| physical BCC defect realizes outgoing clock-error asymptotics | `C:6` model premise |
| deeper derivation of the asymptotic condition from boundary material dynamics | `C:3` |

The one-line paper statement is:

$$
\boxed{
\text{The BB edge update is a unitary local scattering colligation; flavor sees
the retarded survival compression, not the closed recurrent wedge.}
}
$$
