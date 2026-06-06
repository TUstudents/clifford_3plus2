# Edge-Clock Scattering Boundary Update

The previous audits killed three bare routes:

$$
\text{BB spinor alone},\qquad
\text{wedge geometry alone},\qquad
\text{ordinary face exterior alone}.
$$

This note gives the first constructive microscopic boundary update that
survives those controls. It keeps the pinned BB amplitudes exactly, but changes
the target of mixed-normal hops. Instead of becoming recurrent wedge states
$q=\pm2$, they become outgoing **edge-clock error ports**. This is the local
scattering version of the single-clock constraint.

The update is:

$$
\boxed{
\text{same-normal BB amplitudes stay visible;}
\qquad
\text{mixed-normal BB amplitudes enter outgoing clock-error channels.}
}
$$

The follow-up minimality theorem
[EDGE_CLOCK_SCATTERING_MINIMALITY.md](EDGE_CLOCK_SCATTERING_MINIMALITY.md)
shows that the hidden clock-error sector is forced by local unitarity once the
same-normal BB amplitudes are kept. The remaining physical choice is the
retarded/no-incoming condition on that forced hidden sector. The exact causal
compression of that condition is isolated in
[CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md](CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md).

## 1. Local Port Space

At a diagonal edge event $q=0$, split the four normal-sign classes into
orthogonal output ports:

$$
\mathcal P
=
\{r_+,r_-,\chi_+,\chi_-\}.
$$

The ports mean:

| port | normal signs | block | physical reading |
|---|---|---|---|
| $r_+$ | $(+,+)$ | $B_+$ | synchronous outward radial scar |
| $r_-$ | $(-,-)$ | $B_-$ | synchronous inward radial scar |
| $\chi_+$ | $(+,-)$ | $M_{+2}$ | positive clock-error channel |
| $\chi_-$ | $(-,+)$ | $M_{-2}$ | negative clock-error channel |

The local BB scattering isometry is

$$
\boxed{
C_{\rm edge}\psi
=
|r_+\rangle B_+\psi
+|r_-\rangle B_-\psi
+|\chi_+\rangle M_{+2}\psi
+|\chi_-\rangle M_{-2}\psi .
}
$$

The BB matrix identities give

$$
B_+^\dagger B_+ + B_-^\dagger B_-
+M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}=I,
$$

so

$$
\boxed{
C_{\rm edge}^\dagger C_{\rm edge}=I.
}
$$

Thus the edge-clock routing is a local isometry using the exact BB amplitudes.
It does not delete probability. It separates the synchronous edge event from
the asynchronous clock-error event. `C:9`.

## 2. Local Unitary Completion

The output space has dimension $4\times2=8$, while the incoming Weyl spinor has
dimension $2$. Since $C_{\rm edge}$ is an isometry, it can be completed to a
local $8\times8$ unitary scattering matrix

$$
U_{\rm edge}
=
\begin{pmatrix}
C_{\rm edge}&C_\perp
\end{pmatrix},
\qquad
U_{\rm edge}^\dagger U_{\rm edge}=I_8.
$$

The complement $C_\perp$ represents incoming auxiliary boundary modes. It is not
unique, and its non-uniqueness is harmless: the visible BB amplitudes are the
first two columns fixed by the bulk coin, while the remaining columns only
complete local unitarity.

The exact certificate constructs such a unitary completion by Gram-Schmidt over
the BB port isometry. `C:9` for existence and the constructed finite unitary.

## 3. Retarded Clock-Error Leads

Attach $\chi_+$ and $\chi_-$ to outgoing half-line leads:

$$
\chi_\pm(0)\to\chi_\pm(1)\to\chi_\pm(2)\to\cdots .
$$

The retarded boundary condition is the Lax-Phillips scattering condition: no
incoming clock-error wave is supplied from infinity for the mass readout. In
that compression, the clock-error ports carry probability away from the visible
scar but do not feed it back.

The visible compression is therefore

$$
A_{\mathbb N}=S_rB_+ + S_r^\dagger B_-,
$$

and the outgoing clock-error coupling is

$$
E_{\rm clk}
=
L_+M_{+2}+L_-M_{-2}.
$$

The full scattering update is unitary once the outgoing leads and incoming
lead sectors are included. The physical mass readout is the retarded
compression with incoming clock-error data set to zero. `C:6`.

## 4. The First Return Is Removed

In the bare wedge, the first relative return was

$$
R_{\rm rel}^{(2)}
=
M_{-2}M_{+2}+M_{+2}M_{-2}
=
\begin{pmatrix}
-{1+i\over4}&0\\
0&-{1-i\over4}
\end{pmatrix}.
$$

That return exists because $M_{+2}$ and $M_{-2}$ land in physical $q=\pm2$
states that can hop inward again.

In the edge-clock scattering update, the mixed blocks land instead in
$\chi_\pm$ outgoing ports. The retarded visible-to-visible two-step return is

$$
R_{\rm clk}^{(2)}
=
G_+M_{+2}+G_-M_{-2},
$$

where $G_\pm$ are the incoming clock-error-to-visible couplings in the retarded
compression. The retarded condition sets

$$
G_+=G_-=0,
$$

hence

$$
\boxed{
R_{\rm clk}^{(2)}=0.
}
$$

So the scattering update removes exactly the obstruction found in
[BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md](BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md).
It does so without changing any BB amplitude. It changes the boundary
identification of asynchronous hops.

## 5. Relation To The Mismatch Constraint

The clock-error ports are the microscopic version of

$$
K_{\rm rel}=q.
$$

Same-normal hops have $\Delta q=0$ and are synchronous edge events. Mixed-normal
hops have $\Delta q=\pm2$ and are clock-error events. A hard edge-clock
constraint is the limit in which clock-error channels are unresolved outgoing
states for the visible low-energy theory.

Equivalently,

$$
\chi_\pm\text{ outgoing}
\quad\leadsto\quad
H_{\rm lock}=gq^2,\qquad g\to\infty,
$$

after Schur/Feshbach elimination. The two languages are dual:

| scattering language | effective Hamiltonian language |
|---|---|
| $\chi_\pm$ outgoing leads | $q=\pm2$ unresolved sectors |
| no incoming clock-error wave | retarded boundary condition |
| no two-step return | $\Sigma_{\rm mix}^{R}\to0$ |
| clock-error port | mismatch constraint $K_{\rm rel}=q$ |

## 6. What This Closes And What Remains

Closed by this note:

$$
\boxed{
\text{there exists a local unitary BB edge-clock scattering update whose
retarded compression is the }q=0\text{ scar.}
}
$$

Still open:

$$
\boxed{
\text{derive why the physical BCC defect realizes outgoing clock-error leads
rather than recurrent wedge states.}
}
$$

This distinction matters. The present note constructs the microscopic boundary
update and proves its algebra. It does not prove that the bare BB bulk dynamics
forces this boundary condition without an edge-clock defect. The earlier no-gos
show that such a defect is necessary.

## 7. Certainty Ledger

| Claim | Status |
|---|---|
| $C_{\rm edge}^\dagger C_{\rm edge}=I$ using exact BB blocks | `C:9` |
| local $8\times8$ unitary completion exists and is constructible | `C:9` |
| visible compression is $A_{\mathbb N}$ | `C:9` under scattering update |
| clock-error retarded compression gives $R_{\rm clk}^{(2)}=0$ | `C:9` |
| scattering update removes the explicit bare return obstruction | `C:9` |
| retarded/no-incoming compression as causal block-triangular response | `C:9` in [CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md](CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md) |
| physical BCC defect realizes edge-clock scattering | `C:6` model |
| derivation of edge-clock scattering from deeper boundary physics | `C:3` |

The concise result is:

$$
\boxed{
\text{A complete local BB boundary update exists: route asynchronous normal
hops into outgoing edge-clock ports. The unresolved problem is why the physical
defect chooses this scattering boundary condition.}
}
$$
