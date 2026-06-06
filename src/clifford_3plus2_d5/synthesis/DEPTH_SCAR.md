# Depth Scar Sidecar

The `depth_scar` sidecar attacks the free input exposed by the A-track:

$$
d=(0,2,6).
$$

Its main achievement is to replace a diagonal hierarchy spurion by a positive
graph-native object. The family labels are not primitive labels $1,2,3$; they
are normal modes of a residual boundary repair graph.

The basic mechanism is

$$
S_3\ \hbox{residual ports}
\quad\longrightarrow\quad
P_3=(u-a-b)
\quad\longrightarrow\quad
D_{\rm scar}=2\Delta(P_3).
$$

Thus the generation-depth problem becomes the scar-selection problem: why does
the microscopic BCC/QCA boundary leave a path, rather than the unbroken
triangle or a shortcut graph?

Certainty: `C:9` for the exact path-Laplacian depth spectrum. `C:6` for the
path scar as a physical origin of the hierarchy, because the microscopic
selection of the path is still conditional.

## 1. Where This Belongs In The Theory Tree

This sidecar belongs to the family-depth layer, not to color, hypercharge, or
the carrier construction.

The relevant group action is the residual permutation symmetry of the three
boundary ports:

$$
S_3 \curvearrowright \{u,a,b\}.
$$

The path $u-a-b$ preserves only the endpoint reflection

$$
Z_2=\{1,(u\,b)\}.
$$

Color remains $SU(3)_c$ and lives in the sector projection $V_q$; hypercharge
remains $U(1)_Y$ and lives in the Standard Model carrier. Neither is a
generation-depth generator. The BCC/QCA structure enters through the factor of
two in the depth unit and through the one-tick locality/parity rules used later
to forbid the shortcut $b\to u$.

Certainty: `C:9` for the group assignment. `C:7` for the BCC depth-unit
placement as inherited from the boundary-response sidecar.

## 2. The Exact Operator

With port order $(u,a,b)$, take the oriented path incidence matrix

$$
\partial=
\begin{pmatrix}
1&-1&0\\
0&1&-1
\end{pmatrix}.
$$

Then

$$
\Delta(P_3)=\partial^T\partial
=
\begin{pmatrix}
1&-1&0\\
-1&2&-1\\
0&-1&1
\end{pmatrix},
$$

and the BCC-doubled depth operator is

$$
D_{\rm scar}=2\Delta(P_3).
$$

The spectrum is exactly

$$
\operatorname{Spec}\Delta(P_3)=\{0,1,3\},
\qquad
\operatorname{Spec}D_{\rm scar}=\{0,2,6\}.
$$

The normalized modes are

$$
\psi_0=\frac{1}{\sqrt3}(1,1,1),
$$

$$
\psi_2=\frac{1}{\sqrt2}(1,0,-1),
$$

$$
\psi_6=\frac{1}{\sqrt6}(1,-2,1).
$$

So the depth hierarchy is not an arbitrary diagonal assignment. It is the
spectrum of the path graph:

$$
d(\psi_0)=0,\qquad d(\psi_2)=2,\qquad d(\psi_6)=6.
$$

The unbroken triangle control gives the doubled spectrum

$$
\operatorname{Spec}(2\Delta(K_3))=\{0,6,6\},
$$

so full $S_3$ symmetry cannot produce the observed hierarchy. A hand-written
diagonal matrix with spectrum $\{0,2,6\}$ also fails as an origin because it is
not graph-native.

Certainty: `C:9`.

## 3. Transfer Consequence

The same silver root from boundary response appears here:

$$
\epsilon=\sqrt2-1.
$$

Once the path scar is accepted, the transfer operator is

$$
T=\epsilon^{D_{\rm scar}}
 =P_0+\epsilon^2P_2+\epsilon^6P_6,
$$

where $P_i=|\psi_i\rangle\langle\psi_i|$. The pairwise depth gaps are

$$
\Delta d_{12}=2,\qquad
\Delta d_{23}=4,\qquad
\Delta d_{13}=6.
$$

Equivalently, with $\lambda=\epsilon^2$,

$$
|V_{12}|:|V_{23}|:|V_{13}|
\sim
\lambda:\lambda^2:\lambda^3,
$$

before the sector Clebsches and phases are attached. This is a skeleton result:
it explains the exponent pattern, not the full CKM matrix by itself.

Certainty: `C:8` for the transfer powers given the path scar. `C:6` for
identifying this with the physical quark hierarchy before the left/right mass
model is fully integrated.

## 4. The Nilpotent Dual

The path graph has an algebraic dual as a length-3 nilpotent repair flag:

$$
N=|u\rangle\langle a|+|a\rangle\langle b|
=
\begin{pmatrix}
0&1&0\\
0&0&1\\
0&0&0
\end{pmatrix}.
$$

It obeys

$$
N^3=0,\qquad N^2\ne0.
$$

The undirected graph data are recovered as

$$
A=N+N^\dagger,
$$

$$
D=NN^\dagger+N^\dagger N,
$$

$$
\Delta=D-A.
$$

For the canonical real flag, this is exactly $\Delta(P_3)$. Therefore the same
object has three equivalent faces:

$$
\hbox{path Laplacian}
\quad\Longleftrightarrow\quad
\hbox{length-3 nilpotent flag}
\quad\Longleftrightarrow\quad
S_3\to Z_2\ \hbox{repair scar}.
$$

This is the key anti-rediscovery lesson. A later theorem that merely rebuilds
the nilpotent flag is not a new depth mechanism. A genuinely new mechanism must
derive why this flag is selected microscopically.

Certainty: `C:9`.

## 5. Unit Weights And Tree Phases

On the same support, the most general local flag is

$$
N=r e^{i\alpha}|u\rangle\langle a|
 +s e^{i\beta}|a\rangle\langle b|.
$$

The active projections are

$$
N^\dagger N=\operatorname{diag}(0,r^2,s^2),
\qquad
NN^\dagger=\operatorname{diag}(r^2,s^2,0).
$$

If the active block is a partial isometry, then the nonzero singular values are
unit:

$$
r=s=1.
$$

The phases $\alpha,\beta$ live on a tree, so they are removable by port
rephasings. Thus, after assuming the length-3 support and local
partial-isometry saturation, the canonical real unit flag is forced.

Certainty: `C:9` as a conditional algebra theorem. `C:6` as a physical
normalization mechanism, because saturation/no-leakage still needs microscopic
derivation.

## 6. Support Selection

The finite support census is sharp. Among all $2^6=64$ directed no-self-loop
binary supports on three ports, imposing

$$
N^3=0,\qquad N^2\ne0,\qquad \operatorname{rank}N=2,
$$

and exactly two directed edges gives six supports, all in one $S_3$ orbit. That
orbit is represented by

$$
b\to a\to u.
$$

If the two-edge minimality condition is dropped, six shortcut supports survive.
Equivalently, the variational version has twelve feasible supports; minimizing
directed edge count selects six minimizers, again one $S_3$ orbit, with cost
$2$. The shortcut supports have cost $3$.

This is a clean result but also a warning: minimality is load-bearing. The
sidecar proves the finite theorem; it does not yet prove that microscopic QCA
repair minimizes directed edge count.

Certainty: `C:8` for the finite census and variational theorem. `C:4` for
elevating edge count to a microscopic action principle.

## 7. Effective Edge Potential

There is also an effective symmetric potential over nonnegative triangle edge
weights $(w_{ua},w_{ab},w_{ub})$. Let

$$
S_1=w_{ua}+w_{ab}+w_{ub},
$$

$$
S_2=w_{ua}w_{ab}+w_{ab}w_{ub}+w_{ub}w_{ua},
$$

$$
S_3=w_{ua}w_{ab}w_{ub}.
$$

The proposed potential is

$$
V=(S_1-2)^2+(S_2-1)^2+S_3.
$$

On the nonnegative domain, $V=0$ iff

$$
(w_{ua},w_{ab},w_{ub})
\in
\{(1,1,0),(1,0,1),(0,1,1)\}.
$$

Thus the potential selects exactly the three missing-edge path scars. The
$S_3$ term is essential; without it, non-scar zeros appear.

This is useful as an effective selector, but it is not yet a microscopic
selector. The potential has been written down, not derived from the QCA update.

Certainty: `C:9` for the zero-set theorem. `C:4` for interpreting this
potential as physical dynamics.

## 8. Locality And No Leakage

The sidecar then tries to lower the support assumption into microscopic
selection rules. The modeled residual data are

$$
h(u)=0,\qquad h(a)=1,\qquad h(b)=2,
$$

with BCC parities

$$
p(u)=p(b),\qquad p(a)\ne p(u).
$$

Strictly height-lowering repair allows

$$
a\to u,\qquad b\to a,\qquad b\to u.
$$

One-tick locality allows only the adjacent moves

$$
a\to u,\qquad b\to a,
$$

and forbids $b\to u$ because it has residual distance $2$ and does not flip BCC
parity. Under these assumptions, monotone rank-complete repair has exactly the
path support.

For normalization, let $A=\operatorname{span}\{|a\rangle,|b\rangle\}$ and
$R=\operatorname{span}\{|u\rangle,|a\rangle\}$. With a microscopic unitary $U$,
define

$$
N=P_RUP_A,
\qquad
L=(I-P_R)UP_A.
$$

Unitarity gives

$$
N^\dagger N+L^\dagger L=I_A.
$$

Therefore $N$ is an isometry on the active domain iff $L=0$. Unique active
successors,

$$
\Omega(a)=\{u\},\qquad \Omega(b)=\{a\},
$$

would force no leakage and hence unit weights.

The V12 certificate verifies this unique-successor condition on the currently
modeled finite boundary basis: $12$ candidate states, $24$ active transition
rows, one allowed successor per active source, and explicit vetoes for all
forbidden rows. The certificate is strong inside its model. Its physical burden
is basis completeness.

Certainty: `C:9` for the no-leakage/isometry equivalence. `C:8` for the finite
successor certificate inside the modeled basis. `C:6` for the full microscopic
claim, pending a proof that the candidate basis is complete.

## 9. CP Cannot Live On The Pure Scar

The pure path is a tree. Its cycle rank is zero, so all edge phases are gauge
removable. Therefore the depth scar alone cannot be the intrinsic source of CKM
CP violation.

If the missing edge is healed with real weight $\delta$, the doubled real
depth spectrum becomes

$$
\{0,\ 2+4\delta,\ 6\}.
$$

If the healed triangle carries a magnetic loop phase $\phi$, there is exactly
one gauge-invariant holonomy around

$$
u\to a\to b\to u.
$$

Complex conjugation flips this phase:

$$
\phi\mapsto-\phi.
$$

So CP and depth separate cleanly:

$$
\hbox{tree scar} \Rightarrow \hbox{hierarchy without intrinsic CP},
$$

$$
\hbox{healed loop} \Rightarrow \hbox{one possible CP holonomy}.
$$

This also protects the existing boundary-response CP story: the pure depth
scar should not be double-counted as a CP mechanism.

Certainty: `C:9` for the graph-holonomy statements. `C:4` for any physical
claim about the actual values of $\delta$ or $\phi$.

## 10. Synthesis Verdict

The sidecar upgrades the A-track free hierarchy input:

$$
d=(0,2,6)
\quad\hbox{becomes}\quad
d=\operatorname{Spec}(2\Delta(P_3)).
$$

This is a major consolidation. The generation problem is no longer "why these
three numbers?" but

$$
\hbox{why does the boundary repair sector select the unit path scar }P_3?
$$

The current proof stack is:

$$
\hbox{path scar}
\Rightarrow
\{0,2,6\}
\Rightarrow
\{\epsilon^0,\epsilon^2,\epsilon^6\}
\Rightarrow
\lambda:\lambda^2:\lambda^3.
$$

The conditional microscopic stack is:

$$
\hbox{height filtration}
\quad+\quad
\hbox{one-tick locality}
\Rightarrow
\hbox{path support},
$$

$$
\hbox{unique successors}
\Rightarrow
\hbox{no leakage}
\Rightarrow
\hbox{unit weights}.
$$

The open burdens are precise:

1. Derive the three-level defect filtration from the BCC/QCA boundary update.
2. Prove the residual one-tick boundary geometry, including the BCC parity veto.
3. Prove completeness of the V12 local boundary candidate basis.
4. Derive no-leakage or unique successors from the microscopic update, not just
   from the certificate model.
5. Attach the one-sided depth spectrum to the final left/right Yukawa or mass
   construction without mixing it with scalar Clebsches.
6. Keep CP separate unless a healed loop is physically derived.

My synthesis judgement: this sidecar is one of the strongest mathematical
pieces in the repo. It should be central in the paper, but with disciplined
language. The exact theorem is the path/nilpotent depth spectrum. The physical
conjecture is the microscopic selection of the path scar.
