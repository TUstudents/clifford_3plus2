# Causal Edge-Clock Retarded Condition

The edge-clock scattering update proved that a local BB-amplitude-preserving
boundary completion exists. The minimality theorem proved that the hidden
clock-error output is forced by the visible norm deficit. The remaining question
is sharper:

$$
\boxed{
\text{why is the forced clock-error sector outgoing in the mass readout?}
}
$$

This note does not claim that the bare BB wedge itself is retarded. The wedge
audit already killed that. It states the exact causal boundary condition inside
the edge-clock scattering model and separates it from the still-open physical
derivation of that condition.

The point is simple. A clock-error event is not another visible edge state. It
is an asynchronous arrival relative to the single local edge clock. In a
retarded boundary response one prepares a visible excitation and the hidden
clock-error leads in their no-incoming state. The error can leave the visible
scar, but an incoming error is not supplied from the future boundary. That is
the same initial-value principle that selects a retarded Green function rather
than an advanced or recurrent pole.

## 1. Retarded Compression

Let $\mathcal V$ be the visible $q=0$ scar space and let $\mathcal L$ be the
outgoing clock-error lead space. The edge-clock update has visible branch

$$
A=A_{\mathbb N}=S_rB_+ + S_r^\dagger B_-,
$$

and mixed-normal clock-error emission

$$
E=L_+M_{+2}+L_-M_{-2}.
$$

The retarded compression is the block update

$$
\boxed{
T_R=
\begin{pmatrix}
A&0\\
E&S
\end{pmatrix},
}
$$

where $S$ is the outgoing shift on the clock-error leads. The zero block in the
upper right is not a deletion of probability. It is the no-incoming boundary
condition for the retarded observable:

$$
\boxed{
\text{clock-error output is allowed; clock-error input is not prepared.}
}
$$

The full scattering dilation can still be unitary after adding the missing
incoming asymptotic channels. The mass readout is the retarded semigroup
compression of that scattering problem. `C:9` inside the scattering model.

## 2. No Hidden Return Theorem

Because $T_R$ is block lower triangular,

$$
T_R^t=
\begin{pmatrix}
A^t&0\\
\sum_{j=0}^{t-1}S^{t-1-j}EA^j&S^t
\end{pmatrix}.
$$

Therefore

$$
\boxed{
P_{\mathcal V}T_R^t\iota_{\mathcal V}=A^t
\qquad(t\ge0).
}
$$

No path that leaves through $E$ can return to the visible scar in the retarded
compression. The hidden lead stores the emitted clock error, shifts it outward,
and keeps its causal timestamp. The visible survival readout is therefore the
$q=0$ scar branch $A_{\mathbb N}$ itself, not the pole spectrum of a closed
relative-depth bath. `C:9`.

This is the microscopic form of the phrase "outgoing leakage." It is not a
small correction and not a fit. It is a choice of Green-function problem:
initial-value retarded response rather than closed-box recurrence.

## 3. The Control: Allow One Incoming Block

If one allows an incoming hidden-to-visible block $G$, the update becomes

$$
T_G=
\begin{pmatrix}
A&G\\
E&S
\end{pmatrix}.
$$

Already at two steps,

$$
P_{\mathcal V}T_G^2\iota_{\mathcal V}
=A^2+GE.
$$

For the bare wedge identification, the outgoing clock ports are read as
physical $q=\pm2$ states and the return block is

$$
G=\begin{pmatrix}M_{-2}&M_{+2}\end{pmatrix},
\qquad
E=\begin{pmatrix}M_{+2}\\M_{-2}\end{pmatrix}.
$$

Thus

$$
GE=M_{-2}M_{+2}+M_{+2}M_{-2}
=
\begin{pmatrix}
-{1+i\over4}&0\\
0&-{1-i\over4}
\end{pmatrix}
\ne0.
$$

This is exactly the return obstruction found in
[BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md](BARE_BB_RELATIVE_RETURN_OBSTRUCTION.md).
The retarded edge-clock condition kills precisely the term that the bare wedge
keeps. `C:9`.

## 4. Physical Reading

The BCC edge has two normal faces but one local edge clock. Same-normal hops
arrive synchronously and remain in $q=0$. Mixed-normal hops arrive with

$$
\Delta q=\pm2,
$$

so they are clock-error emissions. The causal retarded readout says:

$$
\boxed{
\text{mass is measured by the response of the synchronous scar to a prepared
visible source, with no incoming asynchronous clock-error source.}
}
$$

This is the correct sector for the silver transfer. The silver root is not an
eigenvalue of the closed relative-depth system. It is the decaying transfer root
of the visible survival problem after clock-error emission is made outgoing.

The bare BB wedge is the wrong readout for this purpose because it treats
$q=\pm2$ as recurrent visible-adjacent states. The edge-clock model is the
minimal local repair: it keeps the exact BB mixed amplitudes, keeps local
unitarity in the scattering dilation, but changes the asymptotic condition of
the hidden sector.

## 5. What This Closes

The local boundary chain is now:

$$
\text{exact BB amplitudes}
\Rightarrow
V^\dagger V={1\over2}I
\Rightarrow
\text{hidden clock-error output forced}
\Rightarrow
T_R=
\begin{pmatrix}A&0\\E&S\end{pmatrix}
\Rightarrow
P_{\mathcal V}T_R^t\iota_{\mathcal V}=A^t .
$$

So, inside the edge-clock scattering model, the retarded/no-incoming condition
is no longer a handwave. It is the causal initial-value condition on the forced
clock-error leads. The algebraic consequence is exact: no hidden return enters
the visible survival branch.

What remains open is one layer deeper:

$$
\boxed{
\text{derive from deeper boundary-material dynamics why clock-error channels
are asymptotic outgoing leads rather than recurrent wedge states.}
}
$$

That is not a flavor fit. It is a microscopic boundary-dynamics problem.

## 6. Certainty Ledger

| Claim | Status |
|---|---|
| block retarded update $T_R=\begin{pmatrix}A&0\\E&S\end{pmatrix}$ gives visible powers $A^t$ | `C:9` |
| allowing hidden return $G$ adds the two-step term $GE$ | `C:9` |
| bare wedge return is recovered by $G=(M_{-2},M_{+2})$ and is nonzero | `C:9` |
| retarded/no-incoming is the causal initial-value condition for the edge-clock scattering model | `C:8` |
| physical BCC defect realizes outgoing clock-error leads | `C:6` model premise |
| deeper derivation of outgoing clock-error asymptotics from boundary-material dynamics | `C:3` |

The concise result is:

$$
\boxed{
\text{Retardedness is not an algebraic property of the bare wedge. It is the
causal scattering boundary condition of the forced clock-error sector, and it
exactly removes the BB two-step return.}
}
$$
