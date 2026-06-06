# Edge-Clock Scattering Minimality

The edge-clock scattering update constructs a local retarded boundary
completion. This note proves the accompanying minimality statement:

$$
\boxed{
\text{given exact BB amplitudes and no visible return, the clock-error sector
is not optional.}
}
$$

The theorem is not that nature must choose the retarded condition. The theorem
is narrower and exact: once the visible $q=0$ scar is required to keep the BB
same-normal amplitudes while the mixed-normal first return is absent, a hidden
edge-clock output sector is forced by unitarity. Preserving the two
mixed-normal BB signs identifies that sector with $\chi_\pm$ up to unitary
relabeling.

## 1. Visible Output Is Not Unitary

Let

$$
V=
\begin{pmatrix}
B_+\\
B_-
\end{pmatrix}.
$$

The pinned BB identities give

$$
V^\dagger V
=
B_+^\dagger B_+ + B_-^\dagger B_-
={1\over2}I.
$$

Therefore the visible same-normal scar is not a unitary local output. It is an
isometry only after adding a hidden output $H$ satisfying

$$
V^\dagger V+H^\dagger H=I.
$$

Hence

$$
\boxed{
H^\dagger H={1\over2}I.
}
$$

This already forces

$$
\operatorname{rank}H=2,
$$

so the hidden output space has dimension at least two. A one-dimensional
hidden channel cannot complete the visible BB branch. `C:9`.

## 2. The BB Mixed Sector Supplies Exactly The Missing Norm

Let

$$
M=
\begin{pmatrix}
M_{+2}\\
M_{-2}
\end{pmatrix}.
$$

The exact BB mixed-normal identity is

$$
M^\dagger M
=
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}
={1\over2}I.
$$

Therefore $M$ is a valid minimal hidden completion of the visible scar. The full
local output

$$
C_{\rm edge}=
\begin{pmatrix}
V\\
M
\end{pmatrix}
=
\begin{pmatrix}
B_+\\
B_-\\
M_{+2}\\
M_{-2}
\end{pmatrix}
$$

satisfies

$$
C_{\rm edge}^\dagger C_{\rm edge}=I.
$$

Thus the BB coin itself tells us what the missing norm is: it is exactly the
mixed-normal sector. The only question is whether that sector is read as a
recurrent wedge state or as an outgoing clock-error state. `C:9`.

## 3. Orientation-Resolved Minimality

If one ignores the signs of the clock error, a two-dimensional hidden output
space is enough because $M$ has rank two. But the BCC edge has a reflection
orientation:

$$
q\mapsto-q,
\qquad
M_{+2}\leftrightarrow M_{-2}.
$$

Keeping this local orientation data separates the hidden output into two
clock-error ports:

$$
\chi_+,\qquad \chi_-.
$$

In that orientation-resolved basis the hidden map is exactly

$$
|\chi_+\rangle M_{+2}+|\chi_-\rangle M_{-2}.
$$

Any other orientation-resolved completion preserving the BB mixed amplitudes is
only a unitary relabeling of the clock-error output ports. It cannot remove the
hidden sector, because the visible norm deficit is fixed at ${1\over2}I$.

So the minimal local scattering alternatives are:

1. **collapse orientation:** use one rank-two hidden channel, losing the
   $\pm q$ clock-error labels;
2. **preserve BCC reflection orientation:** use the two clock-error ports
   $\chi_\pm$.

The flavor boundary model chooses 2 because the mismatch sign is the local
reflection-odd data of the two-face edge. `C:8` for the finite classification
under the stated orientation-preserving constraint.

## 4. No-Return Forces Retarded Hidden Output

The bare wedge identifies the hidden output with physical $q=\pm2$ states. Then
the first return is

$$
R_{\rm rel}^{(2)}
=
M_{-2}M_{+2}+M_{+2}M_{-2}
\ne0.
$$

If the model requires the retarded visible compression, the incoming hidden to
visible maps must vanish:

$$
G_+=G_-=0.
$$

Then

$$
R_{\rm clk}^{(2)}=G_+M_{+2}+G_-M_{-2}=0.
$$

Thus no-return does not force a different BB coin. It forces a different
boundary interpretation of the mixed-normal output:

$$
\boxed{
\text{mixed-normal output is outgoing clock error, not recurrent wedge state.}
}
$$

## 5. What This Proves

The chain is now:

$$
V^\dagger V={1\over2}I
\Rightarrow
\text{hidden output required}
\Rightarrow
M^\dagger M={1\over2}I
\Rightarrow
\text{BB mixed sector is the canonical hidden output}
\Rightarrow
\text{retarded readout makes it outgoing}.
$$

So the edge-clock scattering update is not an arbitrary decoration. It is the
minimal local way to preserve the exact BB amplitudes, keep unitarity, and
remove the explicit two-step return obstruction.
The causal block-triangular form of that retarded compression is proved in
[CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md](CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md).

What remains open is one layer deeper:

$$
\boxed{
\text{why does the physical BCC defect realize this forced hidden output as
outgoing clock-error leads rather than recurrent wedge states?}
}
$$

## 6. Certainty Ledger

| Claim | Status |
|---|---|
| visible same-normal branch has norm ${1\over2}I$ | `C:9` |
| hidden output rank at least two is required | `C:9` |
| BB mixed-normal branch supplies exactly the missing norm | `C:9` |
| orientation-resolved completion gives $\chi_\pm$ ports | `C:8` under reflection-sign preservation |
| any BB-amplitude-preserving hidden completion is unitary relabeling plus compression choice | `C:8` |
| no-return requires retarded/no-incoming hidden sector | `C:9` |
| retarded/no-incoming sector gives visible powers $A^t$ | `C:9` in [CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md](CAUSAL_EDGE_CLOCK_RETARDED_CONDITION.md) |
| physical defect realizes outgoing clock-error asymptotics | `C:3` |

The concise result is:

$$
\boxed{
\text{The clock-error sector is forced by local unitarity and exact BB
amplitudes; the causal retarded compression is exact, and only its physical
asymptotic origin remains open.}
}
$$
