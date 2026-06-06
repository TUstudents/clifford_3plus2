# BB-QCA Boundary Update Completion Audit

This note audits the active objective:

$$
\boxed{
\text{complete the bare microscopic BB-QCA boundary update.}
}
$$

Here "bare" is interpreted in the strict sense used throughout the boundary
stack: the local amplitudes are the pinned Bialynicki-Birula BCC Weyl
amplitudes, with no flavor mass number, silver number, or sector coefficient
inserted into the update. The update may include explicit boundary ports needed
for local unitarity. It may not hide probability loss or tune the mixed-normal
sector.

The audit conclusion is:

$$
\boxed{
\text{the local BB-QCA edge update is complete as a unitary scattering
colligation with retarded }q=0\text{ compression.}
}
$$

The remaining open problem is not the local BB-QCA update. It is a deeper
physical-material question:

$$
\boxed{
\text{why does the physical BCC boundary realize the forced clock-error sector
as outgoing asymptotic leads rather than recurrent wedge states?}
}
$$

That distinction matters. The first statement is now a finite algebraic theorem
about the microscopic update. The second is a model-building question about the
physical boundary medium that realizes the scattering asymptotics.

## 1. Requirements Extracted

The boundary-update work generated the following explicit requirements.

| Requirement | Needed evidence | Status |
|---|---|---|
| Pinned BB amplitudes are used unchanged | exact $W_\sigma$ sums for $B_\pm,M_{\pm2}$ | complete, `C:9` |
| The edge coordinate split is microscopic | $\Delta q=\sigma_1-\sigma_2$ gives $q=0\to q=0$ and $q=0\to q=\pm2$ | complete, `C:9` |
| Bare BB spinor alone cannot close the scar | simultaneous kernels of $M_{+2},M_{-2}$ intersect trivially | complete, `C:9` no-go |
| Wedge geometry alone cannot close the scar | $q=0\to q=\pm2\to q=0$ is allowed inside the wedge | complete, `C:9` no-go |
| Ordinary face exterior cannot remove first return | $q=\pm2\to0$ is inward, not exterior | complete, `C:9` no-go |
| Mixed-normal return is explicitly nonzero | $M_{-2}M_{+2}+M_{+2}M_{-2}\ne0$ | complete, `C:9` control |
| Local hidden sector is forced by unitarity | visible branch has norm ${1\over2}I$, so hidden rank at least two | complete, `C:9` |
| Hidden sector is the BB mixed-normal sector | $M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}={1\over2}I$ | complete, `C:9` |
| Orientation-resolved hidden ports are $\chi_\pm$ | $q\mapsto-q$ exchanges $M_{+2}\leftrightarrow M_{-2}$ | complete, `C:8` |
| A finite local unitary completion exists | $C_{\rm edge}$ is an isometry and can be completed to $U_{\rm loc}$ | complete, `C:9` |
| Retarded/no-incoming compression is exact | $T_R=\begin{pmatrix}A&0\\E&S\end{pmatrix}$ gives visible powers $A^t$ | complete, `C:9` |
| Recurrent control restores the obstruction | adding $G$ gives two-step correction $GE=R_{\rm rel}^{(2)}$ | complete, `C:9` |
| The silver transfer is not inserted | visible survival branch at $c=1/\sqrt2$ gives $\lambda_\pm=\sqrt2\pm1$ | complete, `C:9` |
| Mismatch coordinate has microscopic origin | one edge clock gives unique local linear mismatch $K_{\rm rel}=q$ | complete inside edge-clock model, `C:9` |
| Positive locking is a local square | $H_{\rm lock}=gK_{\rm rel}^\dagger K_{\rm rel}=gq^2$ | complete inside edge-clock model, `C:9` |
| Physical outgoing asymptotics are derived from deeper material | boundary medium forces outgoing leads rather than recurrence | not part of local BB update; open, `C:3` |

## 2. Completed Local Update

The completed microscopic edge update is
[MICROSCOPIC_BB_QCA_EDGE_UPDATE.md](MICROSCOPIC_BB_QCA_EDGE_UPDATE.md):

$$
\boxed{
\begin{array}{c}
\sigma_1=\sigma_2:\quad r_\pm\text{ visible radial ports with }B_\pm,\\
\sigma_1=-\sigma_2:\quad \chi_\pm\text{ clock-error ports with }M_{\pm2},\\
C_{\rm edge}^\dagger C_{\rm edge}=I,\\
U_{\rm loc}=(C_{\rm edge},C_\perp)\text{ finite local unitary},\\
T_R=\begin{pmatrix}A_{\mathbb N}&0\\E&S_\chi\end{pmatrix}
\text{ for retarded source preparation.}
\end{array}
}
$$

This is a complete local BB-QCA boundary update. It is local, unitary after
including boundary ports, BB-amplitude preserving, and has an exact retarded
$q=0$ visible compression. `C:9`.

## 3. Killed Alternatives

The update is not chosen because it is convenient. The alternatives were checked
and killed:

1. **BB spinor-only scar:** no nonzero spinor kills both mixed-normal leakage
   channels. `C:1` for spinor-only closure.
2. **Closed relative-depth walk:** the unprojected BB relative channel is a
   two-way unitary channel, not a retarded scar. `C:1` for retardedness from the
   relative cover alone.
3. **Physical wedge alone:** the wedge confines the relative coordinate but
   allows $q=0\to q=\pm2\to q=0$. `C:1` for wedge-only closure.
4. **Face exterior alone:** the first return from $q=\pm2$ is inward and is not
   absorbed by ordinary face exterior channels. `C:1`.
5. **Closed recurrent hidden sector:** adding hidden return $G$ restores
   $R_{\rm rel}^{(2)}\ne0$. `C:1` for recurrent hidden sector as the silver
   readout.

Thus the clock-error scattering update is the minimal surviving local update
that keeps the exact BB amplitudes and removes the explicit return obstruction.

## 4. Boundary-Material Premise

The local update still contains one physical premise:

$$
\boxed{
\text{the clock-error ports are outgoing asymptotic leads in the mass readout.}
}
$$

This is not a flavor fitting parameter. It is a boundary-asymptotic condition.
Inside the scattering model it is exact and causal: no incoming clock-error
source is prepared, so the visible branch is $A_{\mathbb N}^t$. The deeper
question is why a physical BCC boundary material realizes that asymptotic
condition.

That deeper problem should not be recorded as "the BB-QCA update is
incomplete." It should be recorded as the next model-building layer above the
completed local update:

$$
\boxed{
\text{derive the physical medium that implements the already-specified
outgoing clock-error leads.}
}
$$

## 5. Completion Verdict

The objective "complete bare microscopic BB-QCA boundary update" is satisfied
at the local update level:

$$
\boxed{
\text{COMPLETE: exact BB amplitudes, exact local scattering unitarity, exact
retarded }q=0\text{ compression, and exact controls against recurrence.}
}
$$

The remaining open problem is explicitly out of this objective's local-update
scope:

$$
\boxed{
\text{OPEN NEXT LAYER: derive a physical boundary material that selects the
outgoing asymptotic realization.}
}
$$

Certainty: `C:9` for the finite algebraic update and all identities certified
by the scripts below; `C:6` that the outgoing-asymptotic realization is the
right physical boundary model; `C:3` for deriving that realization from deeper
boundary-material dynamics.
