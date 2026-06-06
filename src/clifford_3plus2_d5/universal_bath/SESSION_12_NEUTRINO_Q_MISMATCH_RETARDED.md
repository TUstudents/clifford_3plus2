# Session 12 - Neutrino Q-Mismatch And Retarded Closure

Session 10 built the internal family-port graph.  Session 11 derived the active
projector from selected-port incidence:

$$
e_1=\sqrt{\frac23}\,a+\frac1{\sqrt3}u,
\qquad
\operatorname{detrace}(e_1)=a,
\qquad
P_{\rm act}=I-P_a=P_u+P_b.
$$

Session 12 connects that projector to the microscopic BB edge dynamics under
the single-clock/outgoing-boundary model.

## Relative-Depth Split

At a codimension-two BCC edge define

$$
q=r_1-r_2.
$$

A body-diagonal hop $\sigma=(\sigma_1,\sigma_2,\sigma_3)$ changes it by

$$
\Delta q=\sigma_1-\sigma_2.
$$

Therefore the eight BCC hops split exactly into:

$$
\sigma_1=\sigma_2:\quad \Delta q=0
$$

and

$$
\sigma_1=-\sigma_2:\quad \Delta q=\pm2.
$$

There are four same-normal and four mixed-normal directions.

## Exact BB Norm Split

The same-normal BB blocks are $B_+$ and $B_-$.  The mixed-normal clock-error
blocks are $M_{+2}$ and $M_{-2}$.  The certificate checks:

$$
B_+^\dagger B_+ + B_-^\dagger B_-=\frac12 I,
$$

$$
M_{+2}^\dagger M_{+2}+M_{-2}^\dagger M_{-2}=\frac12 I.
$$

Thus the visible q=0 branch and the mixed-normal branch each carry exactly half
of the local BB norm.  The visible branch is not deleting probability; it is a
compression of a local scattering colligation.

## Q-Mismatch Penalty

The selected-port incidence result identifies the family radial line as $a$.
The single-clock mismatch model says the asynchronous mixed-normal coordinate
is penalized by

$$
H_{\rm lock}=gq^2.
$$

The adjacent mixed-normal sectors have $q=\pm2$, so

$$
H_{\rm leak}=4gI.
$$

With

$$
M=\begin{pmatrix}M_{+2}\\M_{-2}\end{pmatrix},
$$

Schur feedback is

$$
\Sigma_{\rm mix}(z,g)
=
M^\dagger(z-4g)^{-1}M
=
\frac{1}{2(z-4g)}I.
$$

Hence

$$
\lim_{g\to\infty}\Sigma_{\rm mix}(z,g)=0.
$$

In family-port language this is the hard penalty

$$
\Lambda P_a,
$$

leaving the active incidence plane

$$
P_u+P_b.
$$

## Retarded Closure

The outgoing half-line Weyl branch satisfies

$$
m(z)=\frac1{z-m(z)},
$$

with the retarded normalization

$$
m(z)\sim\frac1z.
$$

Thus

$$
m_R(z)=\frac{z-\sqrt{z^2-4}}2.
$$

The certificate checks the Weyl equation, the normalization, and the high-gap
decoupling of the mixed-normal retarded feedback.

The retarded edge update has block-triangular form:

$$
T_R=
\begin{pmatrix}
A_{\mathbb N}&0\\
E&S_\chi
\end{pmatrix}.
$$

Therefore

$$
P_{\rm vis}T_R^t\iota_{\rm vis}=A_{\mathbb N}^t.
$$

The visible mass branch is exactly the q=0 survival compression.

## Recurrent Control

If the mixed-normal ports are instead closed into a recurrent wedge, there is a
hidden-to-visible return block.  The local two-step correction is

$$
M_{-2}M_{+2}+M_{+2}M_{-2}
=
\begin{pmatrix}
-(1+i)/4&0\\
0&-(1-i)/4
\end{pmatrix}
\ne0.
$$

The recurrent update changes the visible powers and is rejected as the mass
readout condition.

## Verdict

```text
NEUTRINO_Q_MISMATCH_RETARDED_COMPRESSION_PASS
```

Meaning:

```text
Exact BB mixed-normal leakage plus a single-clock q^2 hard gap and retarded
outgoing clock-error leads give the Session 10 selected active graph:
H_chain tensor (P_u+P_b) plus a radial penalty on P_a.
```

## Honest Boundary

This closes the executable conditional gate behind Sessions 10 and 11.  It
does not derive the deeper boundary material that realizes the two remaining
physical inputs:

```text
single_clock_locking_field_is_realized_by_boundary_material
mixed_normal_clock_error_ports_are_outgoing_asymptotic_leads
```

So the neutrino family-port boundary graph is now microscopic **inside** the
single-clock/outgoing-boundary model.  The deeper origin of that boundary
material remains open.

## Certainty Ledger

| Statement | Status |
|---|---|
| BCC hop split by $\Delta q$ | `C:9` |
| BB same-normal/mixed-normal norm split | `C:9` |
| $q=\pm2$ hard gap from $gq^2$ | `C:9` under single-clock model |
| mixed Schur feedback $\Sigma=I/[2(z-4g)]$ | `C:9` |
| hard-gap limit $\Sigma\to0$ | `C:9` |
| retarded Weyl branch and normalization | `C:9` |
| block-triangular retarded visible compression | `C:9` |
| recurrent wedge return is nonzero | `C:9` |
| single-clock boundary material exists physically | `C:6` |
| outgoing asymptotic condition from deeper boundary dynamics | `C:3` |
