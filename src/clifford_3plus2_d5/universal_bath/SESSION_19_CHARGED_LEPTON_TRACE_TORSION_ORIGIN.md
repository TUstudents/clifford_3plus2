# Session 19 - Charged-Lepton Trace-Path and Torsion Origin Audit

Session 14 built the minimal colorless active charged-lepton boundary graph:

$$
B_e = V_R^T(z-H_Q)^{-1}V_L,
\qquad
\operatorname{Res} B_e=\sqrt2 P_u + R_\theta P_\perp,
$$

with

$$
\theta=-\frac{2\pi}{3}-\frac29.
$$

It proved the residue and Koide equipartition once two ingredients were
supplied:

```text
microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths
active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics
```

Session 19 audits those two ingredients.

## Trace Paths

The frozen charged-lepton source is

$$
e_1=\sqrt{\frac23}\,a+\frac1{\sqrt3}u,
$$

so the source weights are

$$
p_a=\frac23,\qquad p_u=\frac13,\qquad p_b=0.
$$

If the minimal graph has $n$ coherent trace-return paths, the trace amplitude
is enhanced by $\sqrt n$, while the traceless plane is not.  Hence the
normalized trace and traceless weights are

$$
w_{\rm tr}(n)=\frac{n}{n+2},
\qquad
w_{\perp}(n)=\frac{2}{n+2}.
$$

Koide trace/traceless equipartition requires

$$
w_{\rm tr}=w_\perp=\frac12,
$$

so

$$
n=2.
$$

One trace path gives $1/3$ vs $2/3$.  Three trace paths gives $3/5$ vs $2/5$.
Both controls fail.  The Session 14 graph supplies exactly two trace-only pole
rows, so the minimal graph is internally sharp.

But this is not yet a microscopic origin theorem.  The two trace rows are
present in the minimal graph.  The current BCC/Higgs boundary model does not
derive those rows.

## Torsion

Session 05 derived

$$
p_a p_u = \frac23\frac13=\frac29
$$

as an incoherent source-occupation moment.  Session 14 then used the same
number as a plane rotation angle:

$$
\theta = -\frac{2\pi}{3}-\frac29.
$$

Session 19 confirms that this insertion is exact in the current graph, but no
occupation-to-angle dynamics is derived.  The CMV/holonomy phase word is
distinct:

$$
\arg(\alpha_e)=-\frac{5\pi}{12},
$$

so the $2/9$ torsion is not already hidden inside the CMV phase word.

## Verdict

```text
CHARGED_LEPTON_TRACE_TORSION_ORIGIN_NOT_DERIVED_AUDIT
```

What is now exact:

- two coherent trace paths are mathematically forced by trace/traceless
  equipartition inside the minimal graph ansatz;
- the minimal graph supplies exactly those two trace paths;
- $2/9$ is exactly the frozen-source occupation transition weight;
- one-trace, three-trace, and coherent-amplitude controls are rejected upstream.

What remains open:

```text
microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths
active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics
```

Thus the charged-lepton graph is not closed microscopically.  Session 19 turns
the gap into a sharper pair of finite dynamical questions: why does the
colorless BCC/Higgs boundary produce exactly two coherent trace-return paths,
and what boundary dynamics converts the occupation weight $2/9$ into a real
rotation angle?

## Certainty

- `C:9` for the trace-count algebra $w_{\rm tr}(n)=n/(n+2)$ and the unique
  solution $n=2$.
- `C:9` for $p_a p_u=2/9$ as a source occupation moment.
- `C:3` for the microscopic origin of the two trace paths.
- `C:2` for the current occupation-to-angle dynamics.
