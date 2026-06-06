# Session 14 - Charged-Lepton Minimal Boundary Graph

The charged-lepton branch is not a one-sided positive self-energy.  A
charged-lepton Yukawa is a chiral two-sided Schur kernel:

$$
B_e(z)=V_R^T(z-H_{Q,e})^{-1}V_L.
$$

Session 14 builds the minimal colorless active family-port graph whose pole
residue realizes the charged-lepton BCC ansatz.

## Boundary Graph

Use the visible residual basis ordered as $(u,a,b)$ and the unresolved pole
space

$$
Q_e=\operatorname{span}\{t_+,t_-,p_a,p_b\}.
$$

The two trace states $t_\pm$ are the coherent colorless BCC trace-return paths.
The two plane states $p_a,p_b$ carry the residual traceless plane.

The left coupling is

$$
V_L=
\begin{pmatrix}
1/\sqrt2&0&0\\
1/\sqrt2&0&0\\
0&1&0\\
0&0&1
\end{pmatrix}.
$$

The right coupling is

$$
V_R=
\begin{pmatrix}
1&0&0\\
1&0&0\\
0&\cos\theta&\sin\theta\\
0&-\sin\theta&\cos\theta
\end{pmatrix}.
$$

Then the pole residue is

$$
V_R^TV_L
=
\begin{pmatrix}
\sqrt2&0&0\\
0&\cos\theta&-\sin\theta\\
0&\sin\theta&\cos\theta
\end{pmatrix}
=
\sqrt2P_u+R_\theta P_\perp.
$$

This is exact.

## Angle Input

The angle is assembled from the existing holonomy and torsion gates:

$$
\theta_0=-\frac{2\pi}{3},
\qquad
\delta_e=\frac29,
\qquad
\theta=\theta_0-\delta_e.
$$

The graph does not derive $\delta_e=2/9$.  It uses the Session 05 occupation
moment and the existing holonomy model as inputs.

## Action On The Selected Port

The selected charged-lepton port is

$$
e_1=\frac1{\sqrt3}u+\sqrt{\frac23}a.
$$

Acting with the residue gives

$$
B_e e_1
=
\sqrt{\frac23}
\left[
u+\cos\theta\,a+\sin\theta\,b
\right].
$$

After normalization,

$$
\widehat w_e
=
\frac1{\sqrt2}
\left[
u+\cos\theta\,a+\sin\theta\,b
\right].
$$

Therefore

$$
|\widehat w_{\rm trace}|^2=\frac12,
\qquad
|\widehat w_{\rm traceless}|^2=\frac12.
$$

This is Koide trace/traceless equipartition:

$$
K=\frac{\sum_i w_i^2}{(\sum_i w_i)^2}=\frac23.
$$

## Controls

One trace path is rejected.  If the trace path is not doubled, the trace weight
is $1/3$ and the plane weight is $2/3$, so Koide equipartition is lost.

A one-sided Hermitian self-energy is rejected.  The target plane block is a
nontrivial real rotation, hence it is not a symmetric positive residue.

## Verdict

```text
CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_PASS
```

Meaning:

```text
The minimal two-sided colorless boundary graph realizes the charged-lepton
residue exactly and derives Koide equipartition from two coherent trace paths.
```

## Honest Boundary

This is a minimal graph realization, not the final microscopic BCC/Higgs
derivation.  Remaining inputs:

```text
microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths
active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics
charged_lepton_overall_scale_rho_or_mu_e
```

Certainty: `C:9` for the residue algebra and controls; `C:6` for the physical
two-trace-path interpretation; `C:3` for the microscopic origin of the active
$2/9$ torsion.
