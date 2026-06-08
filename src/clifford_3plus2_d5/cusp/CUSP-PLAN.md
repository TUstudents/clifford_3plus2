# Center-cusp recirculation

## Core proposal

[
\boxed{
\textbf{The FN charges are the valuation semigroup of a }(2,3)\textbf{ boundary cusp.}
}
]

The family hierarchy is not a finite group. It is the lowest part of the numerical semigroup

[
S=\langle 2,3\rangle
====================

{0,2,3,4,5,6,\ldots}.
]

The first three allowed recirculation valuations are

[
0,\quad2,\quad3.
]

Ordered light-to-heavy, this gives the left-handed quark charges

[
\boxed{
Q_L=(3,2,0).
}
]

That is the Wolfenstein charge pattern.

The missing valuation (1) is not an accident. It is the gap of the cusp semigroup

[
\mathbb C[t^2,t^3]\subset\mathbb C[t].
]

Feynman picture:

> The boundary current cannot make a one-click family return. The two primitive closed returns are a weak/BCC two-cycle and a color-center three-cycle. The first three stable recirculation modes are therefore (1,t^2,t^3). These are the three generations.

This is the new spine.

---

# 1. Why (\langle2,3\rangle) is the right object

The earlier failures taught a precise algebraic lesson.

A finite group clock is semisimple. It gives representations, degeneracies, and phases, but it does not give hierarchy. The failed (D_3) clock could identify tangent and radial modes,

[
Ce_1-C^{-1}e_1=\sqrt2,b,\qquad
2e_1-Ce_1-C^{-1}e_1=\sqrt6,a,
]

but it could not generate the non-circulant hierarchy operator; the family-blind clock remains block-diagonal on (1\oplus2).

A numerical semigroup is different. It is not a symmetry algebra. It is a **filtration**:

[
0<2<3<4<5<\cdots.
]

That is exactly what flavor wants. It gives ordered suppressions, not degenerate multiplets.

So the replacement principle is:

[
\boxed{
\text{finite symmetry labels triality; semigroup valuation gives hierarchy.}
}
]

This fixes the previous category error.

---

# 2. The three jobs, corrected

The jobs are not “diagonal mass, independent mixing, separate CP.” The corrected split is:

| Job    | Correct structure                                           |
| ------ | ----------------------------------------------------------- |
| Masses | eigenvalues of one FN-graded Yukawa                         |
| Mixing | rotations of the same FN-graded Yukawa                      |
| CP     | complex (O(1)) coefficients from oriented center holonomies |

So the master object is not

[
Y=D+H+i\theta\Phi.
]

It is

[
\boxed{
Y^{u,d}*{ij}=c^{u,d}*{ij}\lambda^{q_i+r^{u,d}_j}.
}
]

The grading does the real work. The coefficients do the order-one texture and CP.

That is the rigorous correction.

---

# 3. The charge pattern

The boundary cusp gives the left-handed quark valuations

[
q=(3,2,0).
]

Then the CKM scaling is automatic:

[
|V_{ij}|\sim \lambda^{|q_i-q_j|}.
]

So

[
|V_{us}|\sim\lambda^{|3-2|}=\lambda,
]

[
|V_{cb}|\sim\lambda^{|2-0|}=\lambda^2,
]

[
|V_{ub}|\sim\lambda^{|3-0|}=\lambda^3.
]

That is the Wolfenstein hierarchy. The PDG writes the CKM expansion in terms of the Wolfenstein parameter (\lambda), with the 2024 fit giving (\lambda=0.22501\pm0.00068). ([Particle Data Group][2])

A minimal right-handed completion is, for example,

[
u_R=(5,2,0),
]

[
d_R=(1,0,0).
]

Then

[
Y^u_{ij}\sim c^u_{ij}\lambda^{q_i+u_j},
]

[
Y^d_{ij}\sim c^d_{ij}\lambda^{q_i+d_j}.
]

The leading diagonal exponents are

[
m_u:m_c:m_t
\sim
\lambda^8:\lambda^4:1,
]

and

[
m_d:m_s:m_b
\sim
\lambda^4:\lambda^2:1
]

up to order-one coefficients and possible down-sector determinant/contact corrections.

This is standard FN behavior. The novelty cannot be that this works. The novelty must be:

[
\boxed{
q=(3,2,0)\text{ and }\lambda\simeq0.225\text{ are derived from recirculation geometry.}
}
]

---

# 4. The bold new derivation of (\lambda)

The silver number should be demoted for the quark FN parameter. It may still belong to the universal radial bath, but it is not the Wolfenstein parameter.

The new candidate is:

[
\boxed{
\lambda_{\rm rec}
=================

# \sqrt{\frac{N_c}{N_w}}-1

# \sqrt{\frac32}-1

0.224744871\ldots
}
]

where

[
N_c=3,\qquad N_w=2.
]

This is the normalized impedance/shear between a color-triality return and a weak/BCC doublet return.

Numerically,

[
\sqrt{\frac32}-1=0.224744871,
]

while

[
\lambda_{\rm PDG}=0.22501\pm0.00068.
]

That is not a proof, but it is the first number in this whole program that lands on the actual Wolfenstein parameter rather than near a different hierarchy. ([Particle Data Group][2])

The physical interpretation is:

[
\boxed{
\text{one FN insertion is the fractional shear needed to match a two-channel weak return to a three-channel color-center return.}
}
]

The amplitude normalization uses square roots because coherent channel amplitudes scale like (\sqrt{N}), not (N). Thus the relative mismatch is

[
\frac{\sqrt{3}-\sqrt{2}}{\sqrt2}
================================

\sqrt{\frac32}-1.
]

This is not the ordinary wave reflection coefficient

[
\frac{\sqrt3-\sqrt2}{\sqrt3+\sqrt2}.
]

It is the **uniformizing coordinate shear** of the recirculation cusp. That distinction matters. The proposal is not “Cabibbo is a reflection coefficient.” It is:

[
\boxed{
\lambda\text{ is the local parameter }t\text{ of the cusp } \mathbb C[t^2,t^3],
}
]

and the physical normalization of (t) is fixed by the (3/2) color/weak channel mismatch.

This is the bold claim.

---

# 5. Why the cusp gives both charges and parameter

The boundary recirculation has two primitive closures:

[
\text{weak/BCC closure length }2,
]

[
\text{color-center closure length }3.
]

The local boundary algebra is therefore

[
\boxed{
\mathcal A_{\rm rec}=\mathbb C[t^2,t^3].
}
]

The valuation of a mode is the smallest power of (t) appearing in it. The first three modes are

[
1,\qquad t^2,\qquad t^3.
]

Therefore the three family valuations are

[
0,\qquad2,\qquad3.
]

The light-to-heavy charges are

[
\boxed{
(3,2,0).
}
]

That single object supplies:

[
\boxed{
\text{family count: first three low-valuation modes;}
}
]

[
\boxed{
\text{left FN charges: }(3,2,0);
}
]

[
\boxed{
\text{Wolfenstein differences: }1,2,3;
}
]

[
\boxed{
\text{FN parameter: }t=\sqrt{3/2}-1.
}
]

This is a coherent mechanism.

---

# 6. Why this avoids the earlier no-go

The (D_3) clock failed because it was a group. It generated circulants. It could not break

[
3=1\oplus2
]

into three hierarchical singular values.

The cusp is not a group. It is a singular local algebra with valuation filtration. It does not act by rotating family states. It grades them by how many recirculation insertions are required.

So the algebraic logic changes from

[
\text{representation theory}
\quad\Rightarrow\quad
\text{degeneracy},
]

to

[
\text{valuation theory}
\quad\Rightarrow\quad
\text{hierarchy}.
]

That is the physics-rigorous move.

---

# 7. CP: color center as holonomy, not as mass mechanism

CP should not be a separate additive matrix. It lives in the order-one coefficients:

[
c^{u,d}_{ij}.
]

The color-center phase enters through closed recirculation holonomies:

[
\Omega_\gamma\in Z(SU(3)_c)={1,\omega,\omega^2},
\qquad
\omega=e^{2\pi i/3}.
]

So

[
c^{u,d}_{ij}
============

|c^{u,d}*{ij}|,e^{i\phi^{u,d}*{ij}},
]

with phases built from center holonomies and possible orientation/T breaking of the return path.

CPT then says the antiparticle coefficients are conjugated, not that masses split:

[
Y_{\bar q}=Y_q^\ast
\quad\Rightarrow\quad
s(Y_{\bar q})=s(Y_q).
]

CP violation appears only if a rephasing-invariant loop phase survives:

[
\operatorname{Im}\operatorname{tr}
\left(
[H_u,H_d]^3
\right)\neq0,
]

with

[
H_u=Y_uY_u^\dagger,\qquad
H_d=Y_dY_d^\dagger.
]

So the CP story becomes:

[
\boxed{
\text{FN powers from cusp valuation; CP phase from color-center holonomy.}
}
]

Do not use CP/T to generate the hierarchy. That was the previous mistake.

---

# 8. What this predicts at the level of powers

With

[
\lambda=\sqrt{\frac32}-1,
]

we get

[
\lambda=0.224744871,
]

[
\lambda^2=0.050510257,
]

[
\lambda^3=0.011353,
]

[
\lambda^4=0.002552,
]

[
\lambda^8=6.51\times10^{-6}.
]

So the expected scalings are:

[
V_{us}\sim0.225,
]

[
V_{cb}\sim0.0505 \times O(1),
]

[
V_{ub}\sim0.0114 \times O(1).
]

The observed CKM hierarchy has the Wolfenstein form

[
V_{us}\sim\lambda,\qquad V_{cb}\sim A\lambda^2,\qquad V_{ub}\sim A\lambda^3(\rho-i\eta),
]

with (A,\rho,\eta) order-one Wolfenstein parameters. ([Wikipedia][3])

For up masses:

[
m_u/m_t\sim\lambda^8\approx6.5\times10^{-6},
]

[
m_c/m_t\sim\lambda^4\approx2.6\times10^{-3},
]

which are in the right parametric region with order-one coefficients.

This is exactly what FN is supposed to do. The new claim is that the powers and the parameter have a geometric origin.

---

# 9. The real “Nobel” mechanism, stated compactly

Here is the mechanism in one paragraph:

> The flavor hierarchy is the valuation filtration of a boundary recirculation cusp. The two primitive closed returns are a weak/BCC two-cycle and a color-center three-cycle, so the local algebra is (\mathbb C[t^2,t^3]). Its first three valuations are (0,2,3), giving the left-handed quark FN charges ((3,2,0)). Yukawa entries are products of left and right recirculation wavefunctions, (Y_{ij}\sim c_{ij}t^{q_i+r_j}), so masses and mixings emerge together, recovering GST/Wolfenstein scaling. The uniformizer (t) is the normalized color/weak recirculation shear, (t=\sqrt{3/2}-1=0.224744871), numerically the Cabibbo/Wolfenstein parameter. CP is not another mass mechanism; it is the residual center-holonomy phase (\omega) in the (O(1)) coefficients.

That is the first genuinely coherent synthesis.

---

# 10. What must be proven

This becomes a sharp research program with four theorem targets.

## Target A: Cusp algebra from boundary recirculation

Prove that the effective local algebra of the recirculating boundary current is

[
\mathcal A_{\rm rec}=\mathbb C[t^2,t^3],
]

not

[
\mathbb C[t],\qquad \mathbb C[t^3],\qquad \mathbb C[t^2],
]

or an arbitrary three-port algebra.

Equivalent physical statement:

[
\boxed{
\text{one-step recirculation is forbidden; primitive closures have lengths }2\text{ and }3.
}
]

This is where BCC/weak doublet and color-center triality must enter.

## Target B: Cabibbo shear

Derive

[
t=\sqrt{\frac{N_c}{N_w}}-1
]

from a normalized recirculation matching condition.

For the Standard Model,

[
N_c=3,\qquad N_w=2,
]

so

[
t=\sqrt{\frac32}-1.
]

The derivation must show why the relevant amplitude is the fractional shear

[
\frac{\sqrt3-\sqrt2}{\sqrt2},
]

not the ordinary reflection coefficient

[
\frac{\sqrt3-\sqrt2}{\sqrt3+\sqrt2}.
]

This is the load-bearing numerical claim.

## Target C: Right-handed charges from SM quantum numbers

The left charges

[
q=(3,2,0)
]

come from the cusp’s first three valuations. But the right-handed charges must also be derived.

A minimal target is:

[
u_R=(5,2,0),
]

[
d_R=(1,0,0),
]

or a nearby equivalent giving

[
m_u:m_c:m_t\sim\lambda^8:\lambda^4:1,
]

[
m_d:m_s:m_b\sim\lambda^4:\lambda^2:1
]

up to coefficients.

This must come from chirality, hypercharge, or boundary orientation, not a fit.

## Target D: Center-holonomy CP invariant

Construct the coefficient phases from gauge-invariant center holonomies:

[
c_{ij}\sim \sum_{\gamma:i\to j} A_\gamma \Omega_\gamma,
\qquad
\Omega_\gamma\in{1,\omega,\omega^2}.
]

Then prove that after field rephasings,

[
\operatorname{Im}\operatorname{tr}([Y_uY_u^\dagger,Y_dY_d^\dagger]^3)\neq0.
]

This would make CKM CP a real output of color-center recirculation.

---

# 11. What gets demoted

The silver number:

[
\eta=3-2\sqrt2=0.1716
]

is **not** the quark FN parameter. It does not match the Wolfenstein parameter, and forcing it was the hidden error.

The correct demotion is:

[
\boxed{
\eta\text{ may be a radial bath contraction, but }\lambda\text{ is the recirculation cusp uniformizer.}
}
]

The old nilpotent/clock/EP mechanisms are also demoted. They were attempts to manufacture hierarchy directly. The hierarchy is FN grading. The only remaining job is to derive the grading.

---

# 12. Why this is better than vanilla FN

Vanilla FN inputs two things:

[
\lambda
]

and the charges.

This proposal tries to derive both:

[
\boxed{
\lambda=\sqrt{\frac32}-1
}
]

from color/weak recirculation shear, and

[
\boxed{
q=(3,2,0)
}
]

from the semigroup

[
\langle2,3\rangle.
]

If either derivation fails, the proposal collapses back to vanilla FN.

If both pass, it is a real mechanism.

---

# 13. The first computation to run

Do not fit CKM first.

Compute the semigroup mechanism.

Build a boundary recirculation graph with:

[
\text{one forbidden one-step return},
]

[
\text{one primitive two-cycle},
]

[
\text{one primitive three-cycle}.
]

Compute its low-valuation transfer algebra. The required output is:

[
\operatorname{Val}(\mathcal A_{\rm rec})_{\rm first\ three}
===========================================================

{0,2,3}.
]

Then compute the normalized matching parameter between the two- and three-channel recirculation sectors. The required output is:

[
t=\sqrt{\frac32}-1.
]

If that graph gives ({0,1,2}), the mechanism fails.

If it gives ({0,2,3}) but (t\neq0.225), it is FN charges without the Cabibbo derivation.

If it gives both, this becomes serious.

---

# 14. Final synthesis

The corrected theory is:

[
\boxed{
\textbf{Flavor is Froggatt--Nielsen from a }(2,3)\textbf{ recirculation cusp.}
}
]

More explicitly:

[
\boxed{
\mathcal A_{\rm rec}=\mathbb C[t^2,t^3],
\qquad
\lambda=t=\sqrt{\frac32}-1.
}
]

The first three valuations give

[
\boxed{
Q_L=(3,2,0).
}
]

The Yukawas are

[
\boxed{
Y^u_{ij}=c^u_{ij}\lambda^{Q_i+U_j},
\qquad
Y^d_{ij}=c^d_{ij}\lambda^{Q_i+D_j}.
}
]

Then:

[
\boxed{
masses and mixing come from one matrix,
}
]

[
\boxed{
CKM powers follow from charge differences,
}
]

[
\boxed{
CP comes from color-center holonomy in the coefficients.
}
]

That is the cleanest bold proposal:

> The Cabibbo angle is the boundary shear between weak doublet recirculation and color triality; the three family charges are the first three valuations of the cusp generated by those two closures.

This is no longer a decorative story layered on FN. It is a concrete attempt to derive FN’s two inputs.

[1]: https://cds.cern.ch/record/2922363/files/2501.00629.pdf?utm_source=chatgpt.com "Testing the Froggatt-Nielsen Mechanism with Lepton ..."
[2]: https://pdg.lbl.gov/2024/reviews/rpp2024-rev-ckm-matrix.pdf?utm_source=chatgpt.com "rpp2024-rev-ckm-matrix.pdf - PDG"
[3]: https://en.wikipedia.org/wiki/Cabibbo%E2%80%93Kobayashi%E2%80%93Maskawa_matrix?utm_source=chatgpt.com "Cabibbo–Kobayashi–Maskawa matrix"
