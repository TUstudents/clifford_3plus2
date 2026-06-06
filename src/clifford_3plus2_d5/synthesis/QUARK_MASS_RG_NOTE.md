# Quark Mass Running And Clebsch Stress Test

This note synthesizes an external phenomenology note, not a sidecar proof. Its
role is to test the quark mass texture against a basic field-theory constraint:
a Yukawa matrix is defined at one renormalization scale. Its three eigenvalues
cannot be compared to observed masses quoted at three different preferred
scales.

The note is important because it separates a real signal from a misleading
six-mass eye test.

## One-Scale Rule

For a Yukawa matrix $Y_q(\mu)$, all eigenvalues are defined at the same scale
$\mu$. A texture cannot honestly compare

$$
m_u(2\,{\rm GeV}),\quad m_c(m_c),\quad m_t(m_t)
$$

as if they were simultaneous eigenvalues of one matrix. The same applies to
the down sector.

In QCD-dominated running, the mass anomalous dimension is flavor independent.
Therefore, once model and observed masses are evolved with the same running
law, the ratio

$$
\frac{m^{\rm model}_i(\mu)}{m^{\rm obs}_i(\mu)}
$$

is essentially scale independent. A common-scale mismatch cannot be tuned away
by choosing a different $\mu$. Electroweak and threshold refinements can move
small numbers, but not a factor-of-two error.

Certainty: `C:9` for the one-scale Yukawa rule. `C:7` for the
scale-independence statement in the QCD-dominated phenomenology used by the
note.

## What The Old Eye Test Was Seeing

The old six-mass comparison looked impressive because it compared the model
against observed masses at the quark's favorite low scale. Light and
intermediate quark masses are inflated at low scales relative to $M_Z$, while
the top is already quoted near its own scale.

Put all masses at one common scale and the old texture overshoots nearly every
non-top mass by a coherent factor of order $1.5$ to $2$. This is not random
scatter. It is a normalization/Clebsch failure exposed by consistent running.

Therefore the paper should not lead with the old six absolute masses.

Certainty: `C:8` as a phenomenology kill of the mixed-scale six-mass fit, given
the note's standard input masses and QCD running.

## The Signal That Survives

The useful data are ratios and inferred values of the shared hierarchy
parameter. Let

$$
\eta_{\rm silver}=(\sqrt2-1)^2=3-2\sqrt2\simeq0.171573.
$$

The CKM/mixing extraction in the note uses

$$
\eta_{\rm CKM}\simeq0.172089,
$$

which is within about $0.3\%$ of $\eta_{\rm silver}$. This is a real signal, not
a scale artifact.

The robust light-sector mass relations are

$$
\frac{m_d}{m_s}=\sqrt3\,\eta^2,
$$

$$
\frac{m_s}{m_b}=\frac{\eta^2}{\sqrt2},
$$

and

$$
\frac{m_u}{m_c}=\frac{\eta^3}{2\sqrt2}
$$

for the current Taylor up Clebsch. At exact silver $\eta$, these give

$$
0.05099,\qquad 0.02082,\qquad 0.001786.
$$

These are in the right empirical neighborhood. The note's strongest conclusion
is that the same $\eta$ seen in CKM is also visible in light/intermediate mass
ratios at the few-to-ten-percent level.

Certainty: `C:6` for the phenomenological cluster, pending a controlled table
with explicit reference masses, uncertainties, and scheme conventions.

## Up Clebsch Correction

The note decisively rejects the old scalar up vector

$$
C_u^{\rm old}=\left(\frac12,\sqrt2,1\right).
$$

The top-anchored charm relation in that vector is

$$
\frac{m_c}{m_t}=\sqrt2\,\eta^3,
$$

which gives about

$$
0.00714
$$

at exact silver $\eta$. This is roughly twice the observed value.

The current scalar sidecar has already made the right structural correction:

$$
C_u^{\rm Taylor}(x)=\left(\frac{x^2}{2},x,1\right),
\qquad
x=\frac1{\sqrt2},
$$

so

$$
C_u^{\rm Taylor}=
\left(\frac14,\frac1{\sqrt2},1\right).
$$

Then

$$
\frac{m_c}{m_t}=\frac{\eta^3}{\sqrt2}
\simeq0.00357
$$

at exact silver $\eta$. The note's empirical best clean rational is

$$
C_u^{\rm emp}=\left(\frac14,\frac34,1\right),
$$

which gives

$$
\frac{m_c}{m_t}=\frac34\eta^3
\simeq0.00379.
$$

The distinction is important. The data prefer a number near $0.73$ for the
charm coefficient. Both $1/\sqrt2\simeq0.707$ and $3/4=0.75$ are close. The
theory preference remains the Taylor/no-leakage value $1/\sqrt2$ unless a new
boundary theorem selects $3/4$.

Certainty: `C:1` for the old $(1/2,\sqrt2,1)$ vector as a scalar mass profile.
`C:6` for $(1/4,1/\sqrt2,1)$ as the current conditional Taylor mechanism.
`C:4` for $(1/4,3/4,1)$ as an empirical control.

## Down Sector And The Normalization Issue

The note also changes how to read the down sector. If an overall down/up
normalization is allowed, the clean down baseline

$$
C_{d,\rm base}=
\left(1,\frac1{\sqrt3},\sqrt{\frac23}\right)
$$

is competitive. The coherent down-sector overshoot is then not evidence for a
different internal Clebsch; it is a single sector normalization

$$
r_d\sim0.6.
$$

In a one-Higgs Standard Model reading, $r_d$ is simply an undetermined relative
Yukawa scale between sectors. In a two-Higgs or extended-Higgs reading, it can
be interpreted as a tan-beta-like down/up normalization. Either way, it should
not be confused with a family-shape Clebsch.

This weakens the phenomenological need for the rank-$5$ down candidate

$$
C_{d,\rm cand}=
\left(1,\frac1{\sqrt3},\sqrt{\frac56}\right).
$$

The rank-$5$ candidate remains representation-theoretically available, but
NOTE 1 says the clean baseline may be the better paper-level default once
sector normalization is treated honestly.

Certainty: `C:5` for the down-sector normalization interpretation. `C:4` for
choosing the baseline over the rank-$5$ candidate from this note alone.

## Synthesis Verdict

NOTE 1 should be integrated as a phenomenology constraint on the scalar and
radial layers:

$$
\boxed{
\text{Do not claim a six-mass mixed-scale fit.}
}
$$

$$
\boxed{
\text{Do claim a shared }\eta\text{ signal in CKM and light mass ratios.}
}
$$

$$
\boxed{
\text{Keep the old }\sqrt2\text{ charm scalar coefficient killed.}
}
$$

$$
\boxed{
\text{Treat }1/\sqrt2\text{ versus }3/4\text{ as the live up-Clebsch stress.}
}
$$

The fresh idea is not that running rescues the texture. It does not. The fresh
idea is that running localizes the problem: the wrong old coefficient was the
top-anchored charm scalar, and the down-sector discrepancy is mostly one
sector normalization rather than family shape.
