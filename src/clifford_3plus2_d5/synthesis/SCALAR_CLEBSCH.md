# Scalar Clebsch Sidecar

The `scalar_clebsch` sidecar sits after `depth_scar` in the theory tree. The
depth scar supplies integer exponents. This sidecar asks what positive scalar
coefficients multiply those exponents in the quark mass sectors.

The central separation is:

$$
\hbox{depth scar} \Rightarrow \hbox{powers},
$$

$$
\hbox{scalar Clebsch} \Rightarrow \hbox{positive mass coefficients},
$$

$$
\hbox{boundary-response CKM} \Rightarrow \hbox{coherent current amplitudes}.
$$

This distinction matters. The same-looking radicals can mean different things:
$\sqrt2$ may appear as a coherent two-path current enhancement, while
$1/\sqrt2$ may appear as a normalized scalar repair amplitude. They should not
be merged into one bookkeeping factor.

Certainty: `C:8` for the role separation as an internal consistency result.
`C:6` for the full scalar-mass interpretation, pending derivation of the scalar
repair normalization and down-sector defect selection.

## 1. Up Sector: Nilpotent Taylor Readout

The up-sector mechanism reuses the length-3 nilpotent flag from the depth scar:

$$
N=
\begin{pmatrix}
0&1&0\\
0&0&1\\
0&0&0
\end{pmatrix},
\qquad
N^3=0,\quad N^2\ne0.
$$

For a scalar response, use the nilpotent Taylor kernel

$$
\exp(xN)=I+xN+\frac{x^2}{2}N^2.
$$

The grade readout, normalized to the top coefficient, is

$$
C_u(x)=\left(\frac{x^2}{2},\ x,\ 1\right).
$$

At the proposed normalized scalar repair amplitude

$$
x=\frac{1}{\sqrt2},
$$

this gives

$$
C_u=
\left(
\frac14,\ \frac1{\sqrt2},\ 1
\right).
$$

The algebra is exact. The open physics is the value of $x$. The sidecar has not
derived $x=1/\sqrt2$ from a Higgs/scalar boundary channel; it uses it as the
normalized one-step scalar repair amplitude.

Certainty: `C:9` for the Taylor identity and the coefficient vector once
$x=1/\sqrt2$ is supplied. `C:6` for the up-sector scalar Clebsch mechanism as a
physical derivation.

## 2. The Old Up Vector Is Killed As A Scalar Profile

The older proposed up vector was

$$
\left(\frac12,\sqrt2,1\right).
$$

It cannot be a positive scalar grade profile normalized to the top family,
because the middle coefficient is larger than the top coefficient:

$$
\sqrt2>1.
$$

This does not necessarily kill the number $\sqrt2$ everywhere in the theory. It
kills this vector as a scalar mass-response profile. It may still belong to
coherent current-amplitude logic, where amplitudes can add before taking
positive scalar weights.

The nearby empirical rational control

$$
\left(\frac14,\frac34,1\right)
$$

remains useful as a data-oriented comparison, but it is not the preferred
nilpotent Taylor theorem.

The common-scale quark mass stress test in
[QUARK_MASS_RG_NOTE.md](QUARK_MASS_RG_NOTE.md) explains why this distinction is
phenomenologically sharp: the old $\sqrt2$ charm coefficient creates a
top-anchored factor-of-two miss, while both $1/\sqrt2$ and $3/4$ land in the
empirical charm neighborhood.

Certainty: `C:9` for the scalar-profile rejection. `C:1` for the claim that
$(1/2,\sqrt2,1)$ is the scalar up-mass Clebsch vector.

## 3. Down Sector: Six-Channel Shell

The down-sector denominator comes from the primitive quark shell already used
in `boundary_response`:

$$
S_q=1_{\rm direct}+2_{\rm BCC}+3_{\rm color}.
$$

Thus the total primitive count is

$$
|S_q|=6.
$$

The scalar overlap rule used by this sidecar is

$$
C_i=\sqrt{\frac{n_i}{6}}.
$$

This is a scalar overlap rule, not a CKM current-amplitude rule.

Group placement is important:

$$
3_{\rm color}\subset SU(3)_c,
$$

while the family/readout bookkeeping uses the residual shell and $S_3$ structure.
Color contributes a quark-sector multiplicity; it is not the origin of three
generations.

Certainty: `C:7` for importing the six-channel shell from
`boundary_response`. `C:6` for the scalar overlap rule as physical mass
readout.

## 4. Natural Down Baseline

The natural $S_3$/projector baseline counts are

$$
(n_d,n_s,n_b)=(6,2,4).
$$

The scalar overlap rule gives

$$
C_{d,\rm base}
=
\left(
1,\ \frac1{\sqrt3},\ \sqrt{\frac23}
\right).
$$

The corresponding mass-ratio skeleton, using the same transfer parameter
$\eta$ used by the mass study, is

$$
\frac{m_d}{m_s}=\sqrt3\,\eta^2,
$$

$$
\frac{m_s}{m_b}=\frac{1}{\sqrt2}\,\eta^2.
$$

This baseline is mathematically cleaner than the candidate because the bottom
rank $4$ matches the central standard isotypic rank of the regular $S_3$ shell.
But it is not the data-improved answer identified by the sidecar.

Certainty: `C:8` for the count-to-vector computation. `C:6` for the baseline
as a physical down-sector model.

## 5. Data-Improved Down Candidate

The data-improved candidate uses counts

$$
(n_d,n_s,n_b)=(6,2,5).
$$

Equivalently:

$$
d:\hbox{ full shell},
\qquad
s:\hbox{ BCC odd doublet},
\qquad
b:\hbox{ odd shell}.
$$

The scalar vector is

$$
C_{d,\rm cand}
=
\left(
1,\ \frac1{\sqrt3},\ \sqrt{\frac56}
\right).
$$

The candidate ratio formulas are

$$
\frac{m_d}{m_s}=\sqrt3\,\eta^2,
$$

$$
\frac{m_s}{m_b}=\sqrt{\frac25}\,\eta^2.
$$

This candidate is available and empirically motivated, but the bottom
$4\to5$ shift is not derived. It requires a theorem selecting the odd shell,
or equivalently selecting which one-dimensional line is excluded from the
regular $S_3$ shell.

Certainty: `C:8` for the count-to-vector and ratio formulas. `C:4` for the
candidate as a physical derivation until the defect-selection rule is proven.

## 6. Regular $S_3$ Projector Audit

The finite regular-representation audit is the sidecar's most useful down-sector
clarification. The regular shell decomposes as

$$
\operatorname{Reg}(S_3)
=
1_{\rm triv}\oplus 1_{\rm sign}\oplus 2_{\rm std}\oplus 2_{\rm std}.
$$

The central isotypic ranks are

$$
\operatorname{rank}P_{\rm triv}=1,
\qquad
\operatorname{rank}P_{\rm sign}=1,
$$

$$
\operatorname{rank}P_{\rm std}=4,
\qquad
\operatorname{rank}I_{\rm reg}=6.
$$

Therefore rank $5$ is available as

$$
I_{\rm reg}-P_{\rm triv}
$$

or as

$$
I_{\rm reg}-P_{\rm sign}.
$$

But this is also the problem: $S_3$ alone does not choose which line to exclude.

Likewise, a rank-$2$ standard copy exists, but it is not a central $S_3$
projector. Central $S_3$ gives a rank-$4$ standard isotypic block. The rank-$2$
middle coefficient requires a defect-polarized standard copy.

Thus the sidecar proves:

$$
(6,2,5)\ \hbox{is representation-theoretically available},
$$

but not

$$
(6,2,5)\ \hbox{is forced by }S_3\hbox{ alone}.
$$

Certainty: `C:9` for the projector ranks and non-uniqueness statements. `C:8`
for the finite availability audit.

## 7. What This Sidecar Fixes

This sidecar cleans up a previous conceptual conflation:

$$
\hbox{mass scalar coefficient}
\ne
\hbox{coherent current Clebsch}.
$$

The CKM side of `boundary_response` uses coherent amplitudes and phase
holonomy. The scalar mass side should use positive readouts. This explains why
the old up coefficient $\sqrt2$ was attractive but misplaced: it is natural as
a coherent enhancement, not as a top-normalized scalar weight.

The clean architecture is now:

$$
\hbox{depth scar}: \{0,2,6\}\ \hbox{and transfer powers},
$$

$$
\hbox{up scalar Clebsch}: \left(\frac14,\frac1{\sqrt2},1\right),
$$

$$
\hbox{down scalar baseline}: \left(1,\frac1{\sqrt3},\sqrt{\frac23}\right),
$$

$$
\hbox{down scalar candidate}: \left(1,\frac1{\sqrt3},\sqrt{\frac56}\right).
$$

Certainty: `C:8` for the cleaned role separation. `C:5` for the full
architecture until radial response and the common-scale quark mass study are
integrated.

## 8. Synthesis Verdict

The sidecar is not a closed scalar mass theory. It is a successful separation
and availability theorem.

The up sector is strong:

$$
N^3=0,\quad \exp(xN)
\quad\Rightarrow\quad
C_u(x)=\left(\frac{x^2}{2},x,1\right).
$$

It becomes the desired vector once $x=1/\sqrt2$ is supplied.

The down sector is honest:

$$
(6,2,4)
\quad\hbox{is the natural }S_3\hbox{ baseline},
$$

$$
(6,2,5)
\quad\hbox{is available and data-improved, but not forced}.
$$

The open burdens are precise:

1. Derive $x=1/\sqrt2$ from scalar/Higgs boundary repair normalization.
2. Derive or kill the defect-selection rule for the rank-$2$ standard copy.
3. Derive or kill the excluded one-dimensional line that gives rank $5$.
4. Decide baseline versus candidate only after the common-scale mass response is
   integrated.
5. Keep these scalar coefficients separate from CKM current-amplitude
   Clebsches.

My synthesis judgement: this sidecar should be kept, but with moderate claims.
It gives a clean up-sector conditional theorem and a rigorous down-sector
availability theorem. It does not yet derive the down candidate or the scalar
normalization.
