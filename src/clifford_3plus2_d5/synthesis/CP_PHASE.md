# CP and phase slots

The `cp` sidecar is a constraint sidecar, not a flavor mechanism. It asks
whether the already-built carrier,
$
\text{BCC Dirac walk}\otimes S_+(\mathrm{Cl}(0,10),J),
$
contains a CP-breaking, CPT-preserving slot before any phenomenological fit is
attempted.

The answer is positive in two different senses:

1. the BCC walk has a lattice CP-odd correction at order $\epsilon$, localized
   in the cubic $T_{2g}$ irrep;
2. the Higgs-like internal charge-shift space has a universal half
   $J$-anticommuting component.

These are not the same statement. The first lives in the spacetime/QCA
anisotropy sector. The second lives in the internal complex-structure sector.
Neither one by itself is yet a CKM theorem.

## Standard-Model placement

In the Standard Model, observable quark CP violation appears after the Yukawa
matrices are diagonalized. The invariant content needs three generations and a
nonzero phase that cannot be removed by field rephasings. In the synthesis tree
this means:

- color belongs to $SU(3)_c\subset SU(4)$;
- weak chirality belongs to $SU(2)_L$;
- hypercharge belongs to $U(1)_Y$ with
  $Y=T_{3R}+(B-L)/2$ in physical normalization;
- family depth belongs to the residual boundary/scar sector;
- CP phase information must couple a phase slot to a three-generation flavor
  mechanism before it becomes CKM/PMNS phenomenology.

The `cp` sidecar supplies phase slots and constraints. It does not supply the
three-generation embedding, the Yukawa eigenvalues, or the measured Jarlskog
invariant. C:8 for the structural slots; C:4 for their present status as
phenomenological CP sources.

## Alpha slot: BCC walk CP

The discrete symmetry audit of the bare BCC Dirac walk gives

| symmetry | verdict |
|---|---|
| $P$ | exact |
| $T$ | broken |
| $C$ | broken |
| $PT$ | broken |
| $CP$ | broken |
| $CT$ | exact |
| $CPT$ | exact |

The same pattern survives the Yukawa-perturbed audit when the internal action
is trivial. The exact preserved set is $\{P,CT,CPT\}$, and the broken set is
$\{T,C,PT,CP\}$. C:8.

The continuum expansion sharpens the statement. Write
$
B(\epsilon,k)=\exp[-i\epsilon H_{\rm eff}(\epsilon,k)]
$
with
$
H_{\rm eff}(\epsilon,k)=H^{(0)}(k)+\epsilon H^{(1)}(k)+O(\epsilon^2),
\qquad H^{(0)}(k)=\alpha\cdot k.
$
The first correction is Hermitian and chirality-block diagonal:

$$
H^{(1)}(k)=
\begin{pmatrix}
H_R^{(1)}(k)&0\\
0&H_L^{(1)}(k)
\end{pmatrix},
\qquad
H_R^{(1)}(k)=H_L^{(1)}(k)=
\begin{pmatrix}
k_xk_y & k_z(k_x+ik_y)\\
k_z(k_x-ik_y)&-k_xk_y
\end{pmatrix}.
$$

The degree-two cubic-harmonic space decomposes as
$
\mathcal P_2=A_{1g}\oplus E_g\oplus T_{2g}.
$
The sidecar finds all of $H^{(1)}$ in the CP-odd $T_{2g}$ cell:
$
\|H^{(1)}_{\mathrm{CP\ odd},T_{2g}}\|^2=12,
$
with zero $A_{1g}$, zero $E_g$, and zero CP-even component. C:8.

This is the cleanest part of the CP sidecar. The leading Lorentz-restoration
correction is not a vague anisotropy; it is a definite $O_h$ representation.
The mechanism is the Bialynicki-Birula BCC hop structure, whose complex
coefficients $q_\pm=(1\pm i)/4$ choose a lattice time orientation while keeping
CPT exact. C:7 as a physical interpretation of the exact audit.

The group location is therefore:

$$
\text{alpha CP slot}\in O_h\text{ cubic lattice sector},\qquad
\text{irrep}=T_{2g}.
$$

It is not color, not hypercharge, and not family depth.

## Beta slot: internal $J$ misalignment

The beta audit is algebraic. Let $J^2=-I$ be the chosen complex structure on
the chiral internal carrier, and let $M$ be a Higgs-like charge-shift map. The
real-linear decomposition is

$$
M_c={1\over 2}(M-JMJ),\qquad
M_a={1\over 2}(M+JMJ),
$$

where $M_c$ commutes with $J$ and $M_a$ anticommutes with $J$. For the audited
dim-four Higgs-like map space, every one of the four basis elements and every
transpose-derived conjugate component satisfies

$$
{\|M_a\|_F^2\over \|M\|_F^2}={1\over 2},
\qquad
\|M_c\|_F^2=\|M_a\|_F^2.
$$

For the representative component, the exact numbers are
$
\|M\|_F^2=256
$
and
$
\|M_c\|_F^2=\|M_a\|_F^2=128.
$
C:8.

The corrected interpretation is crucial. This is a universal
$J$-anticommuting fraction, not a physical CP-violating fraction by itself.
The earlier label "CP-violating fraction" is killed as an overclaim. C:1 for
the old physical label; C:8 for the algebraic half-split.

The group location is:

$$
\text{beta phase slot}\in \operatorname{End}_{\mathbb R}(S_+)
\text{ relative to }J,
$$

with $J$ coming from the $\mathrm{Cl}(0,4)$ commutant used in the
Pati-Salam carrier. This is an internal complex-structure fact, adjacent to
the Higgs/Yukawa sector, not a finite-family theorem.

## Relation to the flavor sidecars

The depth-scar sidecar gives a path:
$
u-a-b.
$
That path has no loop and hence no intrinsic holonomy. A pure path tree cannot
be the source of CP phase. C:9.

The boundary-response sidecar has a quark current-amplitude phase
$
\delta_q=\arctan\sqrt 5
$
from a five-dimensional odd shell through
$
B_q=(I+i\Gamma_q)/\sqrt 6.
$
That is a sector-specific unitary phase in the proposed CKM current
construction. It should not be merged blindly with the alpha or beta CP slots.
At present, the correct statement is that the theory has several phase-bearing
objects that still need one common flavor action. C:5.

The radial-response sidecar reconstructs masses from positive spectral
measures. Positive Stieltjes data do not generate CP phase. CP must enter
through a complex coupling, holonomy, or internal $J$-misalignment before the
radial poles and residues are assigned. C:7.

## What survives

The strong surviving theorem is:

> The QCA carrier contains a CPT-preserving lattice CP slot at order
> $\epsilon$ in the $T_{2g}$ irrep of $O_h$, and the internal Higgs-like map
> space has an exact universal $1/2$ split into $J$-commuting and
> $J$-anticommuting parts.

Certainty: C:8.

The paper-level conjecture is:

> CKM-like CP violation may be obtained only when one of these phase slots is
> coupled to the family-depth and scalar/radial mass mechanisms in a
> rephasing-invariant three-generation construction.

Certainty: C:3.

The sidecar therefore belongs in the synthesis tree as a phase-constraint node:

$$
\text{QCA/BCC carrier}
\longrightarrow
\begin{cases}
O_h:T_{2g}\text{ lattice CP slot},\\
J\text{-misaligned internal Higgs slot},
\end{cases}
\longrightarrow
\text{possible CKM/PMNS phase after family embedding}.
$$

