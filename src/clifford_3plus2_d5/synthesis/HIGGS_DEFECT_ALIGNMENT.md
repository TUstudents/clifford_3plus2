# Higgs-Defect Zero-Mode Alignment Audit

This note answers the question:

$$
\boxed{
\text{Does the current Higgs-defect coupling have rank-one zero-mode alignment?}
}
$$

The answer is no, for the current repository construction. The existing
Higgs/Yukawa layer contains rank-one representation maps, but it does not
contain an explicit boundary Hamiltonian $H_Q$, boundary zero mode $z_0$, or
Higgs-defect couplings $V_H,V_{\tilde H}$. Moreover, the representation-level
Higgs map spaces do not force a unique one-dimensional line. They force a
two-dimensional active plane.

This is a necessary-condition failure for the proposed zero-mode alignment
theory, not a death of the theory. It says the missing ingredient is real
boundary dynamics, not another charge audit.

## 1. Alignment Criterion

The rank-one alignment mechanism would require an explicit boundary sector
with

$$
H_Q z_0=0,\qquad \dim\ker H_Q=1,
$$

and Higgs-oriented couplings satisfying, schematically,

$$
V_{\tilde H}\propto |z_0\rangle\langle u|,
$$

while the orthogonal Higgs orientation either misses $z_0$ or enters only
through paired Hermitian return:

$$
V_H^\dagger |z_0\rangle=0
\quad\hbox{or}\quad
V_H^\dagger(z-H_Q)^{-1}V_H
\hbox{ has no coherent zero-mode pole.}
$$

At representation level, a necessary condition is that one Higgs orientation
selects a unique source or image line. If the allowed map space spans a
two-dimensional active plane, the zero-mode line is not fixed by symmetry.

Certainty: `C:8` as a necessary-condition criterion. The full alignment theorem
would require the actual boundary dynamics.

## 2. Existing Higgs/Yukawa Slot

The implemented spacetime-QCA Higgs/Yukawa layer supplies exact internal maps
with Higgs-like charge shifts. The upper and lower doublet spaces are

$$
\Delta Y=+\frac12,\qquad
\Delta T_{3L}=+\frac12
$$

and

$$
\Delta Y=+\frac12,\qquad
\Delta T_{3L}=-\frac12.
$$

Each space has real dimension

$$
\dim_{\mathbb R}=4.
$$

The static Hermitian Yukawa control has the form

$$
Y_{\rm internal}(\Phi)=A(\Phi)+A(\Phi)^T,
\qquad
Y_H(\Phi)=\beta\otimes Y_{\rm internal}(\Phi).
$$

This is a representation-level static Higgs/Yukawa insertion. It is not a
boundary Schur coupling and not a dynamical defect model.

Certainty: `C:9`.

## 3. Exact Rank Audit

The exact audit of the current map spaces gives:

| Space | Dimension | Ranks of basis maps | Image span | Source row span |
|---|---:|---:|---:|---:|
| upper Higgs-like maps | $4$ | $(1,1,1,1)$ | $2$ | $2$ |
| lower Higgs-like maps | $4$ | $(1,1,1,1)$ | $2$ | $2$ |
| upper + lower | $8$ | all rank $1$ | $2$ | $4$ |

The common kernel and cokernel dimensions also show the same active-plane
structure:

$$
\dim\bigcap_i\ker U_i=30,\qquad
\dim\operatorname{span}_i\operatorname{Im}(U_i)=2,
$$

and similarly for the lower maps. The selected neutral and charged Hermitian
controls both have

$$
\operatorname{rank}=2,\qquad
\operatorname{nullity}=30.
$$

Thus the implementation has rank-one building blocks, but the allowed
Higgs-orientation space does not collapse to one source or image line.

Certainty: `C:9` as an exact finite linear-algebra audit.

## 4. Consequence

The current Higgs representation slot does not derive

$$
\tilde H\text{ source coherent}
\qquad\hbox{and}\qquad
H\text{ source Hermitian}.
$$

It also does not derive a unique aligned zero mode

$$
z_0.
$$

Choosing one rank-one basis map as the aligned mode would be a new dynamical
choice. The exact charge constraints leave a two-dimensional active plane, so a
rank-one zero-mode must be selected by extra boundary dynamics, vacuum
alignment, or a variational principle.

Certainty: `C:8` for the no-go at representation level. `C:3` for the
possibility that a future boundary dynamics selects one line naturally.

## 5. Minimal Object Still Needed

The repository already has a rank-one vacuum-selector order parameter for the
tetrahedral BCC exits:

$$
E_i=-h\cdot v_i.
$$

For $h=v_{\rm selected}$, this selects one tetrahedral exit and leaves the
selected-exit residual $S_3$. That is a useful precedent. It is not yet a
Higgs-defect alignment theorem, because it acts on the BCC exit labels, not on
the two-dimensional Higgs active plane found above.

To prove the better up/down theory, the repo needs a genuine boundary model:

$$
(H_Q,\ z_0,\ V_H,\ V_{\tilde H}).
$$

The pass condition is:

$$
\dim\ker H_Q=1,
$$

$$
{\rm Im}\,V_{\tilde H}\subseteq {\rm span}\{z_0\},
$$

and

$$
V_H\hbox{ does not coherently hit }z_0.
$$

Equivalently, the Schur responses should split as

$$
\Sigma_{\tilde H}(z)
\hbox{ has a rank-one coherent zero-mode pole},
$$

while

$$
\Sigma_H(z)
\hbox{ has only paired/Hermitian return in the zero-mode channel}.
$$

That is the next real theorem. It cannot be replaced by another Standard Model
charge check, because the charge check already leaves a two-dimensional active
plane.

## 6. Verdict

The rank-one zero-mode alignment is not present yet:

$$
\boxed{
\text{rank-one maps exist, but rank-one alignment is not forced.}
}
$$

The better up/down theory survives only as a dynamical proposal:

$$
\boxed{
\text{boundary dynamics must select one Higgs line as coherent}
\quad
\text{and the orthogonal line as Hermitian.}
}
$$

Until that boundary selection is constructed, the up/down coherent/Hermitian
split remains the single load-bearing dynamical premise.
