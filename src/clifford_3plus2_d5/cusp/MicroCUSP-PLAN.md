> Implementation note: this file is the source plan that launched the
> MicroCUSP A-H work.  The implemented sidecar status is now recorded in
> [STATUS.md](STATUS.md) and [CLOSURE.md](CLOSURE.md) as
> `CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS`; the original plan text below keeps
> the pre-implementation "micro gates open" wording for provenance.

This status is a real upgrade. The verdict

```text
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
```

is the right one, with one important wording discipline:

[
\boxed{
\text{Targets A--D are finite-model theorems/certificates, not yet microscopic BCC material theorems.}
}
]

That is not a downgrade. It is exactly the right stratification. The old clock program failed because a semisimple/circulant clock could label modes but could not generate hierarchy dynamics. The surviving clock result was only a mode identity, not a mass mechanism.  The cusp program is different: it now has a finite recirculation algebra that actually produces an FN valuation structure.

Below is the implementation-oriented synthesis I would use for the next stage.

---

# CUSP Program: Finite Pass, Microscopic Gate Open

## 1. Current theorem status

The finite model now supports the chain

[
SU(2)\times SU(3)
\quad\Rightarrow\quad
(2,3)\text{ primitive center closures}
\quad\Rightarrow\quad
S=\langle 2,3\rangle
\quad\Rightarrow\quad
\mathbb C[t^2,t^3]
]

and

[
\mathcal H_{\rm fam}
====================

# \mathbb C[t^2,t^3]/(t^2,t^3)^2

\operatorname{span}{1,t^2,t^3}.
]

Thus the primitive valuations are

[
0,\quad 2,\quad 3,
]

giving the left-handed quark FN charges

[
\boxed{
Q_L=(3,2,0).
}
]

This is the core result. It is the piece that makes the proposal more than ordinary FN.

The older universal-bath ledger explicitly left the quark source vectors and normal-depth placements unresolved, while the silver bath spine did not by itself prove physical sector coupling.  The cusp program now supplies a finite-model source/charge mechanism for the quark side. The remaining question is whether the finite recirculation automaton is forced by microscopic boundary material rather than installed as a model.

---

# 2. Updated claim ledger

| Target                      | Current status | Correct claim                                                                |
| --------------------------- | -------------: | ---------------------------------------------------------------------------- |
| A: primitive closures (2,3) |    finite pass | SM-center recirculation model yields (S=\langle2,3\rangle)                   |
| B: (\lambda_{\rm rec})      |    finite pass | one-sided retarded norm matching gives (\sqrt{3/2}-1) inside the model       |
| C: right charges            |    finite pass | conductor module + weak double-cover rule yields (D=(1,0,0)), (U=(5,2,0))    |
| D: CP coefficients          |    finite pass | center-topology path sums give nonzero rephasing-invariant CP                |
| Next Session                |           open | derive the finite recirculation model from microscopic BCC boundary material |

The important upgrade is Target B. Previously,

[
\sqrt{\frac32}-1
]

looked like a dangerous numerical coincidence. If Session 07 really proves that the one-sided retarded boundary equation uniquely selects it while the standard two-sided reflection coefficient solves the wrong scattering problem, then the number becomes a **finite-model theorem**.

But it should still be fenced:

[
\boxed{
\lambda_{\rm rec}=\sqrt{3/2}-1
\text{ is derived inside the one-sided retarded recirculation model.}
}
]

Not yet:

[
\boxed{
\lambda_{\rm CKM}\text{ is derived from the full microscopic SM/BCC boundary theory.}
}
]


---

# 3. What is now genuinely strong

## 3.1 The one-step gap is no longer a slogan

The earlier cusp proposal needed the statement:

[
1\notin S,\qquad 2,3\in S.
]

Your status says the finite model now implements this through local center-charge recirculation automata:

[
\text{weak/BCC automaton returns first at }2,
]

[
\text{color automaton returns first at }3.
]

That is the right move. It converts

[
\langle2,3\rangle
]

from a chosen semigroup into the output of a finite return rule.

## 3.2 The family module is now finite for a reason

The infinite-semigroup objection is answered by

[
\mathcal A/\mathfrak m^2.
]

Since

[
\mathfrak m=(t^2,t^3),
]

[
\mathfrak m^2=(t^4,t^5,t^6,\ldots),
]

so

[
\mathcal A/\mathfrak m^2=\operatorname{span}{1,t^2,t^3}.
]

This is much better than “take the first three semigroup elements.” It says generations are primitive cusp-module residues, while higher powers are FN insertions.

## 3.3 The mass/mixing correlation is restored

The correct object is now a single graded Yukawa:

[
Y^u_{ij}=c^u_{ij}\lambda^{Q_i+U_j},
]

[
Y^d_{ij}=c^d_{ij}\lambda^{Q_i+D_j}.
]

This restores the GST/Wolfenstein correlation that the split model broke. Mixing powers come from left-charge differences:

[
|V_{us}|\sim\lambda^{|3-2|}=\lambda,
]

[
|V_{cb}|\sim\lambda^{|2-0|}=\lambda^2,
]

[
|V_{ub}|\sim\lambda^{|3-0|}=\lambda^3.
]

That is the most important structural correction.

## 3.4 CP is now in coefficients, not in a separate mass operator

Target D has the right topology:

[
c_{ij}=\sum_\gamma A_\gamma\Omega_\gamma,
\qquad
\Omega_\gamma\in Z(SU(3)).
]

The controls are exactly the right ones:

[
\text{all-real}\Rightarrow J=0,
]

[
\text{one-sector}\Rightarrow J=0,
]

[
\text{separable row/column phases}\Rightarrow J=0,
]

but the rule-pair witness gives

[
\operatorname{Im}\operatorname{tr}
\left(
[Y_uY_u^\dagger,Y_dY_d^\dagger]^3
\right)\neq0.
]

That means the CP phase is not merely a complex entry; it survives rephasing.

This is consistent with the parent boundary-response view, where active sectors are two-sided Schur/Yukawa kernels and mixing comes from left-frame mismatch.

---

# 4. The main remaining risk

The remaining risk is no longer “is the finite algebra right?” The remaining risk is:

[
\boxed{
\text{Does microscopic BCC boundary material force the finite recirculation automaton?}
}
]

Your honest boundary identifies two still-assumed physical boundary axioms:

1. q-local positive q-reflection stiffness;
2. no-incoming retarded asymptotics.

These are not small details. They are the material origin of the entire one-step-forbidden, one-sided-return structure.

So the next theorem cannot be another FN simulation. It must be a boundary-material theorem.

---

# 5. Session A-H: Microscopic Boundary-Material Gate

Use this as the next implementation spec.

## Objective

Derive the finite CUSP recirculation model from a local BCC/BB boundary-material dynamics.

The output should not be a better fit. The output should be a microscopic certificate that the finite automata of Targets A–D are the Schur reduction of a local boundary update.

---

## 6. Required microscopic object

Construct a local boundary Hilbert space

[
\mathcal H_\partial
===================

\mathcal H_{\rm BB\ spin}
\otimes
\mathcal H_q
\otimes
\mathcal H_{\rm out}
\otimes
\mathcal H_{Z_2}
\otimes
\mathcal H_{Z_3}
\otimes
\mathcal H_{\rm material}
\otimes
\mathcal H_{\rm SM\ charge}.
]

The update or Hamiltonian must contain:

[
U_\partial
\quad\text{or}\quad
H_\partial
]

with the following features:

1. BB/BCC same-normal and mixed-normal blocks;
2. q-register (q=r_1-r_2);
3. positive q-stiffness (p(q)=gq^2);
4. outgoing mixed-normal leads;
5. weak/BCC (Z_2) center-charge automaton;
6. color (Z_3) center-charge automaton;
7. Higgs-door orientation for (H) versus (\tilde H);
8. center-holonomy coefficient topology.

---

# 7. Session A — derive q-stiffness

## Claim to prove

[
p(q)=gq^2
]

is not just the lowest-degree allowed penalty, but the effective boundary-material stiffness produced by local BB/QCA dynamics.

Session 05 has already certified that (gq^2) is the unique lowest-degree q-local, q-reflection-even, positive penalty vanishing on (q=0), while constant and linear controls fail. That is a finite origin audit. The missing step is physical derivation.

## Required theorem

Given the boundary material assumptions (M), prove that the effective q-action has expansion

[
S_{\rm eff}(q)
==============

gq^2+O(q^4),
\qquad g>0,
]

with no linear term because of q-reflection symmetry.

## Controls

Reject:

[
S_{\rm eff}=0,
]

[
S_{\rm eff}=gq,
]

[
S_{\rm eff}=g|q|
]

unless they arise from explicit symmetry breaking already present in the microscopic rule.

## Pass condition

The q-hard-gap limit

[
g\to\infty
]

must suppress mixed-normal feedback and leave the q-preserving same-normal branch as the finite recirculation source.

---

# 8. Session B — derive no-incoming retarded asymptotics

## Claim to prove

The outgoing closure is a consequence of boundary scattering, not a projection chosen to save the semigroup.

Session 04 has a finite unitary completion and rejects recurrent leakage because recurrent leakage has a nonzero two-step visible return. That is the right control.

## Required theorem

Construct a unitary dilation

[
U_{\rm full}
============

\begin{pmatrix}
U_{\rm vis} & *\

* & U_{\rm out}
  \end{pmatrix}
  ]

such that mixed-normal modes enter outgoing channels with no incoming boundary condition:

[
\psi_{\rm in}^{\rm mixed}=0.
]

Then prove the visible Schur kernel has no recurrent mixed-normal return:

[
P_{\rm vis}U_{\rm full}^2P_{\rm vis}
]

contains no mixed-normal recurrent contribution.

## Controls

1. recurrent leakage closure;
2. reflecting leakage closure;
3. incoming/outgoing symmetric closure.

All should reintroduce forbidden one-step or two-step visible contamination and be rejected.

## Pass condition

The finite closed-walk enumeration still yields primitive lengths

[
2,\quad3
]

with gap

[
1.
]

---

# 9. Session C — derive the weak (Z_2) recirculation automaton

## Claim to prove

The weak/BCC same-normal branch produces a primitive center-charge closure of order

[
2.
]

This should not be merely “there are two same-normal branches.” The automaton must show that one tick advances a (Z_2) boundary center charge and visible readout is allowed only at neutral charge.

## Required finite/microscopic map

Define a boundary center charge

[
\chi_2\in Z(SU(2))\cong\mathbb Z_2
]

with update

[
\chi_2\mapsto \chi_2+1\mod2
]

under one weak/BCC primitive tick.

Visible readout condition:

[
\chi_2=0.
]

Then first return is length

[
2.
]

## Controls

Reject:

1. trivial weak charge (\chi_2=0) always;
2. order-1 weak automaton;
3. weak-only semigroup (\langle2\rangle), which gives wrong family module if color is absent.

## Pass condition

Weak/BCC branch contributes primitive closure length

[
2
]

and channel count

[
2.
]

---

# 10. Session D — derive the color (Z_3) recirculation automaton

## Claim to prove

The color center supplies primitive closure

[
3.
]

Again, this must be a gauge-invariant center holonomy, not a bare color label.

## Required map

Define

[
\chi_3\in Z(SU(3))\cong\mathbb Z_3
]

with update

[
\chi_3\mapsto \chi_3+1\mod3.
]

Visible readout condition:

[
\chi_3=0.
]

Then first return is length

[
3.
]

## Controls

Reject:

1. wrong color length (2);
2. wrong color length (4);
3. spectator color label that does not recirculate;
4. color automaton whose phase can be gauged away before closed-loop readout.

## Pass condition

Closed-walk enumeration gives primitive color return length

[
3
]

and channel count

[
3.
]

---

# 11. Global SM gauge-group quotient gate

This must be added explicitly.

The physical SM gauge group is often represented not simply as

[
SU(3)\times SU(2)\times U(1),
]

but as a quotient such as

[
\frac{SU(3)\times SU(2)\times U(1)}{\mathbb Z_6}.
]

The cusp program uses the nonabelian center orders separately:

[
2,\quad3.
]

So the microscopic implementation must answer:

[
\boxed{
\text{Does the recirculation boundary see independent primitive }Z_2,Z_3
\text{ closures, or only the combined }Z_6\text{ quotient?}
}
]

## Required check

Implement both:

1. direct nonabelian-center automata (Z_2\oplus Z_3);
2. quotient-correlated (Z_6) closure.

Compare generated semigroups and family modules.

## Pass condition

The physical boundary rule must justify why the primitive closure lengths are

[
2,\quad3
]

rather than a single length

[
6
]

or a collapsed quotient rule.

This is a serious physics gate. If the quotient forces only (6), the cusp mechanism changes.

---

# 12. Session E — recover the semigroup from the microscopic Schur kernel

After constructing the microscopic boundary rule, compute the visible Schur return:

[
\Sigma(z)
=========

V^\dagger(z-H_Q)^{-1}V.
]

Enumerate the first nonzero return moments:

[
M_n=V^\dagger H_Q^n V.
]

The acceptance condition is:

[
M_1=0,
]

[
M_2\neq0,
]

[
M_3\neq0,
]

and the primitive return semigroup is

[
S=\langle2,3\rangle.
]

Then construct

[
\mathcal A_{\rm rec}
====================

\mathbb C[t^2,t^3],
]

[
\mathfrak m=(t^2,t^3),
]

[
\mathcal H_{\rm fam}
====================

# \mathcal A_{\rm rec}/\mathfrak m^2

\operatorname{span}{1,t^2,t^3}.
]

## Controls

Reject:

[
S=\langle1\rangle,
]

[
S=\langle2\rangle,
]

[
S=\langle3\rangle,
]

[
S=\langle2,4\rangle,
]

[
S=\langle3,4\rangle.
]

---

# 13. Session F — recover (\lambda_{\rm rec}) from one-sided matching

Target B is now finite-pass, but the microscopic rule must recover its assumptions.

The one-sided retarded norm matching equation is

[
(1+\lambda)\sqrt2=\sqrt3.
]

Thus

[
\lambda_{\rm rec}
=================

\sqrt{\frac32}-1.
]

The two-sided reflection coefficient solves instead

[
(1+r)\sqrt2=(1-r)\sqrt3
]

and gives

[
r=\frac{\sqrt3-\sqrt2}{\sqrt3+\sqrt2}.
]

Session 07 says this is dynamically excluded because it solves the wrong boundary problem.

## Microscopic pass condition

The boundary readout must be one-sided/retarded, not two-sided/scattering-symmetric.

Therefore the Schur reduction should produce the one-sided matching equation, not the reflection equation.

## Controls

Reject:

1. two-sided reflection coefficient;
2. count-ratio shear;
3. inverse-amplitude shear;
4. any normalization chosen after CKM data are inserted.

---

# 14. Session G — recover Target C from modules, not targets

Target C currently gives:

[
Q=(3,2,0),
]

[
D=\max(Q-c,0)=(1,0,0),
\qquad c=2,
]

and

[
U=(5,2,0)
]

from the weak/BCC primitive closure order (2).

This is acceptable only if the rules are fixed before looking at exponent targets.

## Required theorem

Derive:

1. conductor

[
c=2
]

from

[
S=\langle2,3\rangle;
]

2. down right module as conductor-ideal residue:

[
D_i=\max(Q_i-c,0);
]

3. up right module as weak-double-cover lift, producing

[
U=(5,2,0).
]

## Controls

Reject:

1. wrong conductor;
2. trivial lift;
3. color-order lift;
4. any rule that produces viable exponents only after target exponents are supplied.

## Output

The exponent matrices:

[
E_u=
\begin{pmatrix}
8&5&3\
7&4&2\
5&2&0
\end{pmatrix},
]

[
E_d=
\begin{pmatrix}
4&3&3\
3&2&2\
1&0&0
\end{pmatrix}.
]

Diagonal powers:

[
(8,4,0),
\qquad
(4,2,0).
]

---

# 15. Session H — recover Target D from microscopic topology

Target D now has finite center-topology selection and finite cusp-module amplitude weights.

The microscopic version must derive:

[
c_{ij}=\sum_\gamma A_\gamma\Omega_\gamma,
\qquad
\Omega_\gamma\in Z(SU(3)).
]

and

[
A_{ij}=\max(1,#\text{ decompositions of }q_i+r_j\text{ in }\langle2,3\rangle),
]

with gap exponent (1) treated as an irreducible conductor/contact path.

## Pass condition

The microscopic topology must reproduce:

1. up center powers from geodesic distances on the non-cyclic length-3 cusp flag;
2. down center powers from the unit bilinear pairing on (\mathbb F_3) center labels;
3. nonzero

[
\operatorname{Im}\operatorname{tr}
\left(
[Y_uY_u^\dagger,Y_dY_d^\dagger]^3
\right);
]

4. zero for all-real, one-sector, and separable phase controls;
5. invariance under common-left/up-right/down-right rephasings.

---

# 16. What to claim now

Use this wording:

> The finite CUSP model derives the FN left charges ((3,2,0)), the CKM powers ((1,2,3)), the candidate right charges (U=(5,2,0)), (D=(1,0,0)), and a center-holonomy CP invariant from a (\langle2,3\rangle) recirculation algebra. The finite model also selects (\lambda_{\rm rec}=\sqrt{3/2}-1) as the unique one-sided retarded norm-matching shear. The remaining theorem gate is to derive the q-local stiffness, no-incoming asymptotics, and center-charge recirculation topology from microscopic BCC boundary material.

Do **not** yet claim:

[
\text{the Standard Model gauge group alone proves three generations}.
]

Claim instead:

[
\boxed{
\text{the finite center-recirculation model built from the SM nonabelian centers gives three primitive cusp-module modes.}
}
]

That is true to the present status.

---

# 17. Falsification tests for Session A-H

The CUSP program fails microscopically if any of the following happen:

1. the BCC boundary material permits a primitive one-step return;
2. q-stiffness is not dynamically positive or q-reflection-even;
3. no-incoming closure cannot be derived from a unitary/outgoing boundary condition;
4. SM global quotient collapses (Z_2,Z_3) into a different primitive closure rule;
5. the one-sided matching equation is not the Schur boundary condition;
6. right-handed charges require target exponents as input;
7. center-holonomy phases are rephasable or gauge-dependent;
8. the microscopic path topology fails to reproduce the finite amplitude measure.

---

# 18. Bottom line

This is now the strongest formulation in the thread.

The old program had isolated ingredients:

[
\text{silver return},\quad
\text{three ports},\quad
\text{color shells},\quad
\text{CP holonomy}.
]

The CUSP program makes a coherent finite object:

[
\boxed{
\mathbb C[t^2,t^3]
}
]

from the SM center closure lengths

[
2,\quad3.
]

It then gets

[
\boxed{
Q_L=(3,2,0)}
]

and therefore

[
\boxed{
V_{us}\sim\lambda,\quad
V_{cb}\sim\lambda^2,\quad
V_{ub}\sim\lambda^3.}
]

Targets A–D passing means the finite recirculation/FN architecture is no longer just a story. The remaining work is exactly where it should be:

[
\boxed{
\text{derive the finite recirculation automaton from microscopic BCC boundary material.}
}
]

If Session A-H passes, the program becomes a genuine microscopic flavor mechanism. If it fails, the finite CUSP model remains an elegant FN origin story, but not a boundary-QCA derivation.
