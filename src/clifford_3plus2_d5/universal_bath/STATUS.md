# universal_bath - Status

**Status**: Session 08A/08B implemented.

## Current theorem-level status

Session 01 proves the common bath spine:

```text
finite Lanczos head + period-one retarded tail
```

with the universal tail

```text
t(z) = (z - sqrt(z^2 - 4)) / 2
t(2 sqrt(2)) = sqrt(2) - 1
```

The silver value is inherited from the BCC/BB band-edge theorem. The sidecar
does not yet prove that any physical sector source has the required finite
head or nonzero coupling to the universal tail.

Session 01 also fixes the reduction taxonomy:

```text
positive one-sided       -> scalar Jacobi
real non-positive shell  -> indefinite look-ahead Jacobi
chiral/unitary phase     -> CMV/OPUC
```

The CMV/OPUC free-tail criterion is `alpha_n = 0` after the finite head.

## Session 02 source dictionary

Session 02 upgrades the source-dictionary scaffold into a write-once source
ledger.  The BB same-normal first-hop identity is:

```text
B_+^* B_+ + B_-^* B_- = I/2
```

Hence every normalized frozen $q=0$ source has radial first-hop survival weight

```text
1/2
```

The frozen anchors are:

```text
neutrino_collective_u       u, depth 1, scalar Jacobi
neutrino_edge_b             b, depth 0, scalar Jacobi
charged_lepton_active_e1    e1, depth 2, CMV/OPUC
```

The charged-lepton port is exactly:

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

with no `b` component.

The quark source records are present but deliberately unresolved:

```text
up_quark_boundary_source
down_quark_boundary_source
```

Their SM charge anchors are known, but the universal-bath BCC source vectors and
normal-depth placements are not yet derived.  Session 02 therefore refuses to
freeze them.

## Session 03 neutrino product bath

Session 03 applies the universal bath spine to the frozen neutrino anchors:

```text
neutrino_collective_u = u, depth 1
neutrino_edge_b       = b, depth 0
```

with the product hidden bath

```text
H_Q = H_chain tensor I_family
```

For the product states `|0> tensor u` and `|0> tensor b`, the certificate
checks:

```text
<u|H_Q^k|b> = 0,      k = 0..4
<u|H_Q^k|u> = <b|H_Q^k|b>,      k = 0..4
```

The semi-infinite chain Weyl readout is the universal tail:

```text
m(z) = t(z) = (z - sqrt(z^2 - 4)) / 2
```

so the normalized neutrino response is:

```text
Sigma_hat_nu(z) = m(z)^2 P_u + P_b
```

At `z = 2 sqrt(2)`:

```text
m(z) = epsilon = sqrt(2) - 1
Sigma_hat_nu = epsilon^2 P_u + P_b
m2/m3 = epsilon^2
Delta m^2_21 / Delta m^2_31 = epsilon^4
```

Rank-one, wrong-source, and alternate-tail controls are rejected.

## Session 04 charged-lepton CMV head

Session 04 applies the CMV/OPUC reduction to the frozen charged-lepton anchor:

```text
charged_lepton_active_e1 = e1, depth 2
```

with

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

The two-step universal-tail leakage gives:

```text
sin(theta_e) = sqrt(3/2) epsilon^2
sin^2(theta_e) = (3/2) epsilon^4
```

The upstream primitive boundary holonomy supplies the phase:

```text
SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2
phase = exp(-5 pi i / 12)
```

The finite Verblunsky coefficient is:

```text
alpha_e = sin(theta_e) exp(-5 pi i / 12)
```

and the two-state CMV/Givens head is exactly unitary.  After the finite head,
the CMV/OPUC tail is free:

```text
alpha_n = 0
```

Depth-one, depth-three, b-leakage, and holonomy subword controls are rejected.

## Session 05 charged-lepton 2/9 torsion gate

Session 05 answers the specific torsion question using the same frozen source:

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

The occupation weights are:

```text
p_a = 2/3
p_u = 1/3
p_b = 0
```

Therefore the incoherent source-transition weight is:

```text
p_a p_u = 2/9
```

The coherent amplitude control is:

```text
sqrt(2)/3
```

and is rejected as the torsion value. Equal-weight and one-port controls are
also rejected. Thus `2/9` is derived as a source-geometry occupation moment,
not as the full CMV phase or charged-lepton mass angle.

## Session 06 up-quark nilpotent CMV head

Session 06 proves the conditional up finite-head algebra. The up source is
still deliberately unresolved in the write-once dictionary:

```text
up_quark_boundary_source: unresolved_not_frozen
```

The BB same-normal survival identity gives:

```text
B_+^* B_+ + B_-^* B_- = I/2
```

so the injection amplitude is:

```text
x = sqrt(1/2) = 1/sqrt(2)
```

The length-3 nilpotent Taylor head gives:

```text
exp(xN) = I + xN + x^2 N^2 / 2
(x^2/2, x, 1) = (1/4, 1/sqrt(2), 1)
```

The geometric control at the same injection gives `(1/2,1/sqrt(2),1)` and is
rejected. The old `sqrt(2)` charm control is also rejected. The finite CMV head
`(1/sqrt(2),1/4)` lies inside the unit disk and is followed by the free tail
`alpha_n=0`.

## Session 07 down-quark indefinite Jacobi head

Session 07 implements the count-level real/symmetric down head and keeps the
source unresolved:

```text
down_quark_boundary_source: unresolved_not_frozen
```

The three-port residual shell gives:

```text
(3,1,2)/3 -> (1, 1/sqrt(3), sqrt(2/3))
```

The six-element regular S3 shell gives the baseline:

```text
(6,2,4)/6 -> (1, 1/sqrt(3), sqrt(2/3))
```

and can host the candidate:

```text
(6,2,5)/6 -> (1, 1/sqrt(3), sqrt(5/6))
```

The three-port shell cannot host the candidate because `5/6 * 3 = 5/2` is not
an integer rank. The regular shell can host it, but S3 does not force it:
rank 2 requires defect polarization and rank 5 is not unique. The bottom
`4 -> 5` line remains an open selection theorem.

## Session 08A quark height-door audit

Session 08A separates the exact SM charge door from the repair-mode premise.
The electroweak charge checks are:

```text
-Y(Q_L) + Y(H_tilde) + Y(u_R) = -1/6 - 1/2 + 2/3 = 0
-Y(Q_L) + Y(H)       + Y(d_R) = -1/6 + 1/2 - 1/3 = 0
```

and both neutral Higgs components obey:

```text
Q_em = Y + T3 = 0
```

The declared height assignment is:

```text
up   -> oriented length-3 nilpotent N
down -> Hermitian closure Delta_N
```

where `N^3=0`, `N^2 != 0`, and

```text
Spec(Delta_N) = {0,1,3}
```

The negative control is important: swapping the repair modes is still
hypercharge-allowed. Thus the coherent-up / Hermitian-down split is not derived
from electroweak charges alone; it remains a named height-dynamics premise.

## Session 08B quark color-lift audit

Session 08B separates visible color covariance from hidden color-return
multiplicity.

The fixed-color control

```text
diag(1,0,0)
```

does not commute with `SU(3)_c` and is rejected.

The color-scalar spectator embedding preserves visible color but stays on the
three-port shell:

```text
(3,1,2)/3 -> (1, 1/sqrt(3), sqrt(2/3))
```

The active hidden color-return lift also preserves visible color, while its
hidden shell reaches:

```text
1_direct + 2_BCC + 3_color
```

with breakdown:

```text
even_direct = 1
bcc_odd     = 2
color_odd   = 3
odd_total   = 5
total       = 6
```

It reproduces the regular-S3 baseline and makes the rank-five candidate
available:

```text
(6,2,4)/6 -> (1, 1/sqrt(3), sqrt(2/3))
(6,2,5)/6 -> (1, 1/sqrt(3), sqrt(5/6))
```

Gauge covariance alone rejects fixed color but does not select active over
spectator. The microscopic active-return selection rule remains open.

## Verdict summary

- Session 01 verdict: **UNIVERSAL_BATH_SPINE_PASS**.
- Session 02 verdict: **SOURCE_DICTIONARY_CORE_PASS**.
- Session 03 verdict: **NEUTRINO_PRODUCT_BATH_CORE_PASS**.
- Session 04 verdict: **CHARGED_LEPTON_CMV_HEAD_PASS**.
- Session 05 verdict: **CHARGED_LEPTON_TORSION_2_OVER_9_PASS**.
- Session 06 verdict: **UP_NILPOTENT_CMV_HEAD_CONDITIONAL_PASS**.
- Session 07 verdict: **DOWN_INDEFINITE_JACOBI_HEAD_CONDITIONAL_PASS**.
- Session 08A verdict: **QUARK_HEIGHT_DOOR_AUDIT_CONDITIONAL_PASS**.
- Session 08B verdict: **QUARK_COLOR_LIFT_AUDIT_CONDITIONAL_PASS**.

## Meaning

The sidecar has converted "universal silver tail" into exact Schur/Jacobi/CMV
algebra, frozen the lepton-side source anchors that are supported upstream,
proved the neutrino core inside the product half-line bath, built the
charged-lepton finite CMV head, derived the charged-lepton `2/9` torsion as an
occupation moment, implemented the conditional up nilpotent finite head, and
implemented the conditional down count-level Jacobi head. It has now also
isolated the two quark-source preconditions: the height-door premise and the
color-lift/active-return bit.

The current open gates are now sharper:

- derive the product factorization from a microscopic BCC/QCA boundary graph;
- derive the charged-lepton holonomy selection dynamics beyond the `2/9`
  occupation moment;
- freeze the up/down quark BCC source vectors without flavor data;
- derive or replace the height-dynamics rule that maps `H_tilde` to the
  oriented nilpotent and `H` to the Hermitian closure;
- derive or replace the active hidden color-return rule that selects the
  regular six-channel shell over the spectator three-port shell;
- derive or kill the down rank-5 bottom-line selection rule;
- assemble mixing from Krylov/CMV basis overlaps.
