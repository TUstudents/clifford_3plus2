# boundary_response — Status

**Status**: V1 through V19 implemented.

- V1 verdict: **BOUNDARY_CORE_KILL_UNBROKEN_K3**.
- V2 verdict: **FRAMED_STERILE_EFFECTIVE_PASS**.
- V3 verdict: **EXPLICIT_HQ_CONVERGENCE_ONLY**.
- V4 verdict: **IMPEDANCE_FREE_PARAMETER**.
- V5 verdict: **PRODUCT_STERILE_CONVERGENCE_PASS**.
- V6 verdict: **PRODUCT_STERILE_LIMIT_PASS**.
- V7 verdict: **CHARGED_LEPTON_LEAKAGE_PASS**.
- V8 verdict: **LEPTONIC_PHASE_WORD_CONDITIONAL_PASS**.
- V9 verdict: **PMNS_CONDITIONAL_ASSEMBLY_PASS**.
- V10 verdict: **LEPTONIC_PHASE_WORD_DERIVED_PASS**.
- V11 verdict: **QUARK_BOUNDARY_SHELL_Q1_PASS**.
- V12 verdict: **QUARK_TRANSFER_HIERARCHY_Q2_PASS**.
- V13 verdict: **QUARK_CLEBSCH_Q3_PASS**.
- V14 verdict: **CKM_CONDITIONAL_ASSEMBLY_PASS**.
- V15 verdict: **QUARK_COIN_RIGIDITY_THEOREM_PASS**.
- V16 verdict: **PRIMITIVE_ERGODICITY_NO_GO_PASS**.
- V17 verdict: **CHIRAL_BOUNDARY_NORMALIZATION_NO_GO_PASS**.
- V18 verdict: **ALGEBRAIC_INTERTWINER_FREE_NORM_KILL**.
- V19 verdict: **MAX_ENTROPY_PRIMITIVE_ERGODICITY_CONDITIONAL_PASS**.

The residual transfer recurrence gives the desired exact invariant:

```text
epsilon = sqrt(2) - 1
epsilon^4 = 17 - 12 sqrt(2)
```

However, the proposed neutrino core operator

```text
K_nu = epsilon^2 P_u + P_b
```

is not invariant under full residual `S_3`; it preserves only the
selected-port `S_2`. The explicit finite `S_3`-equivariant `K_3` tail has an
`S_3`-invariant Schur self-energy, so it cannot produce `K_nu`.

## Meaning

This does not kill the broader boundary-response program. It kills the
strongest unframed `K_3`-tail version of the neutrino core. Later gates derive
the framed neutrino core and assemble conditional PMNS and CKM textures. The
CKM result is conditional on the Q1-Q3 quark boundary-shell model.

## V2 framed sterile result

The corrected effective ansatz now derives the two response directions from
local incidence vectors rather than inserting the target coupling:

```text
(1,1,1) -> u
(0,1,-1) -> b
```

The radial mode `a` is absent from both incidence maps. One extra residual
transfer depth gives:

```text
g_u/g_b = epsilon
```

With equal low-energy sterile return normalization and zero cross-return, the
assembled effective response is:

```text
Sigma_eff ∝ epsilon^2 P_u + P_b
```

This is an effective framed-boundary pass, not a full microscopic proof. The
equal-return and zero-cross-return assumptions remain explicit at V2.

## Next gate

Replace the effective return ansatz with an explicit finite or semi-infinite
sterile boundary Green-function model `H_Q` whose Schur complement produces
the same equal-return structure.

## V3 explicit finite-shell result

The finite transfer-chain Green function supports the transfer-depth part:

```text
G(1,0) / G(0,0) -> epsilon
```

at the transfer probe `z = 2 sqrt(2)`.  The convergence is monotone in shell
depth and reaches the default audit tolerance by `N = 10`.

However, the raw shell-coupled Schur response still fails the return
normalization diagnostic.  Therefore V3 does not yet replace the V2 equal-return
ansatz.  It proves only that the explicit chain can derive the transfer
amplitude.

## V4 impedance catalog result

The minimal endpoint catalog finds no untuned local load that forces equal
`u`/`b` return normalization.  A one-site matched load can reproduce the needed
impedance only by solving for a sector-specific endpoint scalar:

```text
edge_energy = z_transfer - 1 / inferred_collective_return
```

Therefore the current impedance status is `IMPEDANCE_FREE_PARAMETER`, not an
impedance-matching theorem.  The theory still needs either a stronger symmetry,
a less ad hoc boundary completion, or an exact semi-infinite Weyl-function
construction that fixes the endpoint impedance without tuning.

## V5 product sterile-tail result

Mechanism B replaces the two independent endpoint returns by a product bath:

```text
H_Q = H_chain ⊗ I_family
```

The residual family labels live inside the unresolved sector:

```text
|s_u> = |head> ⊗ |u>
|s_b> = |head> ⊗ |b>
```

This makes the `u` and `b` sterile returns equal by tensor structure and makes
their cross-return vanish by family-factor orthogonality.  The remaining
asymmetry is the derived finite-chain transfer amplitude:

```text
amp_N = G_chain(1,0) / G_chain(0,0) -> epsilon
```

The normalized finite response is therefore:

```text
Sigma_N / <head|G_chain|head> = amp_N^2 P_u + P_b
```

At the default `N = 10` audit depth this converges to
`epsilon^2 P_u + P_b` within tolerance, giving
`PRODUCT_STERILE_CONVERGENCE_PASS`.

The sidecar also includes the required negative control: removing the family
factor and coupling both channels to one sterile head gives a rank-one response
with `u`/`b` cross terms.  That model is rejected.

V5 supports the neutrino core gate only.  PMNS and CKM remain parked until
charged-lepton and quark boundary shells are derived explicitly.

## V6 exact Weyl-function theorem

V6 replaces finite-shell convergence with the semi-infinite chain Weyl
function.  For the unit nearest-neighbor half-line chain, the head Green
function is the decaying solution of:

```text
m(z) = 1 / (z - m(z))
```

Equivalently:

```text
m(z) = (z - sqrt(z^2 - 4)) / 2
```

with branch fixed by `m(z) ~ 1/z` at infinity.  At the transfer probe:

```text
z = 2 sqrt(2)
m(z) = sqrt(2) - 1 = epsilon
```

The exact product sterile response is:

```text
Sigma(z) = m(z) [m(z)^2 P_u + P_b]
Sigma(z) / m(z) = m(z)^2 P_u + P_b
```

Therefore at `z = 2 sqrt(2)`:

```text
Sigma / m = epsilon^2 P_u + P_b
```

This upgrades V5 from a finite-shell convergence audit to an exact
semi-infinite limit theorem.  The resulting neutrino-core ratios are:

```text
m_2 / m_3 = epsilon^2
Delta m^2_21 / Delta m^2_31 = epsilon^4
```

V6 does not derive PMNS or CKM.  Those remain parked until explicit
charged-lepton and quark boundary shells are constructed.

## V7 charged-lepton leakage gate

V7 audits only Assumption L1: the charged-lepton/Higgs selected port and the
two-step leakage scalar.  The selected active port is:

```text
e1 = (1,0,0)
```

In the residual `(a,u,b)` basis:

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

with no `b` component.  Reusing the exact V6 Weyl transfer:

```text
m(z_transfer) = epsilon
```

the two-step leakage is:

```text
epsilon^2
```

and the geometric leakage condition gives:

```text
sqrt(2/3) sin(theta_e) = epsilon^2
sin(theta_e) = sqrt(3/2) epsilon^2
sin^2(theta_e) = (3/2) epsilon^4
```

Depth-one and depth-three controls fail, and synthetic `b` leakage is detected.
The verdict is `CHARGED_LEPTON_LEAKAGE_PASS`.

V7 does not derive the leptonic phase word and does not assemble PMNS.  The
next gate is the spin-Coxeter-Schur boundary word:

```text
W_e = -q_A3 q_A2
```

## V8 leptonic phase-word gate

V8 audits only the exact arithmetic of the proposed spin-Coxeter-Schur word:

```text
W_e = -q_A3 q_A2
```

The scalar phase inputs are:

```text
q_A3 = exp(i pi/4)
q_A2 = exp(i pi/3)
Schur sign = exp(i pi)
```

The raw phase angle is:

```text
1 + 1/4 + 1/3 = 19/12
```

whose principal representative is:

```text
-5/12
```

Thus:

```text
W_e = exp(-i 5 pi / 12)
```

No-Schur, `A3`-only, and `A2`-only controls do not match this phase, and the
CP-conjugate branch gives `+5/12`.

The verdict is `LEPTONIC_PHASE_WORD_CONDITIONAL_PASS`.  This verifies the
conditional phase arithmetic only.  It does not derive the boundary loop that
selects the full word.

## V9 conditional PMNS assembly

V9 assembles the already-gated leptonic ingredients:

```text
U_PMNS = R_e^\dagger U_TBM
```

where `U_TBM` is the residual `(a,u,b)` basis, V7 supplies

```text
sin(theta_e) = sqrt(3/2) epsilon^2
```

and V8 supplies the conditional phase branch

```text
phi_e = -5 pi / 12
```

with the CP-conjugate branch `+5 pi / 12`.  The exact and numerical output is:

```text
sin^2(theta13) = (3/4) epsilon^4
sin^2(theta12) ~= 0.304610
sin^2(theta23) ~= 0.488712
delta_CP ~= 261.6 degrees
delta_CP_conjugate ~= 98.4 degrees
```

The verdict is `PMNS_CONDITIONAL_ASSEMBLY_PASS`.  This is not a theorem-level
PMNS derivation because the V8 boundary-loop word selection remains
conditional.  CKM remains parked.

## V10 leptonic boundary-loop holonomy gate

V10 implements the missing V8 selection layer as a minimal boundary-loop
holonomy model.  The admissible charged-lepton loop must contain exactly one
Schur return, one parent `A3` spin lift, and one residual `A2` spin lift in the
oriented primitive order:

```text
SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2
```

The selected word has:

```text
raw angle = 1 + 1/4 + 1/3 = 19/12
principal angle = -5/12
phase = exp(-i 5 pi / 12)
```

The candidate catalog has exactly one admissible primitive word.  The no-Schur,
single-factor, two-factor, reversed-order, and duplicated-cover controls are
rejected.  The verdict is:

```text
LEPTONIC_PHASE_WORD_DERIVED_PASS
```

This upgrades the V8 word selection inside the boundary-loop holonomy model.
It does not construct a quark boundary shell.  CKM remains parked.

## V11 quark primitive boundary-shell gate

V11 implements the first quark-sector gate, Q1.  The primitive shell is:

```text
S_q = 1_even + 5_odd = 1_direct + (2_BCC + 3_color)
```

The five odd channels are represented by exact `Cl_5` generators:

```text
gamma_A^2 = I
{gamma_A, gamma_B} = 0 for A != B
Gamma_q = sum_A gamma_A
Gamma_q^2 = 5 I
```

The flat primitive coin is:

```text
B_q = (I + i Gamma_q) / sqrt(6)
```

and is exactly unitary.  The positive-branch scalar phase is:

```text
phase factor = (1 + i sqrt(5)) / sqrt(6)
delta_q = atan(sqrt(5))
```

Missing-color, non-flat, and commuting-generator controls are rejected.  The
verdict is:

```text
QUARK_BOUNDARY_SHELL_Q1_PASS
```

This derives only the primitive shell count, Clifford closure, flat coin, and
phase source.  CKM magnitudes remain parked until Q2/Q3 gates derive the
transfer-depth hierarchy and color/BCC Clebsches.

## V12 quark transfer-depth hierarchy gate

V12 implements the second quark-sector gate, Q2.  The ordered family-depth
embedding is:

```text
family 1 depth = 0
family 2 depth = 2
family 3 depth = 6
```

Transition depth is the absolute depth difference, so:

```text
1<->2 depth = 2
2<->3 depth = 4
1<->3 depth = 6
```

and the raw transfer amplitudes are:

```text
A_12 = epsilon^2
A_23 = epsilon^4
A_13 = epsilon^6
```

Odd-depth, non-additive direct-depth, and permuted-label controls are rejected.
The verdict is:

```text
QUARK_TRANSFER_HIERARCHY_Q2_PASS
```

This derives only the raw transfer powers.  CKM magnitudes remain parked until
Q3 derives the color-singlet return factor and BCC symmetric/antisymmetric
Clebsches.

## V13 quark color/BCC Clebsch gate

V13 implements the final quark prerequisite gate, Q3.  With exact SU(3)
fundamental generators:

```text
T^A = lambda^A / 2
```

the color-singlet return contraction is:

```text
sum_A T^A T^A = (4/3) I_3
```

The normalized Cabibbo leakage and two BCC path factors are:

```text
L_12 = epsilon^2 / sqrt(1 + epsilon^4)
C_23 = sqrt(2)
C_13 = 1 / sqrt(2)
```

Raw-generator, missing-color-generator, incoherent-path, and
unnormalized-antisymmetric controls are rejected.  The verdict is:

```text
QUARK_CLEBSCH_Q3_PASS
```

This derives the color and BCC prefactors needed for CKM magnitudes.  CKM
matrix assembly remains parked for V14.

## V14 conditional CKM assembly

V14 assembles the passed Q1-Q3 quark gates:

```text
s_12 = (4/3) epsilon^2 / sqrt(1 + epsilon^4)
s_23 = sqrt(2) epsilon^4
s_13 = epsilon^6 / sqrt(2)
delta_q = atan(sqrt(5))
```

using:

```text
V_CKM = R_23(theta_23) R_13(theta_13, delta_q) R_12(theta_12)
```

The audit gives:

```text
s_12 ~= 0.225469
s_23 ~= 0.041631
s_13 ~= 0.003571
delta_q ~= 65.905 degrees
J_q ~= 2.98e-5
```

and the approximate magnitude matrix:

```text
|V_CKM| ~= [[0.97425, 0.22547, 0.00357],
           [0.22533, 0.97340, 0.04163],
           [0.00858, 0.04090, 0.99913]]
```

The verdict is `CKM_CONDITIONAL_ASSEMBLY_PASS`.  This is conditional on the
Q1-Q3 primitive quark boundary-shell model, not a microscopic derivation of
that shell from the QCA update.

## V15 primitive quark coin rigidity theorem

V15 proves the exact isotropic one-parameter family behind the V11 quark coin:

```text
B(r) = (I + i r Gamma_q) / sqrt(1 + 5 r^2)
Gamma_q^2 = 5I
```

For every real ratio `r`, this coin is unitary and its positive-branch phase is:

```text
delta(r) = atan(r sqrt(5))
```

Therefore:

```text
delta_q = atan(sqrt(5))  iff  r = 1
```

The verdict is `QUARK_COIN_RIGIDITY_THEOREM_PASS`.  This is useful negative
information: unitarity and Clifford closure do not force the CKM phase by
themselves.  They force a one-parameter family, and the observed V11 phase is
equivalent to adding flat primitive boundary ergodicity.

## V16 primitive ergodicity no-go

V16 computes invariant subspaces for the primitive shell.  If symmetry
preserves the split:

```text
1_even ⊕ 5_odd
```

and acts transitively on the odd shell, the invariant amplitude space is:

```text
span{|even>, |odd_1> + ... + |odd_5>}
```

Thus odd amplitudes are forced equal, but the even/odd ratio remains free.  The
non-flat ratios `r = 1/2` and `r = 2` satisfy the same parity-preserving
symmetry while producing different phases.

Full six-channel transitivity has only the flat invariant direction and does
force `r = 1`.  The verdict is `PRIMITIVE_ERGODICITY_NO_GO_PASS`: flat
primitive ergodicity is an extra physical principle, not a consequence of
parity-preserving shell symmetry.

## V17 chiral boundary normalization no-go

V17 tests the tempting chiral-boundary route.  Let:

```text
|O> = (|o_1> + ... + |o_5>) / sqrt(5)
```

and define an orthogonal involution exchanging the even channel and normalized
odd collective mode:

```text
Sigma |e> = |O>
Sigma |O> = |e>
```

This `Sigma` is an exact orthogonal involution and commutes with odd-shell
`S_5`.  However, its collective eigenvectors:

```text
|e> ± |O>
```

correspond to primitive-channel ratio:

```text
r = ±1/sqrt(5)
```

The positive branch gives phase `pi/4`, not `atan(sqrt(5))`.  The verdict is
`CHIRAL_BOUNDARY_NORMALIZATION_NO_GO_PASS`: a unitary chiral swap derives
compressed `(e,O)` flatness, not CKM primitive-channel flatness.

## V18 algebraic intertwiner normalization gate

V18 tests whether an algebraic even-to-odd intertwiner can force the CKM
primitive flat ratio.  The exact `S_5`-compatible map is:

```text
T_c |e> = c (|o_1> + ... + |o_5>)
```

and `S_5` fixes only the direction, not the scale:

```text
T_c.T T_c = 5 c^2 P_even
```

The spectral lift:

```text
Gamma_c = T_c + T_c.T
```

satisfies:

```text
Gamma_c^2 = 5 c^2 (P_even + P_odd_collective)
```

on the two-dimensional collective plane, but not `5I` on the full
six-channel shell.  Setting `c = 1` recovers the CKM phase, but that is unit
component normalization, not a consequence of `S_5` or the collective
spectral lift.  The verdict is `ALGEBRAIC_INTERTWINER_FREE_NORM_KILL`.

## V19 primitive max-entropy ergodicity gate

V19 tests whether the flat primitive ratio can be derived from a maximum
entropy principle over the six primitive quark boundary channels.  For:

```text
psi(r) = (|e> + r sum_A |o_A>) / sqrt(1 + 5 r^2)
```

the primitive probabilities are:

```text
p_even = 1 / (1 + 5 r^2)
p_odd_A = r^2 / (1 + 5 r^2)
```

and the six-channel entropy is:

```text
H_6(r) = log(1 + 5 r^2) - (5 r^2 / (1 + 5 r^2)) log(r^2)
```

with derivative:

```text
dH_6/dr = -10 r log(r^2) / (1 + 5 r^2)^2
```

The positive-ratio maximum is `r = 1`, giving the uniform six-channel
distribution and phase `atan(sqrt(5))`.  The compressed `{even, odd_total}`
entropy control instead maximizes at `r = 1/sqrt(5)` and gives phase `pi/4`.
The verdict is `MAX_ENTROPY_PRIMITIVE_ERGODICITY_CONDITIONAL_PASS`: primitive
max entropy derives CKM flatness only if the six primitive channels are the
entropy atoms.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/boundary_response/tests -q
```
