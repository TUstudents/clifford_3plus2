# boundary_response Plan

## Purpose

This sidecar audits the boundary-response neutrino core:

```text
H_Q, V_a -> Sigma(z) -> K_nu = epsilon^2 P_u + P_b
```

It also includes gated conditional PMNS and CKM assemblies from passed
prerequisite gates.  The texture assemblies are kept below theorem status
unless the corresponding boundary-shell assumptions are derived from a
microscopic QCA update.

## V1 Question

Can the residual `K_3` boundary-tail construction alone produce

```text
K_nu = epsilon^2 P_u + P_b
```

by Schur complement?

## Implementation

- `transfer.py`: exact recurrence `x_n = 2 x_{n+1} + x_{n+2}` and
  `epsilon = sqrt(2) - 1`.
- `residual_basis.py`: residual basis `(u,a,b)`, projectors, `K_nu`, and
  residual `S_3` / selected-port `S_2` tests.
- `schur.py`: exact self-energy helper
  `Sigma(z) = V.T (z I - H_Q)^-1 V`.
- `k3_tail.py`: finite `S_3`-equivariant `K_3` tail candidate.
- `audit.py`: combined prove-or-kill verdict.

## Expected V1 Verdict

The transfer invariant should pass:

```text
epsilon = sqrt(2) - 1
epsilon^4 = 17 - 12 sqrt(2)
```

The unbroken residual `K_3` tail should fail the core target. Reason:
`K_nu` splits the residual `S_3` doublet, while any unbroken `S_3`
tail Schur complement remains singlet-plus-degenerate-doublet.

This is a useful kill: the note's neutrino core requires an explicit
framed `S_3 -> S_2` boundary model before PMNS/CKM phenomenology can
resume.

## Next Model If V1 Kills

Define a framed unresolved boundary sector with:

- explicit selected-port defect or sterile boundary state,
- exact `H_Q`,
- exact couplings `V_a`,
- residual selected-port `S_2` but not full `S_3`,
- no direct hardcoding of `P_u` and `P_b`.

Only if that model produces `K_nu` should the sidecar reopen charged-lepton,
PMNS, or CKM boundary scattering.

## V2 Question

Can the corrected framed sterile boundary ansatz derive the target channels
without writing

```text
V_nu = epsilon |s_u><u| + |s_b><b|
```

as an input?

## V2 Implementation

- `framed_sterile.py` derives the channel directions from local incidence in
  the original residual basis:
  - collective tail incidence `(1,1,1)` selects `u`;
  - opposite-edge incidence `(0,1,-1)` selects `b`;
  - the radial mode `a` is absent from both.
- Transfer-depth amplitudes come from `transfer_depth_amplitude(depth)`, with
  `amp(1)/amp(0) = epsilon`.
- The effective response is assembled only after these two ingredients are
  derived:

```text
Sigma_eff = V_derived.T G_return V_derived
```

## V2 Verdict Standard

V2 may report `FRAMED_STERILE_EFFECTIVE_PASS` only if:

- incidence derives `u` and `b`;
- one extra transfer depth derives `g_u/g_b = epsilon`;
- equal diagonal sterile returns and zero cross-return produce
  `epsilon^2 P_u + P_b`;
- breaking the equal-return or zero-cross-return assumptions breaks the target.

This is still not a full microscopic `H_Q` proof.  It is the effective framed
boundary gate before building an explicit Green-function model.

## V3 Question

Can an explicit finite-shell transfer-chain `H_Q` replace the equal-return
ansatz?

## V3 Implementation

- `explicit_hq.py` builds a finite nearest-neighbor transfer chain and evaluates
  its resolvent at the transfer probe `z = 2 sqrt(2)`.
- The Green-function ratio

```text
G(depth=1,0) / G(0,0)
```

converges monotonically to `epsilon`.
- The finite-shell effective response uses that Green-derived amplitude instead
  of `epsilon` as an input.
- The raw shell-coupled Schur response is diagnosed separately; it is not
  promoted to a proof unless its `(a,u,b)` response actually matches the target.

## V3 Verdict Standard

V3 reports:

- `EXPLICIT_HQ_PASS` only if the raw finite `H_Q` Schur response matches all
  framed diagnostics;
- `EXPLICIT_HQ_CONVERGENCE_ONLY` if the Green function derives the transfer
  amplitude but raw shell coupling still fails return-normalization diagnostics;
- `EXPLICIT_HQ_KILL` if convergence or framed diagnostics fail.

## V4 Question

Can a minimal local opposite-edge endpoint load force the missing impedance
condition?

## V4 Implementation

- `impedance.py` evaluates a deterministic small endpoint catalog:
  bare site, self-energy stub, integer dimer, mirrored one-step tail, and a
  parameter-solved matched load.
- Each candidate is scored by raw Schur diagnostics, unresolved-site count,
  graph degree, free parameter count, and whether it uses sector-specific
  tuning.
- V4 reports:
  - `IMPEDANCE_MATCH_PASS` for an untuned local endpoint match;
  - `IMPEDANCE_FREE_PARAMETER` when matching is possible only by solving an
    endpoint scalar;
  - `IMPEDANCE_KILL_MINIMAL_CATALOG` when no catalog candidate matches.

## V5 Question

Can a product sterile bath replace the V2 equal-return and zero-cross-return
assumptions?

## V5 Implementation

- `product_sterile.py` builds:

```text
H_Q = H_chain ⊗ I_family
```

- The family labels live inside the unresolved sector:

```text
|s_u> = |head> ⊗ |u>
|s_b> = |head> ⊗ |b>
```

- This makes the diagonal returns equal and the cross-return zero by tensor
  structure.
- The transfer asymmetry is derived from the finite chain:

```text
amp_N = G_chain(1,0) / G_chain(0,0)
```

- The normalized finite response is:

```text
Sigma_N / G_chain(0,0) = amp_N^2 P_u + P_b
```

- A rank-one negative control removes the family factor.  It must fail by
  producing `u`/`b` cross terms.

## V5 Verdict Standard

V5 reports:

- `PRODUCT_STERILE_CONVERGENCE_PASS` if the product bath gives exact equal
  returns, exact cross-return cancellation, no radial leakage, a passing
  rank-one negative control, and finite transfer convergence to `epsilon`;
- `PRODUCT_STERILE_KILL` if any of those checks fail.

`PRODUCT_STERILE_LIMIT_PASS` is left to V6's semi-infinite Weyl-function
theorem.  PMNS and CKM remain parked after V5.

## V6 Question

Can the V5 finite product-bath convergence be promoted to an exact
semi-infinite theorem?

## V6 Implementation

- `weyl_sterile.py` defines the semi-infinite unit-chain Weyl function:

```text
m(z) = (z - sqrt(z^2 - 4)) / 2
```

- The branch is fixed by:

```text
lim_{z -> infinity} z m(z) = 1
```

- The theorem checks:

```text
m(z)^2 - z m(z) + 1 = 0
m(z) = 1 / (z - m(z))
```

- The exact product sterile response is:

```text
Sigma(z) = m(z) [m(z)^2 P_u + P_b]
```

- At `z = 2 sqrt(2)`, `m(z) = epsilon`, so:

```text
Sigma(z) / m(z) = epsilon^2 P_u + P_b
```

## V6 Verdict Standard

V6 reports:

- `PRODUCT_STERILE_LIMIT_PASS` if the Weyl branch, fixed-point equation,
  transfer-probe value, exact normalized response, V5 convergence regression,
  and rank-one negative control all pass;
- `PRODUCT_STERILE_LIMIT_KILL` otherwise.

This is the exact neutrino-core theorem.  It still does not reopen PMNS or CKM.

## V7 Question

Can the selected charged-lepton/Higgs port and the exact V6 transfer function
derive the two-step charged-lepton leakage scalar?

## V7 Implementation

- `charged_lepton_leakage.py` selects:

```text
e1 = (1,0,0)
```

- In the residual `(a,u,b)` basis:

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

with no `b` component.

- The exact V6 Weyl function gives:

```text
m(z_transfer) = epsilon
```

- The two-step leakage condition gives:

```text
sin(theta_e) = epsilon^2 / sqrt(2/3)
             = sqrt(3/2) epsilon^2
```

## V7 Verdict Standard

V7 reports:

- `CHARGED_LEPTON_LEAKAGE_PASS` if the selected-port decomposition, two-step
  Weyl leakage, depth controls, and `b`-leakage control pass;
- `CHARGED_LEPTON_LEAKAGE_KILL` otherwise.

This derives only Assumption L1.  It does not derive the phase word or touch
CKM.

## V8 Question

Does the proposed leptonic spin-Coxeter-Schur word have the advertised exact
phase?

## V8 Implementation

- `leptonic_phase_word.py` represents phases as exact rational multiples of
  `pi`.
- The scalar inputs are:

```text
q_A3 angle = 1/4
q_A2 angle = 1/3
Schur sign angle = 1
```

- The full raw angle is:

```text
1 + 1/4 + 1/3 = 19/12
```

- The principal angle is:

```text
-5/12
```

- Controls verify that the no-Schur, `A3`-only, and `A2`-only subwords do not
  match the full phase.

## V8 Verdict Standard

V8 reports:

- `LEPTONIC_PHASE_WORD_CONDITIONAL_PASS` if the full word arithmetic passes
  and subword controls differ;
- `LEPTONIC_PHASE_WORD_KILL` otherwise.

`LEPTONIC_PHASE_WORD_DERIVED_PASS` is reserved for a future explicit
boundary-loop holonomy theorem.  V8 does not touch CKM.

## V9 Question

Do the passed V6 neutrino-core theorem, V7 charged-lepton leakage scalar, and
V8 conditional phase word assemble into the advertised PMNS texture?

## V9 Implementation

- `pmns_conditional.py` builds the residual-basis TBM matrix:

```text
U_TBM = [a u b]
```

- The charged-lepton rotation uses:

```text
sin(theta_e) = sqrt(3/2) epsilon^2
phi_e = -5 pi / 12
```

- The PMNS matrix is:

```text
U_PMNS = R_e^\dagger U_TBM
```

- The CP-conjugate branch sends `phi_e -> +5 pi / 12`.
- CKM is not implemented.

## V9 Verdict Standard

V9 reports:

- `PMNS_CONDITIONAL_ASSEMBLY_PASS` if V6, V7, and V8 pass, `U_PMNS` is unitary,
  `sin^2(theta13) = (3/4) epsilon^4`, the numerical PMNS angles match the note,
  and the conjugate branch flips the Jarlskog sign;
- `PMNS_CONDITIONAL_ASSEMBLY_KILL` otherwise.

This is a conditional texture assembly, not a theorem-level PMNS derivation.
The next theorem gate is the boundary-loop holonomy proof selecting the full V8
phase word.

## V10 Question

Can a minimal charged-lepton boundary-loop holonomy model derive the full V8
phase word selection over the no-Schur and subword controls?

## V10 Implementation

- `leptonic_boundary_holonomy.py` defines the exact factor registry:

```text
SCHUR_RETURN angle = 1
PARENT_A3 angle    = 1/4
RESIDUAL_A2 angle  = 1/3
```

- Candidate boundary words are ordered factor words up to length four.
- The primitive charged-lepton word must contain exactly one of each factor in
  the oriented order:

```text
SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2
```

- Negative controls reject no-Schur, single-factor, two-factor, reversed-order,
  and duplicated-cover words.

## V10 Verdict Standard

V10 reports:

- `LEPTONIC_PHASE_WORD_DERIVED_PASS` if exactly one primitive admissible word is
  selected, its phase is `exp(-i 5*pi/12)`, V8 arithmetic still passes, and all
  controls are rejected;
- `LEPTONIC_PHASE_WORD_DERIVED_KILL` otherwise.

This upgrades V8 within the boundary-loop holonomy model.  CKM remains parked.

## V11 Question

Can the primitive quark boundary shell and flat Clifford coin be derived as a
first quark-sector gate?

## V11 Implementation

- `quark_boundary_shell.py` defines:

```text
S_q = 1_even + 5_odd = 1_direct + (2_BCC + 3_color)
```

- The five odd channels are represented by a fixed exact `Cl_5` matrix model:

```text
gamma1 = sigma1 ⊗ I
gamma2 = sigma2 ⊗ I
gamma3 = sigma3 ⊗ sigma1
gamma4 = sigma3 ⊗ sigma2
gamma5 = sigma3 ⊗ sigma3
```

- The audit verifies:

```text
Gamma_q = sum_A gamma_A
Gamma_q^2 = 5 I
B_q = (I + i Gamma_q) / sqrt(6)
B_q^\dagger B_q = I
delta_q = atan(sqrt(5))
```

- Negative controls reject missing color, non-flat amplitude, and commuting
  odd-generator variants.

## V11 Verdict Standard

V11 reports:

- `QUARK_BOUNDARY_SHELL_Q1_PASS` if the shell count, `Cl_5` relations, flat coin
  unitarity, scalar phase, and controls pass;
- `QUARK_BOUNDARY_SHELL_Q1_KILL` otherwise.

This is Q1 only.  It does not assemble CKM.  Q2/Q3 remain future gates for the
transfer-depth hierarchy, color-return factor, and BCC Clebsches.

## V12 Question

Can the quark boundary-depth model derive the raw transfer hierarchy:

```text
1<->2: epsilon^2
2<->3: epsilon^4
1<->3: epsilon^6
```

## V12 Implementation

- `quark_transfer_hierarchy.py` defines the ordered family-depth embedding:

```text
depth(1) = 0
depth(2) = 2
depth(3) = 6
```

- Transition depths are absolute depth differences:

```text
depth(1,2) = 2
depth(2,3) = 4
depth(1,3) = 6
```

- Raw transfer amplitudes are:

```text
A_12 = epsilon^2
A_23 = epsilon^4
A_13 = epsilon^6
```

- Negative controls reject odd-depth, non-additive direct-depth, and
  permuted-label variants.

## V12 Verdict Standard

V12 reports:

- `QUARK_TRANSFER_HIERARCHY_Q2_PASS` if V11 passes, the exact transfer depths
  and amplitudes match, all depths are even, `depth_13 = depth_12 + depth_23`,
  CKM ordering matches, and controls are rejected;
- `QUARK_TRANSFER_HIERARCHY_Q2_KILL` otherwise.

This is Q2 only.  It does not assemble CKM.  Q3 remains future work for the
color-singlet return factor and BCC symmetric/antisymmetric Clebsches.

## V13 Question

Can the quark color-return and BCC Clebsch prefactors be derived exactly?

## V13 Implementation

- `quark_clebsch_factors.py` defines exact SU(3) generators:

```text
T^A = lambda^A / 2
```

- The color-return contraction is:

```text
sum_A T^A T^A = (4/3) I_3
```

- The normalized two-step leakage is:

```text
L_12 = epsilon^2 / sqrt(1 + epsilon^4)
```

- The two-path BCC basis gives:

```text
|S> = (p1 + p2) / sqrt(2),  <S | p1 + p2> = sqrt(2)
|A> = (p1 - p2) / sqrt(2),  <p1 | A> = 1 / sqrt(2)
```

- Negative controls reject raw SU(3) normalization, missing color generator,
  incoherent path sum, and unnormalized antisymmetric vector.

## V13 Verdict Standard

V13 reports:

- `QUARK_CLEBSCH_Q3_PASS` if V12 passes, the SU(3) contraction gives `4/3`,
  the leakage and BCC factors match exactly, and controls are rejected;
- `QUARK_CLEBSCH_Q3_KILL` otherwise.

This is Q3 only.  It does not assemble CKM.  V14 is the CKM matrix assembly
gate.

## V14 Question

Do the passed Q1/Q2/Q3 quark gates assemble into the advertised CKM texture?

## V14 Implementation

- `ckm_conditional.py` computes the exact sine inputs:

```text
s_12 = (4/3) epsilon^2 / sqrt(1 + epsilon^4)
s_23 = sqrt(2) epsilon^4
s_13 = epsilon^6 / sqrt(2)
```

- The quark CP phase is reused from the V11 flat coin:

```text
delta_q = atan(sqrt(5))
exp(i delta_q) = (1 + i sqrt(5)) / sqrt(6)
```

- The CKM matrix uses the standard three-rotation convention:

```text
V_CKM = R_23(theta_23) R_13(theta_13, delta_q) R_12(theta_12)
```

- The audit reports the magnitude matrix and Jarlskog invariant.

## V14 Verdict Standard

V14 reports:

- `CKM_CONDITIONAL_ASSEMBLY_PASS` if V11, V12, and V13 pass and the assembled
  CKM matrix is unitary;
- `CKM_CONDITIONAL_ASSEMBLY_KILL` otherwise.

This is a conditional CKM assembly inside the Q1-Q3 quark boundary-shell
model.  The microscopic derivation of that shell remains the theorem-level
target.

## V15 Question

Does the V11 quark phase follow from unitarity and Clifford closure alone?

## V15 Implementation

- `quark_coin_rigidity.py` proves the exact isotropic coin family:

```text
B(r) = (I + i r Gamma_q) / sqrt(1 + 5 r^2)
Gamma_q^2 = 5I
```

- For every real ratio `r`, `B(r)` is unitary and has positive-branch phase:

```text
delta(r) = atan(r sqrt(5))
```

- The V11 phase satisfies:

```text
delta(r) = atan(sqrt(5))  iff  r = 1
```

## V15 Verdict Standard

V15 reports `QUARK_COIN_RIGIDITY_THEOREM_PASS` if the one-parameter unitary
family, flat specialization, and non-flat controls all pass.

This is a rigidity theorem: the CKM phase requires flat primitive ergodicity.
Unitarity and Clifford closure alone leave one continuous ratio free.

## V16 Question

Can parity-preserving primitive-shell symmetry force the flat ratio `r = 1`?

## V16 Implementation

- `primitive_ergodicity.py` computes exact invariant subspaces for:

```text
1_even ⊕ 5_odd with S_5 acting on the odd shell
6-channel transitive symmetry S_6
```

- The parity-preserving invariant space is:

```text
span{|even>, |odd_1> + ... + |odd_5>}
```

so odd amplitudes are equal, but the even/odd ratio remains free.

- Full six-channel transitivity has only the flat invariant direction.

## V16 Verdict Standard

V16 reports `PRIMITIVE_ERGODICITY_NO_GO_PASS` if parity-preserving symmetry
leaves non-flat ratios invariant while full transitive symmetry rejects them.

This proves the next no-go: the CKM phase requires an extra flat primitive
ergodicity principle or microscopic dynamics that mixes all six channels.

## V17 Question

Does an orthogonal chiral boundary involution exchanging the even channel with
the normalized odd collective mode derive CKM flatness?

## V17 Implementation

- `chiral_boundary_normalization.py` defines:

```text
|O> = (|o_1> + ... + |o_5>) / sqrt(5)
Sigma |e> = |O>
Sigma |O> = |e>
```

- The audit verifies `Sigma^2 = I`, `Sigma.T Sigma = I`, and compatibility
  with odd-shell `S_5`.
- The collective-plane eigenvectors are:

```text
|e> ± |O>
```

which correspond to V15 primitive-channel ratios:

```text
r = ±1/sqrt(5)
```

## V17 Verdict Standard

V17 reports `CHIRAL_BOUNDARY_NORMALIZATION_NO_GO_PASS` if the orthogonal chiral
swap is valid but produces phase `pi/4`, not `atan(sqrt(5))`.

This blocks a false derivation: a unitary chiral swap gives flatness in the
compressed `(e,O)` plane, not flat primitive amplitudes over all six channels.

## V18 Question

Can an algebraic even-to-odd shell intertwiner force the CKM primitive ratio
`r = 1`?

## V18 Implementation

- `intertwiner_normalization.py` computes the exact odd-shell `S_5`
  intertwiner space from the even line into the odd target.
- The invariant map is unique only up to scale:

```text
T_c |e> = c (|o_1> + ... + |o_5>)
```

- Its norm-square is:

```text
T_c.T T_c = 5 c^2 P_even
```

- The collective spectral lift:

```text
Gamma_c = T_c + T_c.T
```

has:

```text
Gamma_c^2 = 5 c^2 (P_even + P_odd_collective)
```

not `5I` on the full primitive shell.

## V18 Verdict Standard

V18 reports `ALGEBRAIC_INTERTWINER_FREE_NORM_KILL` if the intertwiner exists,
is `S_5`-compatible, and has free scale `c`.  The CKM phase is recovered at
`c = 1`, but that is an additional unit component normalization, not an
algebraic theorem from the primitive shell.

## V19 Question

Does maximum entropy over primitive quark boundary channels force the CKM flat
ratio `r = 1`?

## V19 Implementation

- `primitive_entropy_ergodicity.py` defines the six primitive-channel
  probabilities:

```text
p_even = 1 / (1 + 5 r^2)
p_odd_A = r^2 / (1 + 5 r^2)
```

- The primitive Shannon entropy is:

```text
H_6(r) = log(1 + 5 r^2) - (5 r^2 / (1 + 5 r^2)) log(r^2)
```

- Its derivative is:

```text
dH_6/dr = -10 r log(r^2) / (1 + 5 r^2)^2
```

so the positive-ratio maximum is `r = 1`, yielding uniform probability
`1/6` on all six primitive channels and phase `atan(sqrt(5))`.

- The compressed two-channel entropy control over `{even, odd_total}` is also
  audited.  It maximizes at `r = 1/sqrt(5)`, giving phase `pi/4`.

## V19 Verdict Standard

V19 reports `MAX_ENTROPY_PRIMITIVE_ERGODICITY_CONDITIONAL_PASS` if primitive
six-channel max entropy gives `r = 1` while the compressed-channel control
does not.  This is a conditional derivation of CKM flatness from the
primitive-channel entropy principle, not from BCC geometry alone.

## V20 Question

Can V19's primitive max-entropy rule be stated as a Jaynes density theorem?

## V20 Implementation

- `jaynes_primitive_ergodicity.py` defines the odd-shell `S_5` invariant
  density family:

```text
rho(alpha) = alpha P_even + ((1 - alpha) / 5) P_odd
```

- With no retained parity-bias observable, the six-atom entropy is:

```text
S(alpha) = -alpha log(alpha) - (1 - alpha) log((1 - alpha) / 5)
```

- The exact derivatives are:

```text
dS/dalpha = log((1 - alpha) / (5 alpha))
d^2S/dalpha^2 = -1 / (alpha (1 - alpha))
```

so the unique maximum is `alpha = 1/6`, giving:

```text
rho = I_6 / 6
r = sqrt((1 - alpha) / (5 alpha)) = 1
delta = atan(sqrt(5))
```

- The parity-bias controls `alpha = 1/2` and `alpha = 1/3` remain `S_5`
  invariant but do not recover CKM.  The compressed macrochannel entropy
  control maximizes at `alpha = 1/2`, giving `r = 1/sqrt(5)` and phase
  `pi/4`.

## V20 Verdict Standard

V20 reports `JAYNES_PRIMITIVE_ERGODICITY_THEOREM_PASS` if the Jaynes density
is trace-one, `S_5` invariant, has the exact entropy maximum at `I_6/6`,
recovers the CKM phase, and rejects parity-bias and compressed-channel
controls.  This upgrades V19 to a Jaynes theorem conditional on the physical
input that the unresolved primitive shell retains no parity-bias observable.

## V21 Question

Does the six-channel entropy partition follow from conserved-label
distinguishability under boundary scattering?

## V21 Implementation

- `conserved_label_partition.py` assigns each primitive quark boundary channel
  a conserved-label tuple:

```text
(parity, bcc_index, color_index)
```

- The six primitive projectors are exact rank-one label sectors:

```text
P_i P_j = delta_ij P_i
sum_i P_i = I_6
```

- A generic label-preserving boundary scattering operator is diagonal in the
  six conserved-label sectors and commutes with every `P_i`.  A control
  operator mixing two distinct primitive labels fails this conservation test.
- The compressed `{even, odd_total}` partition is rejected because the
  odd_total macrochannel merges five distinct conserved labels.
- The uniform V20 Jaynes density is verified to be the average of the six
  conserved-label projectors.

## V21 Verdict Standard

V21 reports `CONSERVED_LABEL_PARTITION_THEOREM_PASS` if conserved-label tuples
are pairwise distinct, the projectors resolve the identity, label-preserving
scattering commutes with the projectors, label-mixing controls are rejected,
and the compressed partition is coarser than the conserved-label microstate
partition.  This upgrades the V20 entropy partition from declared to derived
from conserved-label distinguishability.

## V22 Question

Can label-conserving boundary dynamics dynamically force the V20 uniform
Jaynes state?

## V22 Implementation

- `label_conserving_dynamics.py` defines generic label-preserving scattering:

```text
U(phi) = sum_i exp(i phi_i) P_i
```

- A generic diagonal primitive population state is:

```text
rho(p) = sum_i p_i P_i
```

- The audit verifies:

```text
[U(phi), rho(p)] = 0
```

for arbitrary populations and phases, so every diagonal population vector is
stationary.
- The label-dephasing channel:

```text
D(rho) = sum_i P_i rho P_i
```

removes off-diagonal coherences but preserves all primitive populations.
- The uniform density `I_6 / 6` is stationary but not unique.  The stationary
trace-one simplex has dimension five.
- A label-mixing control can change populations but fails the V21
conserved-label test.

## V22 Verdict Standard

V22 reports `LABEL_CONSERVING_DYNAMICS_MAX_ENTROPY_NO_GO_PASS` if
label-conserving dynamics preserves all primitive populations, dephasing does
not alter populations, the uniform state is not unique, and any dynamic route
to uniform requires label mixing.  This keeps the max-entropy prior as an
inference principle rather than a consequence of conserved-label dynamics.

## V23 Question

Can the V20 uniform primitive density be derived as a microcanonical reduced
state rather than as dynamical thermalization?

## V23 Implementation

- `microcanonical_reduction.py` models the unresolved primitive shell as:

```text
H_Q = direct_sum_i ( |i>_label tensor B_i )
```

- A full microcanonical state on `H_Q` reduces to primitive-label weights:

```text
p_i = dim(B_i) / sum_j dim(B_j)
rho_label = sum_i p_i P_i
```

- Equal primitive-label bath degeneracy gives:

```text
rho_label = I_6 / 6
r = 1
delta = atan(sqrt(5))
```

- Unequal degeneracy is a negative control: it gives nonuniform primitive
  weights and does not recover the CKM phase.
- The compressed `{even, odd_total}` degeneracy control is also rejected:

```text
r = 1 / sqrt(5)
delta = pi / 4
```

- The audit cross-checks V22's no-go verdict to keep the distinction explicit:
  V22 rejects dynamical selection under label-conserving scattering; V23 proves
  a reduced-state inference theorem.

## V23 Verdict Standard

V23 reports `EQUAL_DEGENERACY_MICROCANONICAL_REDUCTION_PASS` if equal
primitive-label degeneracy reduces to `I_6 / 6`, recovers `r = 1` and
`atan(sqrt(5))`, rejects unequal-degeneracy and compressed-macrochannel
controls, and remains consistent with the V22 label-conserving dynamics no-go.

This closes a sharper version of the primitive ergodicity route: the uniform
primitive density is either a Jaynes prior over conserved-label microstates or
the reduced state of an equal-degeneracy unresolved boundary bath.  The
remaining physical input is not thermalization; it is equal boundary
degeneracy, or equivalently the absence of retained bath-degeneracy bias.
