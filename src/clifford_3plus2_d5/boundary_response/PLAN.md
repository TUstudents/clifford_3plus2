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

## V24 Question

Can the V23 equal-degeneracy input be sharpened into a regular boundary-fiber
principle?

## V24 Implementation

- `regular_boundary_fiber.py` distinguishes two statements:

```text
conserved-label dynamics alone -> arbitrary d_i remain free
regular boundary fiber        -> d_i = D for all primitive labels
```

- The regular unresolved boundary shell is modeled as:

```text
H_Q = direct_sum_i ( |i>_label tensor B )
```

so every conserved primitive label carries the same unresolved bath template
and the same degeneracy.

- Delegating to the V23 microcanonical reduction gives:

```text
rho_label = I_6 / 6
r = 1
delta = atan(sqrt(5))
```

- Unequal-degeneracy and compressed `{even, odd_total}` controls are rejected.
- The audit cross-checks V22 to keep the claim honest: V24 is a structural
  degeneracy theorem, not a label-conserving thermalization theorem.

## V24 Verdict Standard

V24 reports `REGULAR_BOUNDARY_FIBER_EQUAL_DEGENERACY_PASS` if arbitrary
label-preserving degeneracies remain free, a regular primitive boundary fiber
forces equal degeneracy, the reduced density is `I_6 / 6`, the CKM ratio and
phase are recovered, the negative controls fail, and the result remains
compatible with V22.

This replaces the vague V23 input with:

```text
regular_boundary_fiber_or_max_entropy_prior
```

It does not claim that BCC geometry alone forces regularity.

## V25 Question

Can the transfer probe `z = 2 sqrt(2)` be derived from compatibility between
the residual transfer factor and the unit semi-infinite Weyl chain?

## V25 Implementation

- `transfer_probe_theorem.py` encodes the unit-chain compatibility relation:

```text
z = m + 1/m
```

where `m` is the decaying Weyl transfer factor.

- Substituting the residual transfer invariant gives:

```text
z_transfer = epsilon + epsilon^-1 = 2 sqrt(2)
```

- The audit verifies:

```text
semi_infinite_weyl_function(z_transfer) = epsilon
```

- The reciprocal branch `epsilon^-1 = 1 + sqrt(2)` is rejected because it is
  non-decaying.
- A scaled-chain control keeps the normalization input explicit:

```text
z_t = t (epsilon + epsilon^-1)
```

so `z = 2 sqrt(2)` depends on unit sterile-chain hopping.

## V25 Verdict Standard

V25 reports `TRANSFER_PROBE_COMPATIBILITY_PASS` if `epsilon` satisfies the
residual transfer recurrence, `epsilon + epsilon^-1` equals the existing
transfer probe, the Weyl function returns `epsilon` at that probe, the
reciprocal branch is rejected, the symbolic uniqueness residual vanishes only
at the derived probe, and the scaled-chain control shows that unit hopping is
load-bearing.

This upgrades the probe from an inserted value to the unique compatibility
point for the unit half-line Weyl chain.  It does not derive the unit-chain
normalization itself.

## V26 Question

Can the residual transfer recurrence coefficient be derived from the explicit
residual graph quotient?

## V26 Implementation

- `residual_graph_transfer.py` constructs exact complete-graph adjacency
  matrices and derives the residual degree:

```text
deg(K_3) = 2
```

- With unit outward causal continuation, the radial quotient transfer matrix is:

```text
[[degree, 1],
 [1,      0]]
```

For `K_3`, this equals:

```text
[[2, 1],
 [1, 0]]
```

- The derived decaying polynomial is:

```text
epsilon^2 + 2 epsilon - 1 = 0
```

and the decaying root is:

```text
epsilon = sqrt(2) - 1
```

- Negative controls show that `K_2`, `K_4`, and non-unit continuation change
  the transfer root.

## V26 Verdict Standard

V26 reports `RESIDUAL_GRAPH_TRANSFER_RECURRENCE_PASS` if `K_3` adjacency
matches the existing residual tail adjacency, all row sums equal degree two,
the derived transfer matrix matches `transfer_matrix()`, the derived
polynomial matches `transfer_polynomial()`, the derived decaying root matches
`epsilon()`, and the negative controls produce different roots.

This derives the recurrence coefficient from the residual `K_3` graph.  It
does not derive vacuum framing or unit outward continuation normalization.

## V27 Question

Can the `S_4 -> S_3` vacuum-framing orbit be derived from BCC body-diagonal
geometry once one primitive exit is selected?

## V27 Implementation

- `vacuum_framing.py` imports the eight BCC body-diagonal directions and
  quotients by antipodal pairing to four unoriented representatives:

```text
(+++), (+--), (-+-), (--+) / sqrt(3)
```

- The four representatives form a regular tetrahedron:

```text
v_i . v_i = 1
v_i . v_j = -1/3, i != j
```

- Selecting one framed exit leaves three residual exits and the residual
  adjacency:

```text
K_3
```

- The stabilizer of the selected exit inside `S_4` has six elements and
  induces all residual `S_3` permutations.
- No-selection (`K_4`) and two-residual (`K_2`) controls do not reproduce
  `epsilon`.

## V27 Verdict Standard

V27 reports `BCC_VACUUM_FRAMING_ORBIT_PASS` if the eight BCC directions split
into four antipodal pairs, the representatives form a regular tetrahedron,
one selected exit leaves residual `K_3`, the selected-exit stabilizer induces
full residual `S_3`, the wrong-framing controls fail, and V26 applies to the
resulting residual graph.

This upgrades vacuum framing from a bare `S_4 -> S_3` declaration to an exact
BCC orbit quotient.  It does not derive the physical vacuum order parameter
that chooses the selected exit.

## V28 Question

Can the selected-exit input be realized as a concrete vacuum order-parameter
gate rather than a bare label choice?

## V28 Implementation

- `vacuum_selector.py` defines a rank-one selector field over the four V27
  tetrahedral exits:

```text
E_i = - h . v_i
```

- For `h = v_selected`, the regular-tetrahedron Gram matrix gives:

```text
E_selected = -1
E_other = 1/3
gap = 4/3
```

- The selector Hamiltonian is diagonal on exit labels.
- The energy stabilizer of `h = v_selected` has six elements and matches the
  selected-exit stabilizer from V27.
- The induced action on the three residual exits is the full residual `S_3`.
- All four selector choices are `S_4`-conjugate and have the same energy
  multiset.
- Three controls are rejected:
  - zero order parameter: no selected exit;
  - two-exit midpoint field: degenerate ground pair;
  - generic field: unique ground but trivial stabilizer, so no residual `S_3`.

## V28 Verdict Standard

V28 reports `VACUUM_SELECTOR_ORDER_PARAMETER_PASS` if the selector spectrum is
`(-1, 1/3, 1/3, 1/3)`, the selected exit is the unique ground state with gap
`4/3`, the energy stabilizer induces residual `S_3`, all four selectors are
conjugate, the controls fail, and V27 applies after selection.

This upgrades the selected-exit input into an explicit energy-minimization
gate.  It still leaves the existence of the physical vacuum order parameter
as a named input:

```text
physical_vacuum_order_parameter_exists
```

## V29 Question

Can the unit outward-continuation coefficient in V26 be derived from a
primitive shell-matching normalization?

## V29 Implementation

- `unit_continuation.py` treats the outward continuation as an incidence
  operator `M` from one residual shell to the next.
- The scalar radial quotient exists when:

```text
M.T M = c I
```

- The primitive one-to-one outward continuation is:

```text
M = I_3
```

so:

```text
c = 1
```

- In the residual `(u,a,b)` basis, the continuation couplings are all exactly
  one.
- Feeding `c = 1` into V26 recovers the transfer matrix:

```text
[[2, 1],
 [1, 0]]
```

and the decaying root:

```text
epsilon = sqrt(2) - 1
```

- Controls:
  - scaled matching changes `c` and the transfer root;
  - doubled outward matching changes `c` and the transfer root;
  - anisotropic matching has no scalar radial quotient;
  - residual-label permutations are accepted as gauge-equivalent unit
    matchings.

## V29 Verdict Standard

V29 reports `UNIT_OUTWARD_CONTINUATION_NORMALIZATION_PASS` if the primitive
matching has `M.T M = I_3`, all residual-basis continuation couplings are one,
the V26 transfer matrix and `epsilon` are recovered, scaled/doubled/anisotropic
controls fail, and residual-label permutations remain unit gauge-equivalent.

This removes the unit-continuation normalization from the declared-input list.
The remaining named inputs are:

```text
physical_vacuum_order_parameter_exists
regular_boundary_fiber_or_max_entropy_prior
```

## V30 Question

Can the regular boundary-fiber input be replaced by an explicit local
boundary-fiber isomorphism theorem?

## V30 Implementation

- `local_boundary_fiber.py` models the unresolved primitive shell as:

```text
H_Q = C^6_label tensor B_local
```

- The six labels are the V21 conserved primitive labels.  The local unresolved
  boundary patch `B_local` is the same for every label.
- Therefore each label fiber has the same rank:

```text
d_i = D
```

and:

```text
dim(H_Q) = 6D
```

- For concrete local fiber dimension, the module constructs exact block
  projectors and permutation witnesses showing every pair of label fibers is
  isomorphic.
- Tracing out `B_local` gives:

```text
rho_label = I_6 / 6
```

and recovers:

```text
r = 1
delta = atan(sqrt(5))
```

- Controls:
  - sector-dependent fibers fail full local-fiber isomorphism;
  - arbitrary label-preserving degeneracies remain free without local
    isomorphism;
  - compressed macro-fiber counting merges conserved labels and gives the
    wrong branch.

## V30 Verdict Standard

V30 reports `LOCAL_BOUNDARY_FIBER_ISOMORPHISM_PASS` if all six label fibers
have equal rank, explicit pairwise isomorphism witnesses exist, the reduced
density is `I_6/6`, the CKM primitive ratio and phase match the V20/V24 values,
the controls fail, V21 conserved labels remain complete, V22's no-go is
respected, and V24 is recovered.

This removes `regular_boundary_fiber_or_max_entropy_prior` from the
declared-input list.  The remaining named input is:

```text
physical_vacuum_order_parameter_exists
```

## V31 Question

Can arbitrary selected-exit choice be replaced by a tetrahedral
symmetry-breaking selector potential?

## V31 Implementation

- `vacuum_selector_potential.py` defines the tetrahedral cubic invariant:

```text
C(h) = sum_i (h . v_i)^3
```

where `v_i` are the four V27 BCC tetrahedral exits.

- The finite-candidate selector energy is:

```text
E(h) = -a C(h)
```

with positive default anisotropy `a = 1`.

- For each accepted selector candidate:

```text
E(v_i) = -8/9
```

- Zero and midpoint controls have energy zero.  Antipodal controls have energy
  `+8/9`.  The finite-candidate gap is therefore:

```text
8/9
```

- Each selector candidate must reproduce the V28 selector spectrum up to
  permutation and must have selected-exit stabilizer `S_3`.
- Controls:
  - zero anisotropy has no selector gap;
  - wrong-sign anisotropy selects antipodal controls;
  - midpoint controls sit above selector minima;
  - generic V28 field has trivial stabilizer.

## V31 Verdict Standard

V31 reports `TETRAHEDRAL_SELECTOR_POTENTIAL_PASS` if the four accepted
selector candidates are exactly degenerate finite-candidate minima, the gap
against zero/midpoint/antipodal controls is `8/9`, every selector candidate
recovers V28 and residual `S_3`, and all controls fail.

This removes arbitrary selected-exit choice.  The remaining physical input is
only condensation of the tetrahedral selector order parameter:

```text
tetrahedral_selector_order_parameter_condenses
```

## V32 Question

Can V31's finite-candidate selector potential be promoted to a continuous
order-parameter theorem on `R^3`?

## V32 Implementation

- `vacuum_selector_condensation.py` proves the closed-form tetrahedral cubic:

```text
C(h) = sum_i (h . v_i)^3 = 8 x y z / sqrt(3)
```

- On the unit sphere, the audited Lagrange stationary candidates are:
  - four selector maxima with `C = 8/9`;
  - four antipodal minima with `C = -8/9`;
  - six coordinate-axis controls with `C = 0`.

- The nonnegative selector-locking potential is:

```text
V_lock(h) = (|h|^2 - 1)^2 + (C(h) - 8/9)^2
```

- Its zero set is audited against selectors, antipodes, axes, midpoints, zero,
  and generic controls.  Only the four selector directions have zero potential.
- Controls:
  - radial-only locking leaves all unit-sphere directions degenerate;
  - cubic-only energy is unbounded along selector rays;
  - wrong-sign locking selects the antipodal branch.

## V32 Verdict Standard

V32 reports `CONTINUOUS_TETRAHEDRAL_SELECTOR_CONDENSATION_PASS` if the cubic
polynomial identity holds, the unit-sphere stationary audit gives max value
`8/9`, the zero set of `V_lock` is exactly the four accepted selector directions
among audited candidates and controls, radial-only/cubic-only/wrong-sign
controls fail, and V31 is recovered.

This replaces the declared condensation of an abstract selector order parameter
with one narrower physical input:

```text
local_vacuum_realizes_tetrahedral_selector_locking_potential
```

## V33 Question

Can the V32 selector-locking potential be derived as the canonical
lowest-order tetrahedral Landau lock?

## V33 Implementation

- `vacuum_selector_landau.py` constructs the exact proper tetrahedral rotation
  group as 12 signed-permutation matrices preserving the four BCC selectors.
- Homogeneous invariant spaces are computed directly from group invariance:

```text
degree 1: dim 0
degree 2: dim 1, radial |h|^2
degree 3: dim 1, tetrahedral cubic x y z
```

- The first selector-capable anisotropy is therefore degree three:

```text
C(h) = 8 x y z / sqrt(3)
```

- The canonical lowest-order positive Landau lock is:

```text
V_Landau(h) = (|h|^2 - 1)^2 + (C(h) - 8/9)^2
```

- This must exactly recover V32's `selector_locking_potential`.
- Controls:
  - radial-only theory leaves the unit sphere degenerate;
  - cubic-only theory is unbounded without radial stabilization;
  - wrong-sign cubic selects antipodes;
  - ad hoc linear/quadratic anisotropies are not tetrahedral invariants.

## V33 Verdict Standard

V33 reports `TETRAHEDRAL_INVARIANT_LANDAU_MINIMALITY_PASS` if the group has
12 proper selector-preserving rotations, invariant dimensions are `(0,1,1)` in
degrees `(1,2,3)`, degree two is radial, degree three is the tetrahedral cubic,
the lowest selector anisotropy degree is `3`, the Landau lock exactly recovers
V32, controls fail, and V32 is recovered.

This narrows the remaining physical input to:

```text
local_vacuum_enters_lowest_order_positive_tetrahedral_landau_phase
```

## V34 Question

Can the V33 tetrahedral cubic be produced by an explicit local boundary-shell
Schur expansion?

## V34 Implementation

- `vacuum_selector_schur_landau.py` defines the four shell couplings:

```text
d_i(h) = h . v_i
D(h) = diag(d_i(h))
```

- It computes the exact shell power sums:

```text
p_n(h) = Tr(D(h)^n) = sum_i (h . v_i)^n
```

with:

```text
p_1 = 0
p_2 = 4 |h|^2 / 3
p_3 = C(h) = 8 x y z / sqrt(3)
```

- It defines the cubic-order Schur/log-det shell expansion:

```text
F_shell(h; eta, s) = s sum_{n=1..3} eta^n p_n(h) / n
```

so the induced terms are:

```text
quadratic coefficient: 2 s eta^2 / 3
cubic coefficient:    s eta^3 / 3
```

- Controls:
  - `eta -> -eta` flips only the cubic branch;
  - `s -> -s` flips the whole shell contribution;
  - paired `eta` and `-eta` shells cancel the cubic;
  - proper tetrahedral rotations preserve the shell series;
  - inversion flips the cubic but is not a proper tetrahedral symmetry.

## V34 Verdict Standard

V34 reports `SCHUR_SHELL_TETRAHEDRAL_CUBIC_ORIGIN_PASS_SIGN_FREE` if the shell
power sums are exact, the cubic-order series is radial plus tetrahedral cubic,
the cubic is the V33 first selector-capable anisotropy, all sign/orientation
controls pass, and V33 is recovered.

The sign-free verdict is intentional: this gate derives the cubic origin but
does not derive the positive branch.  The remaining input is:

```text
oriented_boundary_shell_selects_positive_cubic_branch
```

## V35 Question

Does the Bialynicki-Birula single-Weyl walk promote the helicity-odd
tetrahedral selector into the real filled-band energy?

## V35 Implementation

- `vacuum_selector_chiral_bb.py` reuses the actual `spacetime_qca.bcc_weyl`
  symbols.  It does not rebuild BB phases.
- It computes the `epsilon^3 kx ky kz` Floquet-trace diagnostic:

```text
right Weyl: -2
left Weyl:  +2
Dirac pair: 0
```

- It expands the matrix logarithm:

```text
U = I + epsilon U1 + epsilon^2 U2 + epsilon^3 U3 + ...
log U = epsilon L1 + epsilon^2 L2 + epsilon^3 L3 + ...
H_eff = i log(U) / epsilon
```

with:

```text
L1 = U1
L2 = U2 - U1^2/2
L3 = U3 - (U1 U2 + U2 U1)/2 + U1^3/3
```

- It verifies:

```text
H0_R = + sigma.k
H0_L = - sigma.k
```

- It applies the conservative scalar trace-energy extraction:

```text
scalar_H2 = Tr(H2) / dim
```

and reads the `kx ky kz` coefficient.  This remains in the audit as a blind
negative control: the scalar trace probe is parity-even and polynomial, so it
cannot detect the parity-odd angular selector.

- It computes the occupied filled-band quasienergy from Bloch eigenphases:

```text
E = -arg(lambda) / epsilon
E_odd(k) = [E_occ(k) - E_occ(-k)] / 2
```

- It verifies the real filled-band selector pattern:

```text
E_odd^R != 0
E_odd^L = -E_odd^R
E_odd^Dirac = 0
E_odd = 0 on xyz = 0
E_odd / (epsilon kx ky kz) is constant on signed permutations
```

- It also verifies that the normalized ratio changes across inequivalent
  radii.  The selector is therefore angular/non-polynomial, not a pure `xyz`
  monomial.

## V35 Verdict Standard

V35 reports `CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_PASS` only if the filled-band
quasienergy contains a real helicity-locked `A2u` selector and the Dirac/vector
pair cancels it.

With the eigenphase-filled-band rule, the verdict is:

```text
CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_PASS
```

The remaining inputs are:

```text
()
```

## V43 Question

Can the vacuum-selector sidecar be closed for polishing with exactly one named
intermediate axiom instead of an open-ended residual input list?

## V43 Implementation

- `vacuum_selector_closure.py` assembles the selector-sector verdict chain:

```text
V35  CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_PASS
V36  CHIRAL_BB_BRANCH_SELECTION_PASS
V37  MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_PASS
V38  FREE_BB_RADIAL_STABILIZATION_NO_GO_PASS
V41  BB_INDUCED_RADIAL_BREAKING_PASS
V42  ANALYTIC_RADIAL_BREAKING_THEOREM_PASS
```

- It declares exactly one intermediate selector axiom:

```text
positive_quartic_backreaction_bounds_selector_radius
```

- It explicitly records that this positive quartic coefficient is not
  microscopically derived.

## V43 Verdict Standard

V43 reports `VACUUM_SELECTOR_CLOSED_WITH_QUARTIC_AXIOM_PASS` only if all
selector-chain prerequisites pass and the remaining axiom set is exactly the
single positive-quartic backreaction axiom.

## V42 Question

Can V41's sampled radial breaking be upgraded to an analytic theorem?

## V42 Implementation

- `vacuum_selector_radial_theorem.py` uses the continuum-leading occupied Weyl
  energy:

```text
E_occ^0(k) = -|k|
```

- Along a unit selector ray it audits:

```text
E(r) = -c r + m^2 r^2 + lambda r^4
c > 0
```

- It proves:
  - `E'(0+) = -c < 0`;
  - `lambda > 0` gives large-radius boundedness;
  - a finite nonzero minimum exists by continuity;
  - for `m^2 = 0`, `lambda = 1`, `c = 1`, the positive stationary point is
    `r = (1/4)^(1/3)`;
  - the V36 branch signs survive at that analytic radius.

## V42 Verdict Standard

V42 reports `ANALYTIC_RADIAL_BREAKING_THEOREM_PASS` only if the analytic
origin-instability, quartic-boundedness, stationary-point, branch-survival, and
V41 regression diagnostics pass.

## V41 Question

Does the Higgs/backreaction sector need to enter an already-broken phase, or
does the free BB filled-band radial energy destabilize the selector origin so
that a merely positive quartic backreaction is enough?

## V41 Implementation

- `vacuum_selector_bb_induced_breaking.py` audits:

```text
V_back(r) = m^2 r^2 + lambda r^4
E_total(r) = E_free_even(r) + V_back(r)
```

- Defaults:

```text
m^2 = 0
lambda = 1
```

- It verifies:
  - `E_free_even(0) = 0` and immediately decreases for `r > 0`;
  - `V_back` alone is unbroken at the origin for `m^2 >= 0`, `lambda > 0`;
  - `E_total` has a finite nonzero minimum on the audited grid;
  - a positive-mass control still has a finite nonzero minimum;
  - `lambda = 0` recovers V38's no-go;
  - `lambda < 0` is rejected as unbounded;
  - V36 branch diagnostics survive at the BB-induced minimum.

## V41 Verdict Standard

V41 reports `BB_INDUCED_RADIAL_BREAKING_PASS` only if the free BB filled-band
even energy destabilizes the origin and a positive quartic backreaction bounds
the radial profile at a finite nonzero selector radius.

This does not derive the positive quartic coefficient.  It removes the need to
insert a negative Higgs mass / broken phase by hand.

The expected verdict is:

```text
BB_INDUCED_RADIAL_BREAKING_PASS
```

The remaining input is:

```text
positive_quartic_backreaction_bounds_selector_radius
```

## V40 Question

Is the V39 Mexican-hat radial stabilizer arbitrary, or is it the unique minimal
bounded broken quartic allowed by local Higgs gauge invariance?

## V40 Implementation

- `vacuum_selector_higgs_origin.py` audits a local Higgs doublet with

```text
rho = |Phi|^2
V(rho) = c + a rho + b rho^2
```

- It verifies the Landau sign criteria:

```text
b > 0       bounded below
a < 0       broken phase
v^2 = -a / (2b)
V(rho) = b (rho - v^2)^2 + constant
```

- It pulls the existing `spacetime_qca` Higgs potential back along the
  gauge-fixed selector representative:

```text
Phi(r) = (0, r)
```

and verifies that this equals the V39 radial term on the audited grid.

- Controls reject:
  - positive quadratic / unbroken phase;
  - negative quartic / unbounded phase;
  - component-anisotropic doublet quadratics that are not functions of
    `|Phi|^2`.

## V40 Verdict Standard

V40 reports `HIGGS_RADIAL_LANDAU_UNIQUENESS_PASS` only if gauge invariance,
boundedness, and the broken-phase sign complete the square to the V39 radial
term, and the existing spacetime-QCA Higgs potential has the same gauge-fixed
pullback.

This is not a microscopic derivation of why the Higgs sector enters the broken
phase.  It narrows the remaining input to that sign/phase choice.

The expected verdict is:

```text
HIGGS_RADIAL_LANDAU_UNIQUENESS_PASS
```

The remaining input is:

```text
higgs_backreaction_sector_enters_broken_quartic_phase
```

## V39 Question

Can the V38 free-BB radial no-go be closed by coupling the selector amplitude
to the existing Higgs/backreaction radial sector without spoiling the chiral
branch-selection diagnostics?

## V39 Implementation

- `vacuum_selector_higgs_stabilizer.py` adds the radial stabilizer

```text
V_H(r) = lambda (r^2 - v^2)^2
```

with defaults `lambda = 1`, `v^2 = 1`.

- It audits the combined radial profile

```text
E_stabilized(r) = E_free_even(r) + V_H(r)
```

on:

```text
r = 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5
```

- It verifies:
  - the Mexican-hat term alone minimizes at `r = 1`;
  - the combined profile has a finite interior minimum;
  - right/left/Dirac branch diagnostics from V36 survive at that minimum;
  - `lambda = 0` recovers the V38 no-go;
  - the pure `v^2 = 0` radial sector is unbroken at the origin.

## V39 Verdict Standard

V39 reports `HIGGS_BACKREACTION_RADIAL_STABILIZER_PASS` only if the radial
stabilizer creates a finite interior selector amplitude and the helicity-locked
branch sign survives there.

This is a sufficiency theorem for coupling to a Higgs/backreaction radial
sector, not a derivation of that sector from the free BB walk.

The expected verdict is:

```text
HIGGS_BACKREACTION_RADIAL_STABILIZER_PASS
```

The remaining input is:

```text
higgs_or_backreaction_sector_supplies_mexican_hat_radial_term
```

## V38 Question

Does the free BB filled-band functional itself stabilize a finite selector
amplitude, or does it only provide the branch direction?

## V38 Implementation

- `vacuum_selector_radial_no_go.py` samples the parity-even filled-band energy
  along one selector ray at:

```text
r = 0, 0.125, 0.25, 0.5, 1.0
```

- It computes forward finite differences and checks for interior local minima.
- It reuses V36's right-Weyl branch-gap helper over the same radii.
- It verifies:

```text
free radial even energy decreases monotonically
no interior finite-radius minimum appears
right-Weyl branch gap is positive for r > 0
Dirac/vector odd selector cancels
V37 is recovered
```

## V38 Verdict Standard

V38 reports `FREE_BB_RADIAL_STABILIZATION_NO_GO_PASS` if the free filled-band
energy is monotone on the audited selector ray, no interior finite minimum is
found, the branch gap remains stable away from the origin, and V37 is
recovered.

If an interior finite-radius minimum is found, the alternative verdict is:

```text
FREE_BB_RADIAL_STABILIZATION_FOUND
```

The expected verdict is:

```text
FREE_BB_RADIAL_STABILIZATION_NO_GO_PASS
```

The remaining input is:

```text
radial_stabilization_requires_interaction_or_backreaction
```

## V37 Question

Does the actual BB occupied-band eigenphase functional split into the parity
even/odd microscopic structure behind the V32-V36 selector chain?

## V37 Implementation

- `vacuum_selector_microscopic_potential.py` defines:

```text
E_occ(h)  = occupied filled-band quasienergy
E_even(h) = [E_occ(h) + E_occ(-h)] / 2
E_odd(h)  = [E_occ(h) - E_occ(-h)] / 2
```

- It verifies that `E_occ = E_even + E_odd` and `E_occ(-h) = E_even - E_odd`.
- It checks the even part on audited samples:
  - selector and antipode candidates are degenerate;
  - coordinate permutations at fixed norm agree;
  - energy decreases monotonically with radius on the selector ray.
- It checks the odd part:
  - V36's right-Weyl branch gap remains positive at radii `0.25, 0.5, 1.0`;
  - Dirac/vector odd selector cancels on sampled points.
- It retains the old scalar trace-polynomial probe as a blind negative control.

## V37 Verdict Standard

V37 reports `MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_PASS` only if the
occupied BB eigenphase energy passes the even/odd split, even-sector symmetry,
radial monotonicity, V36 branch stability, Dirac cancellation, and blind-probe
controls.

The expected verdict is:

```text
MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_PASS
```

The remaining inputs are:

```text
()
```

## V36 Question

Does the V35 single-Weyl filled-band selector choose the actual V27/V32
tetrahedral vacuum branch over its antipode?

## V36 Implementation

- `vacuum_selector_filled_band_potential.py` evaluates V35's occupied-band
  parity-odd energy on:
  - the four accepted tetrahedral selector candidates;
  - the four antipodal controls;
  - origin and coordinate-axis controls.
- It verifies:

```text
right Weyl: E_selector < E_antipode
left Weyl:  E_selector > E_antipode
Dirac pair: E_selector = E_antipode = 0
```

- It checks that all four selector candidates are degenerate, all four
  antipodes are degenerate, and selector/antipode energies are opposite.
- It verifies the normalized candidate ratios match V35's signed-orbit `A2u`
  ratio, proving the branch potential is the same filled-band selector term.

## V36 Verdict Standard

V36 reports `CHIRAL_BB_BRANCH_SELECTION_PASS` only if the filled-band selector
lowers the accepted tetrahedral branch for one helicity, reverses under the
opposite helicity, cancels in the Dirac/vector pair, and vanishes on axis
controls.

The expected verdict is:

```text
CHIRAL_BB_BRANCH_SELECTION_PASS
```

The remaining inputs are:

```text
()
```
