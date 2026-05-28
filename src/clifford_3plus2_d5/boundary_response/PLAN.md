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
