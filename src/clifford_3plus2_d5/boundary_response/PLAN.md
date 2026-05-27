# boundary_response Plan

## Purpose

This sidecar only audits the boundary-response neutrino core:

```text
H_Q, V_a -> Sigma(z) -> K_nu = epsilon^2 P_u + P_b
```

It does not implement PMNS or CKM textures. Those stay parked unless an
explicit unresolved boundary sector `H_Q` and couplings `V_a` produce the
target operator without hardcoding it.

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

This derives only Assumption L1.  It does not derive the phase word, assemble
PMNS, or touch CKM.

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
boundary-loop holonomy theorem.  V8 does not assemble PMNS or touch CKM.
