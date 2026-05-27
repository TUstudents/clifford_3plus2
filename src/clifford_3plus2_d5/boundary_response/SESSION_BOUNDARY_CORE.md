# Session Boundary Core

## Objective

Implement a new sidecar that tests only:

```text
H_Q, V_a -> Sigma(z) -> K_nu = epsilon^2 P_u + P_b
```

The sidecar must not implement PMNS or CKM texture machinery.

## V1 Result

V1 gives a negative but sharp verdict:

```text
BOUNDARY_CORE_KILL_UNBROKEN_K3
```

The transfer recurrence itself passes:

```text
x_n = 2 x_{n+1} + x_{n+2}
epsilon = sqrt(2) - 1
epsilon^4 = 17 - 12 sqrt(2)
```

The Schur-complement core fails for the unbroken residual `K_3` tail.

## Load-Bearing Calculation

The residual family basis is:

```text
u = (1,1,1) / sqrt(3)
a = (2,-1,-1) / sqrt(6)
b = (0,1,-1) / sqrt(2)
```

The proposed target is:

```text
K_nu = epsilon^2 P_u + P_b
```

This target splits the residual `S_3` doublet:

```text
a -> 0
b -> 1
```

An unbroken `S_3`-equivariant tail can only produce an operator of the form:

```text
alpha P_u + beta (P_a + P_b)
```

Therefore it cannot produce `K_nu`.

The finite `K_3` tail model verifies this explicitly using:

```text
Sigma(z) = V.T (z I - H_Q)^-1 V
```

with shell-major residual `K_3` adjacency and identity inter-shell coupling.
The resulting self-energy remains `S_3` invariant.

## Interpretation

This is a useful kill result. The boundary-response flavor program needs an
explicit framed boundary sector that breaks residual `S_3` to selected-port
`S_2` before it can claim the neutrino core.

PMNS and CKM extensions stay parked.

## V2 Effective Framed Sterile Result

V2 implements the corrected non-tautological effective standard. It does not
accept a prebuilt coupling

```text
epsilon |s_u><u| + |s_b><b|
```

as input. Instead:

```text
(1,1,1)      -> u
(0,1,-1)    -> b
depth 1 / 0 -> epsilon
```

The first two statements derive the collective and opposite-edge channels from
local residual incidence. The third derives the coupling ratio from transfer
depth.

With equal low-energy sterile returns and no cross-return, the effective
Schur response is:

```text
Sigma_eff ∝ epsilon^2 P_u + P_b
```

The V2 verdict is:

```text
FRAMED_STERILE_EFFECTIVE_PASS
```

This is not yet a full `H_Q` theorem. It is the next gate: the sidecar now has
an exact effective framed-boundary model whose remaining load-bearing
assumption is equal diagonal sterile return normalization.

## V3 Explicit Finite-Shell Gate

V3 adds a finite nearest-neighbor transfer-chain `H_Q` and computes its
Green-function ratios directly.  At the transfer probe

```text
z = 2 sqrt(2)
```

the finite-chain ratio

```text
G(1,0) / G(0,0)
```

converges monotonically to

```text
epsilon = sqrt(2) - 1.
```

This supports the transfer-depth piece of V2 from an explicit finite resolvent.

The raw shell-coupled Schur response is kept separate and currently fails the
return-normalization diagnostic.  The V3 verdict is:

```text
EXPLICIT_HQ_CONVERGENCE_ONLY
```

So the equal-return layer is not fully replaced. The next model must supply a
sterile return structure whose raw Schur response gives equal `u`/`b` return
normalization without inserting it by hand.

## V4 Impedance-Matching Catalog

V4 tests minimal local opposite-edge endpoint loads:

```text
bare site
self-energy stub
integer dimer
mirrored one-step tail
parameter-solved matched load
```

The untuned local candidates all fail with `UNEQUAL_RETURN`.  The only matching
candidate is the one-site load whose endpoint energy is solved from the
required return:

```text
edge_energy = z_transfer - 1 / inferred_collective_return
```

The V4 verdict is:

```text
IMPEDANCE_FREE_PARAMETER
```

This is not a theorem. It means the target impedance can be fitted by a local
load, but the current catalog does not derive that load from symmetry,
causality, or minimization alone.

## V5 Product Sterile-Tail Gate

V5 tests the preferred Mechanism B:

```text
H_Q = H_chain ⊗ I_family
```

The sterile dynamics is one finite transfer chain.  The residual family factor
is part of the unresolved sector, so the two response states are:

```text
|s_u> = |head> ⊗ |u>
|s_b> = |head> ⊗ |b>
```

This removes the V2 equal-return and zero-cross-return assumptions:

```text
<s_u|G_Q|s_u> = <s_b|G_Q|s_b>
<s_u|G_Q|s_b> = 0
```

The transfer-depth asymmetry is not a free parameter.  It is derived from the
same finite chain used in the product bath:

```text
amp_N = G_chain(1,0) / G_chain(0,0)
amp_N -> epsilon
```

The finite normalized Schur response is:

```text
Sigma_N / <head|G_chain|head> = amp_N^2 P_u + P_b
```

At the default audit depth `N = 10`, this gives:

```text
PRODUCT_STERILE_CONVERGENCE_PASS
```

This is the strongest neutrino-core result in the sidecar so far: the return
normalization is structural, cross-return vanishes structurally, and the
mass-ratio limit is:

```text
m_2 / m_3 -> epsilon^2
Delta m^2_21 / Delta m^2_31 -> epsilon^4
```

The exact semi-infinite Weyl-function theorem is still future work.  V5 is a
finite-shell convergence result, not yet a closed-form limit-point proof.

The required negative control is also implemented.  If the family factor is
removed and both channels couple to one sterile head, the response becomes
rank one and develops `u`/`b` cross terms.  That model fails, confirming that
the family factor must live inside `Q`.

## V6 Exact Weyl-Function Theorem

V6 replaces the finite endpoint with the honest semi-infinite transfer chain.
The head Green function is the Weyl function:

```text
m(z) = (z - sqrt(z^2 - 4)) / 2
```

It is the decaying branch, fixed by:

```text
lim_{z -> infinity} z m(z) = 1
```

and it satisfies the self-consistency equation:

```text
m(z) = 1 / (z - m(z))
```

At the transfer probe:

```text
z = 2 sqrt(2)
m(z) = epsilon
```

The exact product sterile Schur response is:

```text
Sigma(z) = m(z) [m(z)^2 P_u + P_b]
```

so the normalized response is:

```text
Sigma(z) / m(z) = m(z)^2 P_u + P_b
```

At `z = 2 sqrt(2)`, this is exactly:

```text
epsilon^2 P_u + P_b
```

The V6 verdict is:

```text
PRODUCT_STERILE_LIMIT_PASS
```

This supersedes V4's tuned endpoint load.  The endpoint impedance is no longer
fitted; it is the unique decaying Weyl function of the semi-infinite tail.

The theorem is still only the neutrino-core gate.  PMNS and CKM remain parked
until charged-lepton and quark boundary shells are derived.

## V7 Charged-Lepton Leakage Gate

V7 tests only Assumption L1 from the boundary-response note.  It does not
assemble PMNS and it does not derive the leptonic phase word.

The charged-lepton/Higgs response frame selects the first residual port:

```text
e1 = (1,0,0)
```

In the residual neutrino basis:

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

and:

```text
<b|e1> = 0
```

The V6 Weyl theorem supplies:

```text
m(z_transfer) = epsilon
```

So two residual transfer steps give:

```text
m(z_transfer)^2 = epsilon^2
```

The leakage condition is:

```text
sqrt(2/3) sin(theta_e) = epsilon^2
```

therefore:

```text
sin(theta_e) = sqrt(3/2) epsilon^2
sin^2(theta_e) = (3/2) epsilon^4
```

The V7 verdict is:

```text
CHARGED_LEPTON_LEAKAGE_PASS
```

Depth-one and depth-three leakage controls do not match the L1 relation, and a
synthetic `b`-leakage control is detected.  This derives only the leakage
scalar.  PMNS remains parked until the phase word is derived.

## V8 Leptonic Phase-Word Gate

V8 tests only the exact arithmetic of Assumption L2:

```text
W_e = -q_A3 q_A2
```

The scalar spin lifts are:

```text
q_A3 = exp(i pi/4)
q_A2 = exp(i pi/3)
```

The second-order Schur complement contributes the minus sign:

```text
-1 = exp(i pi)
```

Therefore the raw angle in units of `pi` is:

```text
1 + 1/4 + 1/3 = 19/12
```

and the principal representative is:

```text
-5/12
```

so:

```text
W_e = exp(-i 5 pi / 12)
```

The V8 verdict is:

```text
LEPTONIC_PHASE_WORD_CONDITIONAL_PASS
```

No-Schur, `A3`-only, and `A2`-only controls differ from the full phase.  The
CP-conjugate branch has angle `+5/12`.

This is not yet a boundary-loop theorem.  V8 verifies the conditional phase
word only; the selection of the full word over subwords is still an explicit
assumption.  PMNS remains parked.
