# boundary_response — Status

**Status**: V1, V2, V3, V4, V5, V6, V7, and V8 implemented.

- V1 verdict: **BOUNDARY_CORE_KILL_UNBROKEN_K3**.
- V2 verdict: **FRAMED_STERILE_EFFECTIVE_PASS**.
- V3 verdict: **EXPLICIT_HQ_CONVERGENCE_ONLY**.
- V4 verdict: **IMPEDANCE_FREE_PARAMETER**.
- V5 verdict: **PRODUCT_STERILE_CONVERGENCE_PASS**.
- V6 verdict: **PRODUCT_STERILE_LIMIT_PASS**.
- V7 verdict: **CHARGED_LEPTON_LEAKAGE_PASS**.
- V8 verdict: **LEPTONIC_PHASE_WORD_CONDITIONAL_PASS**.

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
strongest unframed `K_3`-tail version of the neutrino core. PMNS and CKM
textures remain parked until an explicit framed `H_Q,V` model derives the
required `S_3 -> S_2` doublet splitting.

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
equal-return and zero-cross-return assumptions remain explicit. PMNS/CKM stay
parked.

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
selects the full word, and it does not assemble PMNS.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/boundary_response/tests -q
```
