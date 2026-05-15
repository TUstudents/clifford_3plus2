# Two-Site Bloch Carrier Design

Status: design note for the next Path-A architecture; not implemented as a
bridge checker yet.

## Purpose

The single-site Bloch Path-A searches now have two failure modes:

- projector-free monomial hops produce the coarse `[0,4,6,10]` center but no
  compatible complex structure in the joint centralizer;
- the first polynomial-hop extension leaves the tractable coarse-center regime
  immediately, exceeding algebra dimension `64` before a center can be checked.

The next architecture should stop asking one `10 x 10` on-site carrier to do
both jobs. A two-site carrier lets the Bloch symbol act on a doubled cell
`R^20`, with the physical `R^10` object recovered by a band projection rather
than placed directly in the coefficient algebra.

## Candidate Form

Use a two-sublattice unit cell:

```text
K_cell = K_A ⊕ K_B
dim_R K_A = dim_R K_B = 10
T(z) =
  [ A_0      B_- z^-1 + B_0 ]
  [ C_0 + C_+ z       D_0  ]
```

with exact real-orthogonal Laurent identity:

```text
T(z)^T T(z) = I_20
T(z) T(z)^T = I_20
```

The first bounded class should use nearest-neighbor split-step blocks:

```text
A_0, D_0: finite-order on-site real orthogonal layers
B_-, B_0, C_0, C_+: partial orthogonal hopping blocks
```

No coefficient may equal, or algebraically generate, `P_alpha` or `P_eta`.

## Verdict Projection

The checker should compute:

```text
samples = (T(zeta^j)) for 0 <= j < period
A_Bloch = R<samples>
center(A_Bloch)
```

Then it should inspect central idempotents whose ranks over `R^20` project to a
rank `(6,4)` pair on a `10`-dimensional effective band. This is deliberately
different from the monomial-hop checker, where `P_alpha/P_eta` live directly
on one site.

## Implementation Steps

1. Add a `dimension=20` Bloch-rule path to `rule_to_verdict`.
2. Add a two-site candidate builder with exact Laurent orthogonality checks
   before closure.
3. Add a guarded projection routine that reports both raw `R^20` idempotent
   ranks and effective `R^10` band ranks.
4. Reuse the existing bounded idempotent-splitting `J` solver only after the
   effective `(6,4)` pair is identified.
5. Keep the seed guardrail on coefficient algebra, not on derived spectral band
   projectors.

## Acceptance Boundary

A two-site candidate is interesting only if it satisfies all of:

```text
coefficient seed guardrail passes
exact Laurent orthogonality passes
joint Bloch algebra closes below the configured cap
effective coarse center has ranks (6,4)
compatible J section is nonempty
rule-generated J section is global ±J, or the gauge-equivalence standard
explicitly accepts the remaining choices
```

If the first bounded two-site class also produces an empty compatible-J set,
that becomes the natural extension of Proposition 4.
