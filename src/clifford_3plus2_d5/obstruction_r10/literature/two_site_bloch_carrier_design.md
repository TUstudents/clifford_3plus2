# Two-Site Bloch Carrier Design

Status: first bounded two-site checker implemented; no bridge candidate.

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

## Implemented Candidate Form

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

The implemented first bounded class is a bipartite forward/inverse carrier:

```text
T(z) =
  [ 0         H(z)       ]
  [ H(z^-1)^T  0         ]

H(z) = sum_s E_s z^s
```

where the `E_s` are projector-free monomial mode-hop matrices. Two variants
are exposed:

```text
uniform: shifts (1,1,1,1,1)
winding-4-3: shifts (4,4,4,3,3)
```

Both pass exact Laurent orthogonality. The uniform variant is a guardrail
sanity check. The winding variant has the Path-A `(4,3)` shape but fails the
coefficient-algebra seed guardrail.

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

## Implementation

```text
src/clifford_3plus2_d5/qca/two_site_bloch.py
scripts/bloch_two_site.py
tests/test_two_site_bloch.py
```

The certificate reports:

```text
raw R^20 central idempotent ranks
sublattice A/B compressed ranks
balanced effective ranks when the idempotent is block-diagonal and equal-rank
coefficient-algebra seed guardrail against embedded P_alpha/P_eta
```

## Current Result

Uniform guardrail sanity:

```text
uv run python scripts/bloch_two_site.py --variant uniform --check

dimension: 20
term_count: 2
shifts: (1, -1)
laurent_orthogonal: true
seed_guardrail_passed: true
coefficient_algebra_dimension: 4
coefficient_algebra_closed: true
generated_algebra_dimension: 4
generated_algebra_closed: true
center_dimension: 1
center_solved: true
central_idempotent_ranks: (0, 20)
effective_rank_6_4_pairs: 0
route_label: two_site_trivial_center_no_effective_split
```

Path-A-shaped winding:

```text
uv run python scripts/bloch_two_site.py --variant winding-4-3 --check

dimension: 20
term_count: 4
shifts: (3, 4, -3, -4)
laurent_orthogonal: true
seed_guardrail_passed: false
coefficient_algebra_dimension: 8
coefficient_algebra_closed: true
generated_algebra_dimension: 8
generated_algebra_closed: true
center_dimension: 2
center_solved: true
central_idempotent_ranks: (0, 20)
effective_rank_6_4_pairs: 0
route_label: two_site_seed_guardrail_rejected
```

Interpretation: doubling the carrier by itself does not solve Path A. The
guardrail-passing uniform carrier has only a trivial center. The first
`(4,3)` winding carrier is exact and computationally tame, but the coefficient
algebra recovers embedded `P_alpha/P_eta` on one sublattice, so it is Route 2
seeding in two-site language.

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

For the next Move-2 attempt, the winding data must be attached through a
two-site coin or split-step layer whose coefficient algebra does not recover
the coarse projector before the Bloch center is computed.

## Split-Step Coin Boundary

The next bounded Move-2b panel adds exact on-site `20 x 20` coins around the
forward/inverse carrier:

```text
T(z) = C_right * Shift(z) * C_left
```

The first panel uses sublattice swaps and shared five-mode cycles. These are
genuine two-site coins, not raw `P_alpha/P_eta` coefficients, and each layer
passes exact Laurent orthogonality. The checker deliberately stops before
forming the sampled Bloch algebra unless the coefficient algebra closes under
the configured cap; otherwise the seed guardrail itself has not been certified.

Bounded panel:

```text
uv run python scripts/bloch_two_site.py --split-step-search --max-candidates 4 --max-generated-algebra-dim 8 --split-step-coefficient-algebra-dim 16 --check

candidate_count: 4
seed_guardrail_rejections: 0
laurent_orthogonal_candidates: 4
closed_candidates: 0
effective_6_4_candidates: 0
strict_bridge_candidates: 0
route_label: split_step_cap_boundary
```

All four candidates hit `split_step_coefficient_cap_boundary`: the exact
coefficients are orthogonal and projector-free at the raw-matrix level, but
their coefficient algebra already exceeds the bounded guardrail cap.

After adding the rational sparse span kernel, the first split-step candidate
can be pushed deeper:

```text
uv run python scripts/bloch_two_site.py --split-step-search --max-candidates 1 --max-generated-algebra-dim 32 --split-step-coefficient-algebra-dim 64 --check

candidate_count: 1
laurent_orthogonal_candidates: 1
closed_candidates: 1
route_label: split_step_cap_boundary
candidate: uniform_sublattice_swap, ..., coef_dim=20, coef_closed=true, dim=10, closed=true, center=10, center_solved=false, label=split_step_center_cap_boundary
```

So the first candidate is no longer stuck at the coefficient guardrail:
coefficient algebra closes at dimension `20`, and the sampled Bloch algebra
closes at dimension `10`. The remaining boundary is the generic central
idempotent solver, whose current polynomial solve is too expensive at center
dimension `10`. This is not a bridge failure theorem. It says the next
calculation improvement should target center-idempotent extraction from a
known finite-dimensional center, not broader candidate enumeration.
