# Real Carrier Report

Phase 1 implements the exact algebraic substrate for the enhanced J-first
attack. It does not prove that QCA dynamics force the complex structure or the
`3+2` split.

## Carrier

The real carrier is:

```text
K_x = R^2_clock ⊗ (R^3 ⊕ R^2)
dim_R K_x = 10
```

The ordered real basis is:

```text
x_1, x_2, x_3, x_4, x_5,
y_1, y_2, y_3, y_4, y_5
```

## Exact Ansatz

The exact quarter-turn matrix is:

```text
epsilon = [[0, -1],
           [1,  0]]
```

The candidate complex structure is:

```text
J = epsilon ⊗ I_5.
```

The candidate block projectors are:

```text
P_3 = I_2 ⊗ diag(1,1,1,0,0)
P_2 = I_2 ⊗ diag(0,0,0,1,1)
```

## Verified Identities

The Phase 1 checker verifies:

```text
J^2 = -I_10
J^T J = I_10
det(J) = 1
P_3 + P_2 = I_10
P_i^2 = P_i
P_3 P_2 = 0
rank_R(P_3) = 6
rank_R(P_2) = 4
[J, P_3] = [J, P_2] = 0
```

## Interpretation

This phase gives the exact real algebra needed for later work. It does not
show that `J` is a microscopic QCA gate, a quarter-period micromotion, or a
forced monodromy. It also does not show that the QCA rule data force only
`P_3` and `P_2`.

Phase 2 must prove or falsify whether QCA data generate `J`. Phase 3 must
prove or falsify whether the split is structural and avoids rank-one
within-block addressability.

Current Phase 1 status:

```text
phase_1_real_carrier_check_passed: true
qca_forces_j: false
load_bearing_qca_bridge: false
```
