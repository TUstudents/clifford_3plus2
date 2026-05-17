# Gate Classification Report

Phase 4 adds the exact SM commutant oracle for geometric internal gates on

```text
W = C^3 ⊕ C^2.
```

The safe target algebra is

```text
Comm(SU(3) x SU(2) x U(1)) = C P_3 ⊕ C P_2.
```

This phase still does not prove that microscopic QCA rules supply only safe
geometric gates.

## Exact Gauge Generators

The classifier uses exact complex `5 x 5` matrices:

```text
SU(3): E_12, E_21, E_23, E_32, H_12, H_23
SU(2): E_45, E_54, H_45
Y: diag(-1/3,-1/3,-1/3,1/2,1/2)
```

A complex matrix is safe iff it commutes with all of these generators. The
implemented expected answer is exactly the block scalar algebra

```text
C P_3 ⊕ C P_2.
```

## Real Gate Classifier

A real `10 x 10` internal gate is first checked against the Phase 1 complex
structure `J`.

```text
[M,J] = 0
```

If this fails, the gate is classified as `antilinear_fail`. Otherwise the gate
is converted to a complex `5 x 5` matrix and classified as:

```text
safe_sm_commutant
block_mixing_fail
color_breaking_fail
weak_breaking_fail
antilinear_fail
unknown_fail
```

## Current Status

```text
P_3: safe_sm_commutant
P_2: safe_sm_commutant
J_P_3: safe_sm_commutant
J_P_2: safe_sm_commutant
block_mixer: block_mixing_fail
rank_one_color_projector: color_breaking_fail
rank_one_weak_projector: weak_breaking_fail
real_conjugation: antilinear_fail
commutant_basis_matches_expected: true
safe_algebra_closure_passed: true
gate_classification_check_passed: true
qca_geometric_gate_algebra_safe: false
load_bearing_qca_bridge: false
```

The default classifier run includes unsafe witnesses on purpose. It verifies
that the oracle rejects them; it does not certify an actual QCA gate set.
