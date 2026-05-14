# Floquet Alpha Report

Status: first physical primitive family implemented.

This report records the Floquet-α family implemented in
[`src/clifford_3plus2_d5/qca/floquet_alpha.py`](../../src/clifford_3plus2_d5/qca/floquet_alpha.py)
and exposed by
[`scripts/floquet_alpha_search.py`](../../scripts/floquet_alpha_search.py).

## Purpose

E1/E2 used abstract primitive classes such as identity, global clock,
whole-block reflection, mode permutation, and rank-one pair controls. Those
classes are now historical controls. The active search should use physical
mechanisms that create `3` versus `2` asymmetry before projector discovery.

Floquet-α uses one mandatory quantized on-site resonance layer. It does not
expose individual pair rotations as independently switchable controls.

## Family

For each mode pair `(x_k, y_k)`, define the exact real rotation:

```text
R_k(theta) = exp(theta J_k)
```

with quantized phases:

```text
alpha phase = 2 pi / 3
eta phase = pi / 2
```

A candidate is one mandatory layer:

```text
U_alpha = product over 3 modes R_k(2 pi / 3)
          product over 2 modes R_k(pi / 2)
```

The search enumerates the ten resonance patterns choosing which three of the
five mode pairs carry the alpha phase.

## Command

```bash
uv run python scripts/floquet_alpha_search.py --check
uv run python scripts/floquet_alpha_plus_search.py --check
uv run python scripts/floquet_alpha_second_layer_search.py --check
```

Current output:

```text
candidate_count: 10
rank_6_4_pair_candidates: 10
rank_one_falsified_candidates: 0
bridge_candidates: 0
verdict_counts: {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge: false
```

Each pattern has:

```text
center_ranks = [0, 4, 6, 10]
complementary_rank_6_4_pairs = 1
forced_j = false
verdict = candidate_only_j_not_forced
```

The alpha-plus polarization extraction computes spectral projectors from the
real minimal polynomials:

```text
f_alpha(x) = x^2 + x + 1
f_eta(x) = x^2 + 1
-x f_alpha + (x + 1) f_eta = 1
```

so:

```text
P_alpha = (U + I)(U^2 + I)
P_eta = I - P_alpha
K_alpha = (2U + I) P_alpha
K_alpha^2 = -3 P_alpha
J_eta = U P_eta
J_eta^2 = -P_eta
J_alpha = K_alpha / sqrt(3)
J = J_alpha + J_eta
```

The implementation declares the exact working field as `QQ(zeta_12)`, with
real matrix entries in `QQ(sqrt(3))`. The load-bearing certificate uses
`K_alpha`, not a rational-basis solve for the normalized `J_alpha`; the
normalized `J` is a derived algebraic-field object after the scaled identity is
certified.

Current alpha-plus output:

```text
candidate_count: 10
polarization_j_candidates: 10
scaled_polarization_certified_candidates: 10
generated_j_moduli_dimension: 0
compatible_centralizer_dimension: 26
compatible_j_moduli_dimension: 9
locality_radius_bound: 0
local_compatible_operator_dimension: 4
local_compatible_j_moduli_dimension: 0
local_compatible_complex_structure_count: 4
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'polarization_j_produced_not_strictly_unique': 10}
load_bearing_qca_bridge: false
```

## Interpretation

Floquet-α is progress relative to E1/E2 because the coarse `6+4` central
idempotent pair is produced by an exact spectral resonance pattern rather than
by seeding `P_3/P_2` directly.

It still does not solve the bridge:

```text
canonical_j_generated_by_floquet = true
generated_j_moduli_dimension = 0
compatible_centralizer_dimension = 26
compatible_j_moduli_dimension = 9
locality_radius_bound = 0
local_compatible_operator_dimension = 4
local_compatible_j_moduli_dimension = 0
local_compatible_complex_structure_count = 4
strict_compatible_j_forced = false
load_bearing_qca_bridge = false
```

The important obstruction is now explicit. The oriented Floquet branch produces
an exact scaled polarization operator as a polynomial in the mandatory rule
operator. The normalized `J` then lives in `QQ(sqrt(3))`. The full
compatible centralizer is still large: `M_3(C)` on the alpha sector and
`M_2(C)` on the eta sector, real dimension `18 + 8 = 26`. The compatible
orthogonal complex structures form a continuous family of dimension
`dim U(3)/O(3) + dim U(2)/O(2) = 6 + 3 = 9`. Alpha-plus is therefore progress
on rule-produced `J`, but not a strict uniqueness proof.

The `local_compatible_*` fields are a rule-generated local-center falsifier,
not a complete search over all on-site local operators. For radius-0 α, all
on-site local operators would be `M_10(R)`. The checker instead asks whether
the rule's own local algebra singles out `J`. It does not: inside the
rule-generated local center the operator dimension is `4`, the moduli
dimension is `0`, and there are four rule-generated local complex structures
rather than a unique `±J`.

The next pressure is to decide whether the physical theorem accepts this
spectral-polarization `J` as forced by the rule, or whether a stricter
microscopic constraint must eliminate the remaining local discrete ambiguity.

## Cycle/Swap Second-Layer Check

The proposed next strict-standard move was to add a second mandatory layer
`V` that commutes with `U`, cycles the three alpha mode-pairs, and swaps the
two eta mode-pairs:

```text
[U,V] = 0
V^3 P_alpha = P_alpha
V^2 P_eta = P_eta
```

This is now implemented as a checked negative via
[`scripts/floquet_alpha_second_layer_search.py`](../../scripts/floquet_alpha_second_layer_search.py).

Current output:

```text
commuting_second_layer_candidates: 10
order_certified_candidates: 10
compatible_centralizer_collapsed_candidates: 10
generated_algebra_dimension: 10
center_dimension: 10
compatible_centralizer_dimension: 10
explicit_lower_rank_projector_ranks: [2, 2, 2]
no_locking_guardrail_passed_candidates: 0
strict_bridge_candidates: 0
load_bearing_qca_bridge: false
```

The layer succeeds at being mandatory, local, real-orthogonal, commuting with
`U`, and reducing the compatible centralizer from `26` to `10`. It does not
reach the bridge. Since `U` and `V` commute, the generated algebra is
commutative; non-scalar spectral data of `V` inside the alpha/eta blocks
produce central idempotents. The explicit projectors include three rank-2
witnesses, so the no-locking guardrail rejects the rule.

This corrects the naive target count. A 3-cycle on the complex alpha block has
three characters, so its centralizer in `M_3(C)` has complex dimension `3`,
not `2`; together with the eta swap block this gives real dimension `10`, not
`8`.
