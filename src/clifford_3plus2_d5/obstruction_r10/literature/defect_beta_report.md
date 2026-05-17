# Defect Beta Report

Status: implemented but parked as a load-bearing route.

This report records the Defect-β family implemented in
[`src/clifford_3plus2_d5/qca/defect_beta.py`](../../src/clifford_3plus2_d5/qca/defect_beta.py)
and exposed by
[`scripts/defect_beta_search.py`](../../scripts/defect_beta_search.py).

## Purpose

Defect-β is retained as a topological comparison route to Floquet-α. Instead
of declaring a block projector, it declares a wall cycle with transition
functions in `O(10)` and computes the round-trip monodromy from their product.
After the commuting second-layer check, β is not treated as the active
load-bearing path; it remains a regression target until it is rebuilt as a
genuine higher-dimensional defect calculation.

The intended spectral pattern is:

```text
omega, omega, omega, i, i
omega = exp(2 pi i / 3)
```

## Family

For each candidate, the wall cycle has two mandatory transition functions:

```text
T_entry
T_exit
```

They are distinct orientation-reversing wall-chart maps. The implementation
uses the clutching reflection:

```text
C = diag(-I_5, I_5)
C^2 = I
det(C) = -1
```

and a monodromy core `M`:

```text
M = diag(omega, omega, omega, i, i) in real pair-rotation form
T_entry = C
T_exit = M C
T_entry != T_exit
det(T_entry) = det(T_exit) = -1
```

Their product is the monodromy by the wall clutching identity:

```text
M_defect = T_exit T_entry
         = M C C
         = M
```

The search enumerates the ten choices of which three mode pairs carry the
`omega` defect charge.

The exact working field is `QQ(zeta_12)`, with real transition entries in
`QQ(sqrt(3))`. The current certificate avoids a half-angle implementation, so
no `sqrt(2)` extension is introduced by the wall transitions.

The round-trip monodromy is exactly the matching Floquet-α operator. β can
therefore differ from α only when the checker is run on the transition
functions themselves, not on the composed monodromy as a single layer. The
current rule-to-verdict path uses `(T_entry, T_exit)`.

## Command

```bash
uv run python scripts/defect_beta_search.py --check
```

Current output:

```text
candidate_count: 10
monodromy_candidates: 10
scaled_monodromy_certified_candidates: 10
monodromy_equals_matching_floquet_alpha: true
transition_functions_commute: false
rule_generated_algebra_dimension: 8
rule_center_dimension: 2
rule_center_solved: true
generated_j_moduli_dimension: 0
compatible_centralizer_dimension: 13
compatible_j_moduli_dimension: None
locality_radius_bound: 1
local_compatible_operator_dimension: 2
local_compatible_j_moduli_dimension: None
local_compatible_complex_structure_count: 0
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge: false
```

Each candidate has:

```text
transition_count = 2
entry_exit_transitions_distinct = true
transition_functions_commute = false
transition_determinants = [-1, -1]
clutching_identity_passed = true
monodromy_equals_matching_floquet_alpha = true
rule_generated_algebra_dimension = 8
rule_center_dimension = 2
omega_projector_rank = 6
i_projector_rank = 4
central_idempotent_ranks = [0, 4, 6, 10]
lower_rank_central_idempotents = 0
scaled_monodromy_certified = true
canonical_j_generated_by_monodromy = true
generated_j_moduli_dimension = 0
compatible_centralizer_dimension = 13
compatible_j_moduli_dimension = None
locality_radius_bound = 1
local_compatible_operator_dimension = 2
local_compatible_j_moduli_dimension = None
local_compatible_complex_structure_count = 0
strict_compatible_j_forced = false
```

## Interpretation

Defect-β is not independent evidence at the monodromy level: the computed
round trip is exactly the matching Floquet-α operator. The entry and exit
transitions are different orientation-reversing wall maps and do not commute,
so the transition-pair rule is algebraically distinct from the one-layer
Floquet rule. Running the checker on `(T_entry, T_exit)` gives generated
algebra dimension `8`, center dimension `2`, and compatible centralizer
dimension `13`. It still does not force `J`: the local compatible center has
dimension `2` and contains no rule-generated compatible complex structures.

So β remains a useful regression target for transition-pair plumbing, but it
is not a load-bearing independent route. A genuine β route must use source
data where the transition algebra, not just the round-trip monodromy, supplies
new microscopic structure.

The bridge remains non-load-bearing:

```text
load_bearing_qca_bridge = false
```

The next decision is conceptual: either build a genuine higher-dimensional
defect rule whose transition algebra is not just a repackaging of α, or drop β
from the active route and keep it only as a regression target.
