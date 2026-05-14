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

## Command

```bash
uv run python scripts/defect_beta_search.py --check
```

Current output:

```text
candidate_count: 10
monodromy_candidates: 10
scaled_monodromy_certified_candidates: 10
generated_j_moduli_dimension: 0
compatible_centralizer_dimension: 26
compatible_j_moduli_dimension: 9
locality_radius_bound: 0
local_compatible_operator_dimension: 4
local_compatible_j_moduli_dimension: 0
local_compatible_complex_structure_count: 4
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'monodromy_j_produced_not_strictly_unique': 10}
load_bearing_qca_bridge: false
```

Each candidate has:

```text
transition_count = 2
entry_exit_transitions_distinct = true
transition_determinants = [-1, -1]
clutching_identity_passed = true
omega_projector_rank = 6
i_projector_rank = 4
central_idempotent_ranks = [0, 4, 6, 10]
lower_rank_central_idempotents = 0
scaled_monodromy_certified = true
canonical_j_generated_by_monodromy = true
generated_j_moduli_dimension = 0
compatible_centralizer_dimension = 26
compatible_j_moduli_dimension = 9
locality_radius_bound = 0
local_compatible_operator_dimension = 4
local_compatible_j_moduli_dimension = 0
local_compatible_complex_structure_count = 4
strict_compatible_j_forced = false
```

## Interpretation

Defect-β confirms that the α+ obstruction is not an artifact of Floquet
notation. It is not merely the old identical-half factorization: the entry and
exit transitions are different orientation-reversing wall maps. The computed
monodromy still has the same spectral data as Floquet-α, so the same strict
compatible-commutant obstruction remains: `M_3(C) ⊕ M_2(C)` has real
dimension `26`, and the compatible orthogonal complex structures contain the
continuous `U(3)/O(3) × U(2)/O(2)` family of dimension `9`.
The locality-restricted compatible search uses the rule-generated local center;
it has dimension `4`, moduli dimension `0`, and four local compatible complex
structures. The strict bridge still fails because this is not a unique `±J`.

The bridge remains non-load-bearing:

```text
load_bearing_qca_bridge = false
```

The next decision is conceptual: accept a canonical spectral/monodromy `J` as
rule-produced, or require an additional microscopic orientation constraint
that removes the remaining local discrete ambiguity.
