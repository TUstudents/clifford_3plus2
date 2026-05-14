# Defect Beta Report

Status: second physical primitive family implemented.

This report records the Defect-β family implemented in
[`src/clifford_3plus2_d5/qca/defect_beta.py`](../../src/clifford_3plus2_d5/qca/defect_beta.py)
and exposed by
[`scripts/defect_beta_search.py`](../../scripts/defect_beta_search.py).

## Purpose

Defect-β is the topological comparison route to Floquet-α. Instead of
declaring a block projector, it declares a wall cycle with transition functions
in `O(10)` and computes the round-trip monodromy from their product.

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

Their product is the monodromy:

```text
M_defect = T_exit T_entry
```

The search enumerates the ten choices of which three mode pairs carry the
`omega` defect charge.

## Command

```bash
uv run python scripts/defect_beta_search.py --check
```

Current output:

```text
candidate_count: 10
monodromy_candidates: 10
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'monodromy_j_produced_not_strictly_unique': 10}
load_bearing_qca_bridge: false
```

Each candidate has:

```text
transition_count = 2
omega_projector_rank = 6
i_projector_rank = 4
central_idempotent_ranks = [0, 4, 6, 10]
lower_rank_central_idempotents = 0
canonical_j_generated_by_monodromy = true
strict_compatible_j_forced = false
```

## Interpretation

Defect-β confirms that the α+ obstruction is not an artifact of Floquet
notation. A computed monodromy can produce the same coarse `6+4` central
idempotent lattice and a canonical spectral `J`, but the strict compatible
commutant still does not collapse to `±J`.

The bridge remains non-load-bearing:

```text
load_bearing_qca_bridge = false
```

The next decision is conceptual: accept a canonical spectral/monodromy `J` as
rule-produced, or require an additional microscopic orientation constraint
that removes block-sign alternatives.
