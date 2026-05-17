# Forcedness Certificate

Status: candidate-only.

This certificate summarizes the Phase 7 forcedness verdict. It is intentionally
separate from the earlier algebraic candidate checks.

## Current Certificate

```text
rule_data_source: candidate_only
candidate_j_valid: true
candidate_split_valid: true
normalizer_preserves_declared_split: true
candidate_j_preserved_by_normalizer: false
continuous_j_alternatives_not_excluded: true
rank_three_projector_family_not_excluded: true
addressability_algebra_safe: true
j_unique_or_forced: false
split_unique_or_forced: false
forcedness_verdict: candidate_only
load_bearing_qca_bridge: false
```

## Meaning

The current data pass the exact candidate checks:

- `J^2 = -I`,
- `J` is real orthogonal,
- `P_3/P_2` are exact complementary projectors,
- `P_3/P_2` commute with `J`,
- the default addressability algebra is only the span of `P_3` and `P_2`.

But the current data do not prove forcedness:

- no source-backed microscopic rule data are present,
- the normalizer does not force the candidate `J`,
- rank-three `J`-invariant alternatives are not excluded by rule data,
- the QCA bridge remains non-load-bearing.

## Strong Pass Target

A future strong pass must satisfy:

```text
rule_data_source != candidate_only
qca_rule_forces_j = true
qca_rule_forces_split = true
j_unique_or_forced = true
split_unique_or_forced = true
addressability_algebra_safe = true
forcedness_verdict = forced
```

Even then, the bridge should not be marked load-bearing until the upstream QCA
rule data are auditable and the earlier finite-depth, locality, and gate
classifier certificates remain valid.

## Immediate Falsifiers

The following are fatal for the current route:

```text
rank_one_color_projectors_addressable = true
rank_one_weak_projectors_addressable = true
off_block_controls_addressable = true
non_scalar_block_controls_addressable = true
```

Any of these means the microscopic controls can see more than the allowed
coarse `3` versus `2` split, or can mix the two blocks directly.
