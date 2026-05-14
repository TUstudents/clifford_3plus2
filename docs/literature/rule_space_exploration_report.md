# Rule-Space Exploration Report

Status: E1 bounded exploration sprint complete.

This report records the first non-bookkeeping exploration layer implemented in
[`src/clifford_3plus2_d5/explore`](../../src/clifford_3plus2_d5/explore) and
exposed by [`scripts/explore_rule_space.py`](../../scripts/explore_rule_space.py).

## Purpose

The previous phases built exact guardrails and candidate certificates. E1
starts bounded search over microscopic rule spaces:

```text
generate primitive sets
enumerate exact finite-depth words
test J and period-four identities
test seeded projector discovery
run addressability and normalizer filters
record rejection reasons and survivors
```

This is still not a bridge proof. It is a reproducible exploration sprint.

## Search Bounds

Default command:

```bash
uv run python scripts/explore_rule_space.py --check --output-dir data/exploration
```

Default bounds:

```text
max_depth = 4
max_primitive_set_size = 4
max_primitive_sets = 200
max_words = 50000
max_survivors = 50
max_rejections = 200
```

## Rule-Space Families

The E1 rule space includes:

```text
identity
global clock tick
minus identity
whole-block reflection
within-block mode swaps
rank-one pair falsifier
rank-one color-control falsifier
rank-one weak-control falsifier
off-block-control falsifier
```

The standard `P_3/P_2` projectors are used as seeded test controls for this
first sprint. That makes projector discovery a sanity check, not a derivation.

## Default Result

Current output:

```text
rule_space_name: e1_bounded_real_rule_space
primitive_family_count: 5
primitive_set_count: 10
primitive_sets_scanned: 10
words_scanned: 170
j_hits: 73
period_four_hits: 73
split_candidates: 170
addressability_safe_hits: 158
normalizer_candidate_hits: 158
forced_candidate_hits: 0
rank_one_rejections: 38
off_block_rejections: 4
normalizer_too_large_rejections: 158
surviving_candidates: 0
exploration_check_passed: true
load_bearing_qca_bridge: false
```

Top rejection reasons:

```text
normalizer_too_large = 158
no_j = 97
no_period_four = 97
rank_one_addressability = 38
unsafe_addressability = 12
normalizer_falsified = 12
off_block_addressability = 4
```

## Artifacts

The committed deterministic artifacts are:

```text
data/exploration/e1_summary.json
data/exploration/e1_survivors.jsonl
data/exploration/e1_rejections.jsonl
```

The survivor file is currently empty:

```text
surviving_candidates = 0
```

## Interpretation

E1 found many algebraic `J` and period-four hits, but none survived forcedness:

```text
forced_candidate_hits = 0
load_bearing_qca_bridge = false
```

This is useful negative evidence for the bounded search space. The explored
candidate words mostly fail because the normalizer remains too large. The
forbidden control families are rejected as expected.

## Follow-Up

E2 performs that unseeded projector pass. See
[Unseeded Projector Discovery Report](unseeded_projector_discovery_report.md).
The current E2 result finds no unseeded complementary `6+4` projector pair.
Until such projectors and `J` are forced by source-backed rule data, the bridge
remains non-load-bearing.
