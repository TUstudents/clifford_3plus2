# Unseeded Projector Discovery Report

Status: E2 bounded projector discovery sprint complete.

This report records the second exploration layer implemented in
[`src/clifford_3plus2_d5/explore`](../../src/clifford_3plus2_d5/explore) and
exposed by
[`scripts/discover_projectors.py`](../../scripts/discover_projectors.py).

## Purpose

E1 still reused the standard `P_3/P_2` projectors as seeded controls. E2
removes those seeded projectors from the main discovery pass and asks whether
small exact rule-generated algebras produce a central complementary
rank-`6+4` projector pair.

The search is still bounded. It is negative evidence inside the declared
search space, not a theorem about every possible QCA.

## Search Modes

The default mode is unseeded:

```bash
uv run python scripts/discover_projectors.py --check --mode unseeded --expect-verdict not_found --output-dir data/exploration
```

Default bounds:

```text
max_word_depth = 2
max_basis_size = 6
coefficient_values = -1, -1/2, 0, 1/2, 1
max_candidates = 200
```

The mode scans primitive sets without standard `P_3/P_2` addressable
operators:

```text
identity_only_unseeded
clock_only_unseeded
clock_plus_minus_identity_unseeded
clock_plus_color_swap_unseeded
clock_plus_weak_swap_unseeded
rank_one_pair_falsifier_unseeded
```

Two non-load-bearing comparison modes are also implemented:

```bash
uv run python scripts/discover_projectors.py --check --mode sanity-seeded --expect-verdict projector_pair_found --output-dir /tmp/e2_sanity
uv run python scripts/discover_projectors.py --check --mode block-reflection-candidate --expect-verdict projector_pair_found --output-dir /tmp/e2_block
```

The seeded sanity mode proves the checker can recognize the standard pair when
`P_3/P_2` are inserted. The block-reflection mode proves a coarse involution
can synthesize the pair. Neither is a QCA derivation unless the inserted data
are themselves forced by microscopic rules.

## Default Result

Current unseeded output:

```text
mode: unseeded
primitive_sets_scanned: 6
algebra_elements_considered: 1924
candidate_projectors: 3
rank_2_projectors: 3
rank_6_projectors: 0
rank_4_projectors: 0
complementary_pairs: 0
unsafe_rank_one_projectors: 3
unseeded_projector_pairs_found: 0
seeded_projector_pairs_found: 0
block_reflection_pairs_found: 0
discovery_verdict: not_found
discovery_check_passed: true
load_bearing_qca_bridge: false
```

The three discovered projectors are rank-2 projectors. They are recorded as
unsafe rank-one structure, not as evidence for a valid `3+2` split.

## Falsifier Control

The explicit rank-one color-control set is deliberately unsafe:

```text
candidate_projectors = 11
rank_2_projectors = 3
rank_6_projectors = 4
rank_4_projectors = 4
complementary_pairs = 4
unsafe_rank_one_projectors = 3
discovery_check_passed = false
```

This demonstrates why discovering `P_3/P_2` is not enough. A rule set can build
the standard block pair by first resolving individual color axes, which
violates the no-locking requirement.

## Artifacts

The committed deterministic E2 artifacts are:

```text
data/exploration/e2_summary.json
data/exploration/e2_projector_candidates.jsonl
data/exploration/e2_rejections.jsonl
```

## Interpretation

E2 removes the biggest E1 weakness: the default projector search no longer
starts with the answer. Within the current bounded unseeded search, no central
complementary rank-`6+4` projector pair is derived.

This keeps the bridge non-load-bearing:

```text
unseeded_projector_pairs_found = 0
load_bearing_qca_bridge = false
```

The next exploration pressure is to broaden the primitive families while
keeping the same guardrail: a valid hit must derive only the coarse `P_3/P_2`
pair and must not derive rank-one projectors inside either block.
