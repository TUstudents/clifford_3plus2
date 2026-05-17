# Real-QCA Branch Report

Status: Phase 8A candidate-only branch checker.

This report records the stronger real-QCA-first branch implemented in
[`src/clifford_3plus2_d5/search/real_qca_branch.py`](../../src/clifford_3plus2_d5/search/real_qca_branch.py)
and exposed by
[`scripts/real_qca_branch_check.py`](../../scripts/real_qca_branch_check.py).

## Purpose

Phase 7 showed that the current candidate data preserve the declared
`P_3/P_2` split but do not force the candidate `J` or the rank-three projector.
Phase 8A therefore starts the fallback program with the preferred branch:

```text
stronger real-QCA first route
```

The branch checker tests finite-depth real gate words on `R^10` together with
the existing no-locking addressability checks. It is an interface for testing
source-backed rule spaces; it is not a new bridge proof.

## Default Output

Command:

```bash
uv run python scripts/real_qca_branch_check.py --check --expect-verdict candidate_only
```

Important output:

```text
candidate_word_found: true
candidate_word: global_clock_tick
word_depth: 1
finite_depth: true
translation_invariant: true
generates_j: true
generates_split: true
j_forced_by_rule_space: false
split_forced_by_rule_space: false
forbidden_rank_one_controls_present: false
forbidden_off_block_controls_present: false
addressability_algebra_safe: true
normalizer_verdict: candidate_only
real_qca_branch_verdict: candidate_only
real_qca_branch_check_passed: true
load_bearing_qca_bridge: false
```

Interpretation:

- The declared period-four clock primitive still produces `J`.
- The declared projectors still provide a candidate split.
- The current rule space does not force `J`.
- The current rule space does not force the split.
- The bridge remains non-load-bearing.

## Falsifiers

The branch checker rejects the standard forbidden controls:

```bash
uv run python scripts/real_qca_branch_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-rank-one-weak --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-rank-one-pair --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-off-block --expect-verdict falsified
```

Rank-one color, rank-one weak, and rank-one pair controls set:

```text
forbidden_rank_one_controls_present: true
real_qca_branch_verdict: falsified
```

Off-block controls set:

```text
forbidden_off_block_controls_present: true
real_qca_branch_verdict: falsified
```

## Acceptance Gap

A future strong pass must use auditable microscopic rule data and satisfy:

```text
candidate_word_found = true
finite_depth = true
translation_invariant = true
generates_j = true
generates_split = true
j_forced_by_rule_space = true
split_forced_by_rule_space = true
addressability_algebra_safe = true
real_qca_branch_verdict = forced_candidate
```

Even that would still need a final bridge review before changing
`load_bearing_qca_bridge`.
