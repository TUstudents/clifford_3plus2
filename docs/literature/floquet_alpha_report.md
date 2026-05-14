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

## Interpretation

Floquet-α is progress relative to E1/E2 because the coarse `6+4` central
idempotent pair is produced by an exact spectral resonance pattern rather than
by seeding `P_3/P_2` directly.

It still does not solve the bridge:

```text
forced_j_found = false
load_bearing_qca_bridge = false
```

The next pressure is to add a microscopic constraint that makes the quantized
resonance layer itself unavoidable and selects a unique compatible complex
structure, rather than merely producing the coarse center.
