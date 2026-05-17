# Forced J Report

Phase 2 adds exact checks for whether declared gate words can produce the
standard clock complex structure

```text
J = epsilon ⊗ I_5.
```

This phase still does not prove that microscopic QCA rules force `J`.

## Candidate

The default candidate gate set contains one declared global clock tick:

```text
global_clock_tick = J.
```

The checker verifies:

```text
J^T J = I
det(J) = 1
J^2 = -I
```

It also confirms that the declared one-gate word generates the Phase 1
standard `J`.

## Period-Four Interpretation

The global clock tick satisfies:

```text
U(T/4) = J
U(T/2) = J^2 = -I on K_x
U(T) = J^4 = I on K_x.
```

This is only an algebraic micromotion candidate. It is not yet a microscopic
finite-depth QCA rule.

## Addressability Falsifier

The checker also exposes the rank-one pair-rotation obstruction. If the rule
set includes independently addressable gates

```text
J_1, J_2, J_3, J_4, J_5,
```

then the candidate is marked `falsified`, because the rule data can resolve
individual pairs inside the future `3` and `2` blocks.

The falsifier is checked with:

```bash
uv run python scripts/forced_j_check.py --include-addressable-rank-one --expect-verdict falsified
```

## Current Status

```text
forced_j_check_passed: true
generated_by_gate_word: true
qca_forces_j: false
forced_j_verdict: candidate_only
load_bearing_qca_bridge: false
```

The checker can certify a declared gate word. Phase 5 must still supply a real
finite-depth microscopic QCA architecture that forces this word as an
indivisible rule-layer.
