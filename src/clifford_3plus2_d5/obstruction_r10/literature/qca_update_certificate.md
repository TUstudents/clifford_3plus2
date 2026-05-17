# QCA Update Certificate

Phase 5 adds an exact certificate framework for finite-depth QCA update
candidates. The default candidate is the minimal period-four carrier drive:

```text
U_1 = J
U_2 = J
U_3 = J
U_4 = J
```

so that

```text
U(T/4) = J
U(T/2) = -I
U(T) = I.
```

This phase still does not prove that microscopic QCA rules force the global
clock tick as an indivisible rule-layer.

## Layer Certificate

Each local gate records:

```text
name
support
matrix shape
locality radius
is_real
is_orthogonal
internal_classification
```

Each layer records:

```text
supports_are_disjoint
locality_check_passed
is_real_orthogonal
```

The default candidate is a one-site carrier micromotion certificate. It is
finite-depth algebraically, but it is not a source-backed lattice QCA.

## No-Locking Falsifiers

The certificate rejects spacetime shifts whose internal actions are not in the
safe SM commutant:

```text
rank-one color shift
rank-one weak shift
block-mixing shift
```

The falsifiers are checked with:

```bash
uv run python scripts/qca_update_check.py --include-rank-one-color-shift --expect-verdict falsified
uv run python scripts/qca_update_check.py --include-rank-one-weak-shift --expect-verdict falsified
```

## Current Status

The default check is:

```bash
uv run python scripts/qca_update_check.py --check
```

```text
finite_depth: true
layer_count: 4
max_locality_radius: 0
all_layers_local: true
all_layers_orthogonal: true
period_four_check_passed: true
quarter_period_is_j: true
half_period_is_minus_identity: true
full_period_is_identity: true
all_internal_actions_safe: true
qca_rule_forces_update: false
finite_depth_qca_verdict: candidate_only
load_bearing_qca_bridge: false
```

The checker certifies the declared update candidate and its no-locking
falsifiers. It does not certify that real microscopic QCA data force the
update.
