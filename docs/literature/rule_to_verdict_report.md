# Rule To Verdict Report

Status: unified physics-interface checker implemented.

This report records the collapsed end-to-end checker implemented in
[`src/clifford_3plus2_d5/qca/rule_verdict.py`](../../src/clifford_3plus2_d5/qca/rule_verdict.py)
and exposed by
[`scripts/rule_to_verdict.py`](../../scripts/rule_to_verdict.py).

## Purpose

The actual bridge question is now evaluated as one rule-to-verdict function.
Input is a finite list of real on-site `10 x 10` layer matrices with locality
metadata. Output is:

```text
Floquet spectrum
generated real algebra dimension
center of the generated algebra
central idempotent ranks
rank-(6,4) complementary pairs
lower-rank central idempotent witnesses
rule-generated complex structures
compatible complex-structure solve status
one bridge-candidate boolean
```

The Spinor-16 and hypercharge table are intentionally outside this verdict.
They remain downstream bookkeeping once `J` and the coarse center are genuinely
rule-derived.

## Current Built-In Controls

Minimal period-four clock:

```bash
uv run python scripts/rule_to_verdict.py --case minimal-period-four --expect-verdict falsified_no_rank_6_4_center
```

Outcome:

```text
generated_algebra_dimension: 2
center_dimension: 2
central_idempotent_ranks: [0, 10]
generated_complex_structures: 2
forced_j_found: false
verdict: falsified_no_rank_6_4_center
load_bearing_qca_bridge: false
```

Clock plus whole-block reflection:

```bash
uv run python scripts/rule_to_verdict.py --case clock-block-reflection --expect-verdict candidate_only_j_not_forced
```

Outcome:

```text
generated_algebra_dimension: 4
center_dimension: 4
central_idempotent_ranks: [0, 4, 6, 10]
complementary_rank_6_4_pairs: 1
lower_rank_central_idempotents: 0
generated_complex_structures: 4
compatible_j_solved: false
forced_j_found: false
verdict: candidate_only_j_not_forced
load_bearing_qca_bridge: false
```

Clock plus rank-one color reflection:

```bash
uv run python scripts/rule_to_verdict.py --case clock-rank-one-color-reflection --expect-verdict falsified_rank_one_center
```

Outcome:

```text
central_idempotent_ranks: [0, 2, 8, 10]
lower_rank_central_idempotents: 1
verdict: falsified_rank_one_center
load_bearing_qca_bridge: false
```

## Honest Limitation

The checker solves the nonlinear `J` equations exactly only when the generated
algebra or compatible centralizer basis is below the configured solve bound.
When the compatible centralizer is too large, it reports `compatible_j_solved:
false` and refuses to mark `J` forced.

This is intentional. A large compatible centralizer is precisely the current
failure mode: the rule has not selected a unique complex structure.

## Interpretation

This script is now the preferred interface for new microscopic candidate
families. E1/E2 remain useful as historical exploration reports, but new
Floquet-polarization or defect-monodromy candidates should be fed through the
single rule-to-verdict checker first.

The first active replacement family is
[Floquet Alpha](floquet_alpha_report.md). It is no longer a search over the
abstract E1 primitives; it exposes one mandatory quantized resonance layer per
candidate. Alpha-plus also extracts a canonical spectral-polarization `J` from
that layer, while the strict rule-to-verdict checker continues to report that
compatible `J` is not unique in the full commutant.
