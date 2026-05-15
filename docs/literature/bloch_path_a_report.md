# Bloch Path-A Report

Status: sampled Bloch sidecar implemented; no unseeded bridge candidate.

This report records the first Path-A checker implemented in
[`src/clifford_3plus2_d5/qca/bloch_rule.py`](../../src/clifford_3plus2_d5/qca/bloch_rule.py)
and exposed by
[`scripts/bloch_path_a_search.py`](../../scripts/bloch_path_a_search.py).

## Purpose

The on-site block-preserving bridge problem is closed by the three no-go
propositions in [Theory](../theory.md). Path A drops the single-site
hypothesis. Instead of one algebra `A subset M_10(R)`, it studies a sampled
Bloch family:

```text
T(z) = sum_s A_s z^s
A_j = R<T(zeta_period^j)>
```

The target is a finite-radius 1D QCA whose sampled Bloch algebras produce a
rank-`(6,4)` band split, a rule-generated compatible `J(k)` section, and
global `±J` transport, without seeding `P_alpha/P_eta` as coefficients.

## Guardrail

The checker rejects both direct and hidden seeding:

```text
raw coefficient equal to P_alpha/P_eta
raw diagonal projector coefficient of intermediate rank
coefficient algebra generates P_alpha/P_eta
```

This matters because the previous spatial sidecar and the combined Route-1/2
sidecar have the right topological sign shape, but their coefficient algebra
already generates the coarse projectors.

## Command

```bash
uv run python scripts/bloch_path_a_search.py --check
```

Current output:

```text
candidate_count: 6
seed_guardrail_rejections: 4
unseeded_candidate_count: 2
stable_6_4_band_candidates: 0
topological_pm_candidates: 4
rule_generated_j_section_candidates: 0
strict_bridge_candidates: 0
route_label: bloch_path_a_seeded_shape_only
load_bearing_qca_bridge: false
projector_free_rule_verdict: not_solved
projector_free_rule_generated_algebra_dimension: 16
projector_free_rule_generated_algebra_closed: false
projector_free_rule_pass_rule_to_bridge: false
```

Candidate labels:

```text
path_a_seeded_projector_shift:
  seeded guardrail rejected
  topological_pm_candidate = true

path_a_combined_route1_route2:
  seeded guardrail rejected
  topological_pm_candidate = true
  rule_generated_transported_j_section_count = 0

path_a_shifted_u2_u1_layers:
  algebraic seeded guardrail rejected
  topological_pm_candidate = true

path_a_shifted_ulocal:
  algebraic seeded guardrail rejected
  topological_pm_candidate = true

path_a_unseeded_uniform_identity_shift:
  seed guardrail passed
  no stable 6+4 band split

path_a_unseeded_clock_shift:
  seed guardrail passed
  no stable 6+4 band split
```

Additional direct `rule_to_verdict` candidate:

```text
path_a_projector_free_cycle_combined:
  Bloch terms have shifts (3,4) from source-mode windings (4,4,4,3,3)
  raw coefficients are partial monomial hops, not P_alpha/P_eta
  on-site update is the Route-1 noncommuting U2 U1
  sampled joint algebra exceeds the bounded exact closure cap
  verdict = not_solved
```

## Interpretation

The checker now separates three things that were previously conflated:

1. A seeded effective rule can have the desired `(4,3)` transport shape.
2. A rule can hide the answer algebraically even if no raw coefficient is
   literally `P_alpha` or `P_eta`.
3. Passing the seed guardrail is not enough; the sampled Bloch algebras must
   produce a stable rank-`(6,4)` band split and a rule-generated `J(k)` section.

The first Path-A result is therefore:

```text
seeded topological shape: yes
unseeded stable 6+4 band split: no
rule-generated J(k) section: no
projector-free combined verdict: bounded not_solved
strict bridge: no
```

The next Path-A step is to enlarge the unseeded finite-radius family beyond
uniform full-site shifts and to optimize the exact algebra closure for
projector-free partial hopping. The useful candidates should have nontrivial
partial hopping structure without allowing the coefficient algebra to generate
constant `P_alpha/P_eta`; the first such candidate already escapes raw
projector seeding but pushes the sampled algebra beyond the current quick
closure bound.
