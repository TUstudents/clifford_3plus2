# Bloch Path-A Report

Status: stepwise projector-free Bloch search finds a coarse center but no
compatible `J`; no unseeded bridge candidate.

This report records the Path-A checkers implemented in
[`src/clifford_3plus2_d5/qca/bloch_rule.py`](../../src/clifford_3plus2_d5/qca/bloch_rule.py)
and exposed by two CLIs:

- [`scripts/bloch_path_a_stepwise.py`](../../scripts/bloch_path_a_stepwise.py):
  the headline calculation for projector-free monomial-hop candidates.
- [`scripts/bloch_path_a_search.py`](../../scripts/bloch_path_a_search.py):
  the mixed candidate-panel check plus the default projector-free structural
  result. Its `--projector-free-max-algebra-dim 16` mode is retained only as a
  legacy cap-boundary regression.

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
uv run python scripts/bloch_path_a_stepwise.py --max-candidates 6 --max-algebra-dim 48 --jobs 2 --check
uv run python scripts/bloch_path_a_stepwise.py --max-candidates 6 --max-algebra-dim 48 --center-top 6 --idempotents --centralizer --j-solve --jobs 4 --check
uv run python scripts/bloch_path_a_search.py --check
uv run python scripts/bloch_path_a_search.py --check --projector-free-max-algebra-dim 16
```

Default mixed-panel output:

```text
candidate_count: 6
seed_guardrail_rejections: 4
unseeded_candidate_count: 2
stable_6_4_band_candidates: 0
topological_pm_candidates: 4
rule_generated_j_section_candidates: 0
strict_bridge_candidates: 0
candidate_panel_route_label: bloch_path_a_seeded_shape_only
route_label: bloch_path_a_projector_free_coarse_center_no_compatible_j
load_bearing_qca_bridge: false
projector_free_rule_verdict: not_solved
projector_free_rule_generated_algebra_dimension: 34
projector_free_rule_generated_algebra_closed: true
projector_free_rule_central_idempotent_ranks: [0, 4, 6, 10]
projector_free_rule_compatible_centralizer_dimension: 4
projector_free_rule_compatible_complex_structure_count: 0
projector_free_rule_pass_rule_to_bridge: false
```

The optional `--projector-free-max-algebra-dim 16` run is retained only as a
legacy cap-boundary regression check. The default now matches the stepwise
headline: the projector-free candidate reaches the coarse center and fails at
compatible `J`.

Focused performance probe after the algebra-kernel optimization:

```text
uv run python scripts/perf_probe.py --max-algebra-dim 16
  bloch_samples: 12 in about 0.1s
  generated_algebra: dimension=16 closed=false in about 3s

uv run python scripts/perf_probe.py --max-algebra-dim 32
  bloch_samples: 12 in about 0.1s
  generated_algebra: dimension=32 closed=false in about 8.5s
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
  reason: the coefficient algebra contains the shifted Route-1 operators; on
  the alpha block they are non-scalar, so Bezout polynomials recover
  P_alpha/P_eta. This is Route 1 wrapped in trivial spatial structure, not a
  genuine Path-A primitive.

path_a_shifted_ulocal:
  algebraic seeded guardrail rejected
  topological_pm_candidate = true
  reason: the coefficient algebra contains U2 U1 itself, so the same
  alpha/eta spectral projectors are algebraically recoverable.

path_a_unseeded_uniform_identity_shift:
  seed guardrail passed
  no stable 6+4 band split

path_a_unseeded_clock_shift:
  seed guardrail passed
  no stable 6+4 band split
```

The six labeled candidates above are a diagnostic panel mixing seeded
controls, Route-1-in-disguise guardrail cases, and two conservative unseeded
full-shift sanity checks. They are not the monomial-hop enumeration. The
actual projector-free enumeration is the stepwise `_five_cycles x
_shift_assignments` scan below.

Additional direct `rule_to_verdict` candidate:

```text
path_a_projector_free_cycle_combined:
  Bloch terms have shifts (3,4) from source-mode windings (4,4,4,3,3)
  raw coefficients are partial monomial hops, not P_alpha/P_eta
  on-site update is the Route-1 noncommuting U2 U1
  sampled joint algebra exceeds the bounded exact closure cap
  verdict = not_solved
```

Stepwise projector-free detailed search:

```text
max_candidates = 6
max_algebra_dim = 48
center_top = 6
jobs = 4

closed_count = 6
coarse_6_4_count = 6
all six closed at generated_algebra_dimension = 34
all six have center_dimension = 4
all six have central_idempotent_ranks = [0,4,6,10]
all six have compatible_centralizer_dimension = 4
all six have generated_j_count = 0
all six have compatible_j_count = 0
all six have bridge_j_status = no_rule_generated_j
```

This is the first non-seeded Path-A positive structural result. The
projector-free monomial-hop family does not blow up to full `M_10`; it closes
to a structured 34-dimensional algebra and has the coarse central idempotent
lattice. The bounded split-center `J` solver settles the sampled family
negatively: neither the rule-generated center nor the compatible centralizer
contains a real orthogonal `J` with `J^2 = -I`.
This is a sharper obstruction than the previous unresolved generic solve.
It is recorded in [Theory](../theory.md) as Proposition 4a, with a
conjectural Proposition 4b naming the broader monomial-hop incompatibility to
prove next.

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
unseeded projector-free closure: yes, dimension 34
unseeded coarse 6+4 center: yes for six sampled candidates
rule-generated J(k) section: no for six sampled candidates
projector-free combined verdict: coarse-center hit, no generated J
strict bridge: no
```

The next Path-A step is to prove or falsify the conjectural monomial-hop
incompatibility: partial monomial hops with winding multiset `(4,4,4,3,3)` may
force the coarse center but appear unable to generate the fixed complex
structure on either coarse block. If that conjecture holds, the next viable
primitive must go beyond monomial hops while still avoiding coefficient-algebra
seeding of `P_alpha/P_eta`.
