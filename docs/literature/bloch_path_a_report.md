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
uv run python scripts/bloch_path_a_stepwise.py --family polynomial-hop --max-candidates 6 --max-algebra-dim 16 --jobs 2 --check
uv run python scripts/bloch_path_a_stepwise.py --family polynomial-hop --max-candidates 1 --max-algebra-dim 64 --check
uv run python scripts/bloch_path_a_search.py --check
uv run python scripts/bloch_path_a_search.py --check --projector-free-max-algebra-dim 16
```

The detailed `--center-top ... --j-solve` command is an archived structural
reproduction, not a fast-loop check; use the headline closure command for
routine validation.

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

## Polynomial-Hop Move 1

The first extension beyond monomial hops allows a coefficient to be a
finite-order rational reflection on two mode edges while preserving the exact
Laurent orthogonality identity. This is still projector-free at the raw
coefficient level and keeps the same `(3,4)` shift support.

Bounded structural run:

```text
uv run python scripts/bloch_path_a_stepwise.py --family polynomial-hop --max-candidates 6 --max-algebra-dim 16 --jobs 2 --check

candidate_count: 6
closed_count: 0
seed_guardrail_checked: false
seed_guardrail_rejections: 0
laurent_orthogonal_count: 6
coarse_6_4_count: 0
all six candidates:
  laurent_orthogonal: true
  generated_algebra_dimension: 16
  generated_algebra_closed: false
  route_label: cap_exceeded_structured
```

Boundary probe on the first polynomial-hop candidate:

```text
uv run python scripts/bloch_path_a_stepwise.py --family polynomial-hop --max-candidates 1 --max-algebra-dim 64 --check

candidate: path_a_polynomial_hop_p0_c12340_s44433_m301
laurent_orthogonal: true
generated_algebra_dimension: 64
generated_algebra_closed: false
route_label: cap_exceeded_structured
time: about 53s
```

A seed-guardrail annotation on that same candidate rejects it algebraically:

```text
uv run python scripts/bloch_path_a_stepwise.py --family polynomial-hop --max-candidates 1 --max-algebra-dim 16 --seed-guardrail --check

seed_guardrail_checked: true
seed_guardrail_rejections: 1
```

Interpretation: the bounded polynomial-hop class does not give a bridge
candidate in the current exact checker. The smallest mixed coefficient already
leaves the tractable coarse-center regime, and the first checked case also
fails the stricter coefficient-algebra seed guardrail. This is not yet a
theorem over all polynomial Bloch hops; it is a computational boundary for the
next architecture.

Additional direct `rule_to_verdict` candidate:

```text
path_a_projector_free_cycle_combined:
  Bloch terms have shifts (3,4) from source-mode windings (4,4,4,3,3)
  raw coefficients are partial monomial hops, not P_alpha/P_eta
  on-site update is the Route-1 noncommuting U2 U1
  sampled joint algebra exceeds the bounded exact closure cap
  verdict = not_solved
```

Stepwise projector-free full monomial-hop census:

```text
candidate_space = 2400
pattern_count = 10
cycle_count = 24
shift_count = 10
max_algebra_dim = 48
jobs = 8

all candidates close below cap 48
center_dim_4_candidates = 1320

dim26 / center4:
  candidates = 360
  central_idempotent_ranks = [0,2,8,10] for all
  coarse_6_4_count = 0
  timeout_count = 0

dim34 / center4:
  candidates = 960
  central_idempotent_ranks = [0,4,6,10] for all
  generated_j_solved = true for all
  generated_j_count = 0 for all
  bridge_j_status = no_rule_generated_j for all
  timeout_count = 0
```

This closes the finite projector-free monomial-hop Path-A class as a strict
bridge route. The dim-`26` branch fails no-locking by producing rank-`2`
central idempotents. The dim-`34` branch has the desired coarse central
idempotent lattice but generates no local-center `J`. The old timeout class was
an idempotent/J diagnostic bottleneck, not an algebra-closure boundary; the
structural filters now classify it directly. The result is recorded in
[Theory](../theory.md) as Proposition 4a.

Reproduction artifacts:

```text
data/scans/bloch_path_a_monomial_center_dim26_rank_filter_v7.jsonl
data/scans/bloch_path_a_monomial_center_dim34_timeout_filter_v6.jsonl
data/scans/bloch_path_a_monomial_center_dim4_bridge_fast_v5.jsonl
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
unseeded projector-free closure: yes for all 2400 monomial-hop candidates
unseeded coarse 6+4 center: yes for 960 dim34 candidates
dim26 no-locking failure: yes for 360 candidates
rule-generated J(k) section: no for all 960 coarse dim34 candidates
projector-free combined verdict: coarse-center hit, no generated J
strict bridge: no
```

The next Path-A theorem step is to decide whether Conjectural Proposition 4b
can be promoted from the finite census to a structural statement about broader
coprime monomial-hop rules. The next bridge-search step, if pursued, must go
beyond monomial hops while still avoiding coefficient-algebra seeding of
`P_alpha/P_eta`.
