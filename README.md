# clifford-3plus2-d5

Do not fool yourself.

This repo tests whether QCA data can produce a real geometry-to-Spin(10)
one-generation bridge. It does not derive the Standard Model, three families,
mirror decoupling, continuum gauge dynamics, or phenomenology.

The active roadmap is the enhanced J-first attack:

```text
real finite-depth QCA data
  -> forced local J on R^10
  -> structural J-invariant 6+4 real split
  -> C^5 = C^3 ⊕ C^2
  -> geometric gate algebra in the SM commutant
  -> Lambda^even(C^5) = Spin(10) chiral 16
```

Current honest status:

```text
phase_0_audit_contract: complete
phase_1_real_carrier_check: passes
forced_j_check: candidate_only
structural_split_check: candidate_only
gate_classification_check: oracle_passes
qca_update_check: candidate_only
spinor16_check: candidate_only
normalizer_check: candidate_only
real_qca_branch_check: candidate_only
rule_space_exploration: no forced survivors
unseeded_projector_discovery: no complementary 6+4 pair
rule_to_verdict_check: unified negative interface with exact field metadata
floquet_alpha_search: coarse 6+4 center, scaled polarization J certificate
defect_beta_search: monodromy 6+4 center, scaled monodromy J certificate
Spin(10) branching check: passes
QCA load-bearing bridge: notation_only
```

## Active Theorem Target

The bridge is accepted only if QCA rule data produce, before invoking
Spin(10):

```text
K_x ~= R^10
J in SO(K_x), J^2 = -I
K_x = K_3 ⊕ K_2
dim_R K_3 = 6
dim_R K_2 = 4
J K_3 = K_3
J K_2 = K_2
A_geom subset Comm(SU(3) x SU(2) x U(1))
```

Then:

```text
W := (K_x, J) ~= C^5 = C^3 ⊕ C^2
S^+ := Lambda^even(W)
Y = -1/3 N_3 + 1/2 N_2
```

If `J`, `P_3/P_2`, or the gate commutant are chosen because they reproduce
Spin(10), the project remains `notation_only`.

## Main Working Docs

- [Roadmap and working index](docs/roadmap.md)
- [Enhanced J-first attack plan](docs/qca_3plus2_j_first_enhanced_attack_plan.md)
- [Real carrier report](docs/literature/real_carrier_report.md)
- [Forced J report](docs/literature/forced_j_report.md)
- [Projector lattice report](docs/literature/projector_lattice_report.md)
- [Gate classification report](docs/literature/gate_classification_report.md)
- [QCA update certificate](docs/literature/qca_update_certificate.md)
- [Spinor 16 report](docs/literature/spinor16_report.md)
- [Normalizer report](docs/literature/normalizer_report.md)
- [Forcedness certificate](docs/literature/forcedness_certificate.md)
- [Real-QCA branch report](docs/literature/real_qca_branch_report.md)
- [Rule-space exploration report](docs/literature/rule_space_exploration_report.md)
- [Unseeded projector discovery report](docs/literature/unseeded_projector_discovery_report.md)
- [Rule-to-verdict report](docs/literature/rule_to_verdict_report.md)
- [Floquet alpha report](docs/literature/floquet_alpha_report.md)
- [Floquet alpha J obstruction](docs/literature/floquet_alpha_j_obstruction.md)
- [Defect beta report](docs/literature/defect_beta_report.md)
- [Theory summary](docs/theory.md)
- [Falsifiers](docs/falsifiers.md)
- [Phase 0 handover compliance](docs/handover_compliance.md)
- [Results](docs/results/index.md)

## Implemented Checks

`scripts/branching_check.py` verifies the standard exterior-algebra branching:

```text
Y = -1/3 N_3 + 1/2 N_2
dim Lambda^even(C^5) = 16
```

It must print:

```text
branching_check_passed: true
load_bearing_qca_bridge: false
```

`scripts/qca_split_audit.py` audits the older exact-input bridge contract.
Without valid exact input at `data/qca_data.json`, it must report:

```text
qca_split_audit_verdict: notation_only
load_bearing_qca_bridge: false
```

The Phase 0 audit contract is closed in
[docs/handover_compliance.md](docs/handover_compliance.md).

`scripts/real_carrier_check.py` verifies the exact Phase 1 real-carrier
ansatz:

```text
K_x = R^2_clock ⊗ (R^3 ⊕ R^2)
J = epsilon ⊗ I_5
P_3 = I_2 ⊗ diag(1,1,1,0,0)
P_2 = I_2 ⊗ diag(0,0,0,1,1)
```

It must also print:

```text
qca_forces_j: false
load_bearing_qca_bridge: false
```

`scripts/forced_j_check.py` verifies that a declared exact gate word can
produce the Phase 1 `J`. By default this is still only a candidate:

```text
generated_by_gate_word: true
qca_forces_j: false
forced_j_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/structural_split_check.py` verifies that the exact Phase 1
projectors form a `J`-compatible `3+2` candidate and rejects rank-one
addressability inside either block. By default this is still only a candidate:

```text
qca_supplies_structural_3plus2_split: false
structural_split_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/gate_classification_check.py` verifies the exact Phase 4 SM
commutant classifier:

```text
safe_sm_commutant
block_mixing_fail
color_breaking_fail
weak_breaking_fail
antilinear_fail
unknown_fail
```

It must also print:

```text
gate_classification_check_passed: true
qca_geometric_gate_algebra_safe: false
load_bearing_qca_bridge: false
```

`scripts/qca_update_check.py` verifies the exact Phase 5 period-four
finite-depth update candidate:

```text
U(T/4) = J
U(T/2) = -I
U(T) = I
```

It must also print:

```text
qca_rule_forces_update: false
finite_depth_qca_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/spinor16_check.py` verifies the guarded Phase 6 reconstruction of
`Lambda^even(C^5)` from the prior `J` and `P_3/P_2` candidates:

```text
spinor16_dimension: 16
hypercharge_check_passed: true
branching_table_check_passed: true
spinor16_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/normalizer_check.py` verifies the Phase 7 forcedness and normalizer
proxies. It checks whether the declared data preserve the split, whether they
force the candidate `J`, and whether switchable controls generate forbidden
within-block or off-block addressability. By default this is still only a
candidate:

```text
candidate_j_preserved_by_normalizer: false
normalizer_preserves_declared_split: true
j_unique_or_forced: false
split_unique_or_forced: false
forcedness_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/real_qca_branch_check.py` verifies the Phase 8A stronger
real-QCA-first branch interface. It composes the gate-word, structural split,
and normalizer checks for a finite-depth real gate-word search space. By
default this is still only a candidate:

```text
candidate_word_found: true
candidate_word: global_clock_tick
generates_j: true
generates_split: true
j_forced_by_rule_space: false
split_forced_by_rule_space: false
real_qca_branch_verdict: candidate_only
load_bearing_qca_bridge: false
```

`scripts/explore_rule_space.py` runs the E1 bounded rule-space exploration.
It enumerates small exact primitive sets and finite-depth words, then records
which constraints reject them:

```text
words_scanned: 170
j_hits: 73
period_four_hits: 73
forced_candidate_hits: 0
surviving_candidates: 0
load_bearing_qca_bridge: false
```

`scripts/discover_projectors.py` runs the E2 bounded unseeded projector
discovery pass. The default mode does not seed the standard `P_3/P_2`
projectors:

```text
primitive_sets_scanned: 6
algebra_elements_considered: 1924
candidate_projectors: 3
rank_2_projectors: 3
rank_6_projectors: 0
rank_4_projectors: 0
complementary_pairs: 0
unseeded_projector_pairs_found: 0
discovery_verdict: not_found
load_bearing_qca_bridge: false
```

`scripts/rule_to_verdict.py` is the collapsed rule-to-verdict interface. It
takes finite-depth real on-site layer data and reports the Floquet spectrum,
generated algebra center, central idempotents, exact `J` solve status, and one
bridge-candidate boolean. The built-in block-reflection control still does not
pass because `J` is not forced:

```text
central_idempotent_ranks: [0, 4, 6, 10]
complementary_rank_6_4_pairs: 1
forced_j_found: false
pass_rule_to_bridge: false
verdict: candidate_only_j_not_forced
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha_search.py` searches the active Floquet-α physical
primitive family. It replaces depth escalation over abstract E1/E2 primitives
with one mandatory quantized resonance layer:

```text
candidate_count: 10
rank_6_4_pair_candidates: 10
rank_one_falsified_candidates: 0
bridge_candidates: 0
verdict_counts: {'candidate_only_j_not_forced': 10}
load_bearing_qca_bridge: false
```

`scripts/floquet_alpha_plus_search.py` extracts the canonical spectral
polarization `J` from the same mandatory Floquet layer:

```text
polarization_j_candidates: 10
scaled_polarization_certified_candidates: 10
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'polarization_j_produced_not_strictly_unique': 10}
load_bearing_qca_bridge: false
```

The load-bearing α certificate is exact over `QQ(zeta_12)`: it verifies
`K_alpha = (2U+I)P_alpha` and `K_alpha^2 = -3P_alpha` before deriving the
normalized `J_alpha = K_alpha/sqrt(3)`.

`scripts/defect_beta_search.py` computes round-trip monodromy from wall
transition functions. It reproduces the same obstruction through a different
physical route:

```text
monodromy_candidates: 10
scaled_monodromy_certified_candidates: 10
strict_compatible_j_forced_candidates: 0
strict_bridge_candidates: 0
verdict_counts: {'monodromy_j_produced_not_strictly_unique': 10}
load_bearing_qca_bridge: false
```

The β wall transitions now avoid half-angle factors: `T_entry = C`,
`T_exit = M C`, and `T_exit T_entry = M`, so no accidental `sqrt(2)` extension
enters the exact certificate.

## QCA Input Contract

The current audit reads nontrivial input only from `data/qca_data.json`.
The expected shape is documented in `data/qca_data.schema.json`.
Exact matrix entries must be strings parseable as rational numbers, such as
`"0"`, `"1"`, `"-1"`, or `"1/2"`. Floating-point matrix entries are rejected.
Internal symbolic primitive families may use declared algebraic fields, such
as `QQ(zeta_12)`, but must certify the relevant polynomial identities exactly.

Do not add `data/qca_data.json` unless it contains real source-backed QCA data.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
uv run python scripts/real_carrier_check.py --check
uv run python scripts/forced_j_check.py --check
uv run python scripts/forced_j_check.py --include-addressable-rank-one --expect-verdict falsified
uv run python scripts/structural_split_check.py --check
uv run python scripts/structural_split_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/structural_split_check.py --include-rank-one-weak --expect-verdict falsified
uv run python scripts/gate_classification_check.py --check
uv run python scripts/qca_update_check.py --check
uv run python scripts/qca_update_check.py --include-rank-one-color-shift --expect-verdict falsified
uv run python scripts/qca_update_check.py --include-rank-one-weak-shift --expect-verdict falsified
uv run python scripts/spinor16_check.py --check
uv run python scripts/normalizer_check.py --check --expect-verdict candidate_only
uv run python scripts/normalizer_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/normalizer_check.py --include-rank-one-weak --expect-verdict falsified
uv run python scripts/normalizer_check.py --include-off-block --expect-verdict falsified
uv run python scripts/normalizer_check.py --include-full-u5-controls --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --check --expect-verdict candidate_only
uv run python scripts/real_qca_branch_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-rank-one-weak --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-rank-one-pair --expect-verdict falsified
uv run python scripts/real_qca_branch_check.py --include-off-block --expect-verdict falsified
uv run python scripts/explore_rule_space.py --check --output-dir data/exploration
uv run python scripts/discover_projectors.py --check --mode unseeded --expect-verdict not_found --output-dir data/exploration
uv run python scripts/discover_projectors.py --check --mode sanity-seeded --expect-verdict projector_pair_found --output-dir /tmp/e2_sanity
uv run python scripts/discover_projectors.py --check --mode block-reflection-candidate --expect-verdict projector_pair_found --output-dir /tmp/e2_block
uv run python scripts/rule_to_verdict.py --case minimal-period-four --expect-verdict falsified_no_rank_6_4_center
uv run python scripts/rule_to_verdict.py --case clock-block-reflection --expect-verdict candidate_only_j_not_forced
uv run python scripts/rule_to_verdict.py --case clock-rank-one-color-reflection --expect-verdict falsified_rank_one_center
uv run python scripts/floquet_alpha_search.py --check
uv run python scripts/floquet_alpha_plus_search.py --check
uv run python scripts/defect_beta_search.py --check
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

## Claim Boundary

Do not call a representation identity a theorem about QCA geometry. Do not flip
a boolean unless code computes it or a proof derives it. Do not treat "`n = 3`
was chosen" as "`n = 3` was selected." Do not hide hand-chosen complex
structures, SU(5) embeddings, gate restrictions, or within-block projectors.
