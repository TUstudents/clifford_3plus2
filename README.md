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

## QCA Input Contract

The current audit reads nontrivial input only from `data/qca_data.json`.
The expected shape is documented in `data/qca_data.schema.json`.
Exact matrix entries must be strings parseable as rational numbers, such as
`"0"`, `"1"`, `"-1"`, or `"1/2"`. Floating-point matrix entries are rejected.

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
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check --expect-verdict notation_only
uv run python scripts/qca_split_audit.py --json --expect-verdict notation_only
```

## Claim Boundary

Do not call a representation identity a theorem about QCA geometry. Do not flip
a boolean unless code computes it or a proof derives it. Do not treat "`n = 3`
was chosen" as "`n = 3` was selected." Do not hide hand-chosen complex
structures, SU(5) embeddings, gate restrictions, or within-block projectors.
