# Roadmap And Working Index

This is the main working index for the `clifford-3plus2-d5` side project.
It organizes the work needed to test whether QCA Clifford geometry can supply
a structural `3+2` bridge to the one-generation Spin(10) spinor.

## Source Documents

- [README](../README.md): project boundary, commands, and current status.
- [Theory Summary](theory.md): current mathematical baseline and audit
  boundary.
- [Falsifiers](falsifiers.md): conditions that force `notation_only` or
  `falsified`.
- [Results Index](results/index.md): reproducible command outputs.
- [Side Project Plan](clifford_3plus2_to_d5_side_project_plan.md): full
  research roadmap and theorem target.
- [Agent Handover](clifford_3plus2_empty_repo_agent_handover.md): bootstrap
  and first implementation contract.

## Current Verdict

```text
branching_check_passed: true
qca_split_audit_verdict: notation_only
load_bearing_qca_bridge: false
```

Meaning:

- The textbook Spin(10) branching arithmetic passes.
- The QCA bridge has not passed.
- No `data/qca_data.json` is present.
- The project must not claim a geometry-to-D5 theorem yet.

## Phase 0: Audit Infrastructure And QCA Data Contract

Status: in progress.

Purpose: make the QCA split audit precise enough that future data can pass or
fail without interpretation.

Completed:

- `uv` package scaffold.
- Exact rational matrix parsing.
- Default-failing QCA split audit.
- One-particle gate algebra safety checks.
- Reproducible scripts:
  - [`scripts/branching_check.py`](../scripts/branching_check.py)
  - [`scripts/qca_split_audit.py`](../scripts/qca_split_audit.py)

Next deliverables:

- Add `data/qca_data.schema.json`.
- Add invalid fixtures for hand-chosen `J`, off-block gates, and color
  projectors.
- Add a clearly labeled synthetic positive fixture, if useful, with no QCA
  evidence claim.
- Add machine-readable JSON output to `qca_split_audit.py`.
- Keep missing or invalid QCA input at `notation_only`.

Acceptance:

```text
The audit interface is documented, tested, and impossible to pass by
omission, floats, unknown origins, or hand-chosen J.
```

## Phase 1: Real Load-Bearing QCA 3+2 Split Audit

Status: not started.

Purpose: decide whether QCA supplies the `3+2` Clifford split structurally.
This is the first scientific gate.

Deliverable:

```text
docs/literature/qca_clifford_3plus2_split_audit.md
```

Questions:

- What are the three spatial QCA Clifford directions?
- What are the two candidate extra directions?
- Are they wall/Floquet directions, Wick-rotated directions, coin matrices, or
  arbitrary decorations?
- What is the exact anticommutation table and signature?
- What is the complex structure `J`, and is it QCA-forced?
- Does `J^2 = -I` exactly?
- Does `J` preserve `P3` and `P2`?
- Is the allowed local gate algebra block-invariant and SM-safe?

Acceptance:

```text
qca_supplies_structural_3plus2_split = true
complex_structure_origin in {qca_chirality, floquet_phase, wick_rotation}
complex_structure_compatible_with_3plus2_split = true
allowed_gate_algebra_block_mixing_absent = true
allowed_gate_algebra_sm_commutant_safe = true
```

Failure status:

```text
notation_only or falsified
```

## Phase 2: Written Theory Baseline

Status: partially started in [Theory Summary](theory.md).

Purpose: separate textbook representation theory from QCA-specific claims.

Deliverable:

```text
docs/literature/clifford_3plus2_to_d5_theory.md
```

Contents:

- Define QCA Clifford data in matrix notation.
- Define the abstract `V_3 ⊕ V_2` split.
- Define `S_+ = Lambda^even(C^5)`.
- Derive the branching table.
- Quote Phase 1 status before any bridge claim.
- State what the construction does and does not explain.

Acceptance:

```text
All representation claims are explicit and checkable.
No BCC-to-Cartan projector appears.
No three-family claim appears.
```

## Phase 3: Branching Calculation / Notation Lemma

Status: implemented.

Purpose: verify the standard Spin(10) `16` branching for the fixed `3+2`
split.

Implemented in:

- [`src/clifford_3plus2_d5/exterior.py`](../src/clifford_3plus2_d5/exterior.py)
- [`src/clifford_3plus2_d5/branching.py`](../src/clifford_3plus2_d5/branching.py)
- [`tests/test_branching.py`](../tests/test_branching.py)

Check:

```bash
uv run python scripts/branching_check.py --check
```

Acceptance:

```text
total multiplicity = 16
Y = -1/3 N_3 + 1/2 N_2
branching_check_passed: true
load_bearing_qca_bridge: false
```

Interpretation:

```text
This is necessary bookkeeping, not QCA evidence.
```

## Phase 4: Intertwiner / No-Projector Lemma

Status: partially implemented as gate algebra tests.

Purpose: prove and test the distinction between safe block invariants and
unsafe color-breaking projectors.

Current implementation:

- [`src/clifford_3plus2_d5/gate_algebra.py`](../src/clifford_3plus2_d5/gate_algebra.py)
- [`tests/test_gate_algebra.py`](../tests/test_gate_algebra.py)

Future deliverable:

```text
docs/literature/clifford_3plus2_intertwiner_theorem.md
```

Tasks:

- Define `gl(3) ⊕ gl(2)` and the SM subgroup action.
- Prove that functions of `N_3,N_2` commute with `SU(3) x SU(2)`.
- Prove that individual basis projectors inside `C^3` break `SU(3)`.
- Connect this to the old BCC-D5 Cartan-locking falsifier.
- Extend tests if the Fock-space gate algebra grows beyond diagonal
  `N_3,N_2` functions.

Acceptance:

```text
The lemma distinguishes safe block invariants from unsafe Cartan projectors.
```

Interpretation:

```text
Still notation-only unless Phase 1 proves QCA gates are actually constrained
to the safe algebra.
```

## Phase 5: QCA Clifford Match

Status: not started.

Purpose: refine the Phase 1 audit into a formal QCA proposition.

Deliverable:

```text
docs/literature/qca_clifford_3plus2_match.md
```

Tasks:

- Prove or falsify structural origin of the two-plane.
- Prove or falsify signature compatibility with the complex `C^5`
  construction.
- Prove or falsify block-invariance of the allowed QCA gate algebra.
- State the symmetry group preserving the split.

Acceptance:

```text
QCA supplies a genuine 3+2 Clifford split, not a post-hoc relabeling.
```

## Phase 6: Combined Geometry / Spin(10) Operation

Status: not started.

Purpose: classify QCA/geometric operations by how they act on the SM subgroup.

Deliverable:

```text
docs/literature/combined_geometry_spin10_operation.md
```

Tasks:

- Define candidate operations: inversion, time reversal, chirality flip,
  wall mass-plane rotation, BCC point-group rotation, defect monodromy.
- Define induced internal action `Omega(g)`.
- Classify each operation as gauge-commuting, gauge-automorphism, or
  gauge-breaking.
- Require the entire allowed gate algebra to be safe, not just examples.

Acceptance:

```text
Every allowed QCA local gate is an SM intertwiner or a declared safe
automorphism.
```

## Phase 7: Theorem Status And Integration Decision

Status: not started.

Purpose: decide what this side project honestly proves.

Deliverable:

```text
docs/literature/clifford_3plus2_to_d5_theorem_status.md
```

Possible final statuses:

- `closed_one_generation_bridge`
- `conditional_bridge`
- `notation_only`
- `falsified`

Acceptance:

```text
The final status states the strongest proved theorem, all assumptions, all
remaining gaps, and whether the result can enter the main BCC-QCA program.
```

## Non-Goals

Do not add claims about:

- three families
- mirror gap
- Golterman-Shamir inertness
- continuum gauge dynamics
- Higgs/Yukawa realism
- phenomenology
- a completed QCA construction

Do not add `data/qca_data.json` unless it contains real candidate QCA data
with exact rational matrices and explicit gate restrictions.

## Working Rule

Phase 3 and Phase 4 can clean up notation. Only Phase 1 and Phase 5 can make
the bridge load-bearing.
