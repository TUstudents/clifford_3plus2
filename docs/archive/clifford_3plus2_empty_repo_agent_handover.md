# New-Agent Handover: 3+2 Clifford Geometry To D5 In An Empty Repo

**Document type:** implementation handover for a fresh agent  
**Target project:** `clifford-3plus2-d5` side project  
**Status:** scaffold and first-proof instructions; no theorem assumed  
**Date:** 2026-05-13

## 0. Mission

Build a small, clean Python research repo that tests one precise idea:

```text
Can QCA Clifford geometry structurally supply a 3+2 split that explains why
Spin(10) 16 is the natural one-generation internal fiber?
```

The project must not claim:

```text
three families,
mirror gap,
full Standard Model,
continuum gauge dynamics,
or a completed QCA construction.
```

The first load-bearing question is:

```text
Does the QCA supply the 3+2 Clifford split structurally,
including a non-hand-chosen complex structure J?
```

If not, the project status is:

```text
notation_only
```

## 1. Highest Mantra

Write this into the repo README:

```text
Do not fool yourself.
```

Operational meaning:

```text
Do not call a representation identity a theorem about QCA geometry.
Do not flip a boolean unless code computes or a proof derives it.
Do not treat "n=3 was chosen" as "n=3 was selected."
Do not hide hand-chosen complex structures, SU(5) embeddings, or gate
restrictions.
Do not claim Spin(10) follows from geometry unless the QCA supplies the
3+2 split and complex structure.
```

## 2. Empty Repo Bootstrap

Use `uv`.

Recommended commands:

```bash
uv init --package
uv add numpy scipy sympy
uv add --dev pytest ruff
```

Expected top-level structure:

```text
clifford-3plus2-d5/
  README.md
  pyproject.toml
  uv.lock
  src/
    clifford_3plus2_d5/
      __init__.py
      exterior.py
      branching.py
      clifford_audit.py
      gate_algebra.py
      status.py
  scripts/
    branching_check.py
    qca_split_audit.py
  tests/
    test_branching.py
    test_clifford_audit.py
    test_gate_algebra.py
    test_scripts.py
  docs/
    theory.md
    roadmap.md
    falsifiers.md
    results/
      index.md
```

## 3. Suggested `pyproject.toml` Settings

Make sure `pyproject.toml` has a real package and strict enough tooling:

```toml
[project]
name = "clifford-3plus2-d5"
version = "0.1.0"
description = "Audit the QCA 3+2 Clifford split to D5/Spin(10) bridge"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "numpy",
  "scipy",
  "sympy",
]

[dependency-groups]
dev = [
  "pytest",
  "ruff",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

## 4. README Requirements

The README must state the claim boundary up front:

```text
This repo tests a possible geometry-to-Spin(10) bridge.
It does not derive the Standard Model.
It does not derive three families.
It does not prove mirror decoupling.
```

Include the current theorem target:

```text
Given QCA Clifford data that structurally supplies

  V = V_3 ⊕ V_2

plus a non-hand-chosen compatible complex structure J, the induced
Spin(10)/SU(5) exterior-algebra construction gives one SM generation:

  S_+ = Lambda^even(C^5).
```

Then include the falsifier:

```text
If the QCA 2-plane or complex structure is chosen by hand,
the project is notation_only.
```

## 5. Core Data Models

Create `src/clifford_3plus2_d5/status.py`.

Use frozen dataclasses.  Suggested objects:

```python
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

ComplexStructureOrigin = Literal[
    "qca_chirality",
    "floquet_phase",
    "wick_rotation",
    "by_hand",
    "unknown",
]

BridgeVerdict = Literal[
    "structural_bridge",
    "conditional_bridge",
    "notation_only",
    "falsified",
]

Signature = Literal[
    "euclidean_5",
    "lorentzian_1_4",
    "lorentzian_4_1",
    "mixed",
    "unknown",
]

StructuralOrigin = Literal[
    "wall_normal_plus_floquet_time",
    "wall_normal_plus_floquet_phase",
    "coin_tau_x_tau_y",
    "arbitrary",
    "unknown",
]

@dataclass(frozen=True)
class QCASplitAudit:
    candidate_generators: tuple[str, ...]
    anticommutation_matrix: tuple[tuple[int, ...], ...]
    signature: Signature
    structural_origin: StructuralOrigin
    complex_structure_operator: tuple[tuple[Fraction, ...], ...] | None
    complex_structure_origin: ComplexStructureOrigin
    complex_structure_squares_to_minus_one: bool
    complex_structure_preserves_3plus2_split: bool
    complex_structure_in_allowed_gate_algebra: bool
    complex_structure_compatible_with_3plus2_split: bool
    off_block_gate_generators_present: bool
    block_diagonal_gate_algebra: bool
    sm_commutant_gate_algebra: bool
    qca_supplies_structural_3plus2_split: bool
    verdict: BridgeVerdict
```

Default values must not pass:

```text
complex_structure_origin = "unknown"
complex_structure_operator = None
qca_supplies_structural_3plus2_split = false
sm_commutant_gate_algebra = false
verdict = "notation_only"
```

## 6. Phase A: Textbook Branching Check

This is necessary but not load-bearing.

Implement in:

```text
src/clifford_3plus2_d5/exterior.py
src/clifford_3plus2_d5/branching.py
scripts/branching_check.py
tests/test_branching.py
```

Calculation:

```text
V = C^3 ⊕ C^2
S_+ = Lambda^even(V)
```

Enumerate even subsets of:

```text
{1,2,3,4,5}
```

with:

```text
N_3 = number selected from {1,2,3}
N_2 = number selected from {4,5}
Y = A + B N_3 + C N_2.
```

Solve:

```text
Y(0,0)=0
Y(0,2)=1
Y(1,1)=1/6
```

Expected:

```text
A = 0
B = -1/3
C = 1/2
```

Expected sector table:

```text
(N_3,N_2)=(0,0): multiplicity 1,  Y=0,     nu^c
(0,2):             multiplicity 1,  Y=1,     e^c
(1,1):             multiplicity 6,  Y=1/6,   Q
(2,0):             multiplicity 3,  Y=-2/3,  u^c
(2,2):             multiplicity 3,  Y=1/3,   d^c
(3,1):             multiplicity 2,  Y=-1/2,  L
```

Test requirements:

```text
total multiplicity = 16
all hypercharges match exactly as rational numbers
script prints "branching_check_passed: true"
script also prints "load_bearing_qca_bridge: false"
```

The script must explicitly say:

```text
This verifies standard Spin(10) branching only.
It does not prove the QCA supplies the split.
```

## 7. Phase B: Load-Bearing QCA Split Audit

This is the first real phase.

Implement in:

```text
src/clifford_3plus2_d5/clifford_audit.py
scripts/qca_split_audit.py
tests/test_clifford_audit.py
```

Input expected from future QCA model:

```text
candidate Clifford generators
microscopic QCA gate word
declared allowed local gate generators
candidate complex structure J
```

The empty repo can start with symbolic placeholders, but placeholders must
fail by default.

The audit must read its nontrivial input from a file.  Use this as the only
escape hatch from the default `notation_only` verdict:

```text
data/qca_data.json
```

If the file is missing, incomplete, or fails schema validation:

```text
verdict = "notation_only"
```

Minimal `qca_data.json` schema:

```json
{
  "candidate_generators": [
    {"name": "e1", "matrix": [[...]], "block": "V3"},
    {"name": "e2", "matrix": [[...]], "block": "V3"},
    {"name": "e3", "matrix": [[...]], "block": "V3"},
    {"name": "m1", "matrix": [[...]], "block": "V2"},
    {"name": "m2", "matrix": [[...]], "block": "V2"}
  ],
  "candidate_complex_structure": {
    "name": "J",
    "origin": "unknown",
    "matrix": [[...]]
  },
  "allowed_gate_generators": [
    {"name": "G0", "matrix": [[...]]}
  ],
  "split_projectors": {
    "P3": [[...]],
    "P2": [[...]]
  }
}
```

Represent exact values as strings parseable as fractions:

```json
"0", "1", "-1", "1/2"
```

Do not use floating point for Clifford anticommutators, projectors, or `J`.

Audit fields:

```text
candidate_generators
anticommutation_matrix
signature
structural_origin
complex_structure_origin
complex_structure_operator
complex_structure_squares_to_minus_one
complex_structure_preserves_3plus2_split
complex_structure_in_allowed_gate_algebra
complex_structure_compatible_with_3plus2_split
off_block_gate_generators_present
block_diagonal_gate_algebra
sm_commutant_gate_algebra
qca_supplies_structural_3plus2_split
verdict
```

Pass conditions:

```text
qca_supplies_structural_3plus2_split = true
signature compatible with Euclidean C^5 or controlled Wick rotation
complex_structure_origin in {qca_chirality, floquet_phase, wick_rotation}
complex_structure_operator is present
J^2 = -I exactly
[J,P3] = 0 and [J,P2] = 0 exactly
J is generated by or included in the allowed QCA gate algebra
complex_structure_compatible_with_3plus2_split = true
off_block_gate_generators_present = false
block_diagonal_gate_algebra = true
sm_commutant_gate_algebra = true
```

Fail conditions:

```text
complex_structure_origin in {by_hand, unknown}
two-plane is tau_x,tau_y by arbitrary declaration
off-block generators Hom(V_3,V_2) or Hom(V_2,V_3) are allowed
gate algebra can address individual color-basis directions
```

Default verdict:

```text
notation_only
```

## 8. Phase C: Gate Algebra Audit

Implement in:

```text
src/clifford_3plus2_d5/gate_algebra.py
tests/test_gate_algebra.py
```

Purpose:

```text
check whether the allowed QCA gate algebra is actually safe for
SU(3) x SU(2), not merely block-diagonal.
```

Given a generator matrix in block form:

```text
G = [ A  B ]
    [ C  D ]
```

the safe condition is:

```text
B = 0
C = 0
```

but this is only the first tier.  Block-diagonal is necessary, not sufficient.
The killed BCC-D5 Cartan-locking branch used operators that lived inside the
`V_3` block but still broke `SU(3)`.

The audit has three tiers:

```text
Tier 1: block mixing
  Hom(V_3,V_2) or Hom(V_2,V_3) is nonzero -> fail.

Tier 2: block-diagonal but not SU(3) x SU(2) invariant
  Examples: |1><1| inside V_3, N_1 = a_1^dagger a_1 -> fail.

Tier 3: SM-commutant / block-scalar invariant
  Functions of N_3, N_2, identity, and allowed SM Casimirs -> pass.
```

The safe algebra is not all of:

```text
gl(V_3) ⊕ gl(V_2)
```

It is the commutant of the gauge action on the representation being audited.
For the one-particle block this means block scalars:

```text
lambda_3 I_{V_3} ⊕ lambda_2 I_{V_2}.
```

For the exterior-algebra/Fock representation, safe diagonal invariants may be
functions of:

```text
N_3, N_2
```

and any explicitly identified SM Casimirs.  Do not classify arbitrary
block-diagonal matrices as safe.

The unsafe algebra includes:

```text
gl(V_3 ⊕ V_2).
```

Tests:

```text
off-block generator fails
block-scalar generator passes
function of N_3,N_2 passes
individual basis projector inside V_3 fails
block-diagonal non-scalar inside V_3 fails
off-block V_3 <-> V_2 generator fails
```

## 9. Phase D: Documentation

Create:

```text
docs/theory.md
docs/roadmap.md
docs/falsifiers.md
docs/results/index.md
```

`docs/theory.md` must include:

```text
Spin(10) branching is textbook.
The only new question is whether QCA supplies the split and J.
```

`docs/falsifiers.md` must list:

```text
1. QCA two-plane is arbitrary.
2. Complex structure J is chosen by hand.
3. SU(5) embedding is chosen independently of QCA.
4. Gate algebra allows V_3 <-> V_2 block mixing.
5. Gate algebra is block-diagonal but not in the SU(3) x SU(2) commutant.
6. Gate algebra resolves individual C^3 basis states.
7. Candidate J has no explicit operator matrix.
8. Candidate J fails J^2=-I or fails to preserve P3,P2.
9. Candidate J is not in the allowed QCA gate algebra.
10. Project only reproduces Spin(10) branching.
```

`docs/results/index.md` should start empty except for reproducible script
results.  Do not add result claims before scripts exist.

## 10. Commands The Agent Should Run

After setup:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check
```

Expected early outcome:

```text
branching_check.py passes.
qca_split_audit.py returns notation_only until real QCA data is supplied.
```

That is not failure.  That is honesty.

## 11. First Pull Request / First Commit Scope

First commit should include only:

```text
uv project scaffold
README with claim boundary
branching calculator and tests
default-failing QCA split audit dataclass/script
qca_data.json schema and parser with exact Fraction matrices
explicit J operator checks
gate algebra block-mixing and SM-commutant tests
docs/theory.md
docs/falsifiers.md
```

Do not include:

```text
full electroweak QCA
mirror gap claims
three-family claims
Spin(10) gauge dynamics
large symbolic GA framework
```

## 12. Handoff Prompt For The New Agent

Use this prompt:

```text
You are starting an empty repo for the 3+2 Clifford-to-D5 side project.

Your job is to build minimal reproducible infrastructure, not to overclaim.

Use uv.  Create a Python package with tests and scripts.

First implement the textbook exterior-algebra branching:

  Lambda^even(C^3 ⊕ C^2) -> one Spin(10) 16 branching table.

Mark this explicitly as notation-only evidence.

Then implement a QCA split audit whose default verdict is notation_only.
It may pass only if supplied QCA data proves:

  structural 3+2 split,
  explicit non-hand-chosen complex structure J with J^2=-I,
  J preserves P3,P2 and lies in the allowed QCA gate algebra,
  compatible signature / Wick rotation,
  allowed gate algebra lies in the SM commutant,
  no V_3 <-> V_2 mixing,
  no individual color-basis projectors.

Use a qca_data.json input file with exact rational matrices.  No qca_data.json,
no bridge claim.

The highest rule is:

  Do not fool yourself.

Do not turn TODOs into booleans.  Do not call Spin(10) branching a QCA theorem.
Do not claim three families.
```

## 13. Success Criteria

Short-term success:

```text
Repo runs.
Branching arithmetic is tested.
Default QCA audit blocks bridge claims.
Gate algebra tests catch block mixing and block-diagonal color breaking.
Complex-structure tests require an explicit J operator.
```

Medium-term success:

```text
Real QCA Clifford data can be inserted into the audit.
The audit gives a nontrivial verdict.
```

Strong success:

```text
QCA data forces the 3+2 split and J, and the full allowed gate algebra is safe.
```

Most likely honest outcome:

```text
The branching is textbook and the QCA split/J are not yet structural.
Verdict: notation_only.
```

That outcome is useful because it prevents a false bridge from entering the
main BCC-QCA program.
