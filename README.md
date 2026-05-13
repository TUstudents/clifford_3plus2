# clifford-3plus2-d5

Do not fool yourself.

This repo tests a possible geometry-to-Spin(10) bridge. It does not derive the
Standard Model, it does not derive three families, and it does not prove mirror
decoupling.

The current honest status is:

```text
Spin(10) branching check: passes
QCA load-bearing bridge: notation_only
```

The branching arithmetic is textbook representation theory. The load-bearing
question is still whether QCA Clifford data structurally supplies the `3+2`
split and a non-hand-chosen compatible complex structure `J`.

## Theorem Target

Given QCA Clifford data that structurally supplies

```text
V = V_3 ⊕ V_2
```

plus a non-hand-chosen compatible complex structure `J`, the induced
Spin(10)/SU(5) exterior-algebra construction gives one Standard Model
generation:

```text
S_+ = Lambda^even(C^5)
```

If the QCA two-plane or complex structure is chosen by hand, the project is
`notation_only`.

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

`scripts/qca_split_audit.py` audits the load-bearing bridge claim. Without
valid exact input at `data/qca_data.json`, it must report:

```text
qca_split_audit_verdict: notation_only
load_bearing_qca_bridge: false
```

`gate_algebra.py` checks that allowed one-particle gates are not merely
block-diagonal, but lie in the `SU(3) x SU(2)` commutant for the audited
one-particle split. Off-block mixing fails. Color-basis projectors inside the
`V_3` block also fail.

## QCA Input Contract

The QCA audit reads its nontrivial input only from `data/qca_data.json`.
Exact matrix entries must be strings parseable as rational numbers, such as
`"0"`, `"1"`, `"-1"`, or `"1/2"`. Floating-point matrix entries are rejected.

Minimal shape, with placeholders that must be replaced by exact rational
matrix entries before the audit can pass:

```json
{
  "structural_origin": "unknown",
  "candidate_generators": [
    {"name": "e1", "matrix": [["..."]], "block": "V3"},
    {"name": "e2", "matrix": [["..."]], "block": "V3"},
    {"name": "e3", "matrix": [["..."]], "block": "V3"},
    {"name": "m1", "matrix": [["..."]], "block": "V2"},
    {"name": "m2", "matrix": [["..."]], "block": "V2"}
  ],
  "candidate_complex_structure": {
    "name": "J",
    "origin": "unknown",
    "matrix": [["..."]]
  },
  "allowed_gate_generators": [
    {"name": "G0", "matrix": [["..."]]}
  ],
  "split_projectors": {
    "P3": [["..."]],
    "P2": [["..."]]
  }
}
```

Passing the audit requires an explicit `J` with `J^2 = -I`, `J` preserving
`P3` and `P2`, `J` included in the allowed gate algebra, structural `3+2`
generator blocks, compatible signature, no `V_3 <-> V_2` gate mixing, and an
SM-commutant-safe gate algebra.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pytest -q
uv run python scripts/branching_check.py --check
uv run python scripts/qca_split_audit.py --check
```

## Claim Boundary

Do not call a representation identity a theorem about QCA geometry. Do not flip
a boolean unless code computes it or a proof derives it. Do not treat "`n = 3`
was chosen" as "`n = 3` was selected." Do not hide hand-chosen complex
structures, SU(5) embeddings, or gate restrictions.
