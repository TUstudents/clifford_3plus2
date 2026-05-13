# clifford-3plus2-d5

Do not fool yourself.

This repo tests a possible geometry-to-Spin(10) bridge.

It does not derive the Standard Model.
It does not derive three families.
It does not prove mirror decoupling.

## Current Theorem Target

Given QCA Clifford data that structurally supplies

```text
V = V_3 oplus V_2
```

plus a non-hand-chosen compatible complex structure `J`, the induced
Spin(10)/SU(5) exterior-algebra construction gives one Standard Model
generation:

```text
S_+ = Lambda^even(C^5)
```

The first load-bearing question is whether the QCA supplies the `3+2`
Clifford split structurally, including a non-hand-chosen complex structure
`J`.

If the QCA two-plane or complex structure is chosen by hand, the project is
`notation_only`.

## Claim Boundary

This repository may verify textbook Spin(10) branching arithmetic. That is
not a theorem about QCA geometry.

Do not flip a boolean unless code computes it or a proof derives it. Do not
treat "`n = 3` was chosen" as "`n = 3` was selected." Do not hide hand-chosen
complex structures, SU(5) embeddings, or gate restrictions.

## Development

```bash
uv run ruff check .
uv run pytest -q
```
