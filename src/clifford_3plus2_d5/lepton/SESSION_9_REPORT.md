# Session 9 Report — Complex-Linear Split Lab

Status: split-first complex-linear pivot implemented for the minimal `C3`
lepton-family carrier, the fixed-basis `C5 = C3 + C2` carrier, and the
discovered `(3,2)` split profile on `C5`.

## Frame Change

This lab stops trying to derive `J`. The complex structure is assumed as
background structure, and the verdict asks whether a complex-linear rule
algebra derives the target central split without extra central locking.

The first pass covered:

```text
C3 = C singlet + C2 doublet
target ranks = (1, 2)

C5 = C3 + C2
target ranks = (3, 2)
```

## Implemented

- Complex carrier helpers for `C2`, `C3`, and `C5`.
- Complex algebra closure over exact SymPy matrices.
- Complex center computation by linear commutator equations.
- Low-dimensional central idempotent solving over the center.
- Split verdict labels:
  - `split_candidate`;
  - `seeded_split_control`;
  - `falsified_no_split`;
  - `falsified_rank_one_locking`;
  - `falsified_forbidden_idempotent`;
  - `not_solved`.
- C3 primitive/control panel.
- C5 primitive/control panel.
- C5 discovered-split profile, which accepts any unique complementary
  central idempotent pair of ranks `(3,2)` and separately records whether it
  matches the canonical SM frame.
- Lepton-local tests and CLI scripts.

## C3 Control Results

The deterministic C3 panel gives:

| Case | Verdict |
|---|---|
| Explicit target projectors | `seeded_split_control` |
| Full irreducible `M3(C)` control | `falsified_no_split` |
| Diagonal rank-one locking control | `falsified_rank_one_locking` |
| Synthetic `C + M2(C)` rule | `split_candidate` |

The synthetic candidate is not a physical bridge yet. It proves the new
complex-linear verdict can distinguish seeded, locked, irreducible, and
split-producing algebra types.

## C5 Control Results

The deterministic C5 panel gives:

| Case | Verdict |
|---|---|
| Explicit `(3,2)` projectors | `seeded_split_control` |
| Full irreducible `M5(C)` control | `falsified_no_split` |
| Diagonal rank-one locking control | `falsified_rank_one_locking` |
| Synthetic `M3(C) + M2(C)` rule | `split_candidate` |

The synthetic C5 candidate has:

- generated algebra dimension `13`;
- center dimension `2`;
- central idempotent ranks `(0, 2, 3, 5)`;
- block dimensions `(9, 4)`;
- block commutativity `(False, False)`.

This reproduces the original `(3,2)` target in the split-first complex-linear
frame. It is still synthetic: the next question is whether a non-seeded,
non-block-declared primitive family can produce the same structure.

## C5 Discovered-Split Results

The discovered profile removes the fixed-projector requirement. It searches
the central idempotents for a unique complementary rank `(3,2)` pair, then
runs the same block invariant checks on the discovered projectors.

The deterministic five-case panel gives:

| Case | Verdict | Canonical split matched |
|---|---|---|
| Explicit `(3,2)` projectors | `seeded_split_control` | `True` |
| Full irreducible `M5(C)` control | `falsified_no_split` | `False` |
| Diagonal rank-one locking control | `falsified_rank_one_locking` | `False` |
| Synthetic canonical `M3(C) + M2(C)` rule | `split_candidate` | `True` |
| Synthetic conjugated `(3,2)` rule | `split_candidate` | `False` |

The fixed C5 profile rejects the conjugated synthetic candidate with
`falsified_no_split`, while the discovered profile accepts it with discovered
projector ranks `(3,2)`, block dimensions `(9,4)`, and block commutativity
`(False, False)`.

This closes the fixed-basis gap in the split-first lab: we can now distinguish
"the rule found a valid structural `(3,2)` split" from "the rule found the
canonical SM frame."

## Validation

```bash
uv run ruff check src/clifford_3plus2_d5/lepton
uv run pytest src/clifford_3plus2_d5/lepton/tests -q
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --max-candidates 4
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_summary --max-candidates 4
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5 --max-candidates 4
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_summary --profile c5 --max-candidates 4
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --max-candidates 5
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_summary --profile c5-discovered --max-candidates 5
```

Full lepton-local test result after C3 implementation:

```text
16 passed
```

Focused complex split test result after C5 implementation:

```text
13 passed
```

Focused complex split test result after discovered-split implementation:

```text
18 passed
```

## Remaining Work

- Add non-synthetic primitive panels for the `C5` case.
- Decide whether block-preserving complex-linear primitives count as a
  mechanism or only as controls.
- Search for non-synthetic discovered `(3,2)` rules instead of only canonical
  or conjugated synthetic witnesses.
