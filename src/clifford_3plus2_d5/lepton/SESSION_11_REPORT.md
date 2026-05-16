# Session 11 Report - Dense Complex-Linear Primitive Search

Status: dense exact `C5` candidate families were added for the discovered-split
profile.

## Goal

Session 10 showed that monomial and permutation-style non-synthetic families
do not produce a discovered `(3,2)` split. Session 11 tests a broader exact
complex-linear class: dense rational generators.

The active target remains:

```text
profile = complex_c5_discovered_3plus2_split
target ranks = (3,2)
required block dimensions = (9,4)
required block commutativity = (False, False)
```

## Added Families

The `c5-discovered` scan now accepts these dense families:

- `dense-conjugated-control`
- `dense-hadamard`
- `dense-householder`
- `dense-fourier-lite`
- `dense-all`

All dense rows carry metadata:

```text
dense=true
synthetic=true|false
control=true|false
```

The success metric for a real mechanism excludes `synthetic=true` and
`control=true`.

## Dense Control

The dense conjugated control takes the synthetic `M3(C) + M2(C)` witness and
conjugates it by an exact rational Householder matrix. This makes the
projectors non-coordinate while preserving the algebraic split.

Result:

```text
verdict = split_candidate
discovered_projector_ranks = (3,2)
block_dimensions = (9,4)
canonical_split_matched = False
synthetic = true
control = true
dense = true
```

The fixed C5 profile rejects the same dense conjugated control with
`falsified_no_split`, as expected. This confirms the discovered-split profile
is not merely checking the canonical coordinate frame.

## Non-Control Dense Results

Small deterministic panels give:

| Family | Candidates | Verdict Pattern |
|---|---:|---|
| `dense-hadamard` | 3 | `not_solved`, `not_solved`, `not_solved` |
| `dense-householder` | 3 | `not_solved`, `not_solved`, `not_solved` |
| `dense-fourier-lite` | 3 | `falsified_no_split`, `falsified_rank_one_locking`, `falsified_rank_one_locking` |

The `dense-all` summary over these 9 non-control dense candidates reports:

```text
split_candidate_count = 0
non_synthetic_split_candidate_count = 0
```

## Interpretation

The dense conjugated control proves the discovered-split machinery can recover
a non-coordinate `(3,2)` split. But the non-control dense rational families do
not produce a genuine split-first mechanism in this first panel.

The failure modes remain familiar:

- dense diagonal-sign conjugates leave the center too large;
- dense Householder reflection pairs leave the center too large;
- dense Fourier-lite conjugates reproduce the monomial obstruction:
  irreducible/no-split or rank-one locking.

This is still not a theorem against complex-linear split-first rules. It is a
negative result for small exact dense rational families.

## Validation

```bash
uv run ruff check src/clifford_3plus2_d5/lepton
uv run pytest src/clifford_3plus2_d5/lepton/tests/test_complex_split.py -q
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --family dense-conjugated-control --max-candidates 1
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --family dense-hadamard --max-candidates 3
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --family dense-householder --max-candidates 3
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --family dense-fourier-lite --max-candidates 3
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_summary --profile c5-discovered --family dense-all --max-candidates 9
```

Focused complex split result:

```text
29 passed
```

## Next Work

- Add per-candidate time/budget guards before trying higher-order exact roots
  or larger dense panels.
- Consolidate the real-carrier, domain-wall, complex split, and dense-search
  outcomes into one obstruction map.
- If continuing the positive search, move from finite algebra panels to a
  designed locality-aware complex QCA primitive, because random exact dense
  generators are not finding the split.
