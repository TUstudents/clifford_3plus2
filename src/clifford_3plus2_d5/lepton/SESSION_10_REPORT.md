# Session 10 Report - C5 Discovered-Split Search

Status: deterministic non-synthetic search families were added for the
complex-linear `C5` discovered-split profile.

## Goal

Session 9 proved the checker can accept a canonical synthetic `(3,2)` split
and a conjugated synthetic `(3,2)` split. Session 10 asks the next question:
can a non-synthetic primitive family produce a unique discovered `(3,2)` split
without explicitly declaring the target projectors?

The active profile is:

```text
complex_c5_discovered_3plus2_split
target ranks = (3,2)
required block dimensions = (9,4)
required block commutativity = (False, False)
```

## Added Search Families

The `c5-discovered` scan now accepts a `--family` switch:

- `controls`: seeded, irreducible, rank-one locking, canonical synthetic, and
  conjugated synthetic controls;
- `phase-permutation`: diagonal finite phases plus permutation generators;
- `monomial`: finite phase-permutation generator pairs;
- `finite-order`: pure finite-order permutation generator pairs;
- `all`: controls plus all non-synthetic families.

All non-synthetic candidates carry metadata:

```text
synthetic=false
family=<family-name>
```

The default exact phase order for the non-synthetic families is `2`. Higher
roots are intentionally not the default because exact symbolic closure and
idempotent solving can become slow before producing new structural information.

## Results

Small deterministic panels give:

| Family | Candidates | Verdict Pattern |
|---|---:|---|
| `phase-permutation` | 4 | `not_solved`, `falsified_no_split`, `not_solved`, `not_solved` |
| `monomial` | 3 | `falsified_no_split`, `falsified_rank_one_locking`, `falsified_no_split` |
| `finite-order` | 4 | `falsified_rank_one_locking`, `falsified_rank_one_locking`, `not_solved`, `falsified_rank_one_locking` |

No non-synthetic `split_candidate` appears in these panels.

The observed failure modes match the earlier obstruction pattern:

- full/transitive generators collapse to irreducible `M5(C)` behavior and
  produce `falsified_no_split`;
- monomial or finite-order degeneracies create extra central rank-one
  idempotents and produce `falsified_rank_one_locking`;
- low-order phase-permutation degeneracies can leave an underdetermined center
  where the idempotent solver returns `not_solved`.

## Interpretation

The discovered-split profile is working: it accepts the synthetic and
conjugated witnesses from Session 9, and it records whether the canonical SM
frame was matched. But the first non-synthetic primitive panels do not produce
a genuine discovered `(3,2)` split.

This is a negative result only for the small deterministic families above. It
does not close arbitrary complex-linear rules, continuous parameter families,
or richer locality-aware QCA constructions.

## Validation

```bash
uv run ruff check src/clifford_3plus2_d5/lepton
uv run pytest src/clifford_3plus2_d5/lepton/tests/test_complex_split.py -q
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --family phase-permutation --max-candidates 4
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --family monomial --max-candidates 3
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_scan --profile c5-discovered --family finite-order --max-candidates 4
uv run python -m clifford_3plus2_d5.lepton.scripts.complex_summary --profile c5-discovered --family monomial --max-candidates 3
```

Focused complex split result:

```text
22 passed
```

## Next Work

- Decide whether to broaden the non-synthetic complex-linear search to
  structured dense unitary families, not just monomial/permutation families.
- Add an optional exact-root budget for order `3` and order `4` scans, with a
  timeout or candidate-level bail-out so one hard symbolic case cannot stall
  the whole panel.
- If the complex-linear route remains negative, compare it directly against the
  earlier real-carrier obstruction table in a consolidated report.
