# Projector Lattice Report

Phase 3 adds exact checks for the structural `3+2` split candidate

```text
P_3 = I_2 ⊗ diag(1,1,1,0,0)
P_2 = I_2 ⊗ diag(0,0,0,1,1).
```

This phase still does not prove that microscopic QCA rules force `P_3/P_2`.

## Candidate Identities

The checker verifies:

```text
P_3 + P_2 = I_10
P_3 P_2 = 0
P_3^2 = P_3
P_2^2 = P_2
rank_R(P_3) = 6
rank_R(P_2) = 4
[J,P_3] = 0
[J,P_2] = 0
```

The default addressability algebra contains only the whole block projectors
`P_3` and `P_2`, so the exact candidate check passes.

## Addressability Falsifiers

The checker rejects microscopic controls that resolve smaller axes inside the
future color or weak blocks:

```text
rank-one color projectors
rank-one weak projectors
off-block mixers
non-scalar within-block controls
```

The falsifiers are checked with:

```bash
uv run python scripts/structural_split_check.py --include-rank-one-color --expect-verdict falsified
uv run python scripts/structural_split_check.py --include-rank-one-weak --expect-verdict falsified
```

## Current Status

```text
projector_identities_passed: true
projectors_commute_with_J: true
addressability_algebra_safe: true
qca_supplies_structural_3plus2_split: false
structural_split_verdict: candidate_only
load_bearing_qca_bridge: false
```

The checker can certify the declared projector candidate and its falsifiers.
Phase 5 must still supply a real finite-depth microscopic QCA architecture
that forces the split as structural data rather than a Standard Model label.
