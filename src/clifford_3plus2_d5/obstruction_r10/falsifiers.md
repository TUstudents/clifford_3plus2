# Falsifiers

The project should return `notation_only` or `falsified` when any load-bearing
datum is missing, arbitrary, or chosen independently of QCA geometry.

## J-First Falsifiers

1. The construction starts from `C^5` instead of deriving a real carrier and
   `J`.
2. `J` is copied from ambient scalar `i`.
3. `J` is chosen by hand.
4. `J` is selected because it gives `SU(5)`.
5. `J^2 != -I`.
6. `J` is not local, finite-depth generated, micromotion-derived, or forced by
   monodromy/rule data.
7. Many inequivalent `J` choices are equally valid.
8. The carrier is not real ten-dimensional, or the real carrier is only
   implicit notation.

## Split And Gate Falsifiers

1. The `3+2` split is arbitrary.
2. `P_3` or `P_2` is chosen because the Standard Model needs ranks `3` and `2`.
3. `P_3` or `P_2` fails to commute with `J`.
4. The rule data generate rank-one projectors inside `C^3`.
5. The rule data generate rank-one projectors inside `C^2`.
6. Gate algebra allows `C^3 <-> C^2` block mixing.
7. Gate algebra is block-diagonal but not in the `SU(3) x SU(2)` commutant.
8. Spacetime shifts use individual internal directions as controls.

## Legacy Audit Falsifiers

1. QCA two-plane is arbitrary.
2. SU(5) embedding is chosen independently of QCA.
3. Candidate `J` has no explicit operator matrix.
4. Candidate `J` is not in the allowed QCA gate algebra.
5. Candidate matrices use floating-point entries for exact Clifford or
   projector checks.
6. Project only reproduces Spin(10) branching.

## Implemented Guardrails

The current code enforces these boundaries:

- Missing `data/qca_data.json` returns `notation_only`.
- Incomplete or invalid JSON returns `notation_only`.
- Float matrix entries are rejected.
- Schema contract requires rational strings for exact matrix entries.
- `complex_structure_origin = "by_hand"` is falsifying.
- `complex_structure_origin = "unknown"` cannot pass.
- Off-block one-particle gate generators fail.
- Block-diagonal one-particle gates that resolve color-basis directions fail.
- The enhanced J-first algebra kernel checks the real carrier, `J`,
  `P_3/P_2`, commutants, finite-depth update candidates, spinor reconstruction,
  and rank-one addressability.
- The Phase 7 normalizer checker rejects rank-one color controls, rank-one
  weak controls, off-block controls, and full-`U(5)`-like mode-resolving
  controls.
- The Phase 8A real-QCA branch checker rejects rank-one color controls,
  rank-one weak controls, independently addressable rank-one pair rotations,
  and off-block controls.
- The E1 exploration runner records rejection reasons for every scanned word
  and currently finds no forced survivors in the bounded rule space.
- The E2 projector discovery runner keeps the default projector search
  unseeded, reports unsafe rank-2 projectors, and treats seeded or
  rank-one-derived projector pairs as controls rather than bridge evidence.

Current forcedness status:

```text
forcedness_verdict: candidate_only
real_qca_branch_verdict: candidate_only
e1_surviving_candidates: 0
e2_unseeded_projector_pairs_found: 0
load_bearing_qca_bridge: false
```
