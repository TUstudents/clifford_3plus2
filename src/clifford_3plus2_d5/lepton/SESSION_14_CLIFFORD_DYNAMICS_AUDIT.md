# Session 14 — Rigid Clifford Dynamics Audit

Status: implemented as an additive lepton-local audit.

## Question

Session 13 showed that `Cl(0,8)` gives an exact real Clifford carrier, an
intrinsic chirality split, and an octonion stabilizer chain once we choose:

- the octonion multiplication table,
- the imaginary direction `e7`,
- which `Cl(0,2)` unit is called `J`.

Session 14 asks whether the rigid Clifford primitive class is rich enough to
see that stabilizer dynamically:

> Among exact Clifford words on `Cl(0,8)`, which preserve the chosen chiral
> octonion `G2` and `SU(3)` structure, and do those primitives generate a
> nontrivial `g2` or `su3` Lie closure?

This is not a split-derivation claim. It is a stabilizer-compatibility audit.

## Fixed Embedding

The triality embedding is pinned:

- identify the positive chiral spinor `R8_+` with the Session 13 octonion
  table,
- define `G2` as the automorphism group of that octonion product,
- define `SU(3)` as the subgroup fixing the imaginary direction `e7`.

This chooses one of the triality-related `G2` subgroups of `Spin(8)`.

## Primitive Family

The rigid v1 family is finite:

- identity: 1 candidate,
- odd reflections `gamma_i`: 8 candidates,
- bivectors `gamma_i gamma_j`: 28 candidates,
- four-vectors `gamma_i gamma_j gamma_k gamma_l`: 70 candidates.

Total: 107 candidates.

Continuous rotations `exp(theta gamma_i gamma_j)` are deliberately out of
scope for this audit.

## Tests

The code separates:

- group membership: `is_octonion_automorphism(T)`, checking `T(xy)=T(x)T(y)`;
- algebra normalization: `normalizes_lie_algebra(T, basis)`, checking
  `T D T^-1` remains in the chosen Lie-algebra span;
- stabilizer class:
  - `su3_fixing_e7`,
  - `su3_flipping_e7`,
  - `g2_beyond_su3`,
  - `spin8_beyond_g2`,
  - `not_chirality_preserving`.

Positive and negative controls are included:

- identity passes the `SU(3) ⊂ G2` tests,
- a fixed nontrivial signed permutation preserves the octonion product,
- `gamma_1` swaps chirality,
- `gamma_1 gamma_2` preserves chirality but fails the octonion automorphism
  test.

## Results

Running:

```bash
uv run python -m clifford_3plus2_d5.lepton.scripts.clifford_dynamics_audit
```

currently reports:

```json
{
  "candidate_count": 107,
  "family_counts": {
    "identity": 1,
    "reflection": 8,
    "bivector": 28,
    "four_vector": 70
  },
  "chirality_preserving_count": 99,
  "stabilizer_class_counts": {
    "su3_fixing_e7": 3,
    "su3_flipping_e7": 4,
    "g2_beyond_su3": 0,
    "spin8_beyond_g2": 92,
    "not_chirality_preserving": 8
  },
  "octonion_automorphism_count": 7,
  "g2_normalizer_count": 15,
  "su3_normalizer_count": 27,
  "g2_algebra_closure_dimension": 0,
  "su3_algebra_closure_dimension": 0,
  "expected_g2_dimension": 14,
  "expected_su3_dimension": 8
}
```

## Reading

The rigid exact-word family does find a small discrete stabilizer signal:

- 7 candidates are octonion automorphisms;
- those are all in the `SU(3)` fixing/flipping buckets, not in a broader
  `G2 beyond SU(3)` bucket;
- 15 candidates normalize `g2`;
- 27 candidates normalize `su3`.

But the Lie closure dimension is zero. This is the load-bearing result:
the exact finite representatives are too discrete to expose infinitesimal
`g2` or `su3` dynamics. They certify compatibility with the chosen stabilizer
chain, but they do not generate the full stabilizer dynamics.

## Consequence

The `Cl(8)` pivot remains useful as a stabilizer framework, but the v1 rigid
primitive class is not rich enough for a QCA dynamics program. Session 15
should add a `clifford_dynamics_profile` and decide whether to admit
continuous or finite-order bivector rotations. The verdict question becomes:

> Does a Clifford-generated rule preserve or reveal the chosen `G2 ⊃ SU(3)`
> stabilizer while supplying a unique compatible `Cl(2)` complex structure?

If Session 15 keeps only rigid words, the expected answer is negative for
dynamic richness. To get full `g2` / `su3` closure, the primitive family likely
needs infinitesimal or finite-order rotations, not only Clifford monomials.
