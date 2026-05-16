# Session 13 Report - Cl(0,8) / Octonion Stabilizer Audit

Status: exact Clifford and octonion stabilizer audit implemented.

## Purpose

This session tests the Clifford/octonion pivot before building any QCA rule
families. The goal is to distinguish real derived structure from notation
change or "Bott magic" handwaving.

The audit pins the signature:

```text
Cl(0,8): gamma_i^2 = -I
gamma_i gamma_j + gamma_j gamma_i = -2 delta_ij I
```

## Derived Clifford Structure

The exact `16x16` real gamma system passes the pinned `Cl(0,8)` relations.

Audit results:

```text
gamma count = 8
gamma dimension = 16
volume square = I
chirality ranks = (8,8)
full Cl(0,8) commutant dimension = 1
even Cl(0,8) commutant dimension = 2
```

Interpretation:

- The full Clifford action is irreducible over `R`; its commutant is scalar.
- The even Clifford action exposes only the intrinsic chirality `Z/2` split.
- Chirality is derived, but it is not yet Standard Model structure.

## Cl(0,2) and J

The `Cl(0,2)` audit uses two anticommuting generators squaring to `-I`.
The basis candidates:

```text
e1, e2, e1 e2
```

all square to `-I` and pairwise anticommute.

Interpretation:

- `Cl(0,2) = H` constrains the complex-structure choice.
- It does not choose a unique `J`.
- Choosing which quaternionic unit is called `i` remains an explicit choice.

This is progress over arbitrary clock choice, but not elimination of choice.

## Octonion Multiplication Choice

The audit fixes one oriented Fano table:

```text
(1,2,3), (1,4,5), (1,6,7),
(4,2,6), (2,5,7), (3,4,7), (3,5,6)
```

This is an explicit choice. The tests verify:

- imaginary basis units square to `-1`;
- selected Fano products and signs;
- alternativity on representative basis samples.

## G2 and SU3 Stabilizer Chain

The octonion derivation algebra is computed directly from:

```text
D(xy) = D(x)y + xD(y)
D(1) = 0
```

Audit results:

```text
dim Der(O) = 14
```

This is the matrix-level `g2` stabilizer of the chosen octonion
multiplication.

Then the stabilizer of the chosen imaginary direction `e7` is computed by:

```text
D(e7) = 0
```

Audit result:

```text
dim stabilizer(e7) = 8
```

This is the expected `su3` subalgebra inside `g2`.

Interpretation:

- `G2` is derived after choosing octonion multiplication.
- `SU3` is derived after additionally choosing an imaginary direction.
- The six-dimensional complement to `e7` in `Im(O)` is the real form of the
  color-triplet space `C3`.

## Derived vs Chosen

Derived:

- exact `Cl(0,8)` anticommutation;
- chirality split `R16 = R8 + R8`;
- scalar full commutant;
- even commutant `R + R`;
- `g2` derivation algebra of the chosen octonion multiplication;
- `su3` stabilizer of the chosen imaginary direction.

Chosen:

- the octonion multiplication table;
- the imaginary direction `e7`;
- which `Cl(0,2)` quaternionic unit is called `J`.

Not yet tested:

- explicit Spin(8) triality automorphism;
- QCA primitives built from Clifford reflections or bivector rotations;
- whether Clifford-generated dynamics preserve or reveal the stabilizer chain.

## Framing

The Clifford/octonion framework does not derive the Standard Model from
nothing. It reduces and intertwines the choices:

```text
old frame: choose (3,2) partition + clock J + commutant target
new frame: choose octonion multiplication + imaginary direction + H-unit J
```

That is a tighter framework. It is not magic.

The useful positive result is that the stabilizer chain

```text
Spin(8) -> G2 -> SU3
```

is constructively visible at the matrix level once the octonion table and
imaginary direction are documented.

## Validation

```bash
uv run ruff check src/clifford_3plus2_d5/lepton
uv run pytest src/clifford_3plus2_d5/lepton/tests/test_clifford_octonion.py -q
uv run python -m clifford_3plus2_d5.lepton.scripts.clifford_octonion_audit
```

Focused audit result:

```text
9 passed
```

CLI payload:

```text
signature = Cl(0,8)
gamma_count = 8
gamma_dimension = 16
clifford_relations_pass = true
chirality_ranks = (8,8)
full_commutant_dimension = 1
even_commutant_dimension = 2
cl02_j_candidate_count = 3
g2_derivation_dimension = 14
su3_stabilizer_dimension = 8
```

## Next Step

If this pivot continues, Session 14 should test Clifford primitive dynamics.
Start with the most rigid class:

```text
individual Cl(0,8) reflections / generator products
```

Only after that should bivector rotations `exp(theta gamma_i gamma_j)` be
introduced.
