# obstruction_r10 — Status

**Status**: frozen for publication.

This module is the R^10 carrier-first obstruction classification program.
It produced five propositions and two corollaries closing specified primitive
classes for the QCA derivation of the Spin(10) chiral-16 spinor structure
from microscopic rule data.

No further research is expected on the closed primitive classes. The open
classes (non-translation-invariant defects, parameterized rule families,
higher-dimensional carriers, the conjectural Proposition 4b structural
extension) are identified as future work in the paper, not as gating items
for publication.

## Five-line summary of what is proven

1. **Block-blind primitives** cannot produce a rule-derived (6,4) center.
2. **Commuting non-scalar second layers** force lower-rank central
   idempotents, violating no-locking.
3. **Floquet-α-type noncommuting on-site rules** force either lower-rank
   center, full internal addressability, or commutativity — closing the
   on-site bridge under the one-quadratic-factor-per-block hypothesis.
4. **Projector-free monomial-hop Bloch rules** of windings (4, 4, 4, 3, 3)
   never produce a rule-generated compatible J (full 2400-candidate census).
5. **The Route-1 four-J orbits** are SM-inequivalent under the
   fixed-`SU(3) × SU(2)_L × U(1)_Y` gauge-equivalence relaxation
   (hypercharge does not survive the η-block Hodge flip).

Conjectural Proposition 4b extends 4a to all coprime monomial hops via the
Conditional Lemma 4b.1 (`split-real algebra contains no real orthogonal J`).

## Reproducibility

Each proposition has a one-line reproduction command:

```bash
# Prop 1 (E1/E2 bounded exploration)
uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.explore_rule_space --check --output-dir /tmp/e1
uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.discover_projectors --check --mode unseeded --expect-verdict not_found --output-dir /tmp/e2

# Prop 3 (Floquet-α exhaustive signed-twist scan)
uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.floquet_alpha --variant noncommuting-exhaustive --check

# Prop 4a (Bloch monomial-hop census; long-running)
uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.bloch_path_a_stepwise \
  --family monomial-hop --pattern-count 10 --cycle-count 24 --shift-count 10 --check

# Prop 5 (SM-inequivalence of Route-1 J orbits)
uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.gauge_equivalence_check --check
```

See [theory.md](theory.md) and [literature/](literature/) for the full
statements, proofs, and witnesses.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/obstruction_r10/tests/ -q
```
