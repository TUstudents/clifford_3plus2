# Session 12 Report - Consolidated Obstruction Map

Status: lepton-lab outcomes consolidated into a static obstruction map and a
machine-readable JSON summary.

## Why Consolidate Now

The lab has accumulated enough evidence that more ad hoc finite matrix panels
are unlikely to change the picture. The repeated pattern is stable:

- the center is too small, so no `J` or no split appears;
- the center is too large, so extra idempotents or underdetermined sectors
  appear;
- rank-one central locking appears;
- the algebra becomes irreducible and loses the target split;
- synthetic controls pass only because the split is engineered.

Session 12 records that map explicitly and separates controls from actual
mechanism candidates.

## Consolidated Outcome Table

| Entry | Frame | Dimension | Status | Obstruction |
|---|---|---:|---|---|
| Lab A clock-plane closure | real carrier | `R4` | toy positive | none toy |
| Lab B strict | real carrier | `R6` | search negative | multiple J orbits |
| Lab B structural wall | real carrier | `R6` | search negative | multiple J orbits |
| Lab B internal domain wall | real carrier | `R6` | domain-wall positive | none toy |
| Lab B physical wall | real carrier | `R12` | physical negative | physical promotion extra center |
| C3 synthetic split | complex split-first | `C3` | control positive | none control |
| C5 fixed synthetic split | complex split-first | `C5` | control positive | none control |
| C5 discovered synthetic split | complex split-first | `C5` | control positive | synthetic only |
| C5 phase/permutation search | complex split-first | `C5` | search negative | center too large / no split |
| C5 monomial search | complex split-first | `C5` | search negative | irreducible no split / rank-one locking |
| C5 finite-order search | complex split-first | `C5` | search negative | rank-one locking |
| C5 dense conjugated control | complex split-first | `C5` | control positive | synthetic only |
| C5 dense sign-conjugate search | complex split-first | `C5` | search negative | center too large |
| C5 dense Householder search | complex split-first | `C5` | search negative | center too large |
| C5 dense Fourier-lite search | complex split-first | `C5` | search negative | rank-one locking / no split |
| Designed locality-aware complex QCA | complex split-first | finite lattice | open | not tested |

The only positive that resembles a mechanism is the internal `R6` domain wall,
but it is explicitly not the original global bridge contract. Its physical
`R12` promotion fails exact-center audit: the center grows to dimension `8`
instead of shrinking to the desired dimension `2`.

## Machine-Readable Map

The static map lives in:

```text
src/clifford_3plus2_d5/lepton/obstruction_map.py
```

The JSON CLI is:

```bash
uv run python -m clifford_3plus2_d5.lepton.scripts.obstruction_summary
uv run python -m clifford_3plus2_d5.lepton.scripts.obstruction_summary --entries
```

The map intentionally does not rerun expensive symbolic scans. It summarizes
stable session outcomes and gives tests a fast invariant surface.

## Known Gaps

- No physical `R12` bridge candidate has passed exact-center audit.
- No non-synthetic `C5` discovered split was found in monomial, permutation,
  finite-order, or dense rational panels.
- Lab A remains clock-plane-aware, not `J`-blind.
- No `R10` lift has been attempted from the lepton module.
- The complex split-first route has not yet tested a designed locality-aware
  QCA primitive.

## Brainstorm - Designed Locality-Aware Complex QCA Primitive

Random exact finite matrices are not finding a split. The next plausible
positive route is to make the split a consequence of locality or transport
structure rather than algebraic accident.

### Design A - Bipartite Sublattice Split

Carrier:

```text
site Hilbert = C5
lattice = 1D bipartite chain A/B
```

Primitive:

- A-site transport mixes three modes.
- B-site transport mixes two modes.
- Hopping respects the bipartite transport pattern.
- No central projector is inserted as a layer.

Possible effect:

- the propagation graph has two transport components of sizes `3` and `2`;
- the central split is recovered from transport connectivity;
- local algebras inside components can still be full, giving block dimensions
  `(9,4)`.

Caveat: if the graph components are literally disconnected, this is close to
declaring the split in graph form.

### Design B - Complex Gauge-Bundle Domain Wall

Carrier:

```text
C5_left + C5_right = C10
```

Primitive:

- left side carries one local `(3,2)` frame;
- right side carries a conjugated `(3,2)` frame;
- wall transition transports one frame into the other;
- bulk layers are full inside transported blocks.

Possible effect:

- the wall identifies relative split frames without solving for `J`;
- the center might stabilize at a global `(6,4)` split or at a recoverable
  local `(3,2)` split.

Caveat: the real `R12` wall failed by center growth, so this must include an
exact center audit from the start.

### Design C - Constraint-Generated Split via Transport Graph

Carrier:

```text
C5 modes as vertices of a colored transport graph
```

Primitive:

- edge color alpha connects exactly three modes;
- edge color beta connects exactly two modes;
- add noncommuting local mixers inside each connected component;
- do not add projectors.

Why it is the best first design:

- graph adjacency algebras naturally expose central idempotents as connected
  components;
- a `(3,2)` split can be explained by transport locality;
- controlled cross-edge perturbations should destroy the split, giving a clean
  falsifier.

First implementation target:

```text
complex_graph_qca.py
3-cycle on modes {0,1,2}
2-edge on modes {3,4}
full local mixing inside each component
then controlled cross-edge perturbation
```

Expected baseline:

- disconnected graph control: `split_candidate`;
- cross-edge perturbation: `falsified_no_split`;
- over-diagonal graph: `falsified_rank_one_locking`.

This is still a kind of structural declaration, but it is physically
interpretable: the split comes from transport connectivity, not an inserted
projector.

## Recommendation

Do Design C first. It gives the cleanest theorem target:

```text
central splits of the complex transport algebra correspond to connected
components of the transport graph.
```

If Design C only reproduces synthetic block declaration in graph language,
then the honest output is a negative report: every tested carrier-first and
split-first route either engineers the target structure or hits the same center
obstructions.
