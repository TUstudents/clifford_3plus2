# clifford-3plus2-d5

Workspace for several QCA-derivation modules. Each module is a self-contained
sub-project with its own source, tests, scripts, and documentation. They share
a single Python environment and a small set of common utilities (rational
matrix algebra, exterior algebra, Spin(10) branching).

## Modules

| Module | Status | What it is |
|---|---|---|
| [`obstruction_r10`](src/clifford_3plus2_d5/obstruction_r10/) | **Frozen for publication** | Carrier-first QCA derivation of the chiral-16 of Spin(10) on `R^10`, with five structural obstruction propositions. The five propositions plus two corollaries close specified primitive classes; conjectural Proposition 4b has a finite census witness. |
| [`lepton`](src/clifford_3plus2_d5/lepton/) | **Positive result, active** | Cl(0,10) Pati-Salam factorization producing the full SM gauge content (`SU(3) × SU(2)_L × U(1)_Y`), a compatible complex structure, and the correct one-generation hypercharge spectrum, with 1+1D massless-Dirac/Weyl continuum dynamics under background gauge links. |
| [`sim`](src/clifford_3plus2_d5/sim/) | **Shared infrastructure** | Generic JAX simulation helpers for array conversion, periodic rolls, state/link layout, diagnostics, and lightweight benchmarks. |
| [`spacetime_qca`](src/clifford_3plus2_d5/spacetime_qca/) | **In progress** | 3D BCC Weyl walk (Bialynicki-Birula 1994), Dirac chiral assembly, hypercubic doubling control, tensor lift to the lepton internal carrier, position-dependent background gauge covariance, JAX numerical backend, and Wilson plaquette observables/action. |
| [`triality`](src/clifford_3plus2_d5/triality/) | **Closed: negative result** | Spin(8) triality kill test. K1 failed: the natural Pati-Salam-aligned SM Cartan is not preserved as a subspace by the order-3 outer automorphism. Three triality copies cannot represent three equivalent SM generations under this embedding. |
| [`broken_triality`](src/clifford_3plus2_d5/broken_triality/) | **Closed: negative result** | Broken-Z/3 follow-on to the triality kill test.  BT-1 passed (3 distinct Yukawa eigenvalues with non-zero mixing) but BT-2 failed: non-zero eigenvalue ratio `360/217 ≈ 1.66`, essentially flat.  Pure triality projection of a hypercharge-aligned vector does not produce SM mass hierarchy. |

## Read the project status

For an at-a-glance overview, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Shared utilities

These live at the package root and are imported by multiple modules:

- [`algebra/`](src/clifford_3plus2_d5/algebra/) — exact rational matrix algebra, real carrier construction, `IncrementalMatrixSpan` and `RationalMatrixSpan` for sparse exact echelon, commutant/centralizer helpers.
- [`exterior.py`](src/clifford_3plus2_d5/exterior.py) — chiral exterior-algebra basis machinery.
- [`branching.py`](src/clifford_3plus2_d5/branching.py) — Spin(10) → SM branching tables.

## Cross-module dependencies

- `lepton` depends on `obstruction_r10.qca.rule_verdict` (VerdictProfile-based verdict path).
- `spacetime_qca` test-level imports from `lepton` (Pati-Salam / SM generators for tensor-lift audits).
- `spacetime_qca` imports reusable JAX infrastructure from `sim`; BCC/QCA physics stays in `spacetime_qca`.
- `triality` imports from `lepton` (Cl(0,10) gammas, SM gauge generators, hypercharge) through its own `reuse.py`.
- `broken_triality` imports from `triality` (which itself imports from `lepton`).  No new octonion / Clifford code.

## Working in a module

Each module is self-contained. Tests and scripts live inside the module:

```bash
# Run all tests across all modules
uv run pytest -q

# Run tests for a single module
uv run pytest src/clifford_3plus2_d5/obstruction_r10/tests/ -q
uv run pytest src/clifford_3plus2_d5/lepton/tests/ -q
uv run pytest src/clifford_3plus2_d5/sim/tests/ -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/ -q
uv run pytest src/clifford_3plus2_d5/triality/tests/ -q
uv run pytest src/clifford_3plus2_d5/broken_triality/tests/ -q

# Run a script via module invocation
uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.gauge_equivalence_check --check
```

## Workspace-level docs

These describe the workspace itself, not module results:

- [`docs/PUBLICATION_PLAN.md`](docs/PUBLICATION_PLAN.md) — plan for the obstruction_r10 paper.
- [`docs/REORG_PLAN.md`](docs/REORG_PLAN.md) — record of the workspace reorganization.
- [`docs/project_conventions.md`](docs/project_conventions.md) — shared coding conventions.

## Honest framing

This workspace does not derive the Standard Model from QCA in any single
module. The `obstruction_r10` module classifies *what cannot be derived* in
specified primitive classes. The `lepton` module *constructs* the SM gauge
content from a declared Clifford framework, accepting specific algebraic
choices as input. These are different questions with different answers, and
the workspace is the honest superposition of both.
