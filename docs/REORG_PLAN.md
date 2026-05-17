# Repository Reorganization Plan

Status: planning. No changes have been made.

## Why reorganize

The repository currently presents the trunk (R^10 carrier-first obstruction
program) as *the* project, with `src/`, `scripts/`, `tests/`, `docs/`, and
`data/` at the top level. Sidecars (`lepton/`, `spacetime_qca/`) live inside
the package as subfolders. That structure was correct while the trunk was the
active frontier.

It is no longer the active frontier. The active state is:

- **Trunk** (R^10 obstruction classification): five propositions complete,
  publishable as a self-contained paper. Frozen for publication. No new
  research, only writeup.
- **Lepton sidecar** (Cl(0,10) Pati-Salam → SM gauge content + hypercharges):
  positive result, 1+1D dynamics verified. Active development possible toward
  mass / Yukawa / Lorentz / etc.
- **Spacetime QCA sidecar** (BCC Weyl walk, Bialynicki-Birula): scaffolding
  in place, Session 20 implementation in progress.

The repo should make this structure visible. A reader cloning the repo today
sees `docs/`, `scripts/`, `tests/` at the top and assumes the trunk is the
single project. That misrepresents the state.

## What the reorganization achieves

After reorg:

1. Each project module has its own self-contained subtree: source, tests,
   scripts, docs, data, reports.
2. The top-level repo presents itself as *"a workspace for several QCA
   modules, each with its own status"*, not *"the R^10 obstruction project
   plus some sidecars"*.
3. The trunk's frozen publication-ready content is clearly labeled and
   organized for the paper, not mixed with active sidecar artifacts.
4. The active sidecars have visible roadmap and don't compete with the trunk
   for the top-level namespace.

## Target layout

```
clifford_3plus2/
├── README.md                          # workspace-level: lists modules
├── CITATION.cff                       # paper citation (after submission)
├── pyproject.toml                     # one shared environment for all modules
├── uv.lock
├── PROJECT_STATUS.md                  # status of each module
│
├── src/clifford_3plus2_d5/
│   ├── __init__.py                    # workspace package marker
│   │
│   ├── algebra/                       # SHARED: rationals, matrices, commutants, real carriers
│   ├── exterior.py                    # SHARED: exterior-algebra utilities
│   ├── branching.py                   # SHARED: Spin(10) branching tables
│   ├── status.py                      # SHARED: cross-module status query
│   │
│   ├── obstruction_r10/               # TRUNK: renamed and self-contained
│   │   ├── __init__.py
│   │   ├── README.md                  # "this module produced 5 obstruction theorems"
│   │   ├── STATUS.md                  # frozen for publication
│   │   ├── theory.md                  # moved from docs/theory.md (trunk's results)
│   │   ├── falsifiers.md              # moved from docs/falsifiers.md
│   │   ├── handover_compliance.md     # moved
│   │   ├── roadmap.md                 # moved (frozen as historical record)
│   │   ├── literature/                # moved from docs/literature/ (19 reports)
│   │   ├── results/                   # moved from docs/results/
│   │   ├── qca/                       # rule_verdict + bloch_rule + spatial_1d + projected_centralizer + profiles + predicates + wall + ...
│   │   ├── search/                    # moved from src/clifford_3plus2_d5/search/
│   │   ├── explore/                   # moved from src/clifford_3plus2_d5/explore/
│   │   ├── sm/                        # moved from src/clifford_3plus2_d5/sm/ (SM commutant, embedding, hypercharge)
│   │   ├── clifford_audit.py          # moved from package root
│   │   ├── gate_algebra.py            # moved
│   │   ├── gauge_equivalence.py       # moved (Prop 5 witness)
│   │   ├── scripts/                   # moved from top-level scripts/ (23 files)
│   │   ├── tests/                     # moved from top-level tests/ (33 files)
│   │   └── data/                      # moved from top-level data/
│   │
│   ├── lepton/                        # SIDECAR: Cl(0,10) → SM (positive result)
│   │   ├── ... (already self-contained; just verify imports)
│   │
│   └── spacetime_qca/                 # SIDECAR: BCC Weyl walk (active)
│       ├── ... (already self-contained)
│
└── docs/                              # WORKSPACE-LEVEL: meta only
    ├── PUBLICATION_PLAN.md            # for obstruction_r10 paper
    ├── REORG_PLAN.md                  # this document
    └── archive/                       # preserved for git history reasons
```

Key principles:

- **Each module owns its own scripts, tests, docs, data.** No top-level
  `scripts/` or `tests/` directory. Tests run via
  `uv run pytest src/clifford_3plus2_d5/<module>/tests/`.
- **`docs/` at top level is workspace-meta only.** Just the publication
  plan, reorg plan, and a workspace README. No technical content.
- **Shared algebra/exterior/branching stays at package root.** Both lepton
  and obstruction_r10 import these.
- **The trunk gets renamed `obstruction_r10`** to make the scope explicit.
  This is the most visible change: the name no longer suggests "the
  project" but "one module about R^10 obstructions."

## What stays shared

These are genuinely used by multiple modules and should not move into any
module:

- `src/clifford_3plus2_d5/algebra/` (rationals, matrices, commutants, real
  carriers, IncrementalMatrixSpan, RationalMatrixSpan)
- `src/clifford_3plus2_d5/exterior.py` (chiral-Fock-basis machinery)
- `src/clifford_3plus2_d5/branching.py` (Spin(10) → SM branching tables)
- `src/clifford_3plus2_d5/__init__.py` (workspace package marker)
- `src/clifford_3plus2_d5/status.py` (cross-module status query)

These are the load-bearing utilities. Moving them into `obstruction_r10`
would break lepton; moving them into `lepton` would break the trunk.

## Coupling preserved

Current coupling that must survive the reorg:

- `lepton/*` imports `clifford_3plus2_d5.qca.rule_verdict` (the VerdictProfile
  machinery). After reorg this becomes
  `clifford_3plus2_d5.obstruction_r10.qca.rule_verdict`.

  This is the only cross-module import the sidecars rely on. After reorg,
  it's a clearly-labeled dependency from `lepton` on the trunk's
  verdict machinery, which is honest: the lepton sidecar used the trunk's
  VerdictProfile path. Document this in `lepton/README.md`.

- `spacetime_qca/*` has no cross-module imports outside `algebra/`. Stays
  clean.

## Naming choices

Three names to pick. Recommendations:

1. **`obstruction_r10`** (recommended over `trunk`, `original`, `r10`).
   Names the result, not the position. After publication, this module's
   name is the paper's subject: "R^10 obstructions."

2. **`lepton`** (keep). Existing name, accurate enough. Could be more
   specific (`pati_salam_cl010`), but the working name is established.

3. **`spacetime_qca`** (keep). Existing name, accurate.

If you'd rather rename `lepton` to `pati_salam_cl010` or
`clifford_pati_salam`, do it as a separate commit after the trunk reorg.
That rename is mechanical (one `git mv` plus import updates) and shouldn't
gate the larger reorg.

## What this is NOT doing

- Not creating separate packages (`pip install obstruction-r10`). All
  modules stay inside `clifford_3plus2_d5` and share one venv. Renaming
  to separate distribution packages would be a different, larger change.
- Not deleting the trunk's git history. All moves preserve history via
  `git mv`.
- Not changing import surfaces *between* shared algebra and modules.
  `algebra/`, `exterior.py`, `branching.py` stay at package root.
- Not rewriting any module's internal code. Only file paths change, plus
  `from clifford_3plus2_d5.qca` → `from clifford_3plus2_d5.obstruction_r10.qca`
  fixups.

## Execution sequence

### Phase 1: branch + dry-run inventory (~1 hour)

1. Create a new branch `repo-reorg` from the current branch.
2. Compute the exact file list:
   ```bash
   git ls-files docs/ scripts/ tests/ data/                       # what moves into obstruction_r10/
   git ls-files src/clifford_3plus2_d5/qca/ src/clifford_3plus2_d5/search/ src/clifford_3plus2_d5/explore/ src/clifford_3plus2_d5/sm/
   ```
3. Grep for all import paths that will need rewriting:
   ```bash
   grep -rn "from clifford_3plus2_d5\.qca" src/ tests/ scripts/
   grep -rn "from clifford_3plus2_d5\.sm" src/ tests/ scripts/
   grep -rn "from clifford_3plus2_d5\.search" src/ tests/ scripts/
   grep -rn "from clifford_3plus2_d5\.explore" src/ tests/ scripts/
   ```
4. Save this inventory in `docs/REORG_INVENTORY.md` for review before
   committing any moves.

### Phase 2: create the new directory shell (~30 min)

5. `mkdir -p src/clifford_3plus2_d5/obstruction_r10/{qca,search,explore,sm,scripts,tests,literature,results,data}`
6. Add stub `__init__.py` files.
7. Add `obstruction_r10/README.md` and `obstruction_r10/STATUS.md`
   describing the module's frozen state.

### Phase 3: move files with `git mv` (~2-3 hours)

The order matters for not breaking imports mid-flight:

8. Move the trunk source modules first, since lepton depends on them
   through the *current* import path:
   ```bash
   git mv src/clifford_3plus2_d5/qca src/clifford_3plus2_d5/obstruction_r10/qca
   git mv src/clifford_3plus2_d5/search src/clifford_3plus2_d5/obstruction_r10/search
   git mv src/clifford_3plus2_d5/explore src/clifford_3plus2_d5/obstruction_r10/explore
   git mv src/clifford_3plus2_d5/sm src/clifford_3plus2_d5/obstruction_r10/sm
   git mv src/clifford_3plus2_d5/clifford_audit.py src/clifford_3plus2_d5/obstruction_r10/
   git mv src/clifford_3plus2_d5/gate_algebra.py src/clifford_3plus2_d5/obstruction_r10/
   git mv src/clifford_3plus2_d5/gauge_equivalence.py src/clifford_3plus2_d5/obstruction_r10/
   ```

9. Move the trunk's scripts, tests, data, docs:
   ```bash
   git mv scripts src/clifford_3plus2_d5/obstruction_r10/scripts
   git mv tests src/clifford_3plus2_d5/obstruction_r10/tests
   git mv data src/clifford_3plus2_d5/obstruction_r10/data
   git mv docs/literature src/clifford_3plus2_d5/obstruction_r10/literature
   git mv docs/results src/clifford_3plus2_d5/obstruction_r10/results
   git mv docs/theory.md src/clifford_3plus2_d5/obstruction_r10/theory.md
   git mv docs/falsifiers.md src/clifford_3plus2_d5/obstruction_r10/falsifiers.md
   git mv docs/handover_compliance.md src/clifford_3plus2_d5/obstruction_r10/handover_compliance.md
   git mv docs/roadmap.md src/clifford_3plus2_d5/obstruction_r10/roadmap.md
   ```

10. Keep `docs/` at top level for workspace-meta:
    - `docs/PUBLICATION_PLAN.md` (already there)
    - `docs/REORG_PLAN.md` (this file)
    - `docs/archive/` (keep as-is)

### Phase 4: rewrite imports (~1-2 hours)

11. Bulk-rewrite imports in all moved trunk files:
    ```bash
    # In all .py files under src/clifford_3plus2_d5/obstruction_r10/
    sed -i 's/from clifford_3plus2_d5\.qca/from clifford_3plus2_d5.obstruction_r10.qca/g' ...
    sed -i 's/from clifford_3plus2_d5\.search/from clifford_3plus2_d5.obstruction_r10.search/g' ...
    sed -i 's/from clifford_3plus2_d5\.sm/from clifford_3plus2_d5.obstruction_r10.sm/g' ...
    sed -i 's/from clifford_3plus2_d5\.explore/from clifford_3plus2_d5.obstruction_r10.explore/g' ...
    sed -i 's/from clifford_3plus2_d5\.gauge_equivalence/from clifford_3plus2_d5.obstruction_r10.gauge_equivalence/g' ...
    sed -i 's/from clifford_3plus2_d5\.clifford_audit/from clifford_3plus2_d5.obstruction_r10.clifford_audit/g' ...
    sed -i 's/from clifford_3plus2_d5\.gate_algebra/from clifford_3plus2_d5.obstruction_r10.gate_algebra/g' ...
    ```

12. Rewrite the *lepton's* imports from `clifford_3plus2_d5.qca` to
    `clifford_3plus2_d5.obstruction_r10.qca`:
    ```bash
    sed -i 's/from clifford_3plus2_d5\.qca/from clifford_3plus2_d5.obstruction_r10.qca/g' src/clifford_3plus2_d5/lepton/*.py src/clifford_3plus2_d5/lepton/tests/*.py
    ```

13. Update `pyproject.toml`:
    - `pytest.ini_options.testpaths` → include all three modules' test
      directories.
    - Any `scripts/` references → point to module-internal scripts.
    - Verify the package discovery still works (it should, since the
      package layout stays under `src/clifford_3plus2_d5/`).

### Phase 5: verify everything still works (~1 hour)

14. Run the full test matrix:
    ```bash
    uv sync
    uv run pytest src/clifford_3plus2_d5/obstruction_r10/tests/ -q
    uv run pytest src/clifford_3plus2_d5/lepton/tests/ -q
    uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/ -q
    ```
    All three suites must pass.

15. Spot-check a few reproduction commands:
    ```bash
    uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.gauge_equivalence_check --check
    uv run python -m clifford_3plus2_d5.obstruction_r10.scripts.branching_check --check
    ```

    Note: scripts will now run via `-m` invocations instead of
    `scripts/<file>.py` paths. Document this convention in each
    module's README.

16. Run `uv run ruff check src/` on the whole tree.

### Phase 6: workspace docs + commit (~1 hour)

17. Write `README.md` at workspace level: "this is a workspace; here are
    the three modules and their status."

18. Write `PROJECT_STATUS.md`: per-module status table.

19. Write `obstruction_r10/README.md` and `obstruction_r10/STATUS.md`.

20. Write `lepton/STATUS.md` and `spacetime_qca/STATUS.md` if not already
    present.

21. One large commit with the whole reorg, message:
    > Reorganize repository into module subtrees.
    >
    > - `obstruction_r10/`: R^10 carrier-first obstruction program (the
    >   trunk; frozen for publication).
    > - `lepton/`: Cl(0,10) Pati-Salam sidecar (positive SM gauge content).
    > - `spacetime_qca/`: BCC Weyl walk sidecar (in progress).
    >
    > All tests pass: obstruction_r10 (33), lepton (120+), spacetime_qca
    > (10).
    >
    > Imports updated: lepton now depends on
    > `clifford_3plus2_d5.obstruction_r10.qca.rule_verdict`.

22. Push to the `repo-reorg` branch. Do not merge to main until reviewed.

## Risks and mitigation

| Risk | Mitigation |
|---|---|
| Import path rewrite misses a file | Run full pytest + ruff after each phase. |
| `pyproject.toml` test discovery breaks | Test in a fresh clone after Phase 5. |
| Lepton imports break silently | Lepton's 120-test suite must run green. |
| Spacetime_qca import paths break | Spacetime_qca has no cross-module imports today; should be unaffected. |
| Git history obscured | Use `git mv` consistently; `git log --follow` traces moves. |
| Scripts referenced from external docs/READMEs break | Update README.md's command list; provide a one-shot search-and-replace for `scripts/` → `clifford_3plus2_d5.obstruction_r10.scripts.`. |

## What gets easier after reorg

- Writing the obstruction-r10 paper: the entire trunk lives in one
  subtree; the paper's supplementary material is exactly the subtree's
  contents.
- Adding new sidecars: clear template; add a new module directory at
  `src/clifford_3plus2_d5/<name>/` with its own subtree.
- Onboarding a reader: workspace README points to three modules with
  clear status. No need to navigate trunk-vs-sidecar.
- Publishing the obstruction paper: the trunk is frozen, the sidecars
  can keep moving without polluting the paper's git history.

## What gets harder

- Cross-module imports become longer (e.g.,
  `clifford_3plus2_d5.obstruction_r10.qca.rule_verdict`). This is the
  cost of explicit module boundaries. The lepton depends on the trunk's
  VerdictProfile machinery and that's now visible in every import line —
  honest, but verbose.
- Reproduction commands use `python -m` invocations instead of
  `python scripts/<file>.py`. Slightly less convenient on the command
  line; more standard Python.

## Open decision (please choose before execution)

The reorg has one branching choice:

**Option A — One commit, one branch, full reorg.** Branch `repo-reorg`,
all phases in one commit, then merge. Cleanest history, but if anything
goes wrong it's all rolled back together.

**Option B — Stepwise: shell first, then moves, then imports, then
verify.** One commit per phase on a `repo-reorg` branch. Bisectable if
something breaks; longer history.

**Option C — Defer until after the obstruction-r10 paper is written.**
Argument: the trunk content is what the paper cites; reorganizing now
changes file paths in citations. Mitigation: the paper cites *content*
not *paths*; supplementary material refers to a git tag, not paths.

Recommendation: **Option A**, *after* tagging the current trunk at
`v0.9-pre-reorg` so the pre-reorg state is permanently recoverable.

## Effort estimate

| Phase | Effort |
|---|---|
| Phase 1 (inventory) | 1 hour |
| Phase 2 (shell) | 30 min |
| Phase 3 (moves) | 2-3 hours |
| Phase 4 (imports) | 1-2 hours |
| Phase 5 (verify) | 1 hour |
| Phase 6 (workspace docs) | 1 hour |
| Internal review | 1 hour |
| **Total** | **~8 hours of focused work** |

This is a one-session reorg. It is not a multi-week effort.

## After the reorg

The repo's top-level `README.md` will read:

> Workspace for QCA-derivation modules.
>
> - **obstruction_r10**: R^10 carrier-first obstruction classification.
>   Five propositions; publication in preparation.
> - **lepton**: Cl(0,10) Pati-Salam → Standard Model gauge content with
>   correct one-generation hypercharge table. 1+1D dynamics verified.
> - **spacetime_qca**: 3D BCC Weyl walk with internal tensor lift.
>   Session 20 in implementation.

Each module's directory will be a self-contained project that can be
read, tested, and reasoned about independently.

This is the structure that matches the project's actual state.
