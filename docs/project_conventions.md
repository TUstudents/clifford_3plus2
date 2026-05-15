# Project Conventions

## Validation Policy

The full `uv run pytest -q` suite is a slow archival check, not the default
commit gate. It contains a mixture of load-bearing algebra checks, active-route
diagnostics, old phase scaffolding, and downstream Spin(10) bookkeeping. A full
pass is useful as a broad regression baseline, but it is not the main signal for
current physics progress.

For normal work, validate the exact surface that changed:

```bash
uv run ruff check .
uv run python scripts/<active_route_check>.py --check
uv run pytest tests/test_<active_route>.py -q
```

Use focused tests when shared code changes. For example, route-local edits
should run the matching route script and test file; algebra-kernel edits should
run the algebra tests plus every route script that depends on the touched
kernel.

Run the full `uv run pytest -q` suite only when one of these is true:

- a shared algebra, QCA verdict, or matrix-solving API changed broadly;
- preparing a cleanup/release baseline;
- intentionally auditing or retiring stale tests;
- checking an unexpected interaction between active and archived routes.

When the full suite is skipped, say so explicitly in the result note instead of
implying it was run.

## Performance Policy

Default route scripts must stay bounded and diagnostic. Expensive exact
searches need explicit caps and should report the reason they stopped, for
example `generated_algebra_closed: false` rather than silently continuing into
an unbounded SymPy solve.

Keep hot algebra kernels separate from CLI/report bookkeeping. Candidate
construction, exact closure, center/J solves, and result formatting should be
separate steps so timing probes can isolate the real bottleneck.

Use the focused performance probe before and after algebra-kernel changes:

```bash
uv run python scripts/perf_probe.py --max-algebra-dim 16
uv run python scripts/perf_probe.py --max-algebra-dim 32
```

Parallel candidate scans should expose a `--jobs` option and preserve
deterministic output ordering. Do not make parallel execution the only path;
`--jobs 1` must remain the reproducible baseline.

## Test Meaning

Treat project checks as three classes:

- **Load-bearing checks**: exact algebra, rule-to-verdict, active route
  certificates, and falsifiers that can change the bridge claim.
- **Route-local checks**: focused scripts and tests for the route under active
  exploration.
- **Bookkeeping/archive checks**: Spin(10) branching, hypercharge counting,
  old phase scaffolding, and documentation consistency checks. These are still
  useful, but they do not by themselves strengthen the QCA bridge.

Do not cite bookkeeping checks as evidence for a microscopic QCA mechanism.
The active claim boundary remains: the rule must produce `J`, the coarse `6+4`
center, and the no-locking condition before Spin(10) accounting is invoked.

## Results Logging

Results pages should record the commands actually run. Historical full-suite
baselines may remain in `docs/results/`, but new entries should distinguish:

```text
ruff: passed
focused route check: passed
focused route tests: passed
full pytest: skipped, slow archival suite
```

This keeps the repository honest about both runtime cost and evidential value.
