# sim — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`sim` (~1.2k lines, 16 tests, all pass) is **pure shared infrastructure** — a
small package of generic JAX simulation helpers (backend/dtype, periodic lattice
rolls, state allocation, generic links, diagnostics, observable stacking,
loop/`scan` recorded runners, `.npz`/JSON IO, benchmarking, profiling). It makes
**no physics claims** and is correctly scoped: all BCC/Pati-Salam/Wilson physics
is explicitly kept out, in `spacetime_qca`. There is nothing to falsify here;
this review is about code quality and boundary discipline, both of which are
good.

- **Verdict:** clean, well-bounded infrastructure; no physics to assess.
- **Confidence:** n/a (no claims); tests pass, boundary is correctly enforced.

## What it is

STATUS: *"generic JAX simulation helpers that can be reused by sidecars … does
not define Pati-Salam, Spin(10), BCC Dirac dynamics, Wilson plaquette
normalization, or gauge-force physics policy."*

| File | Role |
|---|---|
| `backend.py` | dtype defaults, array conversion, device report, JIT timing |
| `lattice.py` | 3D periodic pull-roll helpers, displacement validation |
| `state.py` | SymPy→NumPy, JAX state alloc, flatten, norm diagnostics |
| `links.py` | generic identity/constant pull-links, site-local gauge transforms |
| `diagnostics.py` | finite-value and state-transition metrics |
| `observables.py` | generic observable stacking/selection/finite checks |
| `runner.py` | physics-agnostic Python-loop and `jax.lax.scan` recorded runners |
| `io.py` | `.npz` + JSON sidecar persistence |
| `benchmarks.py`, `profiling.py` | benchmark wrapper, JSON-safe warm profiling |

## Assessment

- **Boundary discipline is the point, and it's enforced.** The package was
  extracted from `spacetime_qca` (Session 47) precisely to separate generic
  mechanics from physics policy, and the STATUS lists what deliberately stays in
  `spacetime_qca` (BCC kernels, Wilson force, tensor lifts). `spacetime_qca`'s
  own review confirmed import-boundary tests pin this split.
- **Code quality is good.** The `runner.py` I read is clean, typed, generic over
  JAX pytrees, with sensible config validation (`steps ≥ 0`, `record_every > 0`),
  a correct recorded-index scheme (step 0, every `record_every`, and always the
  final step), and a scan runner that advances between record points so
  `record_every > 1` genuinely skips observable work (the Session 58 optimization
  reviewed under `spacetime_qca`). The loop and scan runners share the same
  finite-value bookkeeping.
- **16 tests pass.** Appropriate for an infrastructure layer.

## Gaps

- None of consequence for its remit. The only standing note (from STATUS) is that
  the abstraction is held deliberately minimal — "physics-specific code remains
  in sidecars until a more general architecture is confirmed" — which is the
  right conservative call, not a deficiency.

## Verdict

`sim` is exactly what shared infrastructure should be: small, generic, well-typed,
test-covered, and rigorously kept free of physics policy. It carries no claims to
verify and introduces no risk to the project's physics conclusions; its value is
that the `spacetime_qca` simulation arena rests on a clean, reusable mechanics
layer rather than tangling generic JAX plumbing with BCC/Pati-Salam specifics.
Reviewed for completeness; nothing to flag.
