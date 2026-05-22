# Session 62 - Bounded Gauss Projection Sweep

Session 62 adds a tiny-lattice sweep that compares projection disabled against
bounded projection-enabled trajectories.  It is an audit helper for choosing
projection settings, not a broad performance profile.

## Implemented

- `GaussProjectionSweepCase` stores one trajectory and its projection controls.
- `GaussProjectionSweep` stores the baseline, projected cases, best final
  Gauss residual norm, best max sampled Gauss residual norm, and all-finite
  status.
- `jax_gauss_projection_sweep` runs:
  - projection off;
  - projection on for selected bounded step counts;
  - small deterministic trajectories using existing scaling diagnostics.

## Validation

- Zero-step sweeps have stable shape and finite scalar summaries.
- A tiny `(2, 1, 1)` one-step case with Higgs charge shows lower final Gauss
  residual after one projection step.
- The helper rejects negative projection controls.

## Interpretation

The projection path now has a bounded measurement harness.  Projection remains
off by default; the sweep is the tool for deciding when to enable it in lab
runs and what tiny settings are numerically useful.
