# Session 61 - Opt-In Gauss Projection Control

Session 61 adds a bounded, opt-in Gauss-descent projection around the coupled
fermion/gauge/Higgs prototype.  It is a diagnostic control path, not a full
constraint solver.

## Implemented

- `jax_patisalam_apply_gauss_descent_projection` repeatedly applies the
  existing combined fermion/Higgs Gauss-descent step to gauge momenta.
- `jax_patisalam_fermion_gauge_higgs_step` accepts:
  - `gauss_projection_steps=0`
  - `gauss_projection_step_size=0.0`
- Projection is applied after the final fermion/Higgs fields are available, so
  the residual uses the final state of the coupled step.
- Projection is skipped by default and when the physical step size is zero.
- Scaling, lab-runner, and main simulator configs expose the same controls.
- Simulator CLIs accept `--gauss-projection-steps` and
  `--gauss-projection-step-size`.

## Validation

- Projection-off is an exact no-op.
- Negative projection controls raise `ValueError`.
- A tiny `(2, 1, 1)` Higgs-sourced residual decreases after one projection
  step.
- The projected coupled step remains finite on the tiny test fixture.

## Interpretation

This closes the immediate control-loop gap left by Session 60: the simulator
can now optionally reduce the combined Gauss residual after an interacting
step.  The method is still gradient-descent control, not an exact lattice
Gauss-law projector.
