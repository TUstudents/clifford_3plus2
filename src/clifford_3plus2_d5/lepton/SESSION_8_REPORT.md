# Session 8 Report — R12 Wall Audit Surface

Status: gap-closing pass after Session 7.

## What Changed

- Added a physical two-site `R12 = R6_left + R6_right` domain-wall model.
- Added complementary site projectors for the physical wall.
- Added an independent right gauge-frame check: the declared right singlet mode
  must match the frame transported by the wall transition.
- Added `lab-b-physical-domain-wall` scan support.
- Added a bounded R12 physical-wall audit. The audit reports exact-center
  status instead of hanging on expensive center solves.

## Current Fast-Path Results

The first physical R12 wall candidate has:

- verdict `domain_wall_candidate`;
- generated algebra dimension `20`;
- center dimension `2` under the known-center contract;
- two central J candidates;
- `load_bearing_domain_wall_candidate = True`;
- `load_bearing_qca_bridge = False`;
- complementary site projectors;
- independently checked right gauge frame.

This is a known-center fast-path verdict, not an exact-center result.

## Audit Semantics

The new physical audit reports:

- `passed_exact` when actual center computation proves center dimension `2`
  and central idempotent ranks `(0, 12)`;
- `failed` when exact center computation completes but disagrees;
- `not_solved_timeout` when the exact center computation exceeds the configured
  timeout.

Timeout is not a pass. It is a controlled unresolved status.

## Exact Audit Result

With the default 30-second audit budget, the first physical R12 candidate
completed exact center computation and failed the intended center test:

- actual center dimension: `8`;
- expected center dimension: `2`;
- exact audit status: `failed`.

So the current physical R12 wall implementation closes the modeling gap but
does **not** yet close the bridge gap. The extra center must be collapsed by a
stronger physical layer or by a different wall construction.

## Remaining Gaps

- R12 exact-center audit currently falsifies the first physical candidate:
  the center is too large.
- Lab A Bloch is still a simplified clock-phase sampled model, not a full
  Laurent hopping QCA.
- No R10 regression profile is implemented because this module is currently
  constrained to the `lepton/` folder.
- Strict and structural Lab B still use known-center/idempotent shortcuts for
  regression speed.

## Next Recommended Step

If the physical R12 wall remains the main path, optimize or specialize the R12
center computation so `passed_exact` can be obtained without a broad symbolic
center solve.
