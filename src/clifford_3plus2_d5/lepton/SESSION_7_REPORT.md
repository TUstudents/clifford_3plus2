# Session 7 Report: Leptonic Bridge Laboratory

## Scope

The lepton lab is implemented as an experimental, additive path under
`src/clifford_3plus2_d5/lepton/`. It does not migrate the legacy R10
`rule_to_verdict()` call sites. The v2 verdict path is profile-driven and is
used only by the lepton laboratory.

## Implemented Pieces

- Lab A, R4: clock-plane primitive closure to `M_2(C)`.
- Lab A sampled Bloch scan: real sampled clock-plane symbols on R4.
- Lab A same-spectrum wall scan: tiered split-reassignment intertwiners.
- Lab B, R6 strict: fixed global `(2,4)` split regression.
- Lab B structural wall: non-translation-invariant regression preserving the
  same global split.
- Lab B domain wall: single-carrier internal split-reassignment model with
  side-local gauge data.

## Results

Lab A positives, when present, are reported as
`clock_plane_closure_candidate`. This is intentionally not called a J-blind
bridge: the pair-rotation primitives encode the clock planes.

Lab B strict reproduces the Route 1 obstruction at R6:

- algebra dimension `10`;
- center dimension `4`;
- four central J candidates;
- verdict `candidate_only_j_not_forced`;
- commutant verdict `passed_multiple_aligned`.

Lab B structural wall reproduces the same multiple-J outcome, with
split-reassignment transitions.

Lab B domain wall produces an internal side-local candidate:

- verdict `domain_wall_candidate`;
- algebra dimension `18`;
- actual center dimension `2`;
- two central J candidates;
- commutant verdict `passed_unique_pm`;
- `load_bearing_qca_bridge = False`;
- `load_bearing_domain_wall_candidate = True`.

## Non-Claims

The domain-wall candidate is not the original carrier-first global `(2,4)`
bridge. It uses a single R6 carrier with internal split reassignment. It is
also not the physical two-site R12 wall. The result should be read as evidence
that the side-local sign-lock mechanism is worth testing in the physical wall
model.

## Recommended Next Step

Build the physical two-site R12 domain wall. The R10 lift should remain
deferred until the R12 wall reproduces the internal R6 sign-lock with explicit
site-local side projectors.
